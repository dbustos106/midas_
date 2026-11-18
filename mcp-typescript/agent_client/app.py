import asyncio
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from flask import Flask, request
from langchain_core.documents import Document
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import create_model, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent")

# ============================================================
#  CONFIGURACIÓN
# ============================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MCP_URL = os.getenv("MCP_URL", "http://mcp-server:3000/mcp")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "llama3.2")
WORKER_MODEL = os.getenv("WORKER_MODEL", "llama3.2")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # nomic-embed-text = 768
RAG_K = int(os.getenv("RAG_K", "4"))
SEED_SAMPLE = os.getenv("SEED_SAMPLE", "true").lower() == "true"

MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "24"))  # a partir de aquí se resume
KEEP_MESSAGES = int(os.getenv("KEEP_MESSAGES", "8"))  # mensajes recientes que se conservan
MAX_HOPS = int(os.getenv("MAX_HOPS", "6"))  # tope de saltos del supervisor por turno


@dataclass
class AgentSpec:
    description: str  # lo lee el supervisor para decidir a quién delegar
    prompt: str  # system prompt del agente
    tools: list[str] = field(default_factory=list)  # nombres de tools (MCP o locales)
    model: str = WORKER_MODEL


AGENTS: dict[str, AgentSpec] = {
    "math_agent": AgentSpec(
        description="Hace cálculos y sumas de números.",
        prompt="Eres un experto en cálculo. Usa la herramienta add para sumar.",
        tools=["add"],
    ),
    "weather_agent": AgentSpec(
        description="Consulta el clima de una ciudad.",
        prompt="Eres un meteorólogo. Usa la herramienta weather para consultar el clima.",
        tools=["weather"],
    ),
    "rag_agent": AgentSpec(
        description=(
            "Responde preguntas sobre los documentos de la base de conocimiento "
            "(el propio stack, manuales, notas ingestadas)."
        ),
        prompt=(
            "Responde usando SIEMPRE la herramienta search_docs. "
            "Basa tu respuesta solo en los fragmentos recuperados y cita la fuente entre corchetes. "
            "Si no hay información relevante, dilo claramente; no inventes."
        ),
        tools=["search_docs"],
    ),
}

SAMPLE_SOURCE = "guia-del-stack"
SAMPLE_DOC = """\
Guía del stack de agentes

El stack se levanta con Docker Compose y tiene cinco servicios: postgres, qdrant, ollama,
mcp-server y agent-client.

Postgres guarda la memoria de las conversaciones. LangGraph usa un checkpointer que persiste
el estado de cada hilo (thread_id), de modo que una conversación puede continuar aunque el
contenedor se reinicie.

Qdrant es la base de datos vectorial del RAG. Escucha en el puerto 6333 y la colección se llama
knowledge. Los embeddings se calculan con el modelo nomic-embed-text de Ollama, que genera
vectores de 768 dimensiones.

Ollama sirve los modelos de lenguaje en el puerto 11434. Un contenedor auxiliar descarga los
modelos al arrancar. Con el perfil cpu se usa el servicio ollama y con el perfil gpu-nvidia el
servicio ollama-gpu.

El mcp-server está escrito en TypeScript y escucha en el puerto 3000, en la ruta /mcp. Expone
las herramientas add (suma dos números) y weather (clima de una ciudad).

El agent-client es una aplicación Flask que escucha en el puerto 8002 del host. El endpoint
POST /chat recibe un mensaje y devuelve la respuesta del agente. El endpoint POST /ingest
añade documentos nuevos a la base de conocimiento.
"""


# ============================================================
#  ESTADO
# ============================================================


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str  # resumen de la conversación antigua
    trace: list[str]  # agentes que han actuado en este turno (sin reducer: se reinicia por turno)


def text(m) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def with_context(state: State, system: str) -> list:
    """System prompt + resumen previo + mensajes vigentes."""
    if state.get("summary"):
        system += f"\n\nResumen de la conversación anterior:\n{state['summary']}"
    return [SystemMessage(system), *state["messages"]]


def llm(model: str) -> ChatOllama:
    return ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0)


# ============================================================
#  RAG (Qdrant + embeddings de Ollama)
# ============================================================

qdrant: QdrantClient | None = None
store: QdrantVectorStore | None = None
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


async def ingest_text(content: str, source: str) -> int:
    """Trocea, calcula embeddings y guarda. Reemplaza lo que hubiera de la misma fuente."""
    await asyncio.to_thread(
        qdrant.delete,
        COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        ),
    )
    chunks = splitter.split_text(content)
    docs = [
        Document(page_content=c, metadata={"source": source, "chunk": i})
        for i, c in enumerate(chunks)
    ]
    await store.aadd_documents(docs)
    return len(docs)


@tool
async def search_docs(query: str) -> str:
    """Busca en la base de conocimiento (documentos ingestados) y devuelve los fragmentos más relevantes."""
    docs = await store.asimilarity_search(query, k=RAG_K)
    if not docs:
        return "No encontré información relevante en la base de conocimiento."
    return "\n\n".join(f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs)


LOCAL_TOOLS = [search_docs]


async def seed_when_ready():
    """Carga el documento de prueba cuando Ollama ya tiene el modelo de embeddings."""
    for _ in range(90):  # ~15 min
        try:
            n = (await asyncio.to_thread(qdrant.count, COLLECTION)).count
            if n > 0:
                return
            chunks = await ingest_text(SAMPLE_DOC, SAMPLE_SOURCE)
            log.info("Documento de prueba cargado (%d fragmentos)", chunks)
            return
        except Exception as e:
            log.info("Carga del documento de prueba pendiente: %s", e)
            await asyncio.sleep(10)


# ============================================================
#  NODOS
# ============================================================


def make_memory_node(model: ChatOllama):
    """Gestiona el contexto: resume los mensajes viejos y los borra del estado.
    (La persistencia en sí la hace el checkpointer de Postgres, no este nodo.)"""

    async def memory(state: State):
        msgs = state["messages"]
        if len(msgs) <= MAX_MESSAGES:
            return {}

        # Cortamos en un mensaje humano para no separar tool_calls de sus resultados.
        cut = len(msgs) - KEEP_MESSAGES
        while cut < len(msgs) and msgs[cut].type != "human":
            cut += 1
        if cut <= 0 or cut >= len(msgs):
            return {}

        old = msgs[:cut]
        transcript = "\n".join(f"{m.type}: {text(m)}" for m in old if text(m))
        res = await model.ainvoke(
            [
                SystemMessage("Resumes conversaciones de forma fiel y breve, en el idioma original."),
                HumanMessage(
                    f"Resumen previo:\n{state.get('summary') or '(vacío)'}\n\n"
                    f"Mensajes nuevos:\n{transcript}\n\nEscribe el resumen actualizado."
                ),
            ]
        )
        log.info("Resumiendo %d mensajes", len(old))
        return {
            "summary": text(res),
            "messages": [RemoveMessage(id=m.id) for m in old],
        }

    return memory


def make_supervisor(model: ChatOllama, names: list[str]):
    # Cada agente se expone como una "función": llamarla = delegar en ese agente.
    handoffs = [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": a.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n, a in AGENTS.items()
    ]
    router = model.bind_tools(handoffs)

    system = (
        "Eres el supervisor de un equipo de agentes. No respondes la pregunta tú: solo delegas.\n"
        "- Si la última pregunta del usuario necesita un cálculo, el clima de una ciudad "
        "o buscar en los documentos, llama a la función del agente que corresponde.\n"
        "- Si es un saludo o charla, o los resultados que se necesitan ya están en la "
        "conversación, no llames a ninguna función y responde solo: FINISH."
    )

    async def supervisor(state: State):
        trace = state.get("trace", [])
        if len(trace) >= MAX_HOPS:
            return Command(goto="responder")

        actuaron = ", ".join(trace) if trace else "ninguno"
        prompt = system + f"\n\nAgentes que ya actuaron en este turno: {actuaron}."

        reply = await router.ainvoke(with_context(state, prompt))
        calls = getattr(reply, "tool_calls", None) or []
        nxt = calls[0]["name"] if calls else "FINISH"
        if nxt not in names:
            nxt = "FINISH"

        log.info("Supervisor -> %s", nxt)
        if nxt == "FINISH" or nxt in trace:
            return Command(goto="responder")
        return Command(goto=nxt, update={"trace": trace + [nxt]})

    return supervisor


def make_worker(name: str, spec: AgentSpec, all_tools: list):
    tools = [t for t in all_tools if t.name in spec.tools]
    missing = set(spec.tools) - {t.name for t in tools}
    if missing:
        log.warning("%s: tools no encontradas: %s", name, missing)

    base = llm(spec.model)
    bound = base.bind_tools(tools) if tools else base

    async def worker(state: State):
        reply = await bound.ainvoke(with_context(state, spec.prompt))
        reply.name = name
        return {"messages": [reply]}

    def route(state: State) -> str:
        last = state["messages"][-1]
        return f"{name}_tools" if getattr(last, "tool_calls", None) else "supervisor"

    return worker, route, (ToolNode(tools) if tools else None)


def make_responder(model: ChatOllama):
    system = (
        "Redacta la respuesta final para el usuario, en su mismo idioma, usando únicamente "
        "lo que hay en la conversación (incluidos los resultados de herramientas y las fuentes "
        "citadas). Sé claro y breve."
    )

    async def responder(state: State):
        msgs = [
            *with_context(state, system),
            HumanMessage("Con lo anterior, escribe ahora la respuesta final para el usuario."),
        ]
        reply = await model.ainvoke(msgs)
        return {"messages": [reply]}

    return responder


# ============================================================
#  GRAFO
# ============================================================


def build_graph(all_tools: list, checkpointer):
    sup_llm = llm(SUPERVISOR_MODEL)
    g = StateGraph(State)

    g.add_node("memory", make_memory_node(sup_llm))
    g.add_node(
        "supervisor",
        make_supervisor(sup_llm, list(AGENTS)),
        destinations=(*AGENTS, "responder"),
    )
    g.add_node("responder", make_responder(sup_llm))

    for name, spec in AGENTS.items():
        worker, route, tool_node = make_worker(name, spec, all_tools)
        g.add_node(name, worker)
        if tool_node is not None:
            g.add_node(f"{name}_tools", tool_node)
            g.add_edge(f"{name}_tools", name)
            g.add_conditional_edges(name, route, [f"{name}_tools", "supervisor"])
        else:
            g.add_edge(name, "supervisor")

    g.add_edge(START, "memory")
    g.add_edge("memory", "supervisor")
    g.add_edge("responder", END)
    return g.compile(checkpointer=checkpointer)


# ============================================================
#  ARRANQUE (event loop propio + Postgres + Qdrant + MCP)
# ============================================================

loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()


def run(coro, timeout: float = 300):
    """Ejecuta una corutina en el event loop de fondo desde código síncrono (Flask)."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


graph = None
saver: AsyncPostgresSaver | None = None
_seed_task = None


async def init():
    global graph, saver, qdrant, store, _seed_task

    # --- Postgres: memoria de conversaciones ---
    conninfo = make_conninfo(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()  # crea las tablas checkpoints* si no existen

    # --- Qdrant: colección del RAG ---
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    for attempt in range(30):
        try:
            if not qdrant.collection_exists(COLLECTION):
                qdrant.create_collection(
                    COLLECTION,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
            break
        except Exception as e:
            log.warning("Qdrant no disponible (%s), reintento %d/30", e, attempt + 1)
            await asyncio.sleep(2)
    else:
        raise RuntimeError(f"No pude conectar con Qdrant en {QDRANT_URL}")

    store = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION,
        embedding=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST),
        validate_collection_config=False,
    )

    # --- MCP: herramientas del servidor TypeScript ---
    client = MultiServerMCPClient(
        {"demo": {"transport": "streamable_http", "url": MCP_URL}}
    )
    for attempt in range(30):
        try:
            mcp_tools = await client.get_tools()
            break
        except Exception as e:  # el MCP puede tardar en estar listo
            log.warning("MCP no disponible (%s), reintento %d/30", e, attempt + 1)
            await asyncio.sleep(2)
    else:
        raise RuntimeError(f"No pude conectar con el MCP en {MCP_URL}")

    all_tools = [*mcp_tools, *LOCAL_TOOLS]
    log.info("Tools disponibles: %s", [t.name for t in all_tools])
    graph = build_graph(all_tools, saver)

    if SEED_SAMPLE:
        _seed_task = asyncio.create_task(seed_when_ready())


run(init(), timeout=180)

# ============================================================
#  API FLASK
# ============================================================

app = Flask(__name__)
app.json.ensure_ascii = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return {"error": "El campo 'message' es obligatorio"}, 400

    thread_id = body.get("thread_id") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    result = run(
        graph.ainvoke({"messages": [HumanMessage(message)], "trace": []}, config)
    )
    return {
        "thread_id": thread_id,
        "reply": text(result["messages"][-1]),
        "trace": result.get("trace", []),
    }


@app.get("/history/<thread_id>")
def history(thread_id: str):
    state = run(graph.aget_state({"configurable": {"thread_id": thread_id}}))
    values = state.values or {}
    return {
        "thread_id": thread_id,
        "summary": values.get("summary", ""),
        "messages": [
            {"role": m.type, "agent": getattr(m, "name", None), "content": text(m)}
            for m in values.get("messages", [])
            if m.type in ("human", "ai") and text(m)
        ],
    }


@app.delete("/chat/<thread_id>")
def reset(thread_id: str):
    run(saver.adelete_thread(thread_id))
    return {"deleted": thread_id}


@app.post("/ingest")
def ingest():
    if "file" in request.files:
        f = request.files["file"]
        source = f.filename or "archivo"
        content = f.read().decode("utf-8", errors="ignore")
    else:
        body = request.get_json(silent=True) or {}
        content = body.get("text") or ""
        source = body.get("source") or "manual"
    if not content.strip():
        return {"error": "Envía 'text' (JSON) o un archivo en 'file'"}, 400
    chunks = run(ingest_text(content, source))
    return {"source": source, "chunks": chunks}


@app.get("/search")
def search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return {"error": "Falta el parámetro 'q'"}, 400
    hits = run(store.asimilarity_search_with_score(q, k=RAG_K))
    return {
        "query": q,
        "results": [
            {"score": round(float(s), 4), "source": d.metadata.get("source"), "text": d.page_content}
            for d, s in hits
        ],
    }
