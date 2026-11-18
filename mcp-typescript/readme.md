curl localhost:8002/health
docker compose logs agent-client | tail -20

curl --get "http://localhost:8002/search" \
  --data-urlencode "q=qué puerto usa qdrant"

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hola, ¿cómo estás?","thread_id":"b1"}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Cuánto es 17 + 25?","thread_id":"b2"}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Qué clima hace en Cali?","thread_id":"b2"}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿En qué puerto escucha el servidor MCP y qué herramientas expone?","thread_id":"b3"}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Suma 3 y 4, y dime el clima en Bogotá","thread_id":"b4"}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Me llamo Andrés","thread_id":"b5"}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Cómo me llamo?","thread_id":"b5"}'

curl localhost:8002/history/b5

curl -X POST localhost:8002/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"notas","text":"El proyecto Orion tiene como líder a Marta Ruiz. La fecha de entrega es el 15 de marzo. El presupuesto aprobado es de 40 millones."}'

curl --max-time 300 -X POST localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"¿Quién lidera el proyecto Orion y cuándo se entrega?","thread_id":"b7"}'

curl -X DELETE localhost:8002/chat/b5
curl localhost:8002/history/b5

