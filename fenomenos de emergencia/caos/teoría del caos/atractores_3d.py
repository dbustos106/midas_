"""
ATRACTORES EXTRAÑOS en 3D (caos determinista) + ANIMACIÓN LARGA en MP4.

Igual que con Lorenz: NO hay azar en la dinámica. Todos los puntos obedecen la
misma regla fija (un sistema de 3 ecuaciones diferenciales). Los puntos
arrancan casi en el mismo sitio y, por la sensibilidad a las condiciones
iniciales, la nube se estira, se pliega y se reparte por todo el atractor.

Atractores incluidos (cambia ATRACTOR más abajo):
    "aizawa"     - una esfera con un "tubo" que la atraviesa por el centro
    "rossler"    - una cinta espiral que sube y se pliega sobre sí misma
    "thomas"     - simetría cíclica, una maraña de lazos entrelazados
    "chen"       - parecido a Lorenz pero con alas más abiertas y asimétricas

Requisitos:
    pip3 install matplotlib numpy
    ffmpeg instalado en el sistema (brew install ffmpeg / sudo apt install ffmpeg)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------
# 1. LOS SISTEMAS DINÁMICOS  (p tiene forma (N, 3): una fila por punto)
# ---------------------------------------------------------------------

def aizawa(p, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z ** 3 / 3 - (x * x + y * y) * (1 + e * z) + f * z * x ** 3,
    ], axis=1)


def rossler(p, a=0.2, b=0.2, c=5.7):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=1)


def thomas(p, b=0.208186):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z], axis=1)


def chen(p, a=35.0, b=3.0, c=28.0):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z], axis=1)


# Cada atractor: ecuaciones, punto de partida, paso de integración y
# cuántos pasos se simulan entre un frame y el siguiente (cada sistema
# tiene su propia "velocidad"), tamaño de la bolita inicial y largo del rastro.
ATRACTORES = {
    "aizawa":  {"f": aizawa,  "inicio": (0.1, 0.0, 0.0),   "dt": 0.01,  "pasos": 3, "dispersion": 0.03,  "estela": 45, "cmap": "turbo"},
    "rossler": {"f": rossler, "inicio": (1.0, 1.0, 0.0),   "dt": 0.02,  "pasos": 4, "dispersion": 0.2,   "estela": 40, "cmap": "plasma"},
    "thomas":  {"f": thomas,  "inicio": (0.1, 0.0, 0.0),   "dt": 0.05,  "pasos": 3, "dispersion": 0.05,  "estela": 40, "cmap": "viridis"},
    "chen":    {"f": chen,    "inicio": (-3.0, 2.0, 20.0), "dt": 0.002, "pasos": 8, "dispersion": 1e-3,  "estela": 40, "cmap": "cool"},
}

# Elige cuál animar: "aizawa", "rossler", "thomas" o "chen"
ATRACTOR = "aizawa"


# ---------------------------------------------------------------------
# 2. PARÁMETROS
# ---------------------------------------------------------------------

N_PUNTOS = 150
SEMILLA = 42                     # solo decide dónde cae cada punto dentro de la bolita inicial

# --- Animación (más larga) ---
NOMBRE_ARCHIVO_SALIDA = "atractor_3d.mp4"
DURACION_SEGUNDOS = 45
FPS = 25
DPI = 100
FONDO_OSCURO = True
MOSTRAR_PUNTOS = True            # dibuja la "cabeza" de cada rastro

# --- Cámara: da VUELTAS giros completos y "respira" (se acerca y se aleja) ---
VUELTAS = 1
AZIMUT_INICIAL = 30
ELEVACION_BASE = 20
ELEVACION_OSCILACION = 12        # grados que sube y baja la cámara (0 = elevación fija)
ZOOM_MAX = 1.0                   # encuadre amplio (al inicio y al final)
ZOOM_MIN = 0.65                  # encuadre más cercano (a la mitad del video)


# ---------------------------------------------------------------------
# 3. INTEGRACIÓN (Runge-Kutta 4)
# ---------------------------------------------------------------------

def paso_rk4(f, p, dt):
    k1 = f(p)
    k2 = f(p + dt / 2 * k1)
    k3 = f(p + dt / 2 * k2)
    k4 = f(p + dt * k3)
    return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simular(cfg, num_frames):
    """Posiciones (T,N,3) y velocidades (T,N) de todos los puntos en todos los frames."""
    rng = np.random.default_rng(SEMILLA)
    p = np.array(cfg["inicio"]) + rng.normal(0, cfg["dispersion"], (N_PUNTOS, 3))

    pos = np.empty((num_frames, N_PUNTOS, 3))
    vel = np.empty((num_frames, N_PUNTOS))
    for i in range(num_frames):
        pos[i] = p
        vel[i] = np.linalg.norm(cfg["f"](p), axis=1)
        for _ in range(cfg["pasos"]):
            p = paso_rk4(cfg["f"], p, cfg["dt"])
    return pos, vel


# ---------------------------------------------------------------------
# 4. ANIMACIÓN
# ---------------------------------------------------------------------

def crear_animacion():
    cfg = ATRACTORES[ATRACTOR]
    num_frames = int(DURACION_SEGUNDOS * FPS)
    largo = cfg["estela"]
    pos, vel = simular(cfg, num_frames)

    # Encuadre automático según el tamaño real del atractor
    todos = pos.reshape(-1, 3)
    lo, hi = np.percentile(todos, 0.5, axis=0), np.percentile(todos, 99.5, axis=0)
    centro = (lo + hi) / 2
    radio_base = (hi - lo).max() / 2 * 1.1

    # Color según velocidad (percentiles para evitar extremos)
    v_min, v_max = np.percentile(vel, [2, 98])
    vel_norm = np.clip((vel - v_min) / (v_max - v_min + 1e-12), 0, 1)
    cmap = plt.get_cmap(cfg["cmap"])

    # Rastro: cola transparente y fina, cabeza opaca y más gruesa
    n_seg = largo - 1
    fade = np.linspace(0.0, 1.0, n_seg) ** 1.5
    grosor = np.repeat((0.2 + 1.6 * np.linspace(0, 1, n_seg))[:, None], N_PUNTOS, axis=1).ravel()

    fondo = "black" if FONDO_OSCURO else "white"
    fig = plt.figure(figsize=(7, 7), facecolor=fondo)
    ax = fig.add_subplot(111, projection="3d", facecolor=fondo)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    # (se inicia con segmentos de relleno; una colección vacía da error en add_collection3d)
    coleccion = Line3DCollection(np.zeros((n_seg * N_PUNTOS, 2, 3)), linewidths=grosor)
    ax.add_collection3d(coleccion)

    puntos = None
    if MOSTRAR_PUNTOS:
        color_punto = "white" if FONDO_OSCURO else "black"
        (puntos,) = ax.plot([], [], [], linestyle="", marker="o", markersize=2.2, color=color_punto)

    def actualizar(i):
        # --- Rastro: los últimos frames (repitiendo el primero al inicio) ---
        idx = np.clip(np.arange(i - largo + 1, i + 1), 0, None)
        rastro = pos[idx]                                          # (L, N, 3)
        segmentos = np.stack([rastro[:-1], rastro[1:]], axis=2)    # (L-1, N, 2, 3)
        coleccion.set_segments(segmentos.reshape(-1, 2, 3))

        colores = cmap(vel_norm[idx[1:]])                          # (L-1, N, 4)
        colores[..., 3] = fade[:, None]
        coleccion.set_color(colores.reshape(-1, 4))

        if puntos is not None:
            puntos.set_data_3d(pos[i, :, 0], pos[i, :, 1], pos[i, :, 2])

        # --- Cámara: giro continuo, elevación suave y zoom que "respira" ---
        t = i / num_frames
        ax.view_init(
            elev=ELEVACION_BASE + ELEVACION_OSCILACION * np.sin(2 * np.pi * t),
            azim=AZIMUT_INICIAL + 360 * VUELTAS * t,
        )
        respiracion = (1 - np.cos(2 * np.pi * t)) / 2              # 0 -> 1 -> 0
        r = radio_base * (ZOOM_MAX - (ZOOM_MAX - ZOOM_MIN) * respiracion)
        ax.set_xlim(centro[0] - r, centro[0] + r)
        ax.set_ylim(centro[1] - r, centro[1] + r)
        ax.set_zlim(centro[2] - r, centro[2] + r)
        return ()

    anim = animation.FuncAnimation(fig, actualizar, frames=num_frames, interval=1000 / FPS)
    return fig, anim


def exportar_animacion(anim):
    if NOMBRE_ARCHIVO_SALIDA.lower().endswith(".mp4"):
        # yuv420p = formato de color que entienden todos los móviles (WhatsApp incluido)
        # faststart = el video empieza a reproducirse sin esperar a que termine de cargar
        writer = animation.FFMpegWriter(
            fps=FPS, bitrate=4000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
        )
    else:
        writer = animation.PillowWriter(fps=FPS)
    print(f"Atractor '{ATRACTOR}': generando {DURACION_SEGUNDOS} s de video... "
          "esto puede tardar varios minutos.")
    anim.save(NOMBRE_ARCHIVO_SALIDA, writer=writer, dpi=DPI)
    print(f"¡Listo! Guardado como '{NOMBRE_ARCHIVO_SALIDA}'.")


# ---------------------------------------------------------------------
# 5. EJECUCIÓN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    fig, anim = crear_animacion()
    exportar_animacion(anim)
