"""
ATRACTOR DE LORENZ en 3D (caos determinista) + ANIMACIÓN.

Aquí NO hay azar en la dinámica: todos los puntos obedecen la misma regla fija

    dx/dt = SIGMA * (y - x)
    dy/dt = x * (RHO - z) - y
    dz/dt = x * y - BETA * z

Los puntos arrancan casi en el MISMO sitio (una bolita diminuta). Por la
sensibilidad a las condiciones iniciales, esa bolita se va estirando y se
reparte por todo el atractor. Cada punto deja un rastro (línea) que se
desvanece, coloreado según su velocidad.

Requisitos:
    pip3 install matplotlib numpy pillow
    (Para .mp4 necesitas ffmpeg instalado en el sistema)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------
# 1. PARÁMETROS
# ---------------------------------------------------------------------

# --- El sistema (valores clásicos de Lorenz; prueba RHO = 10, 14, 24, 28...) ---
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

# --- Los puntos ---
N_PUNTOS = 200
PUNTO_INICIAL = (-8.0, 8.0, 27.0)   # cerca del atractor
DISPERSION_INICIAL = 1e-3           # tamaño de la bolita inicial (más pequeño = más espectacular)
SEMILLA = 42                        # solo afecta a dónde cae cada punto dentro de la bolita

# --- Simulación ---
DT = 0.01                    # paso de integración (RK4)
PASOS_POR_FRAME = 4          # pasos de simulación entre un frame y el siguiente
LARGO_ESTELA = 30            # cuántos frames hacia atrás dura el rastro

# --- Animación ---
NOMBRE_ARCHIVO_SALIDA = "lorenz_3d.mp4"   # .mp4 (requiere ffmpeg) o .gif (sin ffmpeg, pero pesado)
DURACION_SEGUNDOS = 14
FPS = 25
DPI = 100
FONDO_OSCURO = True
CMAP = "turbo"               # color según velocidad
MOSTRAR_PUNTOS = True        # dibuja también la "cabeza" de cada rastro

# --- Cámara ---
ELEVACION = 20
ROTAR_CAMARA = True
GIRO_TOTAL_GRADOS = 120      # cuánto gira en toda la animación
AZIMUT_INICIAL = 30
ACERCAR_CAMARA = True
ZOOM_INICIAL = 1.0           # 1.0 = encuadre base
ZOOM_FINAL = 0.7             # menor que 1 = más cerca


# ---------------------------------------------------------------------
# 2. EL SISTEMA DINÁMICO Y SU INTEGRACIÓN
# ---------------------------------------------------------------------

def lorenz(p):
    """Velocidad en cada punto. p tiene forma (N, 3)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], axis=1)


def paso_rk4(p, dt):
    """Un paso de Runge-Kutta 4 (mucho más preciso que Euler)."""
    k1 = lorenz(p)
    k2 = lorenz(p + dt / 2 * k1)
    k3 = lorenz(p + dt / 2 * k2)
    k4 = lorenz(p + dt * k3)
    return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simular(num_frames):
    """Calcula de antemano posiciones (T,N,3) y velocidades (T,N) de todos los puntos."""
    rng = np.random.default_rng(SEMILLA)
    p = np.array(PUNTO_INICIAL) + rng.normal(0, DISPERSION_INICIAL, (N_PUNTOS, 3))

    pos = np.empty((num_frames, N_PUNTOS, 3))
    vel = np.empty((num_frames, N_PUNTOS))
    for i in range(num_frames):
        pos[i] = p
        vel[i] = np.linalg.norm(lorenz(p), axis=1)
        for _ in range(PASOS_POR_FRAME):
            p = paso_rk4(p, DT)
    return pos, vel


# ---------------------------------------------------------------------
# 3. ANIMACIÓN
# ---------------------------------------------------------------------

def crear_animacion():
    num_frames = int(DURACION_SEGUNDOS * FPS)
    pos, vel = simular(num_frames)

    # Normalizar velocidades para el color (percentiles para evitar extremos)
    v_min, v_max = np.percentile(vel, [2, 98])
    vel_norm = np.clip((vel - v_min) / (v_max - v_min + 1e-12), 0, 1)
    cmap = plt.get_cmap(CMAP)

    # Desvanecimiento del rastro: la cola es transparente y fina, la cabeza opaca y gruesa
    n_seg = LARGO_ESTELA - 1
    fade = np.linspace(0.0, 1.0, n_seg) ** 1.5
    grosor = np.repeat((0.2 + 1.6 * np.linspace(0, 1, n_seg))[:, None], N_PUNTOS, axis=1).ravel()

    fondo = "black" if FONDO_OSCURO else "white"
    fig = plt.figure(figsize=(7, 7), facecolor=fondo)
    ax = fig.add_subplot(111, projection="3d", facecolor=fondo)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    centro = np.array([0.0, 0.0, 25.0])
    radio_base = 32.0

    # (se inicia con segmentos de relleno; una colección vacía da error en add_collection3d)
    coleccion = Line3DCollection(np.zeros((n_seg * N_PUNTOS, 2, 3)), linewidths=grosor)
    ax.add_collection3d(coleccion)

    puntos = None
    if MOSTRAR_PUNTOS:
        color_punto = "white" if FONDO_OSCURO else "black"
        (puntos,) = ax.plot([], [], [], linestyle="", marker="o", markersize=2.2,
                            color=color_punto)

    def actualizar(i):
        # --- Rastro: los últimos LARGO_ESTELA frames (repitiendo el primero al inicio) ---
        idx = np.clip(np.arange(i - LARGO_ESTELA + 1, i + 1), 0, None)
        rastro = pos[idx]                                   # (L, N, 3)
        segmentos = np.stack([rastro[:-1], rastro[1:]], axis=2)   # (L-1, N, 2, 3)
        coleccion.set_segments(segmentos.reshape(-1, 2, 3))

        colores = cmap(vel_norm[idx[1:]])                   # (L-1, N, 4)
        colores[..., 3] = fade[:, None]
        coleccion.set_color(colores.reshape(-1, 4))

        if puntos is not None:
            puntos.set_data_3d(pos[i, :, 0], pos[i, :, 1], pos[i, :, 2])

        # --- Cámara: giro y zoom suaves ---
        t = i / max(num_frames - 1, 1)
        azim = AZIMUT_INICIAL + (GIRO_TOTAL_GRADOS * t if ROTAR_CAMARA else 0)
        ax.view_init(elev=ELEVACION, azim=azim)

        zoom = ZOOM_INICIAL + (ZOOM_FINAL - ZOOM_INICIAL) * t if ACERCAR_CAMARA else ZOOM_INICIAL
        r = radio_base * zoom
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
    print("Generando animación... esto puede tardar varios minutos.")
    anim.save(NOMBRE_ARCHIVO_SALIDA, writer=writer, dpi=DPI)
    print(f"¡Listo! Guardado como '{NOMBRE_ARCHIVO_SALIDA}'.")


# ---------------------------------------------------------------------
# 4. EJECUCIÓN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    fig, anim = crear_animacion()
    exportar_animacion(anim)