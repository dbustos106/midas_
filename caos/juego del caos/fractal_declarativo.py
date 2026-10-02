"""
Fractal 3D hecho de LÍNEAS (IFS aplicado a segmentos) + ANIMACIÓN DE CRECIMIENTO.

Idea: en vez del "juego del caos" (un punto que salta al azar), partimos de
uno o varios segmentos (el "iniciador") y aplicamos TODAS las transformaciones
afines 3D a TODO el conjunto de segmentos, una y otra vez:

    nivel_(n+1) = T1(nivel_n)  U  T2(nivel_n)  U  ...  U  Tk(nivel_n)

Cada transformación 3D es:  p' = M @ p + t   (M = matriz 3x3, t = traslación)

Requisitos:
    pip3 install matplotlib numpy pillow
    (Para .mp4 necesitas ffmpeg instalado en el sistema)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------------------------------------------------------------
# 1. HERRAMIENTAS PARA CONSTRUIR TRANSFORMACIONES 3D
# ---------------------------------------------------------------------

def rx(grados):
    a = np.radians(grados); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(grados):
    a = np.radians(grados); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(grados):
    a = np.radians(grados); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def T(escala, R=None, t=(0, 0, 0)):
    """Transformación afín 3D: p' = escala * R @ p + t"""
    R = np.eye(3) if R is None else R
    return {"M": escala * R, "t": np.array(t, dtype=float)}


# ---------------------------------------------------------------------
# 2. FRACTALES DE EJEMPLO (cada uno = iniciador + transformaciones)
# ---------------------------------------------------------------------

def preset_arbol():
    """Árbol 3D: tronco que se ramifica en 3 ramas más pequeñas y torcidas."""
    tronco = np.array([[[0, 0, 0], [0, 0, 1]]], dtype=float)
    ramas = [
        T(0.70, rz(az) @ ry(30) @ rz(40), t=(0, 0, 1))
        for az in (0, 120, 240)
    ]
    return {
        "iniciador": tronco, "transformaciones": ramas,
        "niveles": 7,
        "acumular": True,        # dibuja todos los niveles (tronco + ramas + ...)
        "color_por": "nivel", "cmap": "YlGn",
        "grosor": (3.0, 0.4),    # grosor inicial -> final (según nivel)
    }


def preset_koch():
    """Curva de Koch 3D: cada tramo se reemplaza por 4, con giros fuera del plano."""
    s, giro = 1 / 3, 75
    R2 = rx(giro) @ rz(60)
    R3 = rx(giro) @ rz(-60)
    d2 = R2 @ np.array([1.0, 0, 0])
    return {
        "iniciador": np.array([[[0, 0, 0], [1, 0, 0]]], dtype=float),
        "transformaciones": [
            T(s),
            T(s, R2, t=(s, 0, 0)),
            T(s, R3, t=np.array([s, 0, 0]) + s * d2),
            T(s, t=(2 * s, 0, 0)),
        ],
        "niveles": 6,
        "acumular": False,       # solo el último nivel
        "color_por": "orden", "cmap": "plasma",
        "grosor": (0.8, 0.8),
    }


def preset_tetraedro():
    """Tetraedro de Sierpinski dibujado con aristas (6 aristas -> 4 copias a 1/2)."""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    aristas = np.array([[v[i], v[j]] for i in range(4) for j in range(i + 1, 4)])
    return {
        "iniciador": aristas,
        "transformaciones": [T(0.5, t=0.5 * vi) for vi in v],
        "niveles": 5,
        "acumular": False,
        "color_por": "altura", "cmap": "cool",
        "grosor": (0.7, 0.7),
    }


PRESETS = {"arbol": preset_arbol, "koch": preset_koch, "tetraedro": preset_tetraedro}

# Elige cuál animar: "arbol", "koch" o "tetraedro"
PRESET = "tetraedro"


# ---------------------------------------------------------------------
# 3. PARÁMETROS DE LA ANIMACIÓN
# ---------------------------------------------------------------------

NOMBRE_ARCHIVO_SALIDA = "fractal_3d_creciendo.mp4"   # .mp4 (requiere ffmpeg) o .gif
DURACION_SEGUNDOS = 8         # tiempo que tarda en dibujarse todo
PAUSA_FINAL_SEGUNDOS = 1.5    # el fractal terminado se queda quieto un rato
FPS = 24
DPI = 100
ELEVACION = 20                # cámara FIJA: ángulo vertical (grados)
AZIMUT = 35                   # cámara FIJA: ángulo horizontal (grados)
FONDO_OSCURO = True


# ---------------------------------------------------------------------
# 4. MOTOR: APLICAR LAS TRANSFORMACIONES A LOS SEGMENTOS
# ---------------------------------------------------------------------

def generar_segmentos(cfg):
    """Devuelve (segmentos [N,2,3], valores [N] en 0..1, grosores [N], nivel de cada segmento [N])."""
    actual = cfg["iniciador"]
    historial = [actual]

    for _ in range(cfg["niveles"]):
        # (N,2,3) @ (3,3).T -> (N,2,3): transforma ambos extremos de cada segmento
        actual = np.concatenate([actual @ t["M"].T + t["t"] for t in cfg["transformaciones"]])
        historial.append(actual)

    g0, g1 = cfg["grosor"]
    if cfg["acumular"]:
        segs = np.concatenate(historial)
        niveles = np.concatenate([np.full(len(h), i) for i, h in enumerate(historial)])
        frac = niveles / max(cfg["niveles"], 1)
        grosores = g0 + (g1 - g0) * frac
    else:
        segs = historial[-1]
        niveles = np.zeros(len(segs), dtype=int)
        grosores = np.full(len(segs), g0)

    modo = cfg["color_por"]
    if modo == "nivel":
        valores = frac
    elif modo == "orden":
        valores = np.linspace(0, 1, len(segs))
    else:  # altura (coordenada z del punto medio)
        z = segs[:, :, 2].mean(axis=1)
        valores = (z - z.min()) / (np.ptp(z) + 1e-12)

    return segs, valores, grosores, niveles


# ---------------------------------------------------------------------
# 5. ANIMACIÓN: LAS LÍNEAS APARECEN POCO A POCO (cámara fija)
# ---------------------------------------------------------------------

def segmentos_visibles(niveles, progreso):
    """Cuántos segmentos mostrar. Cada nivel ocupa el mismo tiempo de animación."""
    total_niveles = niveles.max() + 1
    p = progreso * total_niveles
    nivel = min(int(p), total_niveles - 1)
    fraccion = 1.0 if progreso >= 1 else p - nivel
    antes = np.sum(niveles < nivel)
    en_nivel = np.sum(niveles == nivel)
    return int(antes + fraccion * en_nivel)


def crear_animacion(segs, valores, grosores, niveles, cmap):
    fondo = "black" if FONDO_OSCURO else "white"
    fig = plt.figure(figsize=(7, 7), facecolor=fondo)
    ax = fig.add_subplot(111, projection="3d", facecolor=fondo)
    ax.set_axis_off()
    ax.view_init(elev=ELEVACION, azim=AZIMUT)

    # Cubo con límites iguales (calculado con el fractal COMPLETO) para que
    # el encuadre no cambie mientras crece
    puntos = segs.reshape(-1, 3)
    centro = (puntos.min(axis=0) + puntos.max(axis=0)) / 2
    radio = (puntos.max(axis=0) - puntos.min(axis=0)).max() / 2 * 1.05
    ax.set_xlim(centro[0] - radio, centro[0] + radio)
    ax.set_ylim(centro[1] - radio, centro[1] + radio)
    ax.set_zlim(centro[2] - radio, centro[2] + radio)
    ax.set_box_aspect((1, 1, 1))

    colores = plt.get_cmap(cmap)(valores)           # RGBA de cada segmento
    coleccion = Line3DCollection(segs, colors=colores, linewidths=grosores)
    ax.add_collection3d(coleccion)

    frames_dibujo = int(DURACION_SEGUNDOS * FPS)
    num_frames = frames_dibujo + int(PAUSA_FINAL_SEGUNDOS * FPS)

    def actualizar(i):
        progreso = min(i / (frames_dibujo - 1), 1.0)
        k = segmentos_visibles(niveles, progreso)
        rgba = colores.copy()
        rgba[k:, 3] = 0.0                            # los que faltan: invisibles
        coleccion.set_color(rgba)
        return ()

    anim = animation.FuncAnimation(fig, actualizar, frames=num_frames, interval=1000 / FPS)
    return fig, anim


def exportar_animacion(anim):
    if NOMBRE_ARCHIVO_SALIDA.lower().endswith(".mp4"):
        # yuv420p = formato de color que entienden todos los móviles (WhatsApp incluido)
        # faststart = el video empieza a reproducirse sin esperar a que termine de cargar
        writer = animation.FFMpegWriter(
            fps=FPS, bitrate=3000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
        )
    else:
        writer = animation.PillowWriter(fps=FPS)
    print("Generando animación... esto puede tardar un poco.")
    anim.save(NOMBRE_ARCHIVO_SALIDA, writer=writer, dpi=DPI)
    print(f"¡Listo! Guardado como '{NOMBRE_ARCHIVO_SALIDA}'.")


# ---------------------------------------------------------------------
# 6. EJECUCIÓN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PRESETS[PRESET]()
    segs, valores, grosores, niveles = generar_segmentos(cfg)
    print(f"Fractal '{PRESET}': {len(segs)} segmentos.")
    fig, anim = crear_animacion(segs, valores, grosores, niveles, cfg["cmap"])
    exportar_animacion(anim)