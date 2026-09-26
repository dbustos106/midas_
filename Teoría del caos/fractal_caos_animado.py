"""
Fractal por el "Juego del Caos" (Chaos Game) + ANIMACIÓN ROTATORIA.

Genera el fractal igual que la versión estática (mediante transformaciones
afines aplicadas al azar) y luego crea un video/GIF donde el fractal
completo gira sobre su propio centro.

Requisitos:
    pip3 install matplotlib numpy pillow
    (Pillow se usa para exportar el GIF; no necesitas ffmpeg)

    Si además quieres exportar en formato MP4, instala ffmpeg en tu
    sistema (no es un paquete de pip, es un programa aparte):
      - Mac:      brew install ffmpeg
      - Ubuntu:   sudo apt install ffmpeg
      - Windows:  https://ffmpeg.org/download.html
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------
# 1. CONFIGURACIÓN DE TRANSFORMACIONES (igual que antes, edítalas aquí)
# ---------------------------------------------------------------------

HELECHO_BARNSLEY = [
    {"a": 0.00, "b": 0.00, "c": 0.00, "d": 0.16, "e": 0.00, "f": 0.00, "prob": 0.01, "color": "#1b5e20"},
    {"a": 0.85, "b": 0.04, "c": -0.04, "d": 0.85, "e": 0.00, "f": 1.60, "prob": 0.85, "color": "#2e7d32"},
    {"a": 0.20, "b": -0.26, "c": 0.23, "d": 0.22, "e": 0.00, "f": 1.60, "prob": 0.07, "color": "#43a047"},
    {"a": -0.15, "b": 0.28, "c": 0.26, "d": 0.24, "e": 0.00, "f": 0.44, "prob": 0.07, "color": "#66bb6a"},
]

SIERPINSKI = [
    {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5, "e": 0.0, "f": 0.0, "prob": 1 / 3, "color": "#0d47a1"},
    {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5, "e": 0.5, "f": 0.0, "prob": 1 / 3, "color": "#1565c0"},
    {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5, "e": 0.25, "f": 0.5, "prob": 1 / 3, "color": "#1976d2"},
]

ESPIRAL = [
    {"a": 0.85, "b": 0.00, "c": 0.00, "d": 0.85, "e": 0.10, "f": 0.10, "prob": 0.80, "color": "#4a148c"},
    {"a": 0.20, "b": -0.30, "c": 0.30, "d": 0.20, "e": 0.10, "f": 0.10, "prob": 0.10, "color": "#7b1fa2"},
    {"a": -0.15, "b": 0.30, "c": 0.30, "d": 0.15, "e": 0.20, "f": 0.05, "prob": 0.10, "color": "#ab47bc"},
]

# Elige el fractal a animar:
TRANSFORMACIONES = HELECHO_BARNSLEY


# ---------------------------------------------------------------------
# 2. PARÁMETROS GENERALES
# ---------------------------------------------------------------------

NUM_PUNTOS = 20000           # menos puntos = animación más fluida y liviana
DESCARTAR_PRIMEROS = 20
PUNTO_INICIAL = (0.0, 0.0)
TAMANO_PUNTO = 0.4
USAR_COLOR_POR_TRANSFORMACION = True

# --- Parámetros de la animación ---
NOMBRE_ARCHIVO_SALIDA = "fractal_rotando.gif"  # usa .mp4 si tienes ffmpeg
DURACION_SEGUNDOS = 6
FPS = 30
VUELTAS_COMPLETAS = 1          # cuántas vueltas de 360° da en toda la animación
FONDO_OSCURO = True            # True = fondo negro, False = fondo blanco


# ---------------------------------------------------------------------
# 3. MOTOR DEL JUEGO DEL CAOS
# ---------------------------------------------------------------------

def validar_transformaciones(transformaciones):
    suma = sum(t["prob"] for t in transformaciones)
    if abs(suma - 1.0) > 1e-6:
        raise ValueError(
            f"Las probabilidades deben sumar 1.0 (suman {suma:.4f})."
        )


def elegir_transformacion(transformaciones):
    r = random.random()
    acumulado = 0.0
    for t in transformaciones:
        acumulado += t["prob"]
        if r <= acumulado:
            return t
    return transformaciones[-1]


def aplicar_transformacion(t, x, y):
    x_nueva = t["a"] * x + t["b"] * y + t["e"]
    y_nueva = t["c"] * x + t["d"] * y + t["f"]
    return x_nueva, y_nueva


def generar_fractal(transformaciones, num_puntos, descartar, punto_inicial):
    validar_transformaciones(transformaciones)
    x, y = punto_inicial
    xs, ys, colores = [], [], []

    for i in range(num_puntos):
        t = elegir_transformacion(transformaciones)
        x, y = aplicar_transformacion(t, x, y)
        if i >= descartar:
            xs.append(x)
            ys.append(y)
            colores.append(t.get("color", "#2e7d32"))

    return np.array(xs), np.array(ys), colores


# ---------------------------------------------------------------------
# 4. ANIMACIÓN: ROTACIÓN SOBRE EL CENTRO
# ---------------------------------------------------------------------

def rotar_puntos(xs, ys, centro_x, centro_y, angulo_rad):
    """Rota (xs, ys) un ángulo dado alrededor de (centro_x, centro_y)."""
    x_centrado = xs - centro_x
    y_centrado = ys - centro_y
    cos_a, sin_a = np.cos(angulo_rad), np.sin(angulo_rad)
    x_rot = x_centrado * cos_a - y_centrado * sin_a
    y_rot = x_centrado * sin_a + y_centrado * cos_a
    return x_rot + centro_x, y_rot + centro_y


def crear_animacion(xs, ys, colores):
    # Centro de rotación = centro del bounding box del fractal
    centro_x = (xs.min() + xs.max()) / 2
    centro_y = (ys.min() + ys.max()) / 2

    # Margen para que el fractal no se salga del cuadro al rotar
    radio = np.sqrt((xs - centro_x) ** 2 + (ys - centro_y) ** 2).max()
    margen = radio * 1.15

    fondo = "black" if FONDO_OSCURO else "white"

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=fondo)
    ax.set_facecolor(fondo)
    ax.set_xlim(centro_x - margen, centro_x + margen)
    ax.set_ylim(centro_y - margen, centro_y + margen)
    ax.set_aspect("equal")
    ax.axis("off")

    color_puntos = colores if USAR_COLOR_POR_TRANSFORMACION else "#2e7d32"
    scatter = ax.scatter(xs, ys, s=TAMANO_PUNTO, c=color_puntos, linewidths=0)

    num_frames = int(DURACION_SEGUNDOS * FPS)
    angulos = np.linspace(0, 2 * np.pi * VUELTAS_COMPLETAS, num_frames, endpoint=False)

    def actualizar(frame_idx):
        angulo = angulos[frame_idx]
        x_rot, y_rot = rotar_puntos(xs, ys, centro_x, centro_y, angulo)
        scatter.set_offsets(np.column_stack([x_rot, y_rot]))
        return (scatter,)

    anim = animation.FuncAnimation(
        fig, actualizar, frames=num_frames, interval=1000 / FPS, blit=True
    )
    return fig, anim


def exportar_animacion(anim):
    if NOMBRE_ARCHIVO_SALIDA.lower().endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=FPS, bitrate=3000)
    else:
        writer = animation.PillowWriter(fps=FPS)

    print(f"Generando video... esto puede tardar un poco.")
    anim.save(NOMBRE_ARCHIVO_SALIDA, writer=writer, dpi=150)
    print(f"¡Listo! Guardado como '{NOMBRE_ARCHIVO_SALIDA}'.")


# ---------------------------------------------------------------------
# 5. EJECUCIÓN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    xs, ys, colores = generar_fractal(
        TRANSFORMACIONES, NUM_PUNTOS, DESCARTAR_PRIMEROS, PUNTO_INICIAL
    )
    fig, anim = crear_animacion(xs, ys, colores)
    exportar_animacion(anim)
