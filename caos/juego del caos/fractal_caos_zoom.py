"""
Fractal por el "Juego del Caos" + ZOOM INFINITO AUTOSIMILAR (en bucle).

LA IDEA (correcta):
    Los puntos del fractal NUNCA se transforman ni se mueven. Lo único
    que se mueve es la CÁMARA (los límites del gráfico), que se va
    encogiendo hacia el punto fijo de una transformación elegida.

    Por qué esto funciona: la región chiquita alrededor del punto fijo
    ya contiene, dentro del mismo fractal, una copia completa de todo
    el fractal (autosimilitud) — es exactamente la parte de puntos que
    "cayeron" en esa rama de la transformación durante el juego del
    caos. Entonces, al encoger la cámara hacia esa región, vemos un
    acercamiento real, y cuando terminamos de encoger exactamente el
    factor de esa transformación, lo que vemos es -de nuevo- una copia
    completa del fractal. Ahí reiniciamos la cámara a su tamaño
    original: como el contenido ya se veía igual, el reinicio es
    invisible, y el ciclo se puede repetir tantas veces como se quiera.

Requisitos:
    pip3 install matplotlib numpy pillow

    Para exportar en MP4 necesitas ffmpeg instalado en tu sistema:
      - Mac:      brew install ffmpeg
      - Ubuntu:   sudo apt install ffmpeg
      - Windows:  https://ffmpeg.org/download.html
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------
# 1. CONFIGURACIÓN DE TRANSFORMACIONES (edítalas para tu propio fractal)
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

# Elige el fractal. Para el efecto de zoom infinito, SIERPINSKI es el más
# limpio (transformaciones de escalado puro, sin rotación ni distorsión).
TRANSFORMACIONES = SIERPINSKI


# ---------------------------------------------------------------------
# 2. PARÁMETROS GENERALES
# ---------------------------------------------------------------------

NUM_PUNTOS_BASE = 80000        # más puntos = la región zoomeada se ve más densa/nítida
DESCARTAR_PRIMEROS = 20
PUNTO_INICIAL = (0.0, 0.0)
TAMANO_PUNTO = 0.6
USAR_COLOR_POR_TRANSFORMACION = True

# --- Parámetros del zoom infinito ---
INDICE_TRANSFORMACION_ZOOM = 0     # cuál transformación usar como "destino" del zoom (prueba 0, 1, 2...)
FPS = 30
DURACION_SEGUNDOS = 1              # duración TOTAL del video, tan larga como quieras
DURACION_CICLO_SEGUNDOS = 4         # cuánto dura UN acercamiento completo antes de reiniciarse
FONDO_OSCURO = True
NOMBRE_ARCHIVO_SALIDA = "fractal_zoom_infinito.mp4"   # .mp4 (requiere ffmpeg) o .gif


# ---------------------------------------------------------------------
# 3. MOTOR DEL JUEGO DEL CAOS
# ---------------------------------------------------------------------

def validar_transformaciones(transformaciones):
    suma = sum(t["prob"] for t in transformaciones)
    if abs(suma - 1.0) > 1e-6:
        raise ValueError(f"Las probabilidades deben sumar 1.0 (suman {suma:.4f}).")


def elegir_transformacion(transformaciones):
    r = random.random()
    acumulado = 0.0
    for t in transformaciones:
        acumulado += t["prob"]
        if r <= acumulado:
            return t
    return transformaciones[-1]


def generar_fractal(transformaciones, num_puntos, descartar, punto_inicial):
    validar_transformaciones(transformaciones)
    x, y = punto_inicial
    xs, ys, colores = [], [], []

    for i in range(num_puntos):
        t = elegir_transformacion(transformaciones)
        x_nuevo = t["a"] * x + t["b"] * y + t["e"]
        y_nuevo = t["c"] * x + t["d"] * y + t["f"]
        x, y = x_nuevo, y_nuevo
        if i >= descartar:
            xs.append(x)
            ys.append(y)
            colores.append(t.get("color", "#2e7d32"))

    return np.array(xs), np.array(ys), np.array(colores)


# ---------------------------------------------------------------------
# 4. MATEMÁTICA DEL ZOOM: punto fijo y factor de escala de la transformación
# ---------------------------------------------------------------------

def matriz_y_vector(t):
    A = np.array([[t["a"], t["b"]], [t["c"], t["d"]]])
    b = np.array([t["e"], t["f"]])
    return A, b


def punto_fijo(A, b):
    """Resuelve p = A*p + b: el punto que la transformación deja igual."""
    I = np.eye(2)
    return np.linalg.solve(I - A, b)


def factor_de_escala(A):
    """
    Qué tanto encoge esta transformación, en promedio (raíz cuadrada del
    valor absoluto del determinante = factor de escala lineal aproximado).
    """
    return np.sqrt(abs(np.linalg.det(A)))


# ---------------------------------------------------------------------
# 5. ANIMACIÓN — LOS PUNTOS NO SE MUEVEN, SOLO LA CÁMARA
# ---------------------------------------------------------------------

def crear_animacion(X0, Y0, colores):
    t_zoom = TRANSFORMACIONES[INDICE_TRANSFORMACION_ZOOM]
    A, b = matriz_y_vector(t_zoom)
    p = punto_fijo(A, b)
    s = factor_de_escala(A)
    print(f"Punto fijo: {p} — factor de escala por ciclo: {s:.3f}")

    # En vez de un radio simétrico alrededor de p (que deja espacio vacío
    # si p está en una esquina del fractal, no en su centro), medimos
    # cuánto se extiende el fractal HACIA CADA LADO de p por separado.
    margen = 1.08
    dx_min = (np.min(X0) - p[0]) * margen
    dx_max = (np.max(X0) - p[0]) * margen
    dy_min = (np.min(Y0) - p[1]) * margen
    dy_max = (np.max(Y0) - p[1]) * margen

    # Tamaño de figura ajustado a la proporción real del fractal,
    # para que no queden franjas negras de relleno.
    ancho, alto = (dx_max - dx_min), (dy_max - dy_min)
    ancho_fig = 7.0
    alto_fig = max(3.0, min(12.0, ancho_fig * (alto / ancho)))

    fondo = "black" if FONDO_OSCURO else "white"
    fig, ax = plt.subplots(figsize=(ancho_fig, alto_fig), facecolor=fondo)
    ax.set_facecolor(fondo)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # sin márgenes blancos/negros extra

    color_puntos = colores if USAR_COLOR_POR_TRANSFORMACION else "#2e7d32"
    # Los puntos se dibujan UNA sola vez y no se vuelven a tocar.
    ax.scatter(X0, Y0, s=TAMANO_PUNTO, c=color_puntos, linewidths=0)

    total_frames = int(DURACION_SEGUNDOS * FPS)
    frames_por_ciclo = int(DURACION_CICLO_SEGUNDOS * FPS)

    def actualizar(frame_idx):
        # progreso va de 0 a 1 dentro de cada ciclo, y se reinicia solo
        progreso = (frame_idx % frames_por_ciclo) / frames_por_ciclo

        # las 4 distancias (a cada lado de p) se encogen EXPONENCIALMENTE
        # con el mismo factor de escala de la transformación elegida
        factor = s ** progreso

        ax.set_xlim(p[0] + dx_min * factor, p[0] + dx_max * factor)
        ax.set_ylim(p[1] + dy_min * factor, p[1] + dy_max * factor)
        return ()

    anim = animation.FuncAnimation(
        fig, actualizar, frames=total_frames, interval=1000 / FPS, blit=False
    )
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

    print("Generando video de zoom infinito... esto puede tardar un poco.")
    anim.save(NOMBRE_ARCHIVO_SALIDA, writer=writer, dpi=150)
    print(f"¡Listo! Guardado como '{NOMBRE_ARCHIVO_SALIDA}'.")


# ---------------------------------------------------------------------
# 6. EJECUCIÓN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    X0, Y0, colores = generar_fractal(
        TRANSFORMACIONES, NUM_PUNTOS_BASE, DESCARTAR_PRIMEROS, PUNTO_INICIAL
    )
    fig, anim = crear_animacion(X0, Y0, colores)
    exportar_animacion(anim)