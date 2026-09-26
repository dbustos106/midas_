"""
Generador de fractales mediante el "Juego del Caos" (Chaos Game).

Idea:
    Se parte de un punto inicial y se le aplica, en cada iteración,
    una transformación afín elegida al azar (con cierta probabilidad)
    de una lista de transformaciones. Tras muchas iteraciones, los
    puntos generados dibujan un fractal (atractor del sistema).

Cada transformación es de la forma:
    x' = a*x + b*y + e
    y' = c*x + d*y + f

Para configurar tu propio fractal, solo edita la lista TRANSFORMACIONES
más abajo. Cada entrada es un diccionario con:
    "a", "b", "c", "d", "e", "f"  -> coeficientes de la transformación
    "prob"                        -> probabilidad de elegir esa transformación
                                      (las probabilidades deben sumar 1)
    "color" (opcional)            -> color para los puntos generados con ella

Se incluyen ejemplos ya listos: Helecho de Barnsley, Triángulo de
Sierpinski y una variante "hoja". Cambia FRACTAL_ACTIVO para elegir cuál
usar, o define el tuyo propio.
"""

import random
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. CONFIGURACIÓN DE TRANSFORMACIONES
# ---------------------------------------------------------------------

# --- Helecho de Barnsley (clásico) ---
HELECHO_BARNSLEY = [
    {"a": 0.00, "b": 0.00, "c": 0.00, "d": 0.16, "e": 0.00, "f": 0.00, "prob": 0.01, "color": "#1b5e20"},
    {"a": 0.85, "b": 0.04, "c": -0.04, "d": 0.85, "e": 0.00, "f": 1.60, "prob": 0.85, "color": "#2e7d32"},
    {"a": 0.20, "b": -0.26, "c": 0.23, "d": 0.22, "e": 0.00, "f": 1.60, "prob": 0.07, "color": "#43a047"},
    {"a": -0.15, "b": 0.28, "c": 0.26, "d": 0.24, "e": 0.00, "f": 0.44, "prob": 0.07, "color": "#66bb6a"},
]

# --- Triángulo de Sierpinski ---
SIERPINSKI = [
    {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5, "e": 0.0, "f": 0.0, "prob": 1 / 3, "color": "#0d47a1"},
    {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5, "e": 0.5, "f": 0.0, "prob": 1 / 3, "color": "#1565c0"},
    {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5, "e": 0.25, "f": 0.5, "prob": 1 / 3, "color": "#1976d2"},
]

# --- Espiral / hoja (otro ejemplo para experimentar) ---
ESPIRAL = [
    {"a": 0.85, "b": 0.00, "c": 0.00, "d": 0.85, "e": 0.10, "f": 0.10, "prob": 0.80, "color": "#4a148c"},
    {"a": 0.20, "b": -0.30, "c": 0.30, "d": 0.20, "e": 0.10, "f": 0.10, "prob": 0.10, "color": "#7b1fa2"},
    {"a": -0.15, "b": 0.30, "c": 0.30, "d": 0.15, "e": 0.20, "f": 0.05, "prob": 0.10, "color": "#ab47bc"},
]

# Elige aquí qué fractal quieres generar:
#   HELECHO_BARNSLEY | SIERPINSKI | ESPIRAL | (tu propia lista)
TRANSFORMACIONES = SIERPINSKI


# ---------------------------------------------------------------------
# 2. PARÁMETROS GENERALES (ajustables)
# ---------------------------------------------------------------------

NUM_PUNTOS = 60000          # cantidad de puntos a generar
DESCARTAR_PRIMEROS = 20     # puntos iniciales que se descartan (transición)
PUNTO_INICIAL = (0.0, 0.0)  # punto de partida
TAMANO_PUNTO = 0.2          # tamaño de cada punto en el gráfico
USAR_COLOR_POR_TRANSFORMACION = True  # False = un solo color


# ---------------------------------------------------------------------
# 3. MOTOR DEL JUEGO DEL CAOS (no hace falta tocar esto)
# ---------------------------------------------------------------------

def validar_transformaciones(transformaciones):
    suma = sum(t["prob"] for t in transformaciones)
    if abs(suma - 1.0) > 1e-6:
        raise ValueError(
            f"Las probabilidades deben sumar 1.0 (suman {suma:.4f}). "
            "Revisa el campo 'prob' de cada transformación."
        )


def elegir_transformacion(transformaciones):
    r = random.random()
    acumulado = 0.0
    for t in transformaciones:
        acumulado += t["prob"]
        if r <= acumulado:
            return t
    return transformaciones[-1]  # margen por errores de redondeo


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

        if i >= descartar:  # se descartan los primeros puntos (transición)
            xs.append(x)
            ys.append(y)
            colores.append(t.get("color", "#2e7d32"))

    return xs, ys, colores


def dibujar_fractal(xs, ys, colores, usar_color=True, tamano_punto=0.2):
    plt.figure(figsize=(8, 10), facecolor="white")
    if usar_color:
        plt.scatter(xs, ys, s=tamano_punto, c=colores, linewidths=0)
    else:
        plt.scatter(xs, ys, s=tamano_punto, c="#2e7d32", linewidths=0)

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("fractal.png", dpi=200, facecolor="white")
    plt.show()


# ---------------------------------------------------------------------
# 4. EJECUCIÓN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    xs, ys, colores = generar_fractal(
        TRANSFORMACIONES,
        NUM_PUNTOS,
        DESCARTAR_PRIMEROS,
        PUNTO_INICIAL,
    )
    dibujar_fractal(xs, ys, colores, USAR_COLOR_POR_TRANSFORMACION, TAMANO_PUNTO)
    print(f"Fractal generado con {len(xs)} puntos. Guardado como 'fractal.png'.")
