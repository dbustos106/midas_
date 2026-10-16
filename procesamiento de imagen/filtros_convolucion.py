"""
Aplica filtros de convolución clásicos a una imagen y los compara en una cuadrícula.

Uso:
    python filtros_convolucion.py foto.jpg
    python filtros_convolucion.py foto.jpg --size 400 --salida resultado.png
    python filtros_convolucion.py            # sin imagen: usa una imagen sintética de prueba

Dependencias:
    pip install numpy pillow matplotlib
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Kernels conocidos
# ---------------------------------------------------------------------------
def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


KERNELS = {
    "Identidad": np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=float),

    "Box blur 5x5": np.ones((5, 5)) / 25,

    "Gaussiano 5x5": gaussian_kernel(5, 1.2),

    "Motion blur (horizontal)": np.pad(np.ones((1, 9)) / 9, ((4, 4), (0, 0))),

    "Sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=float),

    "Unsharp mask": -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                         [4, 16, 24, 16, 4],
                                         [6, 24, -476, 24, 6],
                                         [4, 16, 24, 16, 4],
                                         [1, 4, 6, 4, 1]], dtype=float),

    "Laplaciano (bordes)": np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=float),

    "Laplaciano 8-vecinos": np.array([[1, 1, 1],
                                      [1, -8, 1],
                                      [1, 1, 1]], dtype=float),

    "Sobel X (bordes verticales)": np.array([[-1, 0, 1],
                                             [-2, 0, 2],
                                             [-1, 0, 1]], dtype=float),

    "Sobel Y (bordes horizontales)": np.array([[-1, -2, -1],
                                               [0, 0, 0],
                                               [1, 2, 1]], dtype=float),

    "Prewitt X": np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=float),

    "Emboss (relieve)": np.array([[-2, -1, 0],
                                  [-1, 1, 1],
                                  [0, 1, 2]], dtype=float),

    "Outline": np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]], dtype=float),
}

# Filtros que producen valores centrados en 0 (bordes): se les suma un offset
# para poder visualizarlos, o se muestra el valor absoluto.
FILTROS_DE_BORDE = {
    "Laplaciano (bordes)", "Laplaciano 8-vecinos", "Sobel X (bordes verticales)",
    "Sobel Y (bordes horizontales)", "Prewitt X", "Outline",
}


# ---------------------------------------------------------------------------
# Convolución 2D (implementada a mano con numpy)
# ---------------------------------------------------------------------------
def convolve2d(canal, kernel):
    """Convolución 2D de un canal (H, W) con un kernel, usando padding por reflejo."""
    kernel = np.flipud(np.fliplr(kernel))  # convolución real = correlación con kernel volteado
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(canal, ((ph, ph), (pw, pw)), mode="reflect")
    salida = np.zeros_like(canal, dtype=float)
    h, w = canal.shape
    # Se suma cada "desplazamiento" del kernel: kh*kw iteraciones, todo vectorizado.
    for i in range(kh):
        for j in range(kw):
            salida += kernel[i, j] * padded[i:i + h, j:j + w]
    return salida

    '''
    En vez de recorrer cada píxel y multiplicar su vecindario por el kernel 
    (lo que sería lento, con 4 bucles anidados), se invierte el orden: 
    por cada posición del kernel (i, j), se toma la imagen completa desplazada 
    y se multiplica por ese único número del kernel.

    padded[i:i+h, j:j+w] es una “ventana” del tamaño de la imagen original, 
    pero corrida i filas y j columnas dentro de padded. 
    Esa ventana contiene, para cada píxel de la imagen, el vecino que está en la 
    posición (i, j) del kernel.
    '''


def aplicar_filtro(img, kernel, es_borde=False):
    """Aplica el kernel a cada canal de una imagen (H, W, 3) en float [0, 255]."""
    canales = [convolve2d(img[..., c], kernel) for c in range(img.shape[2])]
    out = np.stack(canales, axis=-1)
    if es_borde:
        out = np.abs(out)
        m = out.max()
        if m > 0:
            out = out / m * 255  # normaliza para que se vea
    return np.clip(out, 0, 255).astype(np.uint8)


def magnitud_sobel(img):
    """Magnitud del gradiente: sqrt(Gx^2 + Gy^2) sobre la imagen en gris."""
    gris = img.mean(axis=2)
    gx = convolve2d(gris, KERNELS["Sobel X (bordes verticales)"])
    gy = convolve2d(gris, KERNELS["Sobel Y (bordes horizontales)"])
    mag = np.hypot(gx, gy)
    mag = mag / mag.max() * 255
    return np.stack([mag] * 3, axis=-1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Imagen de prueba (por si no pasas ninguna)
# ---------------------------------------------------------------------------
def imagen_sintetica(size=400):
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    # degradado
    for x in range(size):
        d.line([(x, 0), (x, size // 6)], fill=(int(255 * x / size), 80, 255 - int(255 * x / size)))
    d.rectangle([40, 90, 180, 230], fill=(220, 60, 60), outline=(0, 0, 0), width=3)
    d.ellipse([210, 100, 350, 240], fill=(60, 160, 220), outline=(0, 0, 0), width=3)
    d.polygon([(200, 380), (270, 270), (340, 380)], fill=(250, 200, 40), outline=(0, 0, 0))
    for k in range(0, size, 12):  # rayas finas: muy útiles para ver blur/sharpen
        d.line([(30, 260 + k // 6), (170, 260 + k // 6)], fill=(0, 0, 0), width=1)
    d.text((40, 360), "Convolucion", fill=(0, 0, 0))
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Filtros de convolución sobre una imagen")
    ap.add_argument("imagen", nargs="?", help="ruta de la imagen (opcional)")
    ap.add_argument("--size", type=int, default=400, help="lado máximo al que se reduce la imagen")
    ap.add_argument("--salida", default="comparacion_filtros.png", help="archivo de salida")
    ap.add_argument("--no-mostrar", action="store_true", help="no abrir la ventana de matplotlib")
    args = ap.parse_args()

    pil = Image.open(args.imagen).convert("RGB") if args.imagen else imagen_sintetica(args.size)
    pil.thumbnail((args.size, args.size))  # reduce para que sea rápido
    img = np.asarray(pil, dtype=float)

    resultados = [("Original", np.asarray(pil))]
    for nombre, k in KERNELS.items():
        resultados.append((nombre, aplicar_filtro(img, k, es_borde=nombre in FILTROS_DE_BORDE)))
    resultados.append(("Sobel (magnitud)", magnitud_sobel(img)))

    n = len(resultados)
    cols = 4
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 3.6 * filas))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (titulo, im) in zip(axes.ravel(), resultados):
        ax.imshow(im)
        ax.set_title(titulo, fontsize=10)
    plt.tight_layout()
    plt.savefig(args.salida, dpi=130)
    print(f"Guardado en {args.salida}")
    if not args.no_mostrar:
        plt.show()


if __name__ == "__main__":
    main()
