import numpy as np
from Letras.t import imageT

gameState = imageT(200, 150)
coords = np.argwhere(gameState == 1)

# Sumar 30 en y
coords[:, 1] += 30

for x, y in coords:
    print(f"gameState[{x}, {y}] = 1")