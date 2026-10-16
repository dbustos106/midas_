from Letras.t import imageT
from Letras.e import imageE
from Letras.a import imageA
from Letras.m import imageM
from Letras.o import imageO
from Letras.i import imageI
from Letras.s import imageS
from Letras.z import imageZ

import numpy as np
import pygame
import ctypes
import time
import os

# -------------------- Configuration --------------------

BG = (126, 120, 119)
GRID = (143, 143, 143)
CELL_COLOR = (62, 1, 74)

PANEL_WIDTH = 58
MAX_FRAMES = None 

SEQUENCE = [
    {
        "name": "ME",
        "nxC": 300,
        "nyC": 100,
        "duration": 10,
        "letters": [
            {"generator": imageM, "x": 0, "y": 0},
            {"generator": imageE, "x": 200, "y": 5},
        ],
    },
    {
        "name": "TIS",
        "nxC": 300,
        "nyC": 125,
        "duration": 20,
        "letters": [
            {"generator": imageT, "x": 0, "y": 10},
            {"generator": imageI, "x": 147, "y": 0},
            {"generator": imageS, "x": 205, "y": 0},
        ],
    },
]

# -------------------- Image helpers --------------------

def load_image(path, width, height):
    image = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(image, (width, height))

# -------------------- Game of Life --------------------

def next_state(state):
    """Calculate one generation of Conway's Game of Life."""

    neighbors = sum(
        np.roll(np.roll(state, dx, axis=0), dy, axis=1)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    )

    return (
        (neighbors == 3)
        | ((state == 1) & (neighbors == 2))
    ).astype(state.dtype)

# -------------------- Grid helpers --------------------

def create_grid(width, height, nxC, nyC, cell_width, cell_height):
    """Draw the grid lines."""

    surface = pygame.Surface((width, height), pygame.SRCALPHA)

    for x in range(nxC + 1):
        px = round(x * cell_width)
        pygame.draw.line(surface, GRID, (px, 0), (px, height))

    for y in range(nyC + 1):
        py = round(y * cell_height)
        pygame.draw.line(surface, GRID, (0, py), (width, py))

    return surface

def draw_cells(screen, state, cell_width, cell_height, color):
    """Draw active cells."""

    for x, y in np.argwhere(state == 1):
        left = round(x * cell_width)
        top = round(y * cell_height)
        right = round((x + 1) * cell_width)
        bottom = round((y + 1) * cell_height)

        screen.fill(color, (left, top, max(1, right - left), max(1, bottom - top)),)

# -------------------- Sequence management --------------------

def load_pattern(state, index):
    """Load multiple letters into a single game state."""

    config = SEQUENCE[index]

    state["sequence_index"] = index
    state["nxC"] = config["nxC"]
    state["nyC"] = config["nyC"]

    # Initialize an empty grid
    game_state = np.zeros(
        (state["nxC"], state["nyC"]),
        dtype=np.uint8,
    )

    # Load and position each letter
    for letter in config["letters"]:

        pattern = letter["generator"](
            state["nxC"], state["nyC"]
        )

        coords = np.argwhere(pattern == 1)

        # Apply horizontal and vertical displacement
        coords[:, 0] += letter["x"]
        coords[:, 1] += letter["y"]

        # Keep only coordinates inside the grid
        valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < state["nxC"])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < state["nyC"])
        )

        coords = coords[valid]

        # Add the letter to the combined grid
        game_state[coords[:, 0], coords[:, 1]] = 1

    state["game_state"] = game_state

    state["dimCW"] = state["draw_width"] / state["nxC"]
    state["dimCH"] = state["draw_height"] / state["nyC"]

    state["grid"] = create_grid(
        state["draw_width"],
        state["draw_height"],
        state["nxC"],
        state["nyC"],
        state["dimCW"],
        state["dimCH"],
    )

    state["elapsed"] = 0.0

def advance_sequence(state):
    """Move to the next pattern when its duration expires."""

    config = SEQUENCE[state["sequence_index"]]

    if state["elapsed"] < config["duration"]:
        return

    next_index = state["sequence_index"] + 1

    if next_index >= len(SEQUENCE):
        state["paused"] = True
        return

    load_pattern(state, next_index)

# -------------------- Event handling --------------------

def handle_events(state, screen_width):
    """Process keyboard, mouse, buttons and speed slider."""

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                positions = np.argwhere(state["game_state"] == 1)

                print("\n# Coordenadas de las celdas activas")
                for x, y in positions:
                    print(f"    gameState[{x}, {y}] = 1")

            else:
                state["paused"] = not state["paused"]

        if event.type == pygame.MOUSEBUTTONUP:
            state["dragging"] = False

    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_buttons = pygame.mouse.get_pressed()

    if any(mouse_buttons):

        if mouse_x < state["draw_width"]:

            # Paint with the left button, erase with the right.
            cell_x = min(
                int(mouse_x / state["dimCW"]),
                state["nxC"] - 1,
            )
            cell_y = min(
                int(mouse_y / state["dimCH"]),
                state["nyC"] - 1,
            )

            state["game_state"][cell_x, cell_y] = (
                0 if mouse_buttons[2] else 1
            )

        else:

            # Reset the current pattern.
            if (
                screen_width - 60 < mouse_x < screen_width
                and 70 < mouse_y < 127
                and not state["dragging"]
            ):
                load_pattern(state, state["sequence_index"])
                state["paused"] = True
                state["frames"] = 0

            # Play.
            elif (
                screen_width - 60 < mouse_x < screen_width
                and 0 < mouse_y < 57
                and not state["dragging"]
            ):
                state["paused"] = False

        # Speed slider.
        if (screen_width - 41 <= mouse_x <= screen_width - 16 and 
            161 <= mouse_y <= 415):
            state["dragging"] = True

    if state["dragging"]:
        state["pointer_y"] = max(161, min(415, mouse_y))

        state["speed_delay"] = (
            (state["pointer_y"] - 161) * 0.3 / 254
        )

    return True

# -------------------- Drawing controls --------------------

def draw_controls(screen, images, screen_width, mouse_pos, pointer_y):
    mouse_x, mouse_y = mouse_pos

    screen.blit(images["background"], (screen_width - 57, 0))
    screen.blit(images["container"], (screen_width - 54, 150))

    pygame.draw.rect(
        screen,
        GRID,
        (screen_width - 41, 161, 25, abs(pointer_y - 160)),
    )

    screen.blit(images["pointer"], (screen_width - 41, pointer_y))

    on_play = screen_width - 60 < mouse_x < screen_width and 0 < mouse_y < 57
    on_reset = screen_width - 60 < mouse_x < screen_width and 70 < mouse_y < 127

    screen.blit(
        images["play_hover"] if on_play else images["play"],
        (screen_width - 57, 0),
    )

    screen.blit(
        images["reset_hover"] if on_reset else images["reset"],
        (screen_width - 57, 70),
    )

# -------------------- Main --------------------

def main():
    pygame.init()

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()

    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1) - 70

    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,27"
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Juego de la vida")

    pygame.display.set_icon(pygame.image.load("Imagenes/play1.png"))

    images = {
        "background": load_image("Imagenes/fondo.png", 57, screen_height),
        "container": load_image("Imagenes/contenedor.png", 50, 300),
        "pointer": load_image("Imagenes/puntero.png", 25, 20),
        "play": load_image("Imagenes/play1.png", 57, 57),
        "play_hover": load_image("Imagenes/play2.png", 57, 57),
        "reset": load_image("Imagenes/reset1.png", 57, 57),
        "reset_hover": load_image("Imagenes/reset2.png", 57, 57),
    }

    state = {
        "draw_width": screen_width - PANEL_WIDTH,
        "draw_height": screen_height,
        "sequence_index": 0,
        "nxC": 0,
        "nyC": 0,
        "game_state": None,
        "dimCW": 0,
        "dimCH": 0,
        "grid": None,
        "paused": False,
        "dragging": False,
        "pointer_y": 288,
        "speed_delay": 0.15,
        "elapsed": 0.0,
        "frames": 0,
    }

    load_pattern(state, 0)

    running = True
    previous_time = time.monotonic()

    while running:

        current_time = time.monotonic()
        delta_time = current_time - previous_time
        previous_time = current_time

        if MAX_FRAMES is not None and state["frames"] >= MAX_FRAMES:
            break

        running = handle_events(state, screen_width)

        if not running:
            break

        if not state["paused"]:

            state["elapsed"] += delta_time
            advance_sequence(state)

            if not state["paused"]:
                state["game_state"] = next_state(state["game_state"])
                state["frames"] += 1

        # Draw the active pattern and its grid.
        screen.fill(BG)

        draw_cells(
            screen,
            state["game_state"],
            state["dimCW"],
            state["dimCH"],
            CELL_COLOR,
        )

        screen.blit(state["grid"], (0, 0))

        draw_controls(
            screen,
            images,
            screen_width,
            pygame.mouse.get_pos(),
            state["pointer_y"],
        )

        pygame.display.flip()

        if state["speed_delay"] > 0:
            time.sleep(state["speed_delay"])

    pygame.quit()


if __name__ == "__main__":
    main()
