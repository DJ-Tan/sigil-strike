"""
constants.py
────────────
All magic numbers, colors, layout geometry, and key bindings live here.
Nothing in this file imports from the rest of the project.
"""

import pygame

# ── Window ────────────────────────────────────────────────────────────────────
WIDTH  = 1280
HEIGHT = 720
FPS    = 60
TITLE  = "SIGIL STRIKE"

# ── Layout (all in pixels) ────────────────────────────────────────────────────
CAM_W  = 390
CAM_H  = 310
CAM_Y  = 60            # top of camera panels

SLOT_Y = CAM_Y + CAM_H          # top of action-buffer strip (310)
SLOT_H = 50

ARENA_X1 = CAM_W                # 330
ARENA_X2 = WIDTH - CAM_W        # 950

QUEUE_Y  = 590                   # top of move-queue bar
QUEUE_H  = HEIGHT - QUEUE_Y     # 130

MAX_QUEUED_MOVES = 5
ACTIONS_PER_COMBO = 3

# ── Colors ────────────────────────────────────────────────────────────────────
BG_DARK      = (10,  10,  20)
BG_MID       = (18,  18,  35)
PANEL_COLOR  = (22,  22,  45)
BORDER_COLOR = (50,  60, 120)
DARK_GRAY    = (40,  40,  60)

WHITE  = (255, 255, 255)
GRAY   = (120, 120, 140)
GREEN  = (80,  220, 120)
RED    = (220,  60,  60)
GOLD   = (255, 200,  60)

P1_COLOR = (255, 120,  60)
P2_COLOR = ( 60, 180, 255)

# Per-action shape colors (keyed by Action.value string)
SHAPE_COLORS = {
    "move1": (255, 120,  60),
    "move2": (255, 200,  60),
    "move3": ( 60, 180, 255),
    "move4": (220,  80, 160),
    "move5": ( 80, 220, 120),
}

# ── Key bindings ──────────────────────────────────────────────────────────────
# Populated at runtime (after pygame.init) so we store raw pygame.K_ constants.
# Imported in game.py after pygame.init().
P1_KEYS = {
    pygame.K_q: "move1",
    pygame.K_w: "move2",
    pygame.K_e: "move3",
    pygame.K_r: "move4",
    pygame.K_t: "move5",
}

P2_KEYS = {
    pygame.K_y: "move1",
    pygame.K_u: "move2",
    pygame.K_i: "move3",
    pygame.K_o: "move4",
    pygame.K_p: "move5",
}
