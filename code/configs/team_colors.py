"""
configs/team_colors.py
──────────────────────
Per-team display colors used by the bracket UI, the HP bars, and the
floating combat text. This file is the single source of truth — `team.env`
no longer stores per-team color, so editing values here is the only
place needed to recolor a team.

Keys are team numbers (1..6), matching `Teams/Team<N>/`. Values are RGB
tuples; each channel must be an int in 0..255.
"""
from __future__ import annotations

TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    1: (255, 120,  60),   # orange
    2: ( 60, 180, 255),   # blue
    3: ( 80, 220, 120),   # green
    4: (255, 200,  60),   # yellow
    5: (220,  80, 160),   # pink
    6: (160, 120, 255),   # purple
}
