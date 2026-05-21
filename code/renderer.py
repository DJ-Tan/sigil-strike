"""
renderer.py
───────────
All pygame drawing lives here.  The Renderer receives read-only snapshots of
game state and paints them onto a surface.  It never mutates game state.
"""

from __future__ import annotations

import math
import os
import random

import pygame

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

from constants import (
    WIDTH, HEIGHT,
    BG_DARK, BG_MID, PANEL_COLOR, BORDER_COLOR, DARK_GRAY,
    WHITE, GRAY, GREEN, RED, GOLD,
    P1_COLOR, P2_COLOR, SHAPE_COLORS,
    CAM_W, CAM_H, CAM_Y, SLOT_Y, SLOT_H,
    ARENA_X1, ARENA_X2, QUEUE_Y, QUEUE_H, MAX_QUEUED_MOVES,
)
from configs.time_config import (
    DEATHMATCH_START_SEC, DEATHMATCH_COUNTDOWN_SEC,
)
from moves import Move, MOVE_COLORS, MOVE_HINTS
from player import Player
from paths import resource_dir


# ── Utility drawing helpers ───────────────────────────────────────────────────

def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGB tuples; t is clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_rounded_rect(surf, color, rect, radius=12, border=0, border_color=None):
    """Draw a rounded rectangle; pass color=None to draw a border-only outline."""
    if color:
        pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def draw_text(surf, text, font, color, cx, cy, anchor="center"):
    """Render text on `surf` anchored at (cx, cy). Anchor: 'center', 'left', or 'right'."""
    img = font.render(text, True, color)
    r   = img.get_rect()
    if   anchor == "center": r.center   = (cx, cy)
    elif anchor == "left":   r.midleft  = (cx, cy)
    elif anchor == "right":  r.midright = (cx, cy)
    surf.blit(img, r)


_ACTION_NUM = {
    "move1": "1",
    "move2": "2",
    "move3": "3",
    "move4": "4",
    "move5": "5",
}

_num_font_cache: dict[int, pygame.font.Font] = {}

def _get_num_font(size: int) -> pygame.font.Font:
    if size not in _num_font_cache:
        try:
            _num_font_cache[size] = pygame.font.SysFont(
                "couriernew,lucidaconsole,monospace", size, bold=True)
        except Exception:
            _num_font_cache[size] = pygame.font.Font(None, size)
    return _num_font_cache[size]


def draw_shape(surf, shape: str, cx: int, cy: int, size: int,
               color: tuple, alpha: int = 255):
    """Draw a circled number (1-5) for the given action, centred at (cx, cy)."""
    dim = size * 2 + 4
    s   = pygame.Surface((dim, dim), pygame.SRCALPHA)
    c   = (*color, alpha)
    h   = size + 2

    pygame.draw.circle(s, c, (h, h), size, 2)
    num = _ACTION_NUM.get(shape, "?")
    font = _get_num_font(int(size * 1.4))
    txt  = font.render(num, True, c)
    tr   = txt.get_rect(center=(h, h))
    s.blit(txt, tr)

    surf.blit(s, (cx - h, cy - h))


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    Stateless-ish renderer.  The only mutable state it holds is
    purely cosmetic (cam_tick for animations).
    """

    FONT_SIZES = [14, 16, 18, 20, 22, 24, 28, 32, 36, 48, 64, 80]

    # Move-icon assets in <project>/images/. Files are white-on-transparent
    # PNGs sourced from game-icons.net (CC BY 3.0).
    # When frozen by PyInstaller, resource_dir() resolves to sys._MEIPASS,
    # where the build script bundles images/ via --add-data.
    _IMAGES_DIR = str(resource_dir() / "images")
    _ICON_FILES = {
        Move.POWER_STRIKE: "power_strike.png",
        Move.COMBO_BLAST:  "combo_blast.png",
        Move.SHIELD_WALL:  "shield_wall.png",
        Move.DODGE_ROLL:   "dodge_roll.png",
        Move.MEND:         "mend.png",
    }

    # Cross-platform chain: Windows → Segoe UI Symbol; macOS → Apple Symbols;
    # Linux → Noto/DejaVu/Free.  Arial is a wide-coverage last resort.
    _SYM_FAMILY = (
        "segoeuisymbol,applesymbols,"
        "notosanssymbols2,notosanssymbols,"
        "dejavusans,freesans,unifont,arial"
    )

    def __init__(self):
        self.fonts: dict[int, pygame.font.Font] = {}
        for sz in self.FONT_SIZES:
            try:
                self.fonts[sz] = pygame.font.SysFont(
                    "couriernew,lucidaconsole,monospace", sz, bold=True)
            except Exception:
                self.fonts[sz] = pygame.font.Font(None, sz)

        # Secondary font with broad Unicode glyph coverage for symbol text
        self.sym_fonts: dict[int, pygame.font.Font] = {}
        for sz in self.FONT_SIZES:
            try:
                sf = pygame.font.SysFont(self._SYM_FAMILY, sz, bold=True)
                self.sym_fonts[sz] = sf if sf is not None else self.fonts[sz]
            except Exception:
                self.sym_fonts[sz] = self.fonts[sz]

        self.cam_tick: float = 0.0

        # Load move icons once; scaled copies are cached in _icon_cache.
        self._icon_base: dict[Move, pygame.Surface] = {}
        for move, fname in self._ICON_FILES.items():
            path = os.path.join(self._IMAGES_DIR, fname)
            try:
                self._icon_base[move] = pygame.image.load(path).convert_alpha()
            except (pygame.error, FileNotFoundError):
                self._icon_base[move] = None
        self._icon_cache: dict[tuple[Move, int], pygame.Surface] = {}

        # Try to open one webcam per player (index 0 → P1, index 1 → P2)
        self._caps: list = []
        self._cam_surfaces: list = [None, None]
        self._cam_rgb_frames: list = [None, None]
        self._reconnect_timers: list = [0.0, 0.0]
        if _cv2 is not None:
            for i in range(2):
                cap = _cv2.VideoCapture(i, _cv2.CAP_DSHOW)
                self._caps.append(cap if cap.isOpened() else None)
                if not cap.isOpened():
                    cap.release()
        else:
            self._caps = [None, None]

    def f(self, size: int) -> pygame.font.Font:
        return self.fonts.get(size, self.fonts[28])

    def move_icon(self, move: Move, size: int) -> pygame.Surface | None:
        """Return a (size × size) cached copy of the icon for `move`."""
        base = self._icon_base.get(move)
        if base is None:
            return None
        key = (move, size)
        cached = self._icon_cache.get(key)
        if cached is None:
            cached = pygame.transform.smoothscale(base, (size, size))
            self._icon_cache[key] = cached
        return cached

    def sym(self, size: int) -> pygame.font.Font:
        """Return the Unicode-capable font at the requested size."""
        return self.sym_fonts.get(size, self.sym_fonts.get(28, self.fonts[28]))

    def close(self) -> None:
        for cap in self._caps:
            if cap is not None:
                cap.release()

    _RECONNECT_INTERVAL = 3.0  # seconds between reconnect attempts

    def update(self, dt: float) -> None:
        self.cam_tick += dt
        if _cv2 is None:
            return
        for i in range(2):
            cap = self._caps[i]
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    # Camera dropped — release and fall back to stick figure
                    cap.release()
                    self._caps[i] = None
                    self._cam_surfaces[i] = None
                    self._reconnect_timers[i] = self._RECONNECT_INTERVAL
                else:
                    frame = _cv2.flip(frame, 1)
                    frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                    self._cam_rgb_frames[i] = frame
                    display = _cv2.resize(frame, (CAM_W, CAM_H))
                    self._cam_surfaces[i] = pygame.image.frombuffer(
                        display.tobytes(), (CAM_W, CAM_H), "RGB"
                    ).convert()
            else:
                # Periodically probe for the camera coming back
                self._reconnect_timers[i] -= dt
                if self._reconnect_timers[i] <= 0:
                    self._reconnect_timers[i] = self._RECONNECT_INTERVAL
                    new_cap = _cv2.VideoCapture(i, _cv2.CAP_DSHOW)
                    if new_cap.isOpened():
                        self._caps[i] = new_cap
                    else:
                        new_cap.release()

    # ── Top-level draw ────────────────────────────────────────────────────────

    def draw_frame(
        self,
        surf: pygame.Surface,
        p1: Player, p2: Player,
        resolve_timer: float,
        resolve_interval: float,
        event_msg: str,
        event_msg_color: tuple,
        event_msg_life: float,
        elapsed: float = 0.0,
        waiting: bool = False,
    ) -> None:
        self._draw_background(surf)
        self._draw_title_banner(surf, resolve_timer, resolve_interval, elapsed)
        self._draw_cam_panel(surf, p1, x=0)
        self._draw_cam_panel(surf, p2, x=WIDTH - CAM_W)
        self._draw_action_circles(surf, p1, x=0, waiting=waiting)
        self._draw_action_circles(surf, p2, x=WIDTH - CAM_W, waiting=waiting)
        self._draw_arena(surf, event_msg, event_msg_color,
                         event_msg_life, resolve_timer)
        self._draw_move_queue_bar(surf, p1, p2)

    def draw_game_over(self, surf: pygame.Surface, winner: Player | None,
                       hint: str = "ENTER → play again     ESC → quit",
                       tiebreaker: bool = False) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        surf.blit(overlay, (0, 0))

        mx, my = WIDTH // 2, HEIGHT // 2
        if winner:
            if tiebreaker:
                draw_text(surf, "TIEBREAKER", self.f(64), winner.color, mx, my - 90)
                draw_text(surf, winner.name,  self.f(48), WHITE,        mx, my +  0)
            else:
                draw_text(surf, "VICTORY",    self.f(80), winner.color, mx, my - 80)
                draw_text(surf, winner.name,  self.f(48), WHITE,        mx, my + 10)
        else:
            draw_text(surf, "DRAW!",          self.f(80), GOLD,         mx, my - 40)

        draw_text(surf, hint, self.f(22), GRAY, mx, my + 100)

    # ── Background ────────────────────────────────────────────────────────────

    def _draw_background(self, surf: pygame.Surface) -> None:
        surf.fill(BG_DARK)
        for x in range(0, WIDTH, 60):
            pygame.draw.line(surf, (20, 20, 42), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 60):
            pygame.draw.line(surf, (20, 20, 42), (0, y), (WIDTH, y))

    # ── Title banner ──────────────────────────────────────────────────────────

    def _draw_title_banner(self, surf, resolve_timer: float,
                            resolve_interval: float,
                            elapsed: float = 0.0) -> None:
        mid_x = (ARENA_X1 + ARENA_X2) // 2
        rect  = pygame.Rect(ARENA_X1, 0, ARENA_X2 - ARENA_X1, CAM_Y)
        draw_rounded_rect(surf, PANEL_COLOR, rect,
                          radius=0, border=2, border_color=BORDER_COLOR)

        # Title (shifted up to make room for the timer line)
        draw_text(surf, "✦  SIGIL STRIKE  ✦", self.sym(24), WHITE, mid_x, 18)

        # Elapsed-time clock — replaced by deathmatch HUD during countdown
        # and once decay is active.
        time_until_dm = DEATHMATCH_START_SEC - elapsed
        if time_until_dm <= 0:
            # Active: alternate every ~1s between the warning banner and the
            # match clock, so players still see how long the round has run.
            if int(elapsed) % 2 == 0:
                pulse = 0.5 + 0.5 * math.sin(elapsed * 6.0)
                color = lerp_color((140, 30, 30), (255, 80, 80), pulse)
                draw_text(surf, "⚠  DEATHMATCH  ⚠",
                          self.sym(20), color, mid_x, 40)
            else:
                mins, secs = int(elapsed // 60), int(elapsed % 60)
                draw_text(surf, f"⏱  {mins:02d}:{secs:02d}",
                          self.sym(18), (255, 100, 100), mid_x, 40)
        elif time_until_dm <= DEATHMATCH_COUNTDOWN_SEC:
            # Countdown: pulsing red text
            pulse = 0.5 + 0.5 * math.sin(elapsed * 4.0)
            color = lerp_color((180, 60, 60), (255, 120, 120), pulse)
            secs_left = int(math.ceil(time_until_dm))
            draw_text(surf, f"DEATHMATCH IN 0:{secs_left:02d}",
                      self.f(18), color, mid_x, 40)
        else:
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            draw_text(surf, f"⏱  {mins:02d}:{secs:02d}", self.sym(18), GOLD, mid_x, 40)

        # Countdown bar
        t       = resolve_timer / resolve_interval
        bar_x   = ARENA_X1 + 10
        bar_y   = CAM_Y - 8
        bar_w   = ARENA_X2 - ARENA_X1 - 20
        bg_rect = pygame.Rect(bar_x, bar_y, bar_w, 6)
        draw_rounded_rect(surf, DARK_GRAY, bg_rect, radius=3)
        fill_w = int(bar_w * max(t, 0))
        if fill_w > 0:
            draw_rounded_rect(surf, lerp_color(RED, GREEN, t),
                              pygame.Rect(bar_x, bar_y, fill_w, 6), radius=3)

    # ── Camera panel ─────────────────────────────────────────────────────────

    def _draw_cam_panel(self, surf, player: Player, x: int) -> None:
        cx = x + CAM_W // 2

        # ── Team name header (above the camera, y = 0 → CAM_Y) ───────────────
        team_rect = pygame.Rect(x, 0, CAM_W, CAM_Y)
        draw_rounded_rect(surf, PANEL_COLOR, team_rect,
                          radius=0, border=2, border_color=player.color)
        draw_text(surf, f"Team  {player.name}", self.f(18), player.color,
                  cx, CAM_Y // 2)

        # ── Live camera / stick-figure area ──────────────────────────────────
        rect = pygame.Rect(x, CAM_Y, CAM_W, CAM_H)
        idx  = 0 if x == 0 else 1
        cam_surf = self._cam_surfaces[idx] if idx < len(self._cam_surfaces) else None

        if cam_surf is not None:
            surf.blit(cam_surf, (x, CAM_Y))
            # Dark strip so text stays readable over the live feed
            header = pygame.Surface((CAM_W, 58), pygame.SRCALPHA)
            header.fill((0, 0, 0, 160))
            surf.blit(header, (x, CAM_Y))
        else:
            draw_rounded_rect(surf, PANEL_COLOR, rect,
                              radius=8, border=2, border_color=player.color)
            for i in range(0, CAM_H, 6):
                noise = pygame.Surface((CAM_W, 2), pygame.SRCALPHA)
                noise.fill((255, 255, 255, random.randint(0, 12)))
                surf.blit(noise, (x, CAM_Y + i))
            self._draw_stick_figure(surf, cx, CAM_Y + 155, player.color)

        # Border and overlays always on top
        pygame.draw.rect(surf, player.color, rect, 2, border_radius=8)
        draw_text(surf, f"[ {player.name} CAM ]", self.f(18), player.color,
                  cx, CAM_Y + 20)
        draw_text(surf, "⬛ LIVE", self.sym(14), RED, cx, CAM_Y + 42)
        self._draw_hp_bar(surf, player, x)

    def _draw_stick_figure(self, surf, cx: int, cy: int, color: tuple) -> None:
        t   = self.cam_tick
        bob = math.sin(t * 2) * 4

        def iy(v): return int(v + bob)

        # Head
        pygame.draw.circle(surf, color, (cx, iy(cy - 50)), 22, 2)
        # Body
        pygame.draw.line(surf, color, (cx, iy(cy - 28)), (cx, iy(cy + 20)), 2)
        # Arms
        aa = math.sin(t * 2) * 0.3
        pygame.draw.line(surf, color,
                         (cx, iy(cy - 10)),
                         (int(cx - 30 * math.cos(aa + 0.5)),
                          iy(cy - 5 + 30 * math.sin(aa + 0.5))), 2)
        pygame.draw.line(surf, color,
                         (cx, iy(cy - 10)),
                         (int(cx + 30 * math.cos(aa - 0.5)),
                          iy(cy - 5 + 30 * math.sin(aa - 0.5))), 2)
        # Legs
        pygame.draw.line(surf, color, (cx, iy(cy + 20)), (cx - 20, iy(cy + 60)), 2)
        pygame.draw.line(surf, color, (cx, iy(cy + 20)), (cx + 20, iy(cy + 60)), 2)

    def _draw_hp_bar(self, surf, player: Player, panel_x: int) -> None:
        bar_w = CAM_W - 20
        bar_h = 18
        by    = CAM_Y + CAM_H - 24
        bg    = pygame.Rect(panel_x + 10, by, bar_w, bar_h)
        draw_rounded_rect(surf, DARK_GRAY, bg, radius=6)
        pct   = max(0.0, player.hp_display / player.max_hp)
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            draw_rounded_rect(surf, lerp_color(RED, GREEN, pct),
                              pygame.Rect(panel_x + 10, by, fill_w, bar_h),
                              radius=6)
        draw_rounded_rect(surf, None, bg, radius=6,
                          border=1, border_color=BORDER_COLOR)
        draw_text(surf, f"{int(player.hp)} / {player.max_hp}",
                  self.f(14), WHITE, panel_x + CAM_W // 2, by + bar_h // 2)

    # ── Action-buffer circles ─────────────────────────────────────────────────

    def _draw_action_circles(self, surf, player: Player, x: int,
                              waiting: bool = False) -> None:
        # ── Strip background ─────────────────────────────────────────────────
        strip = pygame.Rect(x, SLOT_Y, CAM_W, SLOT_H)
        draw_rounded_rect(surf, (15, 15, 35), strip,
                          radius=0, border=1, border_color=BORDER_COLOR)

        # ── 3 circles, horizontally centred inside the strip ─────────────────
        mid_x      = x + CAM_W // 2
        cy         = SLOT_Y + SLOT_H // 2
        CIRCLE_R   = 20
        CIRCLE_GAP = 80          # wider centre-to-centre spacing

        show_buffer = player.display_buffer if player.combo_locked else player.action_buffer

        for i in range(3):
            cx = mid_x + (i - 1) * CIRCLE_GAP
            pygame.draw.circle(surf, DARK_GRAY,    (cx, cy), CIRCLE_R)
            pygame.draw.circle(surf, BORDER_COLOR, (cx, cy), CIRCLE_R, 2)
            if i < len(show_buffer):
                draw_shape(surf, show_buffer[i].value, cx, cy, 14,
                           SHAPE_COLORS[show_buffer[i].value])

        # ── Scoreboard boxes below the strip (count only, no label) ──────────
        BOX_H  = 42
        MARGIN = 5
        GAP    = 6
        box_y  = SLOT_Y + SLOT_H + MARGIN
        box_w  = (CAM_W - 2 * MARGIN - GAP) // 2

        for col_i, (count, fg, bg, border) in enumerate([
            (player.moves_hit,  GREEN, (10, 32, 14), (40, 110, 55)),
            (player.moves_miss, RED,   (32, 10, 10), (110, 40, 40)),
        ]):
            bx       = x + MARGIN + col_i * (box_w + GAP)
            box_rect = pygame.Rect(bx, box_y, box_w, BOX_H)
            draw_rounded_rect(surf, bg,   box_rect, radius=6)
            draw_rounded_rect(surf, None, box_rect, radius=6,
                              border=1, border_color=border)
            draw_text(surf, str(count), self.f(24), fg,
                      bx + box_w // 2, box_y + BOX_H // 2)

        # ── Key hints below the scoreboard boxes ──────────────────────────────
        # While the READY overlay is up, suppress the keyboard hint for any
        # player who already has a working camera — they'll play via gestures.
        idx = 0 if x == 0 else 1
        cam_detected = idx < len(self._caps) and self._caps[idx] is not None
        if not (waiting and cam_detected):
            keys     = "Q W E R T" if player.pid == 1 else "Y U I O P"
            hint_y   = box_y + BOX_H + 14
            draw_text(surf, f"{keys}  →  ① ② ③ ④ ⑤",
                      self.sym(13), player.color, mid_x, hint_y)

    # ── Arena ─────────────────────────────────────────────────────────────────

    def _draw_arena(self, surf, event_msg, event_msg_color,
                    event_msg_life, resolve_timer) -> None:
        rect = pygame.Rect(ARENA_X1, CAM_Y,
                           ARENA_X2 - ARENA_X1, QUEUE_Y - CAM_Y)
        draw_rounded_rect(surf, BG_MID, rect,
                          radius=0, border=1, border_color=BORDER_COLOR)

        mid_x = (ARENA_X1 + ARENA_X2) // 2
        mid_y = (CAM_Y + QUEUE_Y) // 2

        self._draw_vs(surf, mid_x, mid_y)

        if event_msg_life > 0:
            draw_text(surf, event_msg, self.f(20), event_msg_color,
                      mid_x, CAM_Y + 22)

        secs = math.ceil(resolve_timer)
        draw_text(surf, f"NEXT RESOLVE IN  {secs}s",
                  self.f(16), GRAY, mid_x, QUEUE_Y - 16)

    def _draw_vs(self, surf, cx: int, cy: int) -> None:
        t     = self.cam_tick
        scale = 1.0 + 0.05 * math.sin(t * 3)

        # Slash lines
        pygame.draw.line(surf, (55, 55, 75), (cx - 2, cy - 70), (cx + 2, cy + 70), 4)
        pygame.draw.line(surf, (55, 55, 75), (cx - 3, cy - 65), (cx + 3, cy + 65), 2)

        draw_text(surf, "VS", self.f(64), WHITE, cx, cy)

        # Pulsing glow ring
        glow = pygame.Surface((200, 200), pygame.SRCALPHA)
        pa   = int(28 + 18 * math.sin(t * 4))
        pygame.draw.circle(glow, (*GOLD, pa), (100, 100), 80, 4)
        surf.blit(glow, (cx - 100, cy - 100))

    # ── Move-queue bar ────────────────────────────────────────────────────────

    def _draw_move_queue_bar(self, surf, p1: Player, p2: Player) -> None:
        bar = pygame.Rect(0, QUEUE_Y, WIDTH, QUEUE_H)
        draw_rounded_rect(surf, (12, 12, 28), bar,
                          radius=0, border=2, border_color=BORDER_COLOR)

        mid = WIDTH // 2
        pygame.draw.line(surf, BORDER_COLOR, (mid, QUEUE_Y), (mid, HEIGHT), 2)

        draw_text(surf, "P1 MOVE QUEUE", self.f(16), P1_COLOR,
                  mid // 2, QUEUE_Y + 14)
        draw_text(surf, "P2 MOVE QUEUE", self.f(16), P2_COLOR,
                  mid + mid // 2, QUEUE_Y + 14)

        self._draw_queue_slots(surf, p1, start_x=0,   width=mid)
        self._draw_queue_slots(surf, p2, start_x=mid, width=mid)

    def _draw_queue_slots(self, surf, player: Player,
                           start_x: int, width: int) -> None:
        slot_w = width // MAX_QUEUED_MOVES
        slot_h = QUEUE_H - 32
        sy     = QUEUE_Y + 28

        for i in range(MAX_QUEUED_MOVES):
            sx   = start_x + i * slot_w + 8
            sw   = slot_w - 16
            rect = pygame.Rect(sx, sy, sw, slot_h)
            draw_rounded_rect(surf, DARK_GRAY, rect,
                              radius=8, border=1, border_color=BORDER_COLOR)

            if i < len(player.move_queue):
                move  = player.move_queue[i]
                color = MOVE_COLORS[move]
                fill  = pygame.Rect(sx + 2, sy + 2, sw - 4, slot_h - 4)
                draw_rounded_rect(surf, color, fill, radius=6)

                icon_size = max(16, min(sw - 12, slot_h - 12))
                icon = self.move_icon(move, icon_size)
                icon_cy = sy + slot_h // 2
                if icon is not None:
                    surf.blit(icon, icon.get_rect(center=(sx + sw // 2, icon_cy)))
                else:
                    draw_text(surf, move.value, self.f(14), WHITE,
                              sx + sw // 2, icon_cy)
            else:
                draw_text(surf, str(i + 1), self.f(16), (40, 40, 60),
                          sx + sw // 2, sy + slot_h // 2)

