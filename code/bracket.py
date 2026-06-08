"""
bracket.py
──────────
Tournament bracket display for a 6-team Sigil Strike event.

Structure:
  Group stage  – two groups of 3, round-robin (3 matches per group).
  Knockout     – top-2 from each group → two semifinals → grand final.
                 A1 vs B2, B1 vs A2.
  Consolation  – 5th-place playoff: A3 vs B3 (same stage as the semifinals).
                 3rd-place playoff: SF losers (same stage as the final).
  Tiebreaker   – equal wins resolved by shorter total game time.

Run standalone:  python bracket.py
Embed elsewhere: create Tournament, populate match results, call
                 BracketRenderer().draw(surface, tournament).
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Optional

import pygame

import audio
from configs.team_colors import TEAM_COLORS
from game import Game
from paths import external_dir


# ── Palette ───────────────────────────────────────────────────────────────────

BG_DARK      = (8,   8,  18)
BG_MID       = (14,  14, 30)
PANEL_COLOR  = (20,  20, 42)
BORDER_COLOR = (50,  60, 120)
DARK_GRAY    = (35,  35, 55)
WHITE        = (255, 255, 255)
GRAY         = (120, 120, 140)
DIM          = (60,  60,  90)
GREEN        = (80,  220, 120)
RED          = (220,  60,  60)
GOLD         = (255, 200,  60)
SILVER       = (180, 180, 200)

WIDTH, HEIGHT = 1280, 720
FPS = 60

# Layout geometry
_HEADER_H  = 48
_GROUP_H   = 360
_GROUP_Y   = _HEADER_H                          # 48
_KO_Y      = _GROUP_Y + _GROUP_H               # 408
_KO_H      = HEIGHT - _KO_Y                    # 312


# ── Display manager ───────────────────────────────────────────────────────────

class Display:
    """Renders at a fixed 1280×720 logical resolution and scales to any window."""

    def __init__(self):
        self._fullscreen = False
        self.screen  = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.logical = pygame.Surface((WIDTH, HEIGHT))

    def toggle(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    def to_logical(self, pos: tuple) -> tuple:
        """Map a physical-screen position into logical (WIDTH × HEIGHT) space."""
        sw, sh = self.screen.get_size()
        scale  = min(sw / WIDTH, sh / HEIGHT)
        ox     = (sw - int(WIDTH  * scale)) // 2
        oy     = (sh - int(HEIGHT * scale)) // 2
        return ((pos[0] - ox) / scale, (pos[1] - oy) / scale)

    def present(self) -> None:
        """Scale the logical surface to fill the screen (letterboxed) and flip."""
        sw, sh  = self.screen.get_size()
        scale   = min(sw / WIDTH, sh / HEIGHT)
        sw_out  = int(WIDTH  * scale)
        sh_out  = int(HEIGHT * scale)
        scaled  = pygame.transform.smoothscale(self.logical, (sw_out, sh_out))
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, ((sw - sw_out) // 2, (sh - sh_out) // 2))
        pygame.display.flip()


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Team:
    name: str
    color: tuple


@dataclass
class GroupMatch:
    team_a: Team
    team_b: Team
    score_a: Optional[int] = None   # None = not yet played
    score_b: Optional[int] = None
    duration: Optional[float] = None  # game seconds – tiebreaker


@dataclass
class Standing:
    team: Team
    wins: int = 0
    losses: int = 0
    total_time: float = 0.0
    rank: int = 0


@dataclass
class KnockoutMatch:
    label: str
    team_a: Optional[Team] = None
    team_b: Optional[Team] = None
    score_a: Optional[int] = None
    score_b: Optional[int] = None
    winner: Optional[Team] = None


# ── Tournament state ──────────────────────────────────────────────────────────

class Tournament:
    """Holds all match data and computes standings on demand."""

    def __init__(self, group_a: list[Team], group_b: list[Team]):
        assert len(group_a) == 3 and len(group_b) == 3
        self.group_a   = group_a
        self.group_b   = group_b
        self.matches_a = self._rr(group_a)
        self.matches_b = self._rr(group_b)
        self.sf1   = KnockoutMatch("SEMI-FINAL 1")   # A1 vs B2
        self.sf2   = KnockoutMatch("SEMI-FINAL 2")   # B1 vs A2
        self.fifth = KnockoutMatch("5TH PLACE")      # A3 vs B3
        self.third = KnockoutMatch("3RD PLACE")      # SF1 loser vs SF2 loser
        self.final = KnockoutMatch("GRAND FINAL")

    @staticmethod
    def _rr(teams: list[Team]) -> list[GroupMatch]:
        return [GroupMatch(teams[i], teams[j])
                for i in range(len(teams))
                for j in range(i + 1, len(teams))]

    def _standings(self, teams: list[Team],
                   matches: list[GroupMatch]) -> list[Standing]:
        rows: dict[str, Standing] = {t.name: Standing(t) for t in teams}
        for m in matches:
            if m.score_a is None:
                continue
            a, b = rows[m.team_a.name], rows[m.team_b.name]
            if m.score_a > m.score_b:
                a.wins   += 1; b.losses += 1
            elif m.score_b > m.score_a:
                b.wins   += 1; a.losses += 1
            # draw: neither gains a win; time still counted
            if m.duration is not None:
                a.total_time += m.duration
                b.total_time += m.duration
        ranked = sorted(rows.values(), key=lambda s: (-s.wins, s.total_time))
        for i, s in enumerate(ranked):
            s.rank = i + 1
        return ranked

    def standings_a(self) -> list[Standing]:
        return self._standings(self.group_a, self.matches_a)

    def standings_b(self) -> list[Standing]:
        return self._standings(self.group_b, self.matches_b)

    def seed_knockout(self) -> None:
        """Assign SF + 5th-place seeds from group standings (call after group stage)."""
        sa, sb = self.standings_a(), self.standings_b()
        self.sf1.team_a   = sa[0].team   # A1
        self.sf1.team_b   = sb[1].team   # B2
        self.sf2.team_a   = sb[0].team   # B1
        self.sf2.team_b   = sa[1].team   # A2
        self.fifth.team_a = sa[2].team   # A3
        self.fifth.team_b = sb[2].team   # B3

    def seed_final(self) -> None:
        """Assign finalists from semi-final winners (call after both SFs)."""
        self.final.team_a = self.sf1.winner
        self.final.team_b = self.sf2.winner

    # ── Result management (used by bracket UI) ────────────────────────────────

    def set_group_result(self, match: GroupMatch,
                         winner_is_a: bool, duration: float = 0.0) -> None:
        match.score_a = 1 if winner_is_a else 0
        match.score_b = 0 if winner_is_a else 1
        match.duration = duration if duration > 0 else None
        self._reseed()

    def clear_group_result(self, match: GroupMatch) -> None:
        match.score_a = match.score_b = match.duration = None
        self._reseed()

    def set_knockout_result(self, match: KnockoutMatch,
                            winner_is_a: bool) -> None:
        match.score_a = 1 if winner_is_a else 0
        match.score_b = 0 if winner_is_a else 1
        match.winner  = match.team_a if winner_is_a else match.team_b
        loser         = match.team_b if winner_is_a else match.team_a
        if match is self.sf1:
            self.final.team_a = match.winner
            self.third.team_a = loser
            self._clear_final_result()
            self._clear_third_result()
        elif match is self.sf2:
            self.final.team_b = match.winner
            self.third.team_b = loser
            self._clear_final_result()
            self._clear_third_result()

    def clear_knockout_result(self, match: KnockoutMatch) -> None:
        old_winner = match.winner
        old_loser  = (match.team_b if old_winner is match.team_a
                      else match.team_a if old_winner is match.team_b
                      else None)
        match.score_a = match.score_b = match.winner = None
        if match is self.sf1 and old_winner is not None:
            if self.final.team_a is old_winner:
                self.final.team_a = None
            if self.third.team_a is old_loser:
                self.third.team_a = None
            self._clear_final_result()
            self._clear_third_result()
        elif match is self.sf2 and old_winner is not None:
            if self.final.team_b is old_winner:
                self.final.team_b = None
            if self.third.team_b is old_loser:
                self.third.team_b = None
            self._clear_final_result()
            self._clear_third_result()

    def _clear_final_result(self) -> None:
        self.final.score_a = self.final.score_b = self.final.winner = None

    def _clear_third_result(self) -> None:
        self.third.score_a = self.third.score_b = self.third.winner = None

    def _clear_fifth_result(self) -> None:
        self.fifth.score_a = self.fifth.score_b = self.fifth.winner = None

    def _reseed(self) -> None:
        """Recompute knockout seeds whenever a group result changes."""
        all_done = (all(m.score_a is not None for m in self.matches_a) and
                    all(m.score_b is not None for m in self.matches_b))
        if not all_done:
            for ko in (self.sf1, self.sf2, self.fifth, self.third, self.final):
                ko.team_a = ko.team_b = ko.score_a = ko.score_b = ko.winner = None
            return
        sa, sb    = self.standings_a(), self.standings_b()
        new_sf1   = (sa[0].team, sb[1].team)
        new_sf2   = (sb[0].team, sa[1].team)
        new_fifth = (sa[2].team, sb[2].team)
        if (self.sf1.team_a, self.sf1.team_b) != new_sf1:
            self.clear_knockout_result(self.sf1)
        self.sf1.team_a, self.sf1.team_b = new_sf1
        if (self.sf2.team_a, self.sf2.team_b) != new_sf2:
            self.clear_knockout_result(self.sf2)
        self.sf2.team_a, self.sf2.team_b = new_sf2
        if (self.fifth.team_a, self.fifth.team_b) != new_fifth:
            self._clear_fifth_result()
        self.fifth.team_a, self.fifth.team_b = new_fifth


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _rrect(surf: pygame.Surface, color, rect,
           r: int = 8, bw: int = 0, bc=None) -> None:
    if color:
        pygame.draw.rect(surf, color, rect, border_radius=r)
    if bw and bc:
        pygame.draw.rect(surf, bc, rect, bw, border_radius=r)


def _txt(surf: pygame.Surface, text: str, font: pygame.font.Font,
         color, cx: int, cy: int, anchor: str = "center") -> None:
    img = font.render(text, True, color)
    rct = img.get_rect()
    if   anchor == "center": rct.center   = (cx, cy)
    elif anchor == "left":   rct.midleft  = (cx, cy)
    elif anchor == "right":  rct.midright = (cx, cy)
    surf.blit(img, rct)


def _fmt_time(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


# ── Renderer ──────────────────────────────────────────────────────────────────

class BracketRenderer:
    _SIZES = [11, 12, 13, 14, 16, 18, 20, 24, 28, 32, 40]
    _BOX_W = 290   # width of every match box
    _ROW_H = 40    # height of each team row (box height = _ROW_H * 2)

    _SYM_FAMILY = (
        "segoeuisymbol,applesymbols,"
        "notosanssymbols2,notosanssymbols,"
        "dejavusans,freesans,unifont,arial"
    )

    def __init__(self):
        self.fonts: dict[int, pygame.font.Font] = {}
        for sz in self._SIZES:
            try:
                self.fonts[sz] = pygame.font.SysFont(
                    "couriernew,lucidaconsole,monospace", sz, bold=True)
            except Exception:
                self.fonts[sz] = pygame.font.Font(None, sz)

        self.sym_fonts: dict[int, pygame.font.Font] = {}
        for sz in self._SIZES:
            try:
                sf = pygame.font.SysFont(self._SYM_FAMILY, sz, bold=True)
                self.sym_fonts[sz] = sf if sf is not None else self.fonts[sz]
            except Exception:
                self.sym_fonts[sz] = self.fonts[sz]

        self._hit_rects: list[tuple[pygame.Rect, object]] = []

    def f(self, sz: int) -> pygame.font.Font:
        return self.fonts.get(sz, self.fonts[16])

    def sym(self, sz: int) -> pygame.font.Font:
        return self.sym_fonts.get(sz, self.sym_fonts.get(16, self.fonts[16]))

    # ── Top-level ─────────────────────────────────────────────────────────────

    def draw(self, surf: pygame.Surface, t: Tournament,
             highlighted_match: object = None) -> None:
        self._hit_rects = []   # rebuilt every frame
        surf.fill(BG_DARK)
        self._grid(surf)
        self._header(surf)
        pw = WIDTH // 2
        self._group_panel(surf, "GROUP A", t.matches_a, t.standings_a(), 0,   pw,
                          highlighted_match)
        self._group_panel(surf, "GROUP B", t.matches_b, t.standings_b(), pw,  pw,
                          highlighted_match)
        self._knockout(surf, t, highlighted_match)

    # ── Background grid ───────────────────────────────────────────────────────

    def _grid(self, surf: pygame.Surface) -> None:
        for x in range(0, WIDTH, 60):
            pygame.draw.line(surf, (16, 16, 36), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 60):
            pygame.draw.line(surf, (16, 16, 36), (0, y), (WIDTH, y))

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self, surf: pygame.Surface) -> None:
        _rrect(surf, PANEL_COLOR, pygame.Rect(0, 0, WIDTH, _HEADER_H),
               r=0, bw=2, bc=BORDER_COLOR)
        _txt(surf, "✦  SIGIL STRIKE  —  TOURNAMENT BRACKET  ✦",
             self.sym(24), GOLD, WIDTH // 2, _HEADER_H // 2)

    # ── Group panel ───────────────────────────────────────────────────────────

    def _group_panel(self, surf: pygame.Surface, label: str,
                     matches: list[GroupMatch], standings: list[Standing],
                     px: int, pw: int,
                     highlighted_match: object = None) -> None:
        _rrect(surf, BG_MID,
               pygame.Rect(px, _GROUP_Y, pw, _GROUP_H),
               r=0, bw=1, bc=BORDER_COLOR)

        cx = px + pw // 2
        y  = _GROUP_Y + 14

        # Group title
        _txt(surf, label, self.f(20), GOLD, cx, y + 8)
        y += 28
        pygame.draw.line(surf, BORDER_COLOR, (px + 12, y), (px + pw - 12, y), 1)
        y += 10

        # Matches
        _txt(surf, "MATCHES", self.f(11), GRAY, px + 14, y + 6, anchor="left")
        y += 18
        for m in matches:
            self._match_row(surf, m, px + 10, y, pw - 20,
                            highlight=(m is highlighted_match))
            y += 46

        pygame.draw.line(surf, BORDER_COLOR, (px + 12, y), (px + pw - 12, y), 1)
        y += 10

        # Standings
        _txt(surf, "STANDINGS  (tiebreaker: game time)",
             self.f(11), GRAY, px + 14, y + 6, anchor="left")
        y += 18
        for s in standings:
            self._standing_row(surf, s, px + 10, y, pw - 20)
            y += 36

    def _match_row(self, surf: pygame.Surface, m: GroupMatch,
                   x: int, y: int, w: int,
                   highlight: bool = False) -> None:
        h   = 38
        mid = x + w // 2
        self._hit_rects.append((pygame.Rect(x, y, w, h), m))
        border = GOLD if highlight else BORDER_COLOR
        _rrect(surf, DARK_GRAY, pygame.Rect(x, y, w, h), r=6, bw=2 if highlight else 1, bc=border)

        my = y + h // 2

        # Team names
        a_col = m.team_a.color
        b_col = m.team_b.color
        if m.score_a is not None:
            if m.score_a < m.score_b:
                a_col = tuple(max(c - 90, 20) for c in a_col)
            elif m.score_b < m.score_a:
                b_col = tuple(max(c - 90, 20) for c in b_col)

        _txt(surf, m.team_a.name, self.f(13), a_col, mid - 78, my, anchor="right")
        _txt(surf, m.team_b.name, self.f(13), b_col, mid + 78, my, anchor="left")

        if m.score_a is not None:
            sc  = f"{m.score_a}  —  {m.score_b}"
            if m.score_a > m.score_b:
                sc_col = m.team_a.color
            elif m.score_b > m.score_a:
                sc_col = m.team_b.color
            else:
                sc_col = GRAY
            _txt(surf, sc, self.f(14), sc_col, mid, my)
            if m.duration is not None:
                _txt(surf, _fmt_time(m.duration), self.f(11), DIM,
                     x + w - 4, my, anchor="right")
        else:
            _txt(surf, "VS", self.f(13), DIM, mid, my)

    def _standing_row(self, surf: pygame.Surface, s: Standing,
                      x: int, y: int, w: int) -> None:
        h         = 28
        qualifier = s.rank <= 2
        bg        = (18, 42, 22) if qualifier else (22, 22, 44)
        border    = (40, 110, 55) if qualifier else BORDER_COLOR
        _rrect(surf, bg, pygame.Rect(x, y, w, h), r=5, bw=1, bc=border)

        my     = y + h // 2
        r_col  = GOLD if s.rank == 1 else (GREEN if qualifier else GRAY)
        prefix = "▲ " if qualifier else "   "
        _txt(surf, f"{prefix}#{s.rank}", self.sym(12), r_col, x + 22, my, anchor="center")
        _txt(surf, s.team.name, self.f(13), s.team.color, x + 52, my, anchor="left")

        played = s.wins + s.losses
        _txt(surf, f"W {s.wins}   L {s.losses}", self.f(12), WHITE,
             x + w // 2 + 50, my, anchor="center")

        if played > 0 and s.total_time > 0:
            _txt(surf, _fmt_time(s.total_time), self.f(11), GRAY,
                 x + w - 4, my, anchor="right")

    # ── Knockout bracket ──────────────────────────────────────────────────────

    def _ko_box(self, surf: pygame.Surface, match: KnockoutMatch,
                bx: int, by: int,
                seed_a: str = "", seed_b: str = "",
                label: str = "",
                highlight: bool = False) -> None:
        """
        Draw a two-row match box. Top row = team_a, bottom row = team_b.
        Winner row is highlighted; loser row is greyed out.
        """
        BOX_W   = self._BOX_W
        ROW_H   = self._ROW_H
        decided = match.winner is not None
        self._hit_rects.append((pygame.Rect(bx, by, BOX_W, ROW_H * 2), match))

        for idx, (team, score, seed) in enumerate([
            (match.team_a, match.score_a, seed_a),
            (match.team_b, match.score_b, seed_b),
        ]):
            ry  = by + idx * ROW_H
            won = decided and (match.winner is team)

            # Row background
            if not decided:
                bg = DARK_GRAY
            elif won:
                tc = team.color if team else WHITE
                bg = tuple(min(255, c // 5 + 26) for c in tc)  # subtle tint
            else:
                bg = (18, 18, 32)  # near-black for loser

            # Rounded corners only on the outer edges
            if idx == 0:
                pygame.draw.rect(surf, bg, pygame.Rect(bx, ry, BOX_W, ROW_H),
                                 border_top_left_radius=8, border_top_right_radius=8)
            else:
                pygame.draw.rect(surf, bg, pygame.Rect(bx, ry, BOX_W, ROW_H),
                                 border_bottom_left_radius=8, border_bottom_right_radius=8)

            # Text colours
            if decided and not won:
                name_col = (58, 58, 78)
                num_col  = (48, 48, 68)
                seed_col = (48, 48, 68)
            else:
                name_col = team.color if team else DIM
                num_col  = WHITE
                seed_col = GRAY

            my       = ry + ROW_H // 2
            txt_x    = bx + 10

            if seed:
                _txt(surf, seed, self.f(11), seed_col, txt_x, my, anchor="left")
                txt_x += 26

            name = team.name if team else "TBD"
            _txt(surf, name, self.f(13), name_col, txt_x, my, anchor="left")

            if score is not None:
                _txt(surf, str(score), self.f(14), num_col,
                     bx + BOX_W - 12, my, anchor="right")

        # Divider between the two rows
        pygame.draw.line(surf, BORDER_COLOR,
                         (bx, by + ROW_H), (bx + BOX_W, by + ROW_H), 1)

        # Outer border (drawn last so it sits on top of row fills)
        border = GOLD if highlight else BORDER_COLOR
        pygame.draw.rect(surf, border,
                         pygame.Rect(bx, by, BOX_W, ROW_H * 2),
                         3 if highlight else 2, border_radius=8)

        # Label centred above the box
        if label:
            _txt(surf, label, self.f(11), GRAY,
                 bx + BOX_W // 2, by - 10)

    def _knockout(self, surf: pygame.Surface, t: Tournament,
                  highlighted_match: object = None) -> None:
        ROW_H  = self._ROW_H
        BOX_W  = self._BOX_W
        BOX_H  = ROW_H * 2    # 80
        SF_GAP = 76           # vertical gap between SF1 and SF2 (also between 3RD and 5TH)
        COL_GAP = 30          # horizontal gap between consolation column and SF column
        CONN_W  = 140         # horizontal width of the SF→Final connector section

        # ── Horizontal layout (3 columns: consolation | SF | final) ───────────
        total_w = BOX_W + COL_GAP + BOX_W + CONN_W + BOX_W   # 1050
        cons_x  = (WIDTH - total_w) // 2                     # 115
        sf_x    = cons_x + BOX_W + COL_GAP                   # 435
        fin_x   = sf_x + BOX_W + CONN_W                      # 865

        # ── Vertical layout ───────────────────────────────────────────────────
        LABEL_H   = 28
        bracket_h = BOX_H + SF_GAP + BOX_H                   # 236
        avail     = _KO_H - LABEL_H
        sf1_y     = _KO_Y + LABEL_H + (avail - bracket_h) // 2
        sf2_y     = sf1_y + BOX_H + SF_GAP
        fin_y     = sf1_y + (BOX_H + SF_GAP) // 2
        # Consolation column shares the SF rows: 5TH aligns with SF1, 3RD with SF2.
        fifth_y   = sf1_y
        third_y   = sf2_y

        # Section label
        _txt(surf, "KNOCKOUT STAGE", self.f(16), GRAY,
             WIDTH // 2, _KO_Y + LABEL_H // 2)

        # ── Connector lines: SF → Final ───────────────────────────────────────
        sf1_my  = sf1_y + ROW_H
        sf2_my  = sf2_y + ROW_H
        mid_y   = (sf1_my + sf2_my) // 2   # = fin_y + ROW_H
        elbow_x = sf_x + BOX_W + CONN_W // 2

        lc, lw = BORDER_COLOR, 2
        pygame.draw.line(surf, lc, (sf_x + BOX_W, sf1_my), (elbow_x, sf1_my), lw)
        pygame.draw.line(surf, lc, (sf_x + BOX_W, sf2_my), (elbow_x, sf2_my), lw)
        pygame.draw.line(surf, lc, (elbow_x, sf1_my), (elbow_x, sf2_my), lw)
        pygame.draw.line(surf, lc, (elbow_x, mid_y), (fin_x, mid_y), lw)
        aw, ah = 10, 6
        pygame.draw.polygon(surf, lc, [
            (fin_x,      mid_y),
            (fin_x - aw, mid_y - ah),
            (fin_x - aw, mid_y + ah),
        ])

        # ── Match boxes ───────────────────────────────────────────────────────
        # Consolation column (standalone — no connectors)
        self._ko_box(surf, t.third, cons_x, third_y,
                     label="3RD PLACE",
                     highlight=(t.third is highlighted_match))
        self._ko_box(surf, t.fifth, cons_x, fifth_y,
                     seed_a="A3", seed_b="B3", label="5TH PLACE",
                     highlight=(t.fifth is highlighted_match))
        # Semifinals
        self._ko_box(surf, t.sf1,   sf_x,   sf1_y,
                     seed_a="A1", seed_b="B2", label="SEMI-FINAL 1",
                     highlight=(t.sf1 is highlighted_match))
        self._ko_box(surf, t.sf2,   sf_x,   sf2_y,
                     seed_a="B1", seed_b="A2", label="SEMI-FINAL 2",
                     highlight=(t.sf2 is highlighted_match))
        # Final
        self._ko_box(surf, t.final, fin_x,  fin_y,
                     label="GRAND FINAL",
                     highlight=(t.final is highlighted_match))

        # ── Champion banner ───────────────────────────────────────────────────
        if t.final.winner:
            champ_x = fin_x + BOX_W + 16
            champ_y = fin_y + ROW_H
            _txt(surf, "CHAMPION", self.f(12), GOLD,
                 champ_x, champ_y - 13, anchor="left")
            _txt(surf, t.final.winner.name, self.f(18), t.final.winner.color,
                 champ_x, champ_y + 11, anchor="left")


    def get_match_at(self, pos: tuple[int, int]) -> Optional[object]:
        for rect, match in self._hit_rects:
            if rect.collidepoint(pos):
                return match
        return None

    def draw_context_menu(
        self,
        surf: pygame.Surface,
        match: object,
        mx: int, my: int,
        logical_mouse: tuple = None,
    ) -> list[tuple[pygame.Rect, str]]:
        """Draw a right-click context menu; return list of (rect, action) for hit-testing."""
        is_group = isinstance(match, GroupMatch)
        has_teams  = match.team_a is not None and match.team_b is not None
        has_result = (match.score_a is not None if is_group else match.winner is not None)

        options: list[tuple[str, str, tuple]] = []   # (label, action_key, colour)
        if has_teams:
            options.append(("▶  Play match",                        "play",  GREEN))
            options.append((f"■  {match.team_a.name} wins",         "set_a", WHITE))
            options.append((f"■  {match.team_b.name} wins",         "set_b", WHITE))
        if has_result:
            options.append(("✕  Clear result",                      "clear", RED))

        if not options:
            return []

        ITEM_H = 32
        MENU_W = 210
        MENU_H = len(options) * ITEM_H + 8

        px = min(mx + 6, WIDTH  - MENU_W - 4)
        py = min(my,     HEIGHT - MENU_H - 4)

        _rrect(surf, (12, 12, 24), pygame.Rect(px - 2, py - 2, MENU_W + 4, MENU_H + 4),
               r=8, bw=1, bc=BORDER_COLOR)
        _rrect(surf, PANEL_COLOR, pygame.Rect(px, py, MENU_W, MENU_H), r=6)

        mouse      = logical_mouse if logical_mouse is not None else pygame.mouse.get_pos()
        item_rects = []
        for i, (label, action, col) in enumerate(options):
            iy    = py + 4 + i * ITEM_H
            irect = pygame.Rect(px, iy, MENU_W, ITEM_H)
            if irect.collidepoint(mouse):
                _rrect(surf, DARK_GRAY, irect, r=4)
            _txt(surf, label, self.sym(13), col, px + 12, iy + ITEM_H // 2, anchor="left")
            item_rects.append((irect, action))

        return item_rects


# ── Team loading ─────────────────────────────────────────────────────────────

_TEAMS_DIR = external_dir() / "Teams"

_DEFAULT_NAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]
_FALLBACK_COLOR = (180, 180, 180)   # used if TEAM_COLORS is missing a number


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


def load_teams() -> list[Team]:
    """Read Team1..Team6 names from Teams/<TeamN>/team.env; colors come from
    configs/team_colors.py."""
    teams = []
    for i, def_name in enumerate(_DEFAULT_NAMES, start=1):
        env = _parse_env_file(_TEAMS_DIR / f"Team{i}" / "team.env")
        name  = env.get("NAME", def_name)
        color = TEAM_COLORS.get(i, _FALLBACK_COLOR)
        teams.append(Team(name, color))
    return teams


# ── Group-draw lobby ──────────────────────────────────────────────────────────

def run_lobby(disp: Display, rend: BracketRenderer,
              teams: list[Team]) -> tuple[list[Team], list[Team]]:
    """
    Show the group-draw lobby with a shuffle animation.
    Clicking RANDOMIZE (or pressing R) triggers a rapid-then-decelerating
    card cycle before landing on the final grouping.
    Returns (group_a, group_b) when the user confirms with START / Enter.
    """
    # ── State ─────────────────────────────────────────────────────────────────
    cur_order   = teams[:]
    random.shuffle(cur_order)          # initial random draw

    animating       = False
    anim_timer      = 0.0              # time elapsed since animation started
    anim_swap_timer = 0.0              # time since last card swap
    final_order     = cur_order[:]     # grouping we'll land on
    flash_t         = 0.0              # 0-1 white-flash intensity on cards

    ANIM_TOTAL = 1.8                   # total animation length (seconds)
    SWAP_FAST  = 0.045                 # swap interval at start (fast)
    SWAP_SLOW  = 0.30                  # swap interval near end  (slow)

    def start_anim() -> None:
        nonlocal animating, anim_timer, anim_swap_timer, final_order, flash_t
        final_order     = cur_order[:]
        random.shuffle(final_order)    # decide the outcome up front
        animating       = True
        anim_timer      = 0.0
        anim_swap_timer = 0.0
        flash_t         = 1.0

    # ── Layout ────────────────────────────────────────────────────────────────
    PANEL_W = 380
    PANEL_H = 390
    GROUP_Y = 130
    SLOT_H  = (PANEL_H - 44) // 3     # ~115 px per team slot
    ga_x    = WIDTH  // 4 - PANEL_W // 2
    gb_x    = 3 * WIDTH // 4 - PANEL_W // 2

    BTN_W, BTN_H = 210, 58
    BTN_Y     = GROUP_Y + PANEL_H + 28
    rand_btn  = pygame.Rect(WIDTH // 4       - BTN_W // 2, BTN_Y, BTN_W, BTN_H)
    start_btn = pygame.Rect(3 * WIDTH // 4   - BTN_W // 2, BTN_Y, BTN_W, BTN_H)

    clock = pygame.time.Clock()

    while True:
        dt    = clock.tick(FPS) / 1000.0
        mouse = pygame.mouse.get_pos()

        # ── Animation tick ────────────────────────────────────────────────────
        if animating:
            anim_timer      += dt
            anim_swap_timer += dt
            flash_t          = max(0.0, flash_t - dt * 7)

            progress = anim_timer / ANIM_TOTAL
            if progress < 1.0:
                # Quadratic ease-out: interval grows from SWAP_FAST → SWAP_SLOW
                interval = SWAP_FAST + (SWAP_SLOW - SWAP_FAST) * (progress ** 2)
                if anim_swap_timer >= interval:
                    anim_swap_timer = 0.0
                    random.shuffle(cur_order)
                    flash_t = 1.0
            else:
                # Settle: snap to the pre-determined final grouping
                cur_order[:] = final_order
                animating     = False
                flash_t       = 0.7   # one last flash when it settles

        group_a = cur_order[:3]
        group_b = cur_order[3:]

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_F11:
                    disp.toggle()
                if event.key == pygame.K_r and not animating:
                    start_anim()
                if event.key == pygame.K_RETURN and not animating:
                    return group_a, group_b
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                lpos = disp.to_logical(event.pos)
                if not animating:
                    if rand_btn.collidepoint(lpos):
                        start_anim()
                    elif start_btn.collidepoint(lpos):
                        return group_a, group_b

        # ── Draw ──────────────────────────────────────────────────────────────
        surf = disp.logical
        mouse = disp.to_logical(pygame.mouse.get_pos())

        surf.fill(BG_DARK)
        for x in range(0, WIDTH, 60):
            pygame.draw.line(surf, (16, 16, 36), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 60):
            pygame.draw.line(surf, (16, 16, 36), (0, y), (WIDTH, y))

        # Header bar
        _rrect(surf, PANEL_COLOR, pygame.Rect(0, 0, WIDTH, 72),
               r=0, bw=2, bc=BORDER_COLOR)
        _txt(surf, "✦  SIGIL STRIKE  —  TOURNAMENT SETUP  ✦",
             rend.sym(24), GOLD, WIDTH // 2, 36)

        # Hint text – pulses gold while shuffling
        if animating:
            pulse     = (math.sin(anim_timer * 10) + 1) * 0.5
            hint_col  = tuple(int(GRAY[i] + (GOLD[i] - GRAY[i]) * pulse)
                              for i in range(3))
            hint_text = "SHUFFLING..."
        else:
            hint_col  = GRAY
            hint_text = "R = Randomize   ·   Enter = Start"
        _txt(surf, hint_text, rend.f(14), hint_col, WIDTH // 2, 100)

        # Group panels
        for gx, glabel, gteams in [
            (ga_x, "GROUP A", group_a),
            (gb_x, "GROUP B", group_b),
        ]:
            _rrect(surf, BG_MID,
                   pygame.Rect(gx, GROUP_Y, PANEL_W, PANEL_H),
                   r=8, bw=1, bc=BORDER_COLOR)
            _txt(surf, glabel, rend.f(18), GOLD,
                 gx + PANEL_W // 2, GROUP_Y + 20)
            pygame.draw.line(surf, BORDER_COLOR,
                             (gx + 12, GROUP_Y + 38),
                             (gx + PANEL_W - 12, GROUP_Y + 38), 1)

            for i, team in enumerate(gteams):
                ty   = GROUP_Y + 44 + i * SLOT_H
                srct = pygame.Rect(gx + 8, ty + 4, PANEL_W - 16, SLOT_H - 8)
                _rrect(surf, DARK_GRAY, srct, r=6, bw=1, bc=BORDER_COLOR)

                slot_sw = 28
                pygame.draw.rect(surf, team.color,
                                 pygame.Rect(gx + 20, srct.centery - slot_sw // 2,
                                             slot_sw, slot_sw),
                                 border_radius=5)
                _txt(surf, team.name, rend.f(20), team.color,
                     gx + 60, srct.centery, anchor="left")

                # White flash overlay on every swap
                if flash_t > 0:
                    fsurf = pygame.Surface((srct.width, srct.height),
                                          pygame.SRCALPHA)
                    fsurf.fill((255, 255, 255, int(flash_t * 55)))
                    surf.blit(fsurf, srct.topleft)

        # Buttons (RANDOMIZE greyed-out during animation)
        for btn, label, base_bg, hover_bg, base_col in [
            (rand_btn,  "⟳  RANDOMIZE", (30, 35, 85),  (50, 62, 140), WHITE),
            (start_btn, "▶  START",      (20, 65, 35),  (30, 100, 55), GREEN),
        ]:
            active = not animating
            hover  = btn.collidepoint(mouse) and active
            _rrect(surf, hover_bg if hover else base_bg, btn,
                   r=10, bw=2, bc=(GOLD if hover else BORDER_COLOR))
            col = (GOLD if hover else
                   (60, 60, 90) if (btn is rand_btn and animating)
                   else base_col)
            _txt(surf, label, rend.sym(18), col, btn.centerx, btn.centery)

        disp.present()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.display.set_caption("Sigil Strike — Tournament Bracket")
    clock = pygame.time.Clock()
    rend  = BracketRenderer()
    disp  = Display()

    teams = load_teams()
    group_a, group_b = run_lobby(disp, rend, teams)
    t = Tournament(group_a=group_a, group_b=group_b)

    # ── Context-menu state ────────────────────────────────────────────────────
    ctx_match  = None                              # match under the open menu
    ctx_pos    = (0, 0)                            # logical coords of the menu
    ctx_rects: list[tuple[pygame.Rect, str]] = []  # item rects from last draw

    # ── Tab navigation state ──────────────────────────────────────────────────
    tab_index: int = -1                            # -1 = nothing tab-selected

    # ── Helper: launch game for a match and record result ─────────────────────
    def play_match(match) -> None:
        ta, tb = match.team_a, match.team_b
        if ta is None or tb is None:
            return
        pygame.display.set_caption(f"Sigil Strike — {ta.name} vs {tb.name}")
        is_group = isinstance(match, GroupMatch)
        # Pick a BGM track per stage. Group + 5th-place share the same pool;
        # SF / 3rd / final each have a dedicated track.
        if is_group or match is t.fifth:
            bgm_track = random.choice([f"group_battle_{i}.mp3" for i in range(1, 5)])
        elif match is t.sf1 or match is t.sf2:
            bgm_track = "semi_battle.mp3"
        elif match is t.third:
            bgm_track = "third_place_battle.mp3"
        elif match is t.final:
            bgm_track = "final_battle.mp3"
        else:
            bgm_track = None
        on_start = (lambda track=bgm_track: audio.play_music(track)) if bgm_track else None
        try:
            g = Game(
                screen=disp.screen,
                p1_name=ta.name, p1_color=ta.color,
                p2_name=tb.name, p2_color=tb.color,
                tournament_mode=True,
                on_match_start=on_start,
            )
            winner_pid, duration = g.run_once()
            g.renderer.close()
        finally:
            if bgm_track:
                audio.stop_music()
        # Sync display reference in case the game toggled fullscreen
        disp.screen = pygame.display.get_surface()
        pygame.display.set_caption("Sigil Strike — Tournament Bracket")
        if winner_pid is None:
            return                                  # match was abandoned
        winner_is_a = (winner_pid == 1)
        if is_group:
            t.set_group_result(match, winner_is_a, duration)
        else:
            t.set_knockout_result(match, winner_is_a)

    # ── Helper: dispatch a context-menu action ────────────────────────────────
    def apply_action(match, action: str) -> None:
        if action == "play":
            play_match(match)
        elif action == "set_a":
            if isinstance(match, GroupMatch):
                t.set_group_result(match, winner_is_a=True)
            else:
                t.set_knockout_result(match, winner_is_a=True)
        elif action == "set_b":
            if isinstance(match, GroupMatch):
                t.set_group_result(match, winner_is_a=False)
            else:
                t.set_knockout_result(match, winner_is_a=False)
        elif action == "clear":
            if isinstance(match, GroupMatch):
                t.clear_group_result(match)
            else:
                t.clear_knockout_result(match)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    disp.toggle()
                elif event.key == pygame.K_TAB:
                    all_matches = (t.matches_a + t.matches_b +
                                   [t.sf1, t.sf2, t.fifth, t.third, t.final])
                    if event.mod & pygame.KMOD_SHIFT:
                        tab_index = (tab_index - 1) % len(all_matches)
                    else:
                        tab_index = (tab_index + 1) % len(all_matches)
                elif event.key == pygame.K_RETURN and tab_index >= 0:
                    all_matches = (t.matches_a + t.matches_b +
                                   [t.sf1, t.sf2, t.fifth, t.third, t.final])
                    if 0 <= tab_index < len(all_matches):
                        play_match(all_matches[tab_index])
                elif event.key == pygame.K_ESCAPE:
                    if ctx_match:
                        ctx_match = None        # dismiss menu on ESC
                    elif tab_index >= 0:
                        tab_index = -1          # clear tab selection on ESC
                    else:
                        pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                lpos = disp.to_logical(event.pos)
                if event.button == 1:           # left click
                    if ctx_match is not None:
                        # Check menu items first
                        hit = next((a for r, a in ctx_rects
                                    if r.collidepoint(lpos)), None)
                        if hit:
                            m = ctx_match
                            ctx_match = None
                            apply_action(m, hit)
                        else:
                            ctx_match = None    # click outside → dismiss
                    else:
                        # Direct click on a match box → play it
                        m = rend.get_match_at(lpos)
                        if m is not None:
                            play_match(m)

                elif event.button == 3:         # right click → context menu
                    ctx_match = None
                    m = rend.get_match_at(lpos)
                    if m is not None:
                        ctx_match = m
                        ctx_pos   = lpos

        # Determine highlighted match: hover takes priority over Tab
        all_matches = t.matches_a + t.matches_b + [t.sf1, t.sf2, t.fifth, t.third, t.final]
        lmouse_now = disp.to_logical(pygame.mouse.get_pos())
        hover_match = rend.get_match_at(lmouse_now)
        if hover_match is not None:
            highlighted = hover_match
        elif 0 <= tab_index < len(all_matches):
            highlighted = all_matches[tab_index]
        else:
            highlighted = None

        rend.draw(disp.logical, t, highlighted_match=highlighted)
        if ctx_match is not None:
            lmouse = disp.to_logical(pygame.mouse.get_pos())
            ctx_rects = rend.draw_context_menu(
                disp.logical, ctx_match, *ctx_pos, lmouse)
        else:
            ctx_rects = []
        disp.present()


if __name__ == "__main__":
    main()
