"""
moves.py
────────
Defines the two fundamental building blocks of the game:

  Action  – a single gesture recognised by the camera (or a key-press).
  Move    – a 3-action combo that produces an in-game effect.

Combo categories
────────────────
  Power Strike (weak)   – any 3 identical actions.
  Combo Blast  (strong) – 3 different actions in a fixed order.
  Shield Wall           – 3 different actions in a fixed order.
  Dodge Roll            – 3 different actions in a fixed order.
  Mend         (heal)   – 1st and 3rd actions the same, 2nd different.

Resolution rules
────────────────
  Shield Wall:
    vs strong → takes SHIELD_VS_STRONG_DAMAGE_PCT of the damage
    vs weak   → reflects SHIELD_VS_WEAK_REFLECT_PCT of damage back at attacker
    vs def/heal → no effect
  Dodge Roll:
    vs strong → DODGE_VS_STRONG_CHANCE chance to fully dodge
    vs weak   → DODGE_VS_WEAK_CHANCE chance to fully dodge
    vs def/heal → no effect
  Mend:
    if opponent attacked → heal does not go through
    otherwise            → heal HEAL_AMOUNT HP

Tunable numbers (damage, heal, percentages, chances) and the action
sequences themselves live in configs/move_config.py — edit there to rebalance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from configs import move_config as cfg


# ── Round-time modifiers ──────────────────────────────────────────────────────

@dataclass
class RoundContext:
    """Per-round modifiers passed into resolve_moves.

    heal_multiplier: scales the Mend heal output. Used by deathmatch mode to
    nerf healing. Floor is applied after multiplication.
    """
    heal_multiplier: float = 1.0


_DEFAULT_CONTEXT = RoundContext()


# ── Action ────────────────────────────────────────────────────────────────────

class Action(Enum):
    MOVE1 = "move1"   # Q / Y
    MOVE2 = "move2"   # W / U
    MOVE3 = "move3"   # E / I
    MOVE4 = "move4"   # R / O
    MOVE5 = "move5"   # T / P


def _seq_from_names(names: list[str]) -> list[Action]:
    return [Action(n) for n in names]


# ── Move ──────────────────────────────────────────────────────────────────────

class Move(Enum):
    # Offensive
    POWER_STRIKE = "Power Strike"   # weak
    COMBO_BLAST  = "Combo Blast"    # strong
    # Defensive
    SHIELD_WALL  = "Shield Wall"
    DODGE_ROLL   = "Dodge Roll"
    # Healing
    MEND         = "Mend"


# ── Combo definitions (ORDER MATTERS — specific before generic) ───────────────

_STRONG_SEQ = _seq_from_names(cfg.STRONG_MOVE_SEQUENCE)
_SHIELD_SEQ = _seq_from_names(cfg.SHIELD_SEQUENCE)
_DODGE_SEQ  = _seq_from_names(cfg.DODGE_SEQUENCE)


def _all_same(seq: list[Action]) -> bool:
    return seq[0] == seq[1] == seq[2]


def _first_equals_third(seq: list[Action]) -> bool:
    # 1st and 3rd identical, 2nd different (excludes the all-same case which
    # belongs to Power Strike).
    return seq[0] == seq[2] and seq[1] != seq[0]


COMBO_TABLE: list[tuple[Move, callable]] = [
    # Specific 3-different sequences first.
    (Move.COMBO_BLAST, lambda s: s == _STRONG_SEQ),
    (Move.SHIELD_WALL, lambda s: s == _SHIELD_SEQ),
    (Move.DODGE_ROLL,  lambda s: s == _DODGE_SEQ),
    # Generic patterns.
    (Move.POWER_STRIKE, _all_same),
    (Move.MEND,         _first_equals_third),
]


def match_combo(seq: list[Action]) -> Optional[Move]:
    """
    Walk COMBO_TABLE in priority order and return the first matching Move,
    or None if no combo matched (invalid / unrecognised sequence).
    """
    if len(seq) != 3:
        raise ValueError(f"match_combo expects exactly 3 actions, got {len(seq)}")
    for move, check in COMBO_TABLE:
        if check(seq):
            return move
    return None


# ── Move metadata ─────────────────────────────────────────────────────────────

MOVE_TYPE: dict[Move, str] = {
    Move.POWER_STRIKE: "offensive",
    Move.COMBO_BLAST:  "offensive",
    Move.SHIELD_WALL:  "defensive",
    Move.DODGE_ROLL:   "defensive",
    Move.MEND:         "healing",
}

MOVE_COLORS: dict[Move, tuple] = {
    Move.POWER_STRIKE: (255,  80,  50),
    Move.COMBO_BLAST:  (255, 160,  40),
    Move.SHIELD_WALL:  ( 60, 180, 255),
    Move.DODGE_ROLL:   ( 80, 220, 180),
    Move.MEND:         ( 80, 220, 120),
}


# Human-readable combo hints rendered from the configured sequences so the
# UI stays in sync with configs/move_config.py.
_CIRCLED = {"move1": "①", "move2": "②", "move3": "③", "move4": "④", "move5": "⑤"}


def _hint(names: list[str]) -> str:
    return " → ".join(_CIRCLED[n] for n in names)


MOVE_HINTS: dict[Move, str] = {
    Move.POWER_STRIKE: "Any ①①① / ②②② / ③③③ / ④④④ / ⑤⑤⑤",
    Move.COMBO_BLAST:  _hint(cfg.STRONG_MOVE_SEQUENCE),
    Move.SHIELD_WALL:  _hint(cfg.SHIELD_SEQUENCE),
    Move.DODGE_ROLL:   _hint(cfg.DODGE_SEQUENCE),
    Move.MEND:         "Any X → Y → X  (X ≠ Y)",
}


# ── Resolution logic ──────────────────────────────────────────────────────────

ATTACK_MOVES = (Move.POWER_STRIKE, Move.COMBO_BLAST)


def _variance() -> int:
    v = cfg.DAMAGE_VARIANCE
    return random.randint(-v, v) if v > 0 else 0


def _raw_attack(move: Optional[Move]) -> int:
    if move == Move.POWER_STRIKE: return cfg.WEAK_MOVE_DAMAGE   + _variance()
    if move == Move.COMBO_BLAST:  return cfg.STRONG_MOVE_DAMAGE + _variance()
    return 0


def _heal_amount() -> int:
    return cfg.HEAL_AMOUNT + _variance()


def _apply_defense(
    attacker_move: Optional[Move],
    raw_dmg: int,
    defender_move: Optional[Move],
) -> tuple[int, int, Optional[str]]:
    """
    Resolve a single attack against the defender's move.

    Returns (damage_to_defender, damage_reflected_to_attacker, event)
    where event is one of "dodge_success", "dodge_fail", "reflect", or None.
    """
    if raw_dmg <= 0:
        return 0, 0, None

    if defender_move == Move.SHIELD_WALL:
        if attacker_move == Move.COMBO_BLAST:
            return math.floor(raw_dmg * cfg.SHIELD_VS_STRONG_DAMAGE_PCT), 0, None
        if attacker_move == Move.POWER_STRIKE:
            return 0, math.floor(raw_dmg * cfg.SHIELD_VS_WEAK_REFLECT_PCT), "reflect"

    if defender_move == Move.DODGE_ROLL:
        if attacker_move == Move.COMBO_BLAST:
            if random.random() < cfg.DODGE_VS_STRONG_CHANCE:
                return 0, 0, "dodge_success"
            return raw_dmg, 0, "dodge_fail"
        if attacker_move == Move.POWER_STRIKE:
            if random.random() < cfg.DODGE_VS_WEAK_CHANCE:
                return 0, 0, "dodge_success"
            return raw_dmg, 0, "dodge_fail"

    # No defense (or defender used heal / another attack) — full damage lands.
    return raw_dmg, 0, None


def resolve_moves(
    move1: Optional[Move],
    move2: Optional[Move],
    ctx: RoundContext = _DEFAULT_CONTEXT,
) -> tuple[int, int, int, int, list[tuple[str, int]]]:
    """
    Compute HP deltas for a single round of combat.

    Returns
    -------
    (p1_delta, p2_delta, dmg_landed_by_p1, dmg_landed_by_p2, events)
      - positive delta → HP gain
      - negative delta → HP loss
      - dmg_landed_by_X → damage X actually inflicted on the opponent
        (after the opponent's defense), used for VFX.
      - events → list of (event_name, defender_player_num) tuples for UI
        feedback. event_name is one of "dodge_success", "dodge_fail",
        "reflect"; defender_player_num is 1 or 2.
    """
    raw1 = _raw_attack(move1)
    raw2 = _raw_attack(move2)

    # P1 attacks → P2's defense applies. Reflected damage comes back to P1.
    dmg_to_p2, reflect_to_p1, ev_p2 = _apply_defense(move1, raw1, move2)
    # P2 attacks → P1's defense applies. Reflected damage comes back to P2.
    dmg_to_p1, reflect_to_p2, ev_p1 = _apply_defense(move2, raw2, move1)

    events: list[tuple[str, int]] = []
    if ev_p2 is not None: events.append((ev_p2, 2))
    if ev_p1 is not None: events.append((ev_p1, 1))

    p1_delta = -(dmg_to_p1 + reflect_to_p1)
    p2_delta = -(dmg_to_p2 + reflect_to_p2)

    # Healing — only goes through if the opponent did NOT play an attack.
    if move1 == Move.MEND and move2 not in ATTACK_MOVES:
        p1_delta += math.floor(_heal_amount() * ctx.heal_multiplier)
    if move2 == Move.MEND and move1 not in ATTACK_MOVES:
        p2_delta += math.floor(_heal_amount() * ctx.heal_multiplier)

    return p1_delta, p2_delta, dmg_to_p2, dmg_to_p1, events
