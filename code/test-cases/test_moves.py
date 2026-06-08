"""
Tests for moves.py: combo recognition and round resolution logic.

Variance is patched to 0 throughout so damage values are deterministic.
Dodge probability is controlled per-test via mock.
"""
import math
from unittest.mock import patch

import pytest

from moves import Action, Move, RoundContext, match_combo, resolve_moves, MOVE_TYPE

# ── Helpers ───────────────────────────────────────────────────────────────────

M1, M2, M3, M4, M5 = (Action(f"move{i}") for i in range(1, 6))

# Suppresses ±5 variance so every damage/heal value is the base constant.
NO_VARIANCE = patch("moves.random.randint", return_value=0)


# ── Combo recognition ─────────────────────────────────────────────────────────

class TestMatchCombo:
    def test_combo_blast(self):
        assert match_combo([M1, M2, M3]) == Move.COMBO_BLAST

    def test_shield_wall(self):
        assert match_combo([M3, M4, M5]) == Move.SHIELD_WALL

    def test_dodge_roll(self):
        assert match_combo([M1, M3, M5]) == Move.DODGE_ROLL

    def test_power_strike_aba(self):
        # Power Strike: X-Y-X with X != Y.
        assert match_combo([M1, M2, M1]) == Move.POWER_STRIKE

    def test_power_strike_dfd(self):
        assert match_combo([M3, M5, M3]) == Move.POWER_STRIKE

    def test_mend_all_same_move1(self):
        # Mend: any three identical actions.
        assert match_combo([M1, M1, M1]) == Move.MEND

    def test_mend_all_same_move4(self):
        assert match_combo([M4, M4, M4]) == Move.MEND

    def test_no_match_returns_none(self):
        # [M1, M2, M4] matches no specific sequence and no generic pattern.
        assert match_combo([M1, M2, M4]) is None

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            match_combo([M1, M2])

    def test_all_same_is_mend_not_power_strike(self):
        # [M2, M2, M2] fits the X-X-X pattern; Mend now owns that combo.
        assert match_combo([M2, M2, M2]) == Move.MEND

    def test_specific_sequence_beats_power_strike(self):
        # COMBO_BLAST sequence must win over any generic pattern.
        assert match_combo([M1, M2, M3]) == Move.COMBO_BLAST


# ── Resolution: basic attacks ─────────────────────────────────────────────────

class TestResolveBasicAttacks:
    def test_both_idle(self):
        with NO_VARIANCE:
            d1, d2, dmg2, dmg1, events = resolve_moves(None, None)
        assert d1 == 0 and d2 == 0
        assert events == []

    def test_power_strike_vs_power_strike(self):
        with NO_VARIANCE:
            d1, d2, dmg2, dmg1, events = resolve_moves(Move.POWER_STRIKE, Move.POWER_STRIKE)
        assert d1 == -25 and d2 == -25
        assert dmg1 == 25 and dmg2 == 25
        assert events == []

    def test_power_strike_vs_idle(self):
        with NO_VARIANCE:
            d1, d2, dmg2, dmg1, _ = resolve_moves(Move.POWER_STRIKE, None)
        assert d1 == 0 and d2 == -25

    def test_combo_blast_vs_idle(self):
        with NO_VARIANCE:
            d1, d2, dmg2, dmg1, _ = resolve_moves(Move.COMBO_BLAST, None)
        assert d1 == 0 and d2 == -45 and dmg2 == 45

    def test_combo_blast_vs_power_strike(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.COMBO_BLAST, Move.POWER_STRIKE)
        assert d1 == -25 and d2 == -45


# ── Resolution: Shield Wall ───────────────────────────────────────────────────

class TestShieldWall:
    def test_shield_reflects_power_strike(self):
        with NO_VARIANCE:
            d1, d2, dmg2, dmg1, events = resolve_moves(Move.POWER_STRIKE, Move.SHIELD_WALL)
        assert d2 == 0
        assert d1 == -math.floor(25 * 0.30)  # reflect = -7
        assert any(e[0] == "reflect" for e in events)

    def test_shield_absorbs_combo_blast_partially(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.COMBO_BLAST, Move.SHIELD_WALL)
        assert d1 == 0
        assert d2 == -math.floor(45 * 0.20)  # 20 % leaks through = -9

    def test_shield_vs_idle_no_effect(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.SHIELD_WALL, None)
        assert d1 == 0 and d2 == 0

    def test_shield_vs_shield_no_effect(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.SHIELD_WALL, Move.SHIELD_WALL)
        assert d1 == 0 and d2 == 0


# ── Resolution: Dodge Roll ────────────────────────────────────────────────────

class TestDodgeRoll:
    def test_dodge_power_strike_success(self):
        with NO_VARIANCE, patch("moves.random.random", return_value=0.0):  # 0.0 < 0.70
            d1, d2, _, _, events = resolve_moves(Move.POWER_STRIKE, Move.DODGE_ROLL)
        assert d2 == 0
        assert any(e[0] == "dodge_success" for e in events)

    def test_dodge_power_strike_fail(self):
        with NO_VARIANCE, patch("moves.random.random", return_value=1.0):  # 1.0 >= 0.70
            d1, d2, _, _, events = resolve_moves(Move.POWER_STRIKE, Move.DODGE_ROLL)
        assert d2 == -25
        assert any(e[0] == "dodge_fail" for e in events)

    def test_dodge_combo_blast_success(self):
        with NO_VARIANCE, patch("moves.random.random", return_value=0.0):  # 0.0 < 0.33
            d1, d2, _, _, events = resolve_moves(Move.COMBO_BLAST, Move.DODGE_ROLL)
        assert d2 == 0
        assert any(e[0] == "dodge_success" for e in events)

    def test_dodge_combo_blast_fail(self):
        with NO_VARIANCE, patch("moves.random.random", return_value=1.0):  # 1.0 >= 0.33
            d1, d2, _, _, _ = resolve_moves(Move.COMBO_BLAST, Move.DODGE_ROLL)
        assert d2 == -45

    def test_dodge_vs_idle_no_effect(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.DODGE_ROLL, None)
        assert d1 == 0 and d2 == 0


# ── Resolution: Mend (healing) ────────────────────────────────────────────────

class TestMend:
    def test_mend_heals_vs_idle(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.MEND, None)
        assert d1 == 35 and d2 == 0

    def test_both_mend_both_heal(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.MEND, Move.MEND)
        assert d1 == 35 and d2 == 35

    def test_mend_vs_shield_heals(self):
        # Shield Wall is not an attack — heal should go through.
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.MEND, Move.SHIELD_WALL)
        assert d1 == 35

    def test_mend_vs_dodge_heals(self):
        with NO_VARIANCE, patch("moves.random.random", return_value=1.0):
            d1, d2, _, _, _ = resolve_moves(Move.MEND, Move.DODGE_ROLL)
        assert d1 == 35

    def test_mend_cancelled_by_power_strike(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.MEND, Move.POWER_STRIKE)
        assert d1 == -25   # takes full damage; no heal
        assert d2 == 0

    def test_mend_cancelled_by_combo_blast(self):
        with NO_VARIANCE:
            d1, d2, _, _, _ = resolve_moves(Move.MEND, Move.COMBO_BLAST)
        assert d1 == -45

    def test_heal_multiplier_reduces_heal(self):
        ctx = RoundContext(heal_multiplier=0.5)
        with NO_VARIANCE:
            d1, _, _, _, _ = resolve_moves(Move.MEND, None, ctx)
        assert d1 == math.floor(35 * 0.5)  # 17

    def test_heal_multiplier_zero_gives_no_heal(self):
        ctx = RoundContext(heal_multiplier=0.0)
        with NO_VARIANCE:
            d1, _, _, _, _ = resolve_moves(Move.MEND, None, ctx)
        assert d1 == 0


# ── Move metadata ─────────────────────────────────────────────────────────────

class TestMoveMetadata:
    def test_move_types(self):
        assert MOVE_TYPE[Move.POWER_STRIKE] == "offensive"
        assert MOVE_TYPE[Move.COMBO_BLAST]  == "offensive"
        assert MOVE_TYPE[Move.SHIELD_WALL]  == "defensive"
        assert MOVE_TYPE[Move.DODGE_ROLL]   == "defensive"
        assert MOVE_TYPE[Move.MEND]         == "healing"
