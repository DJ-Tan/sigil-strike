"""
Tests for player.py: action buffering, combo recognition, HP management,
queue capacity, and state reset.
"""
import pytest

from moves import Action, Move
from player import Player
from constants import MAX_QUEUED_MOVES

M1, M2, M3, M4, M5 = (Action(f"move{i}") for i in range(1, 6))


@pytest.fixture
def player():
    return Player(pid=1, name="ALPHA", color=(255, 120, 60))


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialState:
    def test_hp_at_max(self, player):
        assert player.hp == player.max_hp == 100

    def test_queues_and_buffers_empty(self, player):
        assert player.action_buffer == []
        assert player.move_queue == []
        assert player.display_buffer == []

    def test_counters_zero(self, player):
        assert player.moves_hit == 0
        assert player.moves_miss == 0

    def test_not_defeated(self, player):
        assert not player.is_defeated


# ── Action buffering ──────────────────────────────────────────────────────────

class TestAddAction:
    def test_first_action_returns_none(self, player):
        assert player.add_action(M1) is None

    def test_second_action_returns_none(self, player):
        player.add_action(M1)
        assert player.add_action(M2) is None

    def test_valid_combo_returned(self, player):
        # COMBO_BLAST = move1 → move2 → move3
        player.add_action(M1)
        player.add_action(M2)
        result = player.add_action(M3)
        assert result == Move.COMBO_BLAST

    def test_valid_combo_enqueued(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        assert Move.COMBO_BLAST in player.move_queue

    def test_power_strike_recognized(self, player):
        # Power Strike is now triggered by X-Y-X (X != Y).
        player.add_action(M4); player.add_action(M5)
        result = player.add_action(M4)
        assert result == Move.POWER_STRIKE

    def test_invalid_combo_returns_none(self, player):
        player.add_action(M1); player.add_action(M2)
        result = player.add_action(M4)   # [M1,M2,M4] — no match
        assert result is None

    def test_invalid_combo_not_enqueued(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M4)
        assert player.move_queue == []

    def test_buffer_cleared_after_valid_combo(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        assert player.action_buffer == []

    def test_buffer_cleared_after_invalid_combo(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M4)
        assert player.action_buffer == []

    def test_display_buffer_set_after_combo_attempt(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        assert len(player.display_buffer) == 3

    def test_buffer_builds_up_mid_combo(self, player):
        player.add_action(M1)
        assert len(player.action_buffer) == 1
        player.add_action(M2)
        assert len(player.action_buffer) == 2


# ── Move queue ────────────────────────────────────────────────────────────────

class TestMoveQueue:
    def _enqueue_power_strike(self, player):
        # Power Strike: X-Y-X with X != Y.
        player.add_action(M1); player.add_action(M2); player.add_action(M1)

    def test_pop_returns_first_in(self, player):
        # COMBO_BLAST enqueued first, then POWER_STRIKE
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        player.add_action(M1); player.add_action(M2); player.add_action(M1)
        assert player.pop_move() == Move.COMBO_BLAST
        assert player.pop_move() == Move.POWER_STRIKE

    def test_pop_empty_returns_none(self, player):
        assert player.pop_move() is None

    def test_queue_capped_at_max(self, player):
        for _ in range(MAX_QUEUED_MOVES + 2):
            self._enqueue_power_strike(player)
        assert len(player.move_queue) == MAX_QUEUED_MOVES

    def test_sixth_move_dropped(self, player):
        for _ in range(MAX_QUEUED_MOVES + 1):
            self._enqueue_power_strike(player)
        assert len(player.move_queue) == MAX_QUEUED_MOVES


# ── HP management ─────────────────────────────────────────────────────────────

class TestHP:
    def test_apply_damage(self, player):
        player.apply_delta(-25)
        assert player.hp == 75

    def test_hp_floors_at_zero(self, player):
        player.apply_delta(-9999)
        assert player.hp == 0

    def test_hp_caps_at_max(self, player):
        player.hp = 90
        player.apply_delta(+9999)
        assert player.hp == player.max_hp

    def test_heal_partial(self, player):
        player.hp = 60
        player.apply_delta(+10)
        assert player.hp == 70

    def test_is_defeated_at_zero(self, player):
        player.apply_delta(-100)
        assert player.is_defeated

    def test_is_not_defeated_at_one(self, player):
        player.apply_delta(-99)
        assert player.hp == 1
        assert not player.is_defeated


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_restores_hp(self, player):
        player.apply_delta(-50)
        player.reset()
        assert player.hp == player.max_hp

    def test_reset_clears_move_queue(self, player):
        player.add_action(M1); player.add_action(M1); player.add_action(M1)
        player.reset()
        assert player.move_queue == []

    def test_reset_clears_action_buffer(self, player):
        player.add_action(M1)
        player.reset()
        assert player.action_buffer == []

    def test_reset_clears_display_buffer(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        player.reset()
        assert player.display_buffer == []

    def test_reset_clears_counters(self, player):
        player.moves_hit  = 5
        player.moves_miss = 3
        player.reset()
        assert player.moves_hit == 0
        assert player.moves_miss == 0


# ── Per-frame update / timers ─────────────────────────────────────────────────

class TestUpdate:
    def test_display_timer_decrements(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        before = player.display_timer
        player.update(0.1)
        assert player.display_timer < before

    def test_display_buffer_cleared_when_timer_expires(self, player):
        player.add_action(M1); player.add_action(M2); player.add_action(M3)
        player.update(player.display_timer + 0.01)
        assert player.display_buffer == []

    def test_action_buffer_timeout_clears_partial_combo(self, player):
        player.add_action(M1)
        assert len(player.action_buffer) == 1
        player.update(9999.0)   # far past the timeout
        assert player.action_buffer == []
