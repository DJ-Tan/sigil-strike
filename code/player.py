"""
player.py
─────────
Player state: HP, action buffer, move queue, and smooth HP display animation.

Knows nothing about rendering or pygame — pure game logic.
"""

from __future__ import annotations

from typing import Optional

from moves import Action, Move, match_combo
from constants import MAX_QUEUED_MOVES, ACTIONS_PER_COMBO
from configs.time_config import ACTION_BUFFER_TIMEOUT


class Player:
    def __init__(self, pid: int, name: str, color: tuple):
        self.pid   = pid
        self.name  = name
        self.color = color

        self.hp      = 100
        self.max_hp  = 100
        self.hp_display = 100.0   # animated value for smooth HP bar

        # 3-action rolling buffer; cleared after every complete combo attempt
        # or after ACTION_BUFFER_TIMEOUT seconds of inactivity.
        self.action_buffer: list[Action] = []
        self.action_buffer_timer: float = 0.0

        # Snapshot of the last completed combo for display purposes
        self.display_buffer: list[Action] = []
        self.display_timer: float = 0.0

        # Confirmed moves waiting to be resolved (FIFO, max 5)
        self.move_queue: list[Move] = []

        # Move-attempt counters (used for tiebreaker and HUD display)
        self.moves_hit:  int = 0   # combos that resolved to a valid Move
        self.moves_miss: int = 0   # combos that cleared with no match

    # ── Input ─────────────────────────────────────────────────────────────────

    def add_action(self, action: Action, display_time: float = 1.0) -> Optional[Move]:
        """
        Append one action to the buffer.
        When the buffer reaches ACTIONS_PER_COMBO (3) the combo is evaluated:
          - If a valid Move is recognised it is returned AND enqueued.
          - Otherwise None is returned (invalid combo, buffer is still cleared).
        The completed 3-action sequence is kept in display_buffer for
        display_time seconds before the UI clears it.
        """
        self.action_buffer.append(action)
        self.action_buffer_timer = ACTION_BUFFER_TIMEOUT

        if len(self.action_buffer) < ACTIONS_PER_COMBO:
            return None   # combo not yet complete

        seq  = list(self.action_buffer)
        self.display_buffer = list(seq)
        self.display_timer = display_time
        self.action_buffer.clear()
        self.action_buffer_timer = 0.0
        move = match_combo(seq)

        if move is not None:
            self._enqueue_move(move)

        return move   # None on invalid combo

    def _enqueue_move(self, move: Move) -> None:
        if len(self.move_queue) < MAX_QUEUED_MOVES:
            self.move_queue.append(move)

    # ── Queue management ──────────────────────────────────────────────────────

    def pop_move(self) -> Optional[Move]:
        """Remove and return the next queued move, or None if queue is empty."""
        return self.move_queue.pop(0) if self.move_queue else None

    # ── HP management ─────────────────────────────────────────────────────────

    def apply_delta(self, delta: int) -> None:
        self.hp = max(0, min(self.max_hp, self.hp + delta))

    @property
    def is_defeated(self) -> bool:
        return self.hp <= 0

    # ── Per-frame update ──────────────────────────────────────────────────────

    @property
    def combo_locked(self) -> bool:
        """True while the completed combo is still being displayed."""
        return self.display_timer > 0

    def update(self, dt: float) -> None:
        """Smoothly animate the displayed HP toward the real value."""
        if self.display_timer > 0:
            self.display_timer -= dt
            if self.display_timer <= 0:
                self.display_buffer.clear()
        if self.action_buffer_timer > 0:
            self.action_buffer_timer -= dt
            if self.action_buffer_timer <= 0:
                self.action_buffer.clear()
        diff = self.hp - self.hp_display
        self.hp_display += diff * min(dt * 5.0, 1.0)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.hp         = self.max_hp
        self.hp_display = float(self.max_hp)
        self.action_buffer.clear()
        self.action_buffer_timer = 0.0
        self.display_buffer.clear()
        self.display_timer = 0.0
        self.move_queue.clear()
        self.moves_hit  = 0
        self.moves_miss = 0
