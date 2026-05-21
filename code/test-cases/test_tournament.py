"""
Tests for the Tournament data model in bracket.py.

bracket.py imports pygame, the game engine, and audio — none of which are
needed to test the pure-logic Tournament class.  We stub those modules in
sys.modules before the import so bracket.py loads cleanly.
"""
import sys
import types
import unittest.mock

# Stub heavy dependencies before importing bracket.
for _mod_name in ("game", "audio"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = unittest.mock.MagicMock()

from bracket import Tournament, Team, GroupMatch, KnockoutMatch  # noqa: E402


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_teams(n: int = 6) -> list[Team]:
    return [Team(name=chr(65 + i), color=(255, 255, 255)) for i in range(n)]


def make_tournament() -> Tournament:
    teams = make_teams()
    return Tournament(group_a=teams[:3], group_b=teams[3:])


def complete_group(t: Tournament) -> None:
    """Make team_a win every match in both groups (A wins all, D wins all)."""
    for m in t.matches_a + t.matches_b:
        t.set_group_result(m, winner_is_a=True)


# ── Round-robin generation ────────────────────────────────────────────────────

class TestRoundRobin:
    def test_three_teams_produce_three_matches(self):
        t = make_tournament()
        assert len(t.matches_a) == 3
        assert len(t.matches_b) == 3

    def test_all_pairs_unique(self):
        t = make_tournament()
        pairs = {(m.team_a.name, m.team_b.name) for m in t.matches_a}
        assert len(pairs) == 3

    def test_every_team_appears_in_matches(self):
        t = make_tournament()
        involved = {m.team_a.name for m in t.matches_a} | {m.team_b.name for m in t.matches_a}
        assert involved == {"A", "B", "C"}


# ── Standings ─────────────────────────────────────────────────────────────────

class TestStandings:
    def test_all_unplayed_gives_zero_wins(self):
        t = make_tournament()
        for s in t.standings_a():
            assert s.wins == 0 and s.losses == 0

    def test_winner_appears_first(self):
        t = make_tournament()
        m = t.matches_a[0]   # A vs B
        t.set_group_result(m, winner_is_a=True)
        standings = t.standings_a()
        assert standings[0].team is m.team_a

    def test_loser_ranks_below_winner(self):
        t = make_tournament()
        m = t.matches_a[0]
        t.set_group_result(m, winner_is_a=True)
        standings = t.standings_a()
        winner_rank = next(s.rank for s in standings if s.team is m.team_a)
        loser_rank  = next(s.rank for s in standings if s.team is m.team_b)
        assert winner_rank < loser_rank

    def test_tiebreaker_by_shortest_total_time(self):
        # Three-way tie on wins; team with lowest total game time ranks first.
        #
        # matches_a:  [0] A vs B,  [1] A vs C,  [2] B vs C
        # Results:
        #   A beats B in 10 s  → A: W=1, t=10;  B: L=1, t=10
        #   C beats A in 100 s → C: W=1, t=100; A: L=1, t=110
        #   B beats C in 50 s  → B: W=1, t=60;  C: L=1, t=150
        # Final: A(1W,110s), B(1W,60s), C(1W,150s) → rank: B < A < C
        t = make_tournament()
        t.set_group_result(t.matches_a[0], winner_is_a=True,  duration=10.0)
        t.set_group_result(t.matches_a[1], winner_is_a=False, duration=100.0)
        t.set_group_result(t.matches_a[2], winner_is_a=True,  duration=50.0)
        standings = t.standings_a()
        names = [s.team.name for s in standings]
        assert names.index("B") < names.index("A") < names.index("C")

    def test_ranks_assigned_sequentially(self):
        t = make_tournament()
        complete_group(t)
        ranks = [s.rank for s in t.standings_a()]
        assert ranks == [1, 2, 3]


# ── Group result management ───────────────────────────────────────────────────

class TestGroupResults:
    def test_set_result_team_a_wins(self):
        t = make_tournament()
        m = t.matches_a[0]
        t.set_group_result(m, winner_is_a=True)
        assert m.score_a == 1 and m.score_b == 0

    def test_set_result_team_b_wins(self):
        t = make_tournament()
        m = t.matches_a[0]
        t.set_group_result(m, winner_is_a=False)
        assert m.score_a == 0 and m.score_b == 1

    def test_duration_stored(self):
        t = make_tournament()
        m = t.matches_a[0]
        t.set_group_result(m, winner_is_a=True, duration=42.5)
        assert m.duration == 42.5

    def test_zero_duration_stored_as_none(self):
        t = make_tournament()
        m = t.matches_a[0]
        t.set_group_result(m, winner_is_a=True, duration=0.0)
        assert m.duration is None

    def test_clear_result_resets_scores(self):
        t = make_tournament()
        m = t.matches_a[0]
        t.set_group_result(m, winner_is_a=True)
        t.clear_group_result(m)
        assert m.score_a is None and m.score_b is None

    def test_reseed_clears_knockout_while_group_incomplete(self):
        t = make_tournament()
        t.set_group_result(t.matches_a[0], winner_is_a=True)  # only 1 of 3 group A matches
        assert t.sf1.team_a is None


# ── Knockout seeding ──────────────────────────────────────────────────────────

class TestKnockoutSeeding:
    def test_sf_teams_assigned_after_full_group_stage(self):
        t = make_tournament()
        complete_group(t)
        assert t.sf1.team_a is not None and t.sf1.team_b is not None
        assert t.sf2.team_a is not None and t.sf2.team_b is not None

    def test_fifth_place_teams_assigned(self):
        t = make_tournament()
        complete_group(t)
        assert t.fifth.team_a is not None and t.fifth.team_b is not None

    def test_a1_in_sf1_as_team_a(self):
        t = make_tournament()
        complete_group(t)
        assert t.sf1.team_a is t.standings_a()[0].team

    def test_b2_in_sf1_as_team_b(self):
        t = make_tournament()
        complete_group(t)
        assert t.sf1.team_b is t.standings_b()[1].team

    def test_b1_in_sf2_as_team_a(self):
        t = make_tournament()
        complete_group(t)
        assert t.sf2.team_a is t.standings_b()[0].team

    def test_a2_in_sf2_as_team_b(self):
        t = make_tournament()
        complete_group(t)
        assert t.sf2.team_b is t.standings_a()[1].team

    def test_a3_in_fifth_as_team_a(self):
        t = make_tournament()
        complete_group(t)
        assert t.fifth.team_a is t.standings_a()[2].team

    def test_b3_in_fifth_as_team_b(self):
        t = make_tournament()
        complete_group(t)
        assert t.fifth.team_b is t.standings_b()[2].team


# ── Knockout results ──────────────────────────────────────────────────────────

class TestKnockoutResults:
    def test_sf1_winner_stored(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        assert t.sf1.winner is t.sf1.team_a

    def test_sf1_winner_seeded_into_final(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        assert t.final.team_a is t.sf1.team_a

    def test_sf1_loser_seeded_into_third_place(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        assert t.third.team_a is t.sf1.team_b

    def test_sf2_winner_seeded_into_final(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf2, winner_is_a=True)
        assert t.final.team_b is t.sf2.team_a

    def test_sf2_loser_seeded_into_third_place(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf2, winner_is_a=True)
        assert t.third.team_b is t.sf2.team_b

    def test_clear_sf1_removes_winner(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        t.clear_knockout_result(t.sf1)
        assert t.sf1.winner is None

    def test_clear_sf1_cascades_final_team_a_to_none(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        t.set_knockout_result(t.sf2, winner_is_a=True)
        t.clear_knockout_result(t.sf1)
        assert t.final.team_a is None

    def test_seed_final_after_both_sfs(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        t.set_knockout_result(t.sf2, winner_is_a=False)
        t.seed_final()
        assert t.final.team_a is t.sf1.winner
        assert t.final.team_b is t.sf2.winner

    def test_grand_final_winner(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        t.set_knockout_result(t.sf2, winner_is_a=True)
        t.set_knockout_result(t.final, winner_is_a=True)
        assert t.final.winner is t.final.team_a

    def test_scores_set_on_knockout_result(self):
        t = make_tournament()
        complete_group(t)
        t.set_knockout_result(t.sf1, winner_is_a=True)
        assert t.sf1.score_a == 1 and t.sf1.score_b == 0
