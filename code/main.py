"""
main.py
───────
Entry point.

Tournament mode (default):

    python main.py

Test mode — skip the bracket and play a single match between two teams:

    python main.py <p1_team_id> <p2_team_id>

Team IDs are integers 1..6, mapping to Teams/Team<N>/ folders.
Example:  python main.py 1 4    # Team1 vs Team4

To enable real camera prediction instead of (or alongside) keyboard input,
run camera.py in a separate terminal:

    python code/camera.py --player 1
    python code/camera.py --player 2
"""

import os
import sys


def _print_banner() -> None:
    """Print the same launcher banner that start_game.bat shows in dev mode.

    Runs for both `python code/main.py` and the frozen `sigil_strike.exe`, so
    the .exe behaves the same way as the .bat launcher.
    """
    exe_name = os.path.basename(sys.executable) if getattr(sys, "frozen", False) else "main.py"
    print("[start] Launching SIGIL STRIKE ...")
    print("[start] Hand-sign detection runs automatically if model files are present.")
    print("[start] Keyboard input is always available.")
    print("[start] Tip: pass two team IDs to skip the bracket and play a single match,")
    print(f"[start]      e.g.   {exe_name} 1 4   (Team1 vs Team4)")
    print()


def _parse_test_mode_args(args: list[str]) -> list[int] | None:
    """Return [p1_id, p2_id] if args look like a valid test-mode invocation, else None."""
    if len(args) != 2:
        return None
    try:
        ids = [int(x) for x in args]
    except ValueError:
        return None
    if not all(1 <= i <= 6 for i in ids):
        return None
    return ids


def _run_test_match(test_ids: list[int]) -> None:
    import pygame
    from bracket import load_teams
    from game import Game

    teams = load_teams()
    p1 = teams[test_ids[0] - 1]
    p2 = teams[test_ids[1] - 1]
    print(f"[test mode] {p1.name} (Team{test_ids[0]}) vs "
          f"{p2.name} (Team{test_ids[1]})")
    g = Game(
        p1_name=p1.name, p1_color=p1.color,
        p2_name=p2.name, p2_color=p2.color,
        tournament_mode=False,
    )
    pygame.display.set_caption(f"Sigil Strike — Test: {p1.name} vs {p2.name}")
    g.run_once()
    pygame.quit()


def main() -> None:
    _print_banner()
    args = sys.argv[1:]
    if not args:
        from bracket import main as bracket_main
        bracket_main()
        return

    test_ids = _parse_test_mode_args(args)
    if test_ids is None:
        print("usage: python main.py [<p1_team_id> <p2_team_id>]")
        print("  no args     : run the tournament bracket")
        print("  two IDs     : skip the bracket and run a single match")
        print("                IDs are integers 1..6")
        print("                e.g.  start_game.bat 1 4   (Team1 vs Team4)")
        sys.exit(2)

    _run_test_match(test_ids)


if __name__ == "__main__":
    main()
