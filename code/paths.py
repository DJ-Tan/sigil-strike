"""
paths.py
────────
Runtime path resolution for assets and user-editable folders.

The game has two kinds of on-disk dependencies:

  resource_dir()  — read-only bundled assets (audio/, images/, the
                    mediapipe hand_landmarker.task file). When frozen by
                    PyInstaller these are extracted under sys._MEIPASS; in
                    source mode they live under the project root.

  external_dir()  — user-editable folders (configs/, Teams/) that ship
                    NEXT to the .exe rather than inside it, so they can
                    be modified after release without rebuilding.

Use these helpers anywhere the game needs to locate a file on disk —
do NOT use `os.path.dirname(__file__)` directly, since that points into
the PyInstaller temp dir when frozen.
"""

from __future__ import annotations

import pathlib
import sys


def is_frozen() -> bool:
    """True when running inside a PyInstaller-built executable."""
    return getattr(sys, "frozen", False)


def _source_root() -> pathlib.Path:
    # code/paths.py → code/ → project root
    return pathlib.Path(__file__).resolve().parent.parent


def resource_dir() -> pathlib.Path:
    """Where bundled, read-only assets live (audio/, images/, model task file)."""
    if is_frozen():
        return pathlib.Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return _source_root()


def external_dir() -> pathlib.Path:
    """Where user-editable folders live (configs/, Teams/).

    Frozen: the directory containing the .exe.
    Source: the project root, so the existing `configs/` and `Teams/`
            folders work as-is during development.
    """
    if is_frozen():
        return pathlib.Path(sys.executable).resolve().parent
    return _source_root()
