"""
configs/move_config.py
──────────────────────
Loader for move balance tunables. The actual values live in
`move_config.ini` next to the game executable (or in `code/configs/`
during source-mode development), so they can be edited post-release
without rebuilding.

Backwards-compatible API: this module still exposes every value as a
top-level attribute (e.g. `cfg.WEAK_MOVE_DAMAGE`), so callers that did
`from configs import move_config as cfg` continue to work unchanged.

Missing keys, missing files, and malformed values all fall back to the
built-in defaults below, with a warning printed once at startup.
"""

from __future__ import annotations

import configparser
import pathlib
import sys

# Allow this module to import paths.py whether the project is run as
# a script (code/main.py) or frozen (PyInstaller).
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from paths import external_dir, is_frozen  # noqa: E402


# ── Built-in defaults (used when the INI is missing or a key is absent) ──────
_DEFAULTS: dict[str, object] = {
    "WEAK_MOVE_DAMAGE":             25,
    "STRONG_MOVE_DAMAGE":           45,
    "HEAL_AMOUNT":                  35,
    "DAMAGE_VARIANCE":              5,
    "DEATHMATCH_HEAL_MULT":         0.25,
    "SHIELD_VS_STRONG_DAMAGE_PCT":  0.20,
    "SHIELD_VS_WEAK_REFLECT_PCT":   0.30,
    "DODGE_VS_STRONG_CHANCE":       0.33,
    "DODGE_VS_WEAK_CHANCE":         0.70,
    "STRONG_MOVE_SEQUENCE":         ["move1", "move2", "move3"],
    "SHIELD_SEQUENCE":              ["move3", "move4", "move5"],
    "DODGE_SEQUENCE":               ["move1", "move3", "move5"],
}

_SECTION = "balance"


def _ini_path() -> pathlib.Path:
    """Where the INI file lives. Next to the .exe when frozen; next to
    this loader during source-mode dev."""
    if is_frozen():
        return external_dir() / "configs" / "move_config.ini"
    return _THIS_DIR / "move_config.ini"


def _coerce(key: str, raw: str, default: object) -> object:
    """Convert a raw string from the INI into the type of the default value."""
    # bool must be checked before int (bool is a subclass of int in Python).
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return raw


def _load() -> dict[str, object]:
    values = dict(_DEFAULTS)
    path = _ini_path()
    if not path.exists():
        print(f"[move_config] {path} not found — using built-in defaults.")
        return values

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error as e:
        print(f"[move_config] failed to parse {path}: {e} — using defaults.")
        return values

    if _SECTION not in parser:
        print(f"[move_config] section [{_SECTION}] missing in {path} — using defaults.")
        return values

    for key, default in _DEFAULTS.items():
        if key not in parser[_SECTION]:
            continue
        raw = parser[_SECTION][key]
        try:
            values[key] = _coerce(key, raw, default)
        except (ValueError, TypeError) as e:
            print(f"[move_config] bad value for {key}={raw!r}: {e} — using default.")
    return values


# Expose every loaded value at module scope so `cfg.WEAK_MOVE_DAMAGE` works.
globals().update(_load())
