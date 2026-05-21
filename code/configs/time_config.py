"""
configs/time_config.py
──────────────────────
Loader for gameplay timing tunables. Values live in `time_config.ini`
next to the game executable (or in `code/configs/` during source-mode
development), and can be edited post-release without rebuilding.

Backwards-compatible API: this module still exposes every value as a
top-level attribute (e.g. `from configs.time_config import RESOLVE_INTERVAL`),
so existing imports continue to work unchanged.

Missing keys, missing files, and malformed values all fall back to the
built-in defaults below, with a warning printed once at startup.
"""

from __future__ import annotations

import configparser
import pathlib
import sys

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from paths import external_dir, is_frozen  # noqa: E402


_DEFAULTS: dict[str, object] = {
    "RESOLVE_INTERVAL":         5.0,
    "COMBO_DISPLAY_TIME":       1.0,
    "ACTION_COOLDOWN":          0.5,
    "EVENT_MESSAGE_DURATION":   4.0,
    "ACTION_BUFFER_TIMEOUT":    5.0,
    "DEATHMATCH_START_SEC":     180.0,
    "DEATHMATCH_HP_PER_SEC":    2.0,
    "DEATHMATCH_COUNTDOWN_SEC": 30.0,
}

_SECTION = "timing"


def _ini_path() -> pathlib.Path:
    if is_frozen():
        return external_dir() / "configs" / "time_config.ini"
    return _THIS_DIR / "time_config.ini"


def _coerce(raw: str, default: object) -> object:
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _load() -> dict[str, object]:
    values = dict(_DEFAULTS)
    path = _ini_path()
    if not path.exists():
        print(f"[time_config] {path} not found — using built-in defaults.")
        return values

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error as e:
        print(f"[time_config] failed to parse {path}: {e} — using defaults.")
        return values

    if _SECTION not in parser:
        print(f"[time_config] section [{_SECTION}] missing in {path} — using defaults.")
        return values

    for key, default in _DEFAULTS.items():
        if key not in parser[_SECTION]:
            continue
        raw = parser[_SECTION][key]
        try:
            values[key] = _coerce(raw, default)
        except (ValueError, TypeError) as e:
            print(f"[time_config] bad value for {key}={raw!r}: {e} — using default.")
    return values


globals().update(_load())
