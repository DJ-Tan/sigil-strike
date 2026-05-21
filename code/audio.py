"""
audio.py
────────
Tiny SFX loader / player.  Loads audio/sfx/*.mp3 once and plays them by name.
All errors degrade to silent no-ops so the game still runs without audio.
"""
from __future__ import annotations

import os
from typing import Optional

import pygame

from paths import resource_dir

_SFX_DIR = str(resource_dir() / "audio" / "sfx")
_BGM_DIR = str(resource_dir() / "audio" / "bgm")

_FILES: dict[str, str] = {
    "power_strike":   "power_strike.mp3",
    "combo_blast":    "combo_blast.mp3",
    "shield_block":   "shield_block.mp3",
    "shield_reflect": "shield_reflect.mp3",
    "dodge_success":  "dodge_success.mp3",
    "dodge_fail":     "dodge_fail.mp3",
    "mend":           "mend.mp3",
}

# Per-SFX attenuation so every clip plays at the same perceived loudness.
# Values were measured by computing RMS of each file and normalizing to the
# quietest one (mend), since pygame's Sound.set_volume can only attenuate
# (range 0..1). Re-run the measurement if you swap audio files in/out.
_VOLUMES: dict[str, float] = {
    "power_strike":   0.92,
    "combo_blast":    0.57,
    "shield_block":   0.46,
    "shield_reflect": 0.84,
    "dodge_success":  0.83,
    "dodge_fail":     0.71,
    "mend":           1.00,
}

# Per-BGM volumes, measured the same way and capped to the SFX baseline RMS
# so background music never drowns the sound effects.
_BGM_VOLUMES: dict[str, float] = {
    "final_battle.mp3":        0.41,
    "group_battle_1.mp3":      0.26,
    "group_battle_2.mp3":      0.24,
    "group_battle_3.mp3":      0.24,
    "group_battle_4.mp3":      0.32,
    "semi_battle.mp3":         0.24,
    "third_place_battle.mp3":  0.28,
}

_sounds: dict[str, Optional[pygame.mixer.Sound]] = {}
_initialized = False
_enabled = False


def init() -> None:
    """Idempotent.  Safe to call before or after pygame.init()."""
    global _initialized, _enabled
    if _initialized:
        return
    _initialized = True

    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except pygame.error as e:
            print(f"[audio] mixer init failed: {e} — SFX disabled")
            return

    for name, fname in _FILES.items():
        path = os.path.join(_SFX_DIR, fname)
        if not os.path.exists(path):
            print(f"[audio] missing {fname} — '{name}' SFX disabled")
            _sounds[name] = None
            continue
        try:
            snd = pygame.mixer.Sound(path)
            snd.set_volume(_VOLUMES.get(name, 1.0))
            _sounds[name] = snd
        except pygame.error as e:
            print(f"[audio] failed to load {fname}: {e}")
            _sounds[name] = None

    _enabled = any(s is not None for s in _sounds.values())
    if _enabled:
        loaded = [n for n, s in _sounds.items() if s is not None]
        print(f"[audio] loaded {len(loaded)} SFX: {', '.join(loaded)}")


def play(name: str) -> None:
    if not _enabled:
        return
    s = _sounds.get(name)
    if s is not None:
        s.play()


def play_music(filename: str, loops: int = -1,
               volume: Optional[float] = None) -> None:
    """Loop a track from audio/bgm/.  No-op if mixer or file is unavailable.

    If `volume` is None, uses the per-file value from `_BGM_VOLUMES` (which
    keeps BGM at or below the SFX baseline), falling back to 0.5 for unknown
    tracks. Pass an explicit value to override.
    """
    init()
    if not pygame.mixer.get_init():
        return
    path = os.path.join(_BGM_DIR, filename)
    if not os.path.exists(path):
        print(f"[audio] missing BGM {filename}")
        return
    if volume is None:
        volume = _BGM_VOLUMES.get(filename, 0.5)
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(loops=loops)
    except pygame.error as e:
        print(f"[audio] failed to play BGM {filename}: {e}")


def stop_music() -> None:
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
