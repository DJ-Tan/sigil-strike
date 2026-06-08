# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — 2026-06-08

### Added
- **CNN data collection: 3-second countdown.** When `1`–`5` is pressed in
  CNN mode, the webcam dims to 50% brightness and a large white `3 / 2 / 1`
  countdown overlays the frame for three seconds before any frame is
  saved. Gives the user time to settle into the pose. Landmark mode is
  unchanged. (`code/model/collect_data.py`, `code/model/cnn/collect_data.py`)
- **Pre-collection reset prompt.** Before the camera opens, the collector
  asks whether to wipe existing data for specific moves
  (`y/N` → comma-separated move numbers like `1,3,4`). Only shown when
  existing data exists for the active pipeline; displays a per-move counts
  table so the user can decide informed.
  (all three `collect_data.py` entry points)
- **Auto-zip after collection.** `collect_data.py` now bundles each
  session's output into `Team<N>.zip`, rooted at `Team<N>/...` inside the
  archive, filtered by mode (`.jpg` for CNN, `.csv` for landmark). The
  archive layout matches what the Colab training notebook expects on
  upload.
- **Notebook Section 4D — per-move confidence thresholds.** New cells in
  `code/model/sigil_strike_colab.ipynb` that let the team decide
  `THRESHOLD_MOVE<N>` values after observing the live-prediction table
  that Section 4C now emits (10 predictions at 0.5 s intervals over 5 s).
  Section 4D prints the chosen values as ready-to-paste `team.env` lines.
- **Notebook Section 5 — team.env export.** The download bundle now
  includes a `team.env` populated from `NAME`, `MODEL_TYPE`, and the
  Section 4D thresholds, in the same format used by `Teams/Team<N>/team.env`.
- **`code/configs/team_colors.py`** — new single source of truth for the
  six team display colors as a `TEAM_COLORS` dict keyed by team number.
- **Legible counter overlay.** The per-move counters in `collect_data.py`'s
  GUI now render over a semi-transparent dark panel
  (`cv2.convertScaleAbs` + `cv2.addWeighted`) so the text stays readable
  against any webcam background.

### Changed
- **Swapped Power Strike and Mend trigger sequences.**
  - Power Strike → any `X → Y → X` (X ≠ Y)
  - Mend → any 3 identical actions
  - Updated `code/moves.py`, `code/configs/move_config.ini`,
    `dist/sigil_strike/configs/move_config.ini`, and the README's combo
    table + resolution notes.
- **Team colors moved out of `team.env`.** `code/bracket.py` no longer
  parses `COLOR=` from each team's env file — colors are looked up from
  `TEAM_COLORS` in the new `code/configs/team_colors.py`. The `COLOR=`
  line was removed from all six `Teams/Team<N>/team.env` files, and the
  README's `team.env` example was updated to match.

### Fixed
- **Landmark detection silently disabled in dev mode.** `code/game.py`
  looked for `hand_landmarker.task` at `<resource_dir>/model/landmark/`,
  which only exists in the frozen PyInstaller layout. Dev mode
  (`start_game.bat`) now falls back to `<resource_dir>/code/model/landmark/`,
  so MediaPipe initializes correctly when running from source. Symptom
  was "nothing is being predicted by the models" for landmark teams.
- **Keyboard hint visible while a camera is active.** `code/renderer.py`
  now only renders the `Q W E R T → ① ② ③ ④ ⑤` mapping for a player whose
  camera slot is empty. When the camera is connected, gesture input
  takes over and the hint is suppressed.

### Verified (no code change)
- **Game runs gracefully without BGM files.** Confirmed by removing every
  `audio/bgm/*.mp3` and running through the audio path — `audio.play_music`
  returns silently with a single `[audio] missing BGM <filename>` warning,
  and `audio.stop_music` is a safe no-op. The existing defensive code in
  `code/audio.py:113-116` already handles this; no change was needed.
