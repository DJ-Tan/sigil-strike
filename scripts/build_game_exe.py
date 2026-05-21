"""
build_game_exe.py
─────────────────
Builds `sigil_strike.exe`: a single-file standalone executable that wraps
the entire game (main.py + bracket.py + game runtime + mediapipe + torch
+ pygame). User-editable folders (configs/, Teams/) are NOT bundled into
the .exe — they're copied next to it so they can be modified without a
rebuild.

Usage:
    python scripts/build_game_exe.py            # build into dist/sigil_strike/
    python scripts/build_game_exe.py --clean    # wipe build/sigil_strike + spec first
    python scripts/build_game_exe.py --no-cnn   # skip torch (much smaller exe, landmark-only)

Resulting layout under <repo>/dist/sigil_strike/:
    sigil_strike.exe
    configs/move_config.ini   (edit to rebalance)
    configs/time_config.ini   (edit to retune timing)
    Teams/Team1..Team6/       (per-team configs + trained models)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


HERE             = pathlib.Path(__file__).resolve().parent           # scripts/
PROJECT_ROOT     = HERE.parent                                       # repo root
CODE_DIR         = PROJECT_ROOT / "code"

ENTRY_SCRIPT     = CODE_DIR / "main.py"
LANDMARKER_FILE  = CODE_DIR / "model" / "landmark" / "hand_landmarker.task"
AUDIO_DIR        = PROJECT_ROOT / "audio"
IMAGES_DIR       = PROJECT_ROOT / "images"
CONFIGS_SRC_DIR  = CODE_DIR / "configs"
TEAMS_SRC_DIR    = PROJECT_ROOT / "Teams"

EXE_NAME         = "sigil_strike"
DIST_DIR         = PROJECT_ROOT / "dist"  / EXE_NAME
BUILD_DIR        = PROJECT_ROOT / "build" / EXE_NAME
SPEC_FILE        = HERE / f"{EXE_NAME}.spec"


def ensure_pyinstaller() -> None:
    """Install PyInstaller into the current interpreter if it's missing."""
    try:
        import PyInstaller  # noqa: F401
        return
    except ImportError:
        pass
    print("[build] PyInstaller missing — installing ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"])


def ensure_runtime_deps(include_cnn: bool) -> None:
    """Make sure every runtime import succeeds in the build interpreter.
    PyInstaller analyses the script via this interpreter, so missing
    runtime packages cause silent fall-throughs in the resulting .exe."""
    missing: list[str] = []

    def _check(import_name: str, install_name: str) -> None:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(install_name)

    _check("pygame",        "pygame")
    _check("cv2",           "opencv-python")
    _check("mediapipe",     "mediapipe")
    _check("joblib",        "joblib")
    _check("sklearn",       "scikit-learn")
    _check("numpy",         "numpy")
    if include_cnn:
        _check("torch",        "torch")
        _check("torchvision",  "torchvision")

    if missing:
        print(f"[build] Installing runtime deps: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", *missing])


def _add_data(src: pathlib.Path, dest: str, sep: str) -> list[str]:
    return ["--add-data", f"{src}{sep}{dest}"]


def build(clean: bool, include_cnn: bool) -> pathlib.Path:
    for path, label in [
        (ENTRY_SCRIPT,    "entry script"),
        (LANDMARKER_FILE, "hand_landmarker.task"),
        (AUDIO_DIR,       "audio/ folder"),
        (IMAGES_DIR,      "images/ folder"),
    ]:
        if not path.exists():
            sys.exit(f"[build] {label} not found: {path}")

    if clean:
        for path in (DIST_DIR, BUILD_DIR):
            if path.exists():
                print(f"[build] Removing {path}")
                shutil.rmtree(path)
        if SPEC_FILE.exists():
            print(f"[build] Removing {SPEC_FILE}")
            SPEC_FILE.unlink()

    sep = ";" if sys.platform.startswith("win") else ":"

    # Modules PyInstaller might pull in speculatively but the game doesn't
    # actually use. Excluding them keeps exe size sane.
    excludes = [
        "pytest", "jupyter", "notebook", "ipykernel", "ipywidgets",
        "tensorflow", "tensorflow_hub", "tensorboard", "keras",
        "transformers", "tokenizers", "huggingface_hub",
        "pydantic", "pydantic_core",
    ]
    if not include_cnn:
        excludes += ["torch", "torchvision", "torchaudio"]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--console",                              # keep stdout for debug logging
        "--name", EXE_NAME,
        "--paths", str(CODE_DIR),                 # make code/ the import root
        # ── Bundled read-only assets — accessed via paths.resource_dir() ──
        *_add_data(AUDIO_DIR,        "audio",                       sep),
        *_add_data(IMAGES_DIR,       "images",                      sep),
        *_add_data(LANDMARKER_FILE,  "model/landmark",              sep),
        # MediaPipe ships native binaries + data files that PyInstaller's
        # static analysis misses. Pull them in narrowly.
        "--collect-binaries", "mediapipe",
        "--collect-data",     "mediapipe",
        "--collect-submodules", "mediapipe.tasks",
    ]
    if include_cnn:
        cmd += [
            "--collect-binaries", "torch",
            "--collect-data",     "torch",
            "--collect-binaries", "torchvision",
            "--collect-data",     "torchvision",
        ]

    for mod in excludes:
        cmd += ["--exclude-module", mod]

    cmd += [
        "--workpath", str(BUILD_DIR),
        "--distpath", str(DIST_DIR.parent),       # PyInstaller appends --name, so → dist/sigil_strike.exe
        "--specpath", str(HERE),
        str(ENTRY_SCRIPT),
    ]

    print("[build] Running PyInstaller (this can take a few minutes) ...")
    subprocess.check_call(cmd)

    # PyInstaller writes to <distpath>/<name>.exe — we then move/copy the
    # .exe + sibling external folders under dist/<name>/ for a clean layout.
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""
    interim_exe = DIST_DIR.parent / f"{EXE_NAME}{exe_suffix}"
    if not interim_exe.exists():
        sys.exit(f"[build] PyInstaller finished but {interim_exe} is missing.")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    final_exe = DIST_DIR / interim_exe.name
    if final_exe.exists():
        final_exe.unlink()
    shutil.move(str(interim_exe), str(final_exe))
    return final_exe


def stage_external_dirs() -> None:
    """Copy user-editable folders (configs/, Teams/) into dist/sigil_strike/."""
    target_configs = DIST_DIR / "configs"
    target_configs.mkdir(parents=True, exist_ok=True)
    for ini in CONFIGS_SRC_DIR.glob("*.ini"):
        shutil.copy2(ini, target_configs / ini.name)
        print(f"[build] staged {ini.name} → {target_configs}")

    target_teams = DIST_DIR / "Teams"
    if target_teams.exists():
        shutil.rmtree(target_teams)
    shutil.copytree(TEAMS_SRC_DIR, target_teams,
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"[build] staged Teams/ → {target_teams}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build sigil_strike.exe via PyInstaller.")
    p.add_argument("--clean", action="store_true",
                   help="Wipe build/sigil_strike, dist/sigil_strike, and the .spec file first.")
    p.add_argument("--no-cnn", action="store_true",
                   help="Skip PyTorch — much smaller exe, but CNN-mode teams won't work.")
    args = p.parse_args()

    include_cnn = not args.no_cnn

    ensure_pyinstaller()
    ensure_runtime_deps(include_cnn=include_cnn)
    exe_path = build(clean=args.clean, include_cnn=include_cnn)
    stage_external_dirs()

    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print()
    print("─" * 64)
    print(f"  Built : {exe_path}")
    print(f"  Size  : {size_mb:.1f} MB")
    print(f"  CNN   : {'included' if include_cnn else 'EXCLUDED (--no-cnn)'}")
    print("─" * 64)
    print(f"  Release layout under {DIST_DIR}:")
    print(f"    {exe_path.name}")
    print( "    configs/move_config.ini    ← edit to rebalance")
    print( "    configs/time_config.ini    ← edit to retune timing")
    print( "    Teams/Team1..Team6/        ← per-team configs + models")
    print()
    print(f"  Ship the entire {DIST_DIR.name}/ folder. Users can edit the .ini")
    print(f"  files and Teams/ contents without rebuilding the .exe.")


if __name__ == "__main__":
    main()
