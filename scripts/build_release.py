"""
build_release.py
────────────────
Packages the output of build_game_exe.py into a shippable release folder
and a single .zip archive.

Usage:
    python scripts/build_release.py                       # package existing dist/sigil_strike/
    python scripts/build_release.py --build               # run build_game_exe.py first
    python scripts/build_release.py --build --no-cnn      # slim landmark-only build
    python scripts/build_release.py --name v1.2           # → sigil_strike_v1.2.zip

Output (under <repo>/release/):
    sigil_strike/
        sigil_strike.exe
        README.txt
        configs/move_config.ini
        configs/time_config.ini
        Teams/Team1..Team6/
    sigil_strike.zip          (the entire folder above, zipped)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import zipfile


HERE         = pathlib.Path(__file__).resolve().parent       # scripts/
PROJECT_ROOT = HERE.parent
EXE_NAME     = "sigil_strike"

DIST_DIR     = PROJECT_ROOT / "dist" / EXE_NAME      # build_game_exe.py output
RELEASE_DIR  = PROJECT_ROOT / "release"
BUILD_SCRIPT = HERE / "build_game_exe.py"

EXE_FILE     = f"{EXE_NAME}.exe" if sys.platform.startswith("win") else EXE_NAME


README_TEMPLATE = """SIGIL STRIKE — release build
============================

To play:
    Double-click {exe_name}

User-editable files (next to the .exe):

    configs/move_config.ini
        Balance tuning: damage, heal, defence percentages, combo sequences.
        Edit values and restart the game to apply.

    configs/time_config.ini
        Timing: resolve interval, deathmatch start/decay, cooldowns.

    Teams/Team1 ... Team6/
        Per-team configs and trained models.
        - team.env       Name, colour, model type, per-move thresholds.
        - models/        Trained classifier or CNN checkpoint.
        - model_arch.py  (CNN only, optional) custom architecture.

If a configs/*.ini file or a Teams/TeamN/ folder is missing, the game
prints a warning and falls back to built-in defaults — it does not crash.

Tournament mode (default):
    {exe_name}

Test mode — skip the bracket and run one match between two teams:
    {exe_name} 1 4      # Team1 vs Team4

Controls:
    P1: Q W E R T
    P2: Y U I O P
    F11 = fullscreen, ESC = quit
"""


def _run_build(no_cnn: bool, clean: bool) -> None:
    """Invoke build_game_exe.py in the current interpreter."""
    if not BUILD_SCRIPT.exists():
        sys.exit(f"[release] build script not found: {BUILD_SCRIPT}")
    cmd = [sys.executable, str(BUILD_SCRIPT)]
    if no_cnn:
        cmd.append("--no-cnn")
    if clean:
        cmd.append("--clean")
    print(f"[release] Running build: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _verify_dist() -> None:
    """Fail fast if dist/sigil_strike/ doesn't have the expected files."""
    required = [
        DIST_DIR / EXE_FILE,
        DIST_DIR / "configs" / "move_config.ini",
        DIST_DIR / "configs" / "time_config.ini",
        DIST_DIR / "Teams",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[release] {DIST_DIR} is incomplete. Missing:")
        for p in missing:
            print(f"    {p}")
        sys.exit("[release] Run `python scripts/build_game_exe.py` first, or pass --build.")


def _stage_release(folder_name: str) -> pathlib.Path:
    """Copy dist/sigil_strike/ contents into release/<folder_name>/ and write a README."""
    target = RELEASE_DIR / folder_name
    if target.exists():
        print(f"[release] removing stale {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    shutil.copy2(DIST_DIR / EXE_FILE, target / EXE_FILE)
    shutil.copytree(DIST_DIR / "configs", target / "configs",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(DIST_DIR / "Teams", target / "Teams",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (target / "README.txt").write_text(
        README_TEMPLATE.format(exe_name=EXE_FILE), encoding="utf-8")

    print(f"[release] staged → {target}")
    return target


def _zip_release(folder: pathlib.Path, zip_name: str) -> pathlib.Path:
    """Zip `folder` (and its tree) into release/<zip_name>.zip."""
    zip_path = RELEASE_DIR / f"{zip_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"[release] zipping → {zip_path}")
    base = folder.parent  # so arcnames start with the folder name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(folder.rglob("*")):
            arcname = path.relative_to(base)
            zf.write(path, arcname)
    return zip_path


def main() -> None:
    p = argparse.ArgumentParser(description="Package sigil_strike.exe into a release zip.")
    p.add_argument("--build",   action="store_true",
                   help="Run build_game_exe.py before packaging.")
    p.add_argument("--no-cnn",  action="store_true",
                   help="Passed through to build_game_exe.py when --build is given.")
    p.add_argument("--clean",   action="store_true",
                   help="Passed through to build_game_exe.py when --build is given.")
    p.add_argument("--name", default=EXE_NAME,
                   help=f"Release folder + zip basename (default: {EXE_NAME}). "
                        f"Use this to tag versions, e.g. --name sigil_strike_v1.2.")
    args = p.parse_args()

    if args.build:
        _run_build(no_cnn=args.no_cnn, clean=args.clean)

    _verify_dist()
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    folder = _stage_release(args.name)
    zip_path = _zip_release(folder, args.name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print()
    print("─" * 64)
    print(f"  Folder : {folder}")
    print(f"  Zip    : {zip_path}")
    print(f"  Size   : {size_mb:.1f} MB")
    print("─" * 64)
    print("  Ship the .zip — recipients unzip and double-click the .exe.")


if __name__ == "__main__":
    main()
