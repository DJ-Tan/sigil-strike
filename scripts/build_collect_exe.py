"""
build_collect_exe.py
────────────────────
Builds `collect_data.exe`: a single-file standalone executable that bundles
the Python runtime, OpenCV, MediaPipe (with its hand-landmarker model),
NumPy, and the unified collect_data.py launcher.

Usage:
    python scripts/build_collect_exe.py            # build into dist/collect_data/
    python scripts/build_collect_exe.py --clean    # wipe build/collect_data + spec first

The resulting .exe has no dependencies — drop it on any 64-bit Windows machine
and double-click. Training data is written to a `teams/` folder created next
to the .exe.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


HERE             = pathlib.Path(__file__).resolve().parent           # scripts/
PROJECT_ROOT     = HERE.parent                                       # repo root
MODEL_DIR        = PROJECT_ROOT / "code" / "model"

ENTRY_SCRIPT     = MODEL_DIR / "collect_data.py"
LANDMARKER_FILE  = MODEL_DIR / "landmark" / "hand_landmarker.task"

EXE_NAME         = "collect_data"
DIST_DIR         = PROJECT_ROOT / "dist"  / EXE_NAME
BUILD_DIR        = PROJECT_ROOT / "build" / EXE_NAME
SPEC_FILE        = HERE / f"{EXE_NAME}.spec"


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
        return
    except ImportError:
        pass
    print("[build] PyInstaller missing — installing ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"])


def ensure_runtime_deps() -> None:
    """Make sure cv2/mediapipe/numpy are importable in the build interpreter.
    PyInstaller introspects the script using *this* interpreter, so missing
    runtime imports fail at build time, not just at exe runtime.
    """
    missing: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        missing.append("mediapipe")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        print(f"[build] Installing runtime deps: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", *missing])


def build(clean: bool) -> pathlib.Path:
    if not ENTRY_SCRIPT.exists():
        sys.exit(f"[build] Entry script not found: {ENTRY_SCRIPT}")
    if not LANDMARKER_FILE.exists():
        sys.exit(f"[build] hand_landmarker.task not found at {LANDMARKER_FILE}")

    if clean:
        for path in (DIST_DIR, BUILD_DIR):
            if path.exists():
                print(f"[build] Removing {path}")
                shutil.rmtree(path)
        if SPEC_FILE.exists():
            print(f"[build] Removing {SPEC_FILE}")
            SPEC_FILE.unlink()

    add_data_sep = ";" if sys.platform.startswith("win") else ":"

    # Heavy packages that may live in the same site-packages but are NOT
    # used by this script. Excluding them stops PyInstaller from following
    # speculative imports (e.g. matplotlib backends → torch, transformers).
    EXCLUDES = [
        "torch", "torchvision", "torchaudio",
        "transformers", "tokenizers", "huggingface_hub", "hf_xet",
        "scipy", "sklearn", "pandas", "seaborn", "plotly",
        # NOTE: matplotlib stays — mediapipe.tasks.python.vision.drawing_utils
        # imports it directly. It doesn't pull in torch on its own.
        "sympy", "networkx", "imageio", "imageio_ffmpeg", "PIL.ImageQt",
        "IPython", "notebook", "jupyter", "jupyter_client", "jupyter_core",
        "ipykernel", "ipywidgets", "jedi", "pygments", "pytest",
        "tensorflow", "tensorflow_hub", "tensorboard", "keras",
        "pygame", "pydantic", "pydantic_core", "babel",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--console",
        "--name", EXE_NAME,
        # Bundle the MediaPipe hand-landmarker model at the bundle root so
        # resource_dir() (which returns sys._MEIPASS when frozen) finds it.
        "--add-data", f"{LANDMARKER_FILE}{add_data_sep}.",
        # MediaPipe needs its native .pyd/.dll modules plus internal data
        # files at runtime. Narrow collect calls keep speculative imports
        # from pulling in torch / transformers.
        "--collect-binaries", "mediapipe",
        "--collect-data", "mediapipe",
        "--collect-submodules", "mediapipe.tasks",
    ]
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    cmd += [
        "--workpath", str(BUILD_DIR),
        "--distpath", str(DIST_DIR.parent),       # PyInstaller appends --name; final → dist/collect_data.exe
        "--specpath", str(HERE),
        str(ENTRY_SCRIPT),
    ]
    print("[build] Running:")
    print("        " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.check_call(cmd)

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


def main() -> None:
    p = argparse.ArgumentParser(description="Build collect_data.exe via PyInstaller.")
    p.add_argument("--clean", action="store_true",
                   help="Wipe build/collect_data, dist/collect_data, and the spec file first.")
    args = p.parse_args()

    ensure_pyinstaller()
    ensure_runtime_deps()
    exe_path = build(clean=args.clean)

    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print()
    print("-" * 60)
    print(f"  Built: {exe_path}")
    print(f"  Size : {size_mb:.1f} MB")
    print("-" * 60)
    print("  Run with no args for an interactive prompt, or:")
    print(f"    {exe_path.name} --mode cnn --team 1")
    print(f"    {exe_path.name} --mode landmark --team 3 --cam 0")
    print("  Training data is written to ./teams/ next to the .exe.")


if __name__ == "__main__":
    main()
