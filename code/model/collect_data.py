"""
collect_data.py  (unified launcher — bundles both CNN and landmark pipelines)
─────────────────────────────────────────────────────────────────────────────
Single entry point that can collect training data for either the CNN pipeline
(raw 224x224 JPEGs per class) or the landmark pipeline (126-dim feature rows
in a CSV), depending on user selection.

Run as a script:
    python collect_data.py                       # interactive mode + team prompt
    python collect_data.py --mode cnn --team 1
    python collect_data.py --mode landmark --team 3 --cam 0

Run as a frozen .exe (built by build_collect_exe.py):
    collect_data.exe                             # interactive
    collect_data.exe --mode landmark --team 2

Controls (in the camera window, both modes):
    1-5    Start recording samples for that class (1=move1 … 5=move5)
    SPACE  Stop recording
    Q      Quit and save

Output layout (written next to the .exe / script):
    CNN       → teams/Team<N>/images/<class>/frame_XXXXX.jpg
    Landmark  → teams/Team<N>/hand_sign_data.csv

After each session ends, the team folder is also bundled into Team<N>.zip
(rooted at Team<N>/...) for easy upload to the Colab training notebook.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
import time
import zipfile

# Silence OpenCV's verbose backend chatter before cv2 is imported.
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"

import cv2  # noqa: E402  (intentional: env vars must be set first)


# ───────────────────────────────── path helpers ──────────────────────────────
# When PyInstaller freezes the script, __file__ points inside a temp extraction
# directory, which is wiped between runs. We need two anchors:
#   • app_dir()      — stable on-disk location, used for writing output data
#   • resource_dir() — read-only bundle root, used to find hand_landmarker.task

def app_dir() -> pathlib.Path:
    """Where to write the `teams/` output folder (persistent across runs)."""
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).parent
    return pathlib.Path(__file__).parent


def resource_dir() -> pathlib.Path:
    """Where bundled read-only data files live."""
    if getattr(sys, "frozen", False):
        # PyInstaller extracts --add-data files into sys._MEIPASS.
        return pathlib.Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # In dev mode the landmark task ships under landmark/.
    return pathlib.Path(__file__).parent / "landmark"


# ───────────────────────────────── constants ─────────────────────────────────
CLASS_KEYS = {
    ord("1"): "move1",
    ord("2"): "move2",
    ord("3"): "move3",
    ord("4"): "move4",
    ord("5"): "move5",
}
CLASS_NAMES       = list(CLASS_KEYS.values())
CAM_SCAN_RANGE    = 10
CNN_SAVE_SIZE     = (224, 224)
MAX_HANDS         = 4
COUNTDOWN_SECONDS = 3       # CNN-only: prep window before frames start saving
COUNTDOWN_DIM     = 0.5     # frame brightness multiplier during the countdown


# ───────────────────────────────── camera helpers ────────────────────────────
def find_camera() -> int | None:
    """Probe device indices [0, CAM_SCAN_RANGE) and return the first that opens."""
    for idx in range(CAM_SCAN_RANGE):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            print(f"[collect] Found camera at index {idx}")
            return idx
        cap.release()
    return None


def _draw_counter_panel(img, lines: list[str], x: int, y: int,
                        line_height: int = 22, font_scale: float = 0.5,
                        thickness: int = 1) -> None:
    """Render `lines` over a semi-transparent dark panel so text stays legible
    against any webcam background."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_w = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0]
                for line in lines)
    (_, text_h), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
    pad_x, pad_y = 8, 6
    x1, y1 = max(0, x - pad_x), max(0, y - text_h - pad_y)
    x2 = x + max_w + pad_x
    y2 = y + line_height * (len(lines) - 1) + pad_y
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), font, font_scale,
                    (240, 240, 240), thickness, cv2.LINE_AA)
        yy += line_height


def _print_counts_table(counts: dict, mode_label: str) -> None:
    """Render existing per-move counts as a small ASCII table."""
    print()
    print(f"Existing {mode_label} data:")
    print("  +---------+--------+")
    print("  | Move    |  Count |")
    print("  +---------+--------+")
    for cls in CLASS_NAMES:
        print(f"  | {cls:<7} | {counts.get(cls, 0):>6} |")
    print("  +---------+--------+")


def _prompt_reset_moves(counts: dict, mode_label: str) -> list[str]:
    """Show a counts table and ask the user which moves to clear.

    If `counts` has no entries > 0 (no existing data for this pipeline), skip
    the prompt entirely and return [].
    """
    if not any(c > 0 for c in counts.values()):
        return []
    _print_counts_table(counts, mode_label)
    ans = input("Reset data for any moves before starting? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        return []
    raw = input("Which moves to clear? (comma-separated, e.g. 1,3,4): ").strip()
    if not raw:
        print("[collect] No moves resetted.")
        return []
    moves: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or not 1 <= int(tok) <= 5:
            print(f"[collect] Skipping invalid move '{tok}' (must be 1-5).")
            continue
        name = f"move{int(tok)}"
        if name not in moves:
            moves.append(name)
    if not moves:
        print("[collect] No moves resetted.")
    return moves


def _reset_cnn_classes(class_dirs: dict, moves: list[str]) -> None:
    """Delete every .jpg under each named class directory."""
    for m in moves:
        d = class_dirs.get(m)
        if d is None or not d.exists():
            print(f"[collect] {m}: nothing to clear.")
            continue
        n = 0
        for jpg in d.glob("*.jpg"):
            jpg.unlink()
            n += 1
        print(f"[collect] {m}: cleared {n} image(s).")


def _reset_landmark_rows(data_file: pathlib.Path, moves: list[str]) -> None:
    """Drop rows whose first column matches any name in `moves`."""
    if not data_file.exists():
        print(f"[collect] {data_file.name} not found — nothing to clear.")
        return
    targets = set(moves)
    kept: list[list[str]] = []
    removed = 0
    with open(data_file, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] in targets:
                removed += 1
            else:
                kept.append(row)
    with open(data_file, "w", newline="") as f:
        csv.writer(f).writerows(kept)
    print(f"[collect] Cleared {removed} row(s) for {sorted(targets)}.")


def _zip_team_folder(team: int, suffix: str) -> None:
    """Bundle teams/Team{N}/ into Team{N}.zip with Team{N}/... archive paths.

    Only files matching `suffix` (e.g. ".jpg" for CNN, ".csv" for landmark)
    are included, so stray notes / OS metadata don't end up in the upload.
    """
    team_dir = app_dir() / "teams" / f"Team{team}"
    if not team_dir.exists():
        return
    suffix_lower = suffix.lower()
    files = [p for p in team_dir.rglob(f"*{suffix}")
             if p.is_file() and p.suffix.lower() == suffix_lower]
    if not files:
        print(f"[zip] No {suffix} files in {team_dir} — skipping zip.")
        return
    zip_path = app_dir() / f"Team{team}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f"Team{team}/{f.relative_to(team_dir).as_posix()}"
            zf.write(f, arcname)
    print(f"[zip] Wrote {len(files)} file(s) -> {zip_path}")


def open_camera(cam_arg: int | None) -> cv2.VideoCapture:
    """Open the requested camera index, or auto-detect; exits on failure."""
    cam_index = cam_arg if cam_arg is not None else find_camera()
    if cam_index is None:
        sys.exit("[collect] No camera found. Pass --cam <index> to specify one.")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"[collect] Cannot open camera {cam_index}")
    return cap


# ───────────────────────────────── CNN pipeline ──────────────────────────────
def run_cnn(team: int, cam_arg: int | None) -> None:
    """Capture raw 224x224 JPEG frames per class into teams/Team<N>/images/."""
    images_dir = app_dir() / "teams" / f"Team{team}" / "images"
    class_dirs = {cls: images_dir / cls for cls in CLASS_NAMES}
    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    class_counts = {cls: len(list(d.glob("*.jpg"))) for cls, d in class_dirs.items()}

    moves_to_reset = _prompt_reset_moves(class_counts, "CNN")
    if moves_to_reset:
        _reset_cnn_classes(class_dirs, moves_to_reset)
        class_counts = {cls: len(list(d.glob("*.jpg"))) for cls, d in class_dirs.items()}

    cap = open_camera(cam_arg)

    print(f"[collect_cnn] Team {team} — existing counts: {class_counts}")
    print("[collect_cnn] Show BOTH hands forming the shape.")
    print("[collect_cnn] Press 1-5 to record, SPACE to stop, Q to quit.")

    recording_class: str | None = None
    countdown_class: str | None = None
    countdown_end:   float      = 0.0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            if countdown_class is not None and now >= countdown_end:
                recording_class = countdown_class
                countdown_class = None
                print(f"[collect_cnn] Recording: {recording_class}")

            if recording_class is not None:
                count     = class_counts[recording_class]
                save_path = class_dirs[recording_class] / f"frame_{count:05d}.jpg"
                cv2.imwrite(str(save_path), cv2.resize(frame, CNN_SAVE_SIZE))
                class_counts[recording_class] += 1

            display = frame
            if countdown_class is not None:
                display = cv2.convertScaleAbs(frame, alpha=COUNTDOWN_DIM, beta=0)
                seconds_left = max(1, math.ceil(countdown_end - now))
                text = str(seconds_left)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 6.0, 12)
                cx = (display.shape[1] - tw) // 2
                cy = (display.shape[0] + th) // 2
                cv2.putText(display, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 12)

            if recording_class is not None:
                status_text, status_color = f"RECORDING: {recording_class.upper()}", (0, 0, 255)
            elif countdown_class is not None:
                status_text, status_color = f"GET READY: {countdown_class.upper()}", (0, 200, 255)
            else:
                status_text, status_color = "IDLE  (press 1-5)", (200, 200, 200)
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            _draw_counter_panel(
                display,
                [f"  {cls}: {class_counts[cls]}" for cls in CLASS_NAMES],
                10, 60,
            )

            cv2.imshow(f"Collect CNN Images - Team {team}", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording_class = None
                countdown_class = None
            elif key in CLASS_KEYS:
                countdown_class = CLASS_KEYS[key]
                countdown_end   = time.time() + COUNTDOWN_SECONDS
                recording_class = None
                print(f"[collect_cnn] Countdown {COUNTDOWN_SECONDS}s -> {countdown_class}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[collect_cnn] Final counts: {class_counts}")
        _zip_team_folder(team, ".jpg")


# ─────────────────────────────── landmark pipeline ───────────────────────────
def _landmark_bbox_area(lms) -> float:
    """Area of the 2D bounding box around a hand's landmarks (in normalized coords)."""
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _select_two_largest_hands(result):
    """Pick the two largest-area hands from a MediaPipe result and order them left-to-right."""
    if not result.hand_landmarks or len(result.hand_landmarks) < 2:
        return None
    top2 = sorted(result.hand_landmarks, key=_landmark_bbox_area, reverse=True)[:2]
    top2.sort(key=lambda lms: lms[0].x)
    return top2


def _normalize_hand(landmarks):
    """Translation + scale-invariant 63-d feature: wrist-centered, scaled by middle-finger MCP distance."""
    import numpy as np
    raw      = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    centered = raw - raw[0]
    scale    = np.linalg.norm(centered[9])
    if scale > 1e-6:
        centered /= scale
    return centered.flatten()


def run_landmark(team: int, cam_arg: int | None) -> None:
    """Capture two-hand landmark rows into teams/Team<N>/hand_sign_data.csv."""
    import numpy as np  # noqa: F401  (used inside _normalize_hand)
    import mediapipe as mp
    from mediapipe import tasks

    landmarker_path = resource_dir() / "hand_landmarker.task"
    if not landmarker_path.exists():
        sys.exit(f"[collect_lm] Landmarker model not found: {landmarker_path}")

    team_dir = app_dir() / "teams" / f"Team{team}"
    team_dir.mkdir(parents=True, exist_ok=True)
    data_file = team_dir / "hand_sign_data.csv"

    def _read_counts() -> dict:
        counts = {cls: 0 for cls in CLASS_NAMES}
        if data_file.exists():
            with open(data_file, newline="") as f:
                for row in csv.reader(f):
                    if row:
                        counts[row[0]] = counts.get(row[0], 0) + 1
        return counts

    class_counts = _read_counts()

    moves_to_reset = _prompt_reset_moves(class_counts, "Landmark")
    if moves_to_reset:
        _reset_landmark_rows(data_file, moves_to_reset)
        class_counts = _read_counts()

    drawing_utils    = tasks.vision.drawing_utils
    hand_connections = tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    cap = open_camera(cam_arg)

    samples: list[list] = []
    if any(c > 0 for c in class_counts.values()):
        print(f"[collect_lm] Existing data: {class_counts}")

    print(f"[collect_lm] Team {team}")
    print("[collect_lm] Show BOTH hands. Press 1-5 to record, SPACE to stop, Q to quit.")

    opts = tasks.vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(landmarker_path)),
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    detector = tasks.vision.HandLandmarker.create_from_options(opts)

    recording_class: str | None = None
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            two_hands   = _select_two_largest_hands(result)
            hands_ready = two_hands is not None

            if hands_ready:
                for hand_lm in two_hands:
                    drawing_utils.draw_landmarks(frame, hand_lm, hand_connections)
                if recording_class is not None:
                    feats = np.concatenate([
                        _normalize_hand(two_hands[0]),
                        _normalize_hand(two_hands[1]),
                    ])
                    samples.append([recording_class] + feats.tolist())
                    class_counts[recording_class] = class_counts.get(recording_class, 0) + 1
            elif result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    drawing_utils.draw_landmarks(frame, hand_lm, hand_connections)

            status_color = (0, 0, 255) if recording_class else (200, 200, 200)
            status_text  = (f"RECORDING: {recording_class.upper()}"
                            if recording_class else "IDLE  (press 1-5 to record)")
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            hands_text  = "2 HANDS OK" if hands_ready else "NEED 2 HANDS"
            hands_color = (0, 255, 100) if hands_ready else (0, 100, 255)
            cv2.putText(frame, hands_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hands_color, 2)

            _draw_counter_panel(
                frame,
                [f"  {cls}: {class_counts.get(cls, 0)}" for cls in CLASS_NAMES],
                10, 85,
            )

            cv2.imshow(f"Collect Landmark Data - Team {team}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording_class = None
            elif key in CLASS_KEYS:
                recording_class = CLASS_KEYS[key]
                print(f"[collect_lm] Recording: {recording_class}")
    except KeyboardInterrupt:
        print("\n[collect_lm] Interrupted — saving...")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

        if samples:
            with open(data_file, "a", newline="") as f:
                csv.writer(f).writerows(samples)
            print(f"[collect_lm] Saved {len(samples)} new samples to {data_file}")
        else:
            print("[collect_lm] No samples recorded.")
        print(f"[collect_lm] Totals: {class_counts}")
        _zip_team_folder(team, ".csv")


# ───────────────────────────────── interactive prompts ───────────────────────
def _prompt_mode() -> str:
    print()
    print("┌─────────────────────────────────────────────┐")
    print("│  Sigil Strike — Training Data Collector     │")
    print("├─────────────────────────────────────────────┤")
    print("│  1) CNN        (raw image frames)           │")
    print("│  2) Landmark   (MediaPipe keypoint vectors) │")
    print("└─────────────────────────────────────────────┘")
    while True:
        ans = input("Pick a model type [1/2]: ").strip().lower()
        if ans in ("1", "cnn"):
            return "cnn"
        if ans in ("2", "landmark", "lm"):
            return "landmark"
        print("  Please enter 1 or 2.")


def _prompt_team() -> int:
    while True:
        ans = input("Team number [1-6]: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= 6:
            return int(ans)
        print("  Please enter a number between 1 and 6.")


def _pause_before_exit() -> None:
    """Keep the console open when launched via a Windows double-click."""
    if getattr(sys, "frozen", False) and sys.platform.startswith("win"):
        try:
            input("\nPress Enter to close ... ")
        except EOFError:
            pass


# ───────────────────────────────── entry point ───────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect Sigil Strike training data (CNN or landmark mode).")
    p.add_argument("--mode", choices=("cnn", "landmark"),
                   help="Which pipeline to collect for (default: ask).")
    p.add_argument("--team", type=int, choices=range(1, 7),
                   help="Team number 1-6 (default: ask).")
    p.add_argument("--cam", type=int, default=None,
                   help="Camera index (default: auto-detect).")
    args = p.parse_args()

    mode = args.mode or _prompt_mode()
    team = args.team if args.team is not None else _prompt_team()

    try:
        if mode == "cnn":
            run_cnn(team, args.cam)
        else:
            run_landmark(team, args.cam)
    finally:
        _pause_before_exit()


if __name__ == "__main__":
    main()
