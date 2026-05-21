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
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

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
CLASS_NAMES    = list(CLASS_KEYS.values())
CAM_SCAN_RANGE = 10
CNN_SAVE_SIZE  = (224, 224)
MAX_HANDS      = 4


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

    cap = open_camera(cam_arg)
    class_counts = {cls: len(list(d.glob("*.jpg"))) for cls, d in class_dirs.items()}

    print(f"[collect_cnn] Team {team} — existing counts: {class_counts}")
    print("[collect_cnn] Show BOTH hands forming the shape.")
    print("[collect_cnn] Press 1-5 to record, SPACE to stop, Q to quit.")

    recording_class: str | None = None
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            if recording_class is not None:
                count     = class_counts[recording_class]
                save_path = class_dirs[recording_class] / f"frame_{count:05d}.jpg"
                cv2.imwrite(str(save_path), cv2.resize(frame, CNN_SAVE_SIZE))
                class_counts[recording_class] += 1

            status_color = (0, 0, 255) if recording_class else (200, 200, 200)
            status_text  = (f"RECORDING: {recording_class.upper()}"
                            if recording_class else "IDLE  (press 1-5)")
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            y = 60
            for cls in CLASS_NAMES:
                cv2.putText(frame, f"  {cls}: {class_counts[cls]}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y += 22

            cv2.imshow(f"Collect CNN Images - Team {team}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording_class = None
            elif key in CLASS_KEYS:
                recording_class = CLASS_KEYS[key]
                print(f"[collect_cnn] Recording: {recording_class}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[collect_cnn] Final counts: {class_counts}")


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

    drawing_utils    = tasks.vision.drawing_utils
    hand_connections = tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    cap = open_camera(cam_arg)

    samples: list[list] = []
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    if data_file.exists():
        with open(data_file, newline="") as f:
            for row in csv.reader(f):
                if row:
                    class_counts[row[0]] = class_counts.get(row[0], 0) + 1
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

            y = 85
            for cls in CLASS_NAMES:
                cv2.putText(frame, f"  {cls}: {class_counts.get(cls, 0)}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y += 22

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
