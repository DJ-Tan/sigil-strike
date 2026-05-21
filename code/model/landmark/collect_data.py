"""
collect_data.py  (landmark model)
──────────────────────────────────
Record TWO-HAND landmark samples for each of the 5 gesture classes.

Usage:
    python collect_data.py --team 1
    python collect_data.py --team 2 --cam 1

Controls (in the camera window):
    1-5   Start recording samples for that class
          1=move1  2=move2  3=move3  4=move4  5=move5
    SPACE Stop recording
    Q     Quit and save

Both hands must be visible to record a sample.
Data is saved to:  teams/TeamN/hand_sign_data.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"

import cv2
import mediapipe as mp
from mediapipe import tasks
import numpy as np

SCRIPT_DIR    = pathlib.Path(__file__).parent
LANDMARKER_PATH = SCRIPT_DIR / "hand_landmarker.task"

CLASS_KEYS = {
    ord("1"): "move1",
    ord("2"): "move2",
    ord("3"): "move3",
    ord("4"): "move4",
    ord("5"): "move5",
}

MAX_HANDS     = 4
CAM_SCAN_RANGE = 10


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


def landmark_bbox_area(landmarks: list) -> float:
    """Area of the 2D bounding box around a hand's landmarks (in normalized coords)."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def select_two_largest_hands(result) -> list[list] | None:
    """Pick the two largest-area hands from a MediaPipe result and order them left-to-right."""
    if not result.hand_landmarks or len(result.hand_landmarks) < 2:
        return None
    top2 = sorted(result.hand_landmarks, key=landmark_bbox_area, reverse=True)[:2]
    top2.sort(key=lambda lms: lms[0].x)
    return top2


def normalize_hand(landmarks: list) -> np.ndarray:
    """Translation + scale-invariant 63-d feature: wrist-centered, scaled by middle-finger MCP distance."""
    raw     = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist   = raw[0]
    centered = raw - wrist
    scale   = np.linalg.norm(centered[9])
    if scale > 1e-6:
        centered /= scale
    return centered.flatten()


def normalize_two_hands(left: list, right: list) -> np.ndarray:
    """Concatenate the normalized features of both hands into a single 126-d vector."""
    return np.concatenate([normalize_hand(left), normalize_hand(right)])


def main() -> None:
    """Parse CLI args, open the camera, and run the landmark-capture loop."""
    parser = argparse.ArgumentParser(description="Collect 2-hand landmark training data")
    parser.add_argument("--team", type=int, required=True, choices=range(1, 7),
                        help="Team number (1-6)")
    parser.add_argument("--cam", type=int, default=None,
                        help="Camera index (default: auto-detect)")
    args = parser.parse_args()

    team_dir  = SCRIPT_DIR / "teams" / f"Team{args.team}"
    team_dir.mkdir(parents=True, exist_ok=True)
    data_file = team_dir / "hand_sign_data.csv"

    if not LANDMARKER_PATH.exists():
        print(f"[collect] Landmarker model not found: {LANDMARKER_PATH}")
        sys.exit(1)

    drawing_utils   = tasks.vision.drawing_utils
    hand_connections = tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    cam_index = args.cam if args.cam is not None else find_camera()
    if cam_index is None:
        print("[collect] No camera found. Use --cam <index> to specify one.")
        sys.exit(1)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[collect] Cannot open camera {cam_index}")
        sys.exit(1)

    samples: list[list] = []
    recording_class: str | None = None
    class_counts: dict[str, int] = {v: 0 for v in CLASS_KEYS.values()}

    if data_file.exists():
        with open(data_file, newline="") as f:
            for row in csv.reader(f):
                if row:
                    class_counts[row[0]] = class_counts.get(row[0], 0) + 1
        print(f"[collect] Existing data: {class_counts}")

    print(f"[collect] Team {args.team} — cam {cam_index}")
    print("[collect] Show BOTH hands. Press 1-5 to record, SPACE to stop, Q to quit.")

    opts = tasks.vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    detector = tasks.vision.HandLandmarker.create_from_options(opts)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            two_hands   = select_two_largest_hands(result)
            hands_ready = two_hands is not None

            if hands_ready:
                for hand_lm in two_hands:
                    drawing_utils.draw_landmarks(frame, hand_lm, hand_connections)
                if recording_class is not None:
                    features = normalize_two_hands(two_hands[0], two_hands[1])
                    samples.append([recording_class] + features.tolist())
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
            for cls in ["move1", "move2", "move3", "move4", "move5"]:
                cv2.putText(frame, f"  {cls}: {class_counts.get(cls, 0)}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y += 22

            cv2.imshow(f"Collect Landmark Data — Team {args.team}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                recording_class = None
            elif key in CLASS_KEYS:
                recording_class = CLASS_KEYS[key]
                print(f"[collect] Recording: {recording_class}")

    except KeyboardInterrupt:
        print("\n[collect] Interrupted — saving...")

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

        if samples:
            with open(data_file, "a", newline="") as f:
                csv.writer(f).writerows(samples)
            print(f"[collect] Saved {len(samples)} new samples to {data_file}")
        else:
            print("[collect] No samples recorded.")
        print(f"[collect] Totals: {class_counts}")


if __name__ == "__main__":
    main()
