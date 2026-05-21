"""
collect_data.py  (CNN model)
──────────────────────────────
Capture raw camera frames for each gesture class — no landmark extraction.
Both hands forming a shape should be visible in the frame.

Usage:
    python collect_data.py --team 1
    python collect_data.py --team 2 --cam 1

Controls:
    1-5   Start recording frames for that class
          1=move1  2=move2  3=move3  4=move4  5=move5
    SPACE Stop recording
    Q     Quit

Images are saved to:  teams/TeamN/images/{class}/frame_XXXXX.jpg
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"

import cv2

SCRIPT_DIR = pathlib.Path(__file__).parent

CLASS_KEYS = {
    ord("1"): "move1",
    ord("2"): "move2",
    ord("3"): "move3",
    ord("4"): "move4",
    ord("5"): "move5",
}

SAVE_SIZE     = (224, 224)
CAM_SCAN_RANGE = 10


def find_camera() -> int | None:
    """Probe device indices [0, CAM_SCAN_RANGE) and return the first that opens."""
    for idx in range(CAM_SCAN_RANGE):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            print(f"[collect_cnn] Found camera at index {idx}")
            return idx
        cap.release()
    return None


def main() -> None:
    """Parse CLI args, open the camera, and capture per-class JPEG frames."""
    parser = argparse.ArgumentParser(description="Collect CNN hand-sign images")
    parser.add_argument("--team", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--cam",  type=int, default=None)
    args = parser.parse_args()

    team_dir   = SCRIPT_DIR / "teams" / f"Team{args.team}"
    images_dir = team_dir / "images"

    class_dirs: dict[str, pathlib.Path] = {}
    for cls in CLASS_KEYS.values():
        d = images_dir / cls
        d.mkdir(parents=True, exist_ok=True)
        class_dirs[cls] = d

    cam_index = args.cam if args.cam is not None else find_camera()
    if cam_index is None:
        print("[collect_cnn] No camera found. Use --cam <index>.")
        sys.exit(1)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[collect_cnn] Cannot open camera {cam_index}")
        sys.exit(1)

    recording_class: str | None = None
    class_counts = {cls: len(list(d.glob("*.jpg"))) for cls, d in class_dirs.items()}

    print(f"[collect_cnn] Team {args.team} — existing counts: {class_counts}")
    print("[collect_cnn] Show BOTH hands forming the shape.")
    print("[collect_cnn] Press 1-5 to record, SPACE to stop, Q to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        if recording_class is not None:
            count     = class_counts[recording_class]
            save_path = class_dirs[recording_class] / f"frame_{count:05d}.jpg"
            cv2.imwrite(str(save_path), cv2.resize(frame, SAVE_SIZE))
            class_counts[recording_class] += 1

        status_color = (0, 0, 255) if recording_class else (200, 200, 200)
        status_text  = (f"RECORDING: {recording_class.upper()}"
                        if recording_class else "IDLE  (press 1-5)")
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        y = 60
        for cls in ["move1", "move2", "move3", "move4", "move5"]:
            cv2.putText(frame, f"  {cls}: {class_counts[cls]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 22

        cv2.imshow(f"Collect CNN Images — Team {args.team}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            recording_class = None
        elif key in CLASS_KEYS:
            recording_class = CLASS_KEYS[key]
            print(f"[collect_cnn] Recording: {recording_class}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[collect_cnn] Final counts: {class_counts}")


if __name__ == "__main__":
    main()
