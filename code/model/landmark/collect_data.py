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

After the session ends, the folder is also bundled into TeamN.zip
(rooted at TeamN/...) for easy upload to the Colab training notebook.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import zipfile

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


def draw_counter_panel(img, lines: list[str], x: int, y: int,
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


def print_counts_table(counts: dict) -> None:
    """Render existing per-move counts as a small ASCII table."""
    print()
    print("Existing Landmark data:")
    print("  +---------+--------+")
    print("  | Move    |  Count |")
    print("  +---------+--------+")
    for cls in CLASS_KEYS.values():
        print(f"  | {cls:<7} | {counts.get(cls, 0):>6} |")
    print("  +---------+--------+")


def read_csv_counts(data_file: pathlib.Path) -> dict:
    """Count rows per class in the landmark CSV (empty/missing -> all zeros)."""
    counts = {cls: 0 for cls in CLASS_KEYS.values()}
    if data_file.exists():
        with open(data_file, newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def prompt_reset_moves(counts: dict) -> list[str]:
    """Show a counts table and ask which moves to clear.

    If no existing data for any move, skip the prompt entirely and return [].
    """
    if not any(c > 0 for c in counts.values()):
        return []
    print_counts_table(counts)
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


def reset_csv_rows(data_file: pathlib.Path, moves: list[str]) -> None:
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


def zip_team_folder(team_dir: pathlib.Path, zip_path: pathlib.Path) -> None:
    """Bundle .csv files under team_dir into zip_path, rooted at Team<N>/..."""
    if not team_dir.exists():
        return
    files = [p for p in team_dir.rglob("*.csv")
             if p.is_file() and p.suffix.lower() == ".csv"]
    if not files:
        print(f"[collect] No .csv files in {team_dir} — skipping zip.")
        return
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f"{team_dir.name}/{f.relative_to(team_dir).as_posix()}"
            zf.write(f, arcname)
    print(f"[collect] Wrote {len(files)} file(s) -> {zip_path}")


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

    class_counts = read_csv_counts(data_file)

    moves_to_reset = prompt_reset_moves(class_counts)
    if moves_to_reset:
        reset_csv_rows(data_file, moves_to_reset)
        class_counts = read_csv_counts(data_file)

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

    if any(c > 0 for c in class_counts.values()):
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

            draw_counter_panel(
                frame,
                [f"  {cls}: {class_counts.get(cls, 0)}"
                 for cls in ["move1", "move2", "move3", "move4", "move5"]],
                10, 85,
            )

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

        zip_team_folder(team_dir, SCRIPT_DIR / f"Team{args.team}.zip")


if __name__ == "__main__":
    main()
