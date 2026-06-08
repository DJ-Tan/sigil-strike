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

After the session ends, the folder is also bundled into TeamN.zip
(rooted at TeamN/...) for easy upload to the Colab training notebook.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time
import zipfile

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

SAVE_SIZE         = (224, 224)
CAM_SCAN_RANGE    = 10
COUNTDOWN_SECONDS = 3
COUNTDOWN_DIM     = 0.5   # frame brightness multiplier during countdown


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
    print("Existing CNN data:")
    print("  +---------+--------+")
    print("  | Move    |  Count |")
    print("  +---------+--------+")
    for cls in CLASS_KEYS.values():
        print(f"  | {cls:<7} | {counts.get(cls, 0):>6} |")
    print("  +---------+--------+")


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
        print("[collect_cnn] No moves resetted.")
        return []
    moves: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or not 1 <= int(tok) <= 5:
            print(f"[collect_cnn] Skipping invalid move '{tok}' (must be 1-5).")
            continue
        name = f"move{int(tok)}"
        if name not in moves:
            moves.append(name)
    if not moves:
        print("[collect_cnn] No moves resetted.")
    return moves


def reset_classes(class_dirs: dict, moves: list[str]) -> None:
    """Delete every .jpg under each named class directory."""
    for m in moves:
        d = class_dirs.get(m)
        if d is None or not d.exists():
            print(f"[collect_cnn] {m}: nothing to clear.")
            continue
        n = 0
        for jpg in d.glob("*.jpg"):
            jpg.unlink()
            n += 1
        print(f"[collect_cnn] {m}: cleared {n} image(s).")


def zip_team_folder(team_dir: pathlib.Path, zip_path: pathlib.Path) -> None:
    """Bundle .jpg files under team_dir into zip_path, rooted at Team<N>/..."""
    if not team_dir.exists():
        return
    files = [p for p in team_dir.rglob("*.jpg")
             if p.is_file() and p.suffix.lower() == ".jpg"]
    if not files:
        print(f"[collect_cnn] No .jpg files in {team_dir} — skipping zip.")
        return
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f"{team_dir.name}/{f.relative_to(team_dir).as_posix()}"
            zf.write(f, arcname)
    print(f"[collect_cnn] Wrote {len(files)} file(s) -> {zip_path}")


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

    class_counts = {cls: len(list(d.glob("*.jpg"))) for cls, d in class_dirs.items()}

    moves_to_reset = prompt_reset_moves(class_counts)
    if moves_to_reset:
        reset_classes(class_dirs, moves_to_reset)
        class_counts = {cls: len(list(d.glob("*.jpg"))) for cls, d in class_dirs.items()}

    cam_index = args.cam if args.cam is not None else find_camera()
    if cam_index is None:
        print("[collect_cnn] No camera found. Use --cam <index>.")
        sys.exit(1)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[collect_cnn] Cannot open camera {cam_index}")
        sys.exit(1)

    recording_class: str | None = None
    countdown_class: str | None = None
    countdown_end:   float      = 0.0

    print(f"[collect_cnn] Team {args.team} — counts: {class_counts}")
    print("[collect_cnn] Show BOTH hands forming the shape.")
    print("[collect_cnn] Press 1-5 to record, SPACE to stop, Q to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # Countdown → recording transition.
        now = time.time()
        if countdown_class is not None and now >= countdown_end:
            recording_class = countdown_class
            countdown_class = None
            print(f"[collect_cnn] Recording: {recording_class}")

        if recording_class is not None:
            count     = class_counts[recording_class]
            save_path = class_dirs[recording_class] / f"frame_{count:05d}.jpg"
            cv2.imwrite(str(save_path), cv2.resize(frame, SAVE_SIZE))
            class_counts[recording_class] += 1

        # Dim the displayed frame during the countdown and overlay the number.
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

        draw_counter_panel(
            display,
            [f"  {cls}: {class_counts[cls]}"
             for cls in ["move1", "move2", "move3", "move4", "move5"]],
            10, 60,
        )

        cv2.imshow(f"Collect CNN Images — Team {args.team}", display)
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

    cap.release()
    cv2.destroyAllWindows()
    print(f"[collect_cnn] Final counts: {class_counts}")

    zip_team_folder(team_dir, SCRIPT_DIR / f"Team{args.team}.zip")


if __name__ == "__main__":
    main()
