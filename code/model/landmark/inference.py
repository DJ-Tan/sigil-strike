"""
inference.py  (landmark model)
───────────────────────────────
Real-time TWO-HAND sign classification using MediaPipe landmarks + sklearn.

Usage:
    python inference.py --team 1 --player 1
    python inference.py --team 2 --player 2 --cam 1
    python inference.py --team 3 --player 1 --threshold 0.8
"""

from __future__ import annotations

import argparse
import os
import pathlib
import socket
import sys
import time

os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"

import cv2
import joblib
import mediapipe as mp
from mediapipe import tasks
import numpy as np

SCRIPT_DIR      = pathlib.Path(__file__).parent
LANDMARKER_PATH = SCRIPT_DIR / "hand_landmarker.task"
TEAMS_ENV_DIR   = SCRIPT_DIR.parent.parent.parent / "Teams"

GAME_HOST = "127.0.0.1"
GAME_PORT = 5001

STABLE_FRAMES_REQUIRED = 8
COOLDOWN_SECONDS       = 1.2
MAX_HANDS              = 4
CAM_SCAN_RANGE         = 10

ACTION_BYTE = {
    "move1": b"T", "move2": b"D", "move3": b"S",
    "move4": b"H", "move5": b"P",
}
ACTION_DISPLAY = {
    "move1": "MOVE1", "move2": "MOVE2", "move3": "MOVE3",
    "move4": "MOVE4", "move5": "MOVE5",
}


def _parse_env(path: pathlib.Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file; missing files return an empty dict."""
    env: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


def find_camera() -> int | None:
    """Probe device indices [0, CAM_SCAN_RANGE) and return the first that opens."""
    for idx in range(CAM_SCAN_RANGE):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            print(f"[inference] Found camera at index {idx}")
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
    raw      = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    centered = raw - raw[0]
    scale    = np.linalg.norm(centered[9])
    if scale > 1e-6:
        centered /= scale
    return centered.flatten()


def normalize_two_hands(left: list, right: list) -> np.ndarray:
    """Concatenate the normalized features of both hands into a single 126-d vector."""
    return np.concatenate([normalize_hand(left), normalize_hand(right)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Landmark inference for SIGIL STRIKE")
    parser.add_argument("--team",      type=int, required=True, choices=range(1, 7))
    parser.add_argument("--player",    type=int, choices=[1, 2], default=1)
    parser.add_argument("--cam",       type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold (default: read from team.env or 0.6)")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    team_models  = TEAMS_ENV_DIR / f"Team{args.team}" / "models"
    model_file   = team_models / "hand_sign_classifier.pkl"
    encoder_file = team_models / "label_encoder.pkl"

    for f in (model_file, encoder_file):
        if not f.exists():
            print(f"[inference] Missing: {f}")
            print("  Run train_model.py --team", args.team, "first.")
            sys.exit(1)

    if not LANDMARKER_PATH.exists():
        print(f"[inference] Landmarker not found: {LANDMARKER_PATH}")
        sys.exit(1)

    team_env = _parse_env(TEAMS_ENV_DIR / f"Team{args.team}" / "team.env")
    threshold = args.threshold
    if threshold is None:
        try:
            threshold = float(team_env.get("CONFIDENCE_THRESHOLD", "0.6"))
        except ValueError:
            threshold = 0.6

    model = joblib.load(model_file)
    le    = joblib.load(encoder_file)
    print(f"[inference] Team {args.team} | {type(model).__name__} | classes: {list(le.classes_)}")

    # Warn loudly if the trained labels don't match what the game can route.
    # The game's _LABEL_TO_ACTION (code/game.py) only maps move1..move5.
    unmapped = [str(c) for c in le.classes_ if str(c) not in ACTION_BYTE]
    if unmapped:
        print(f"[inference] WARNING: these classes won't fire in-game: {unmapped}")
        print("[inference] Retrain with labels in {move1, move2, move3, move4, move5}, "
              "or extend ACTION_BYTE / game._LABEL_TO_ACTION to map them.")

    cam_index = args.cam if args.cam is not None else find_camera()
    if cam_index is None:
        print("[inference] No camera found. Use --cam <index>.")
        sys.exit(1)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[inference] Cannot open camera {cam_index}")
        sys.exit(1)

    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    prefix = str(args.player).encode()

    drawing_utils    = tasks.vision.drawing_utils
    hand_connections = tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    opts = tasks.vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    detector = tasks.vision.HandLandmarker.create_from_options(opts)

    print(f"[inference] Player {args.player} | cam {cam_index} | threshold {threshold:.0%} | → {GAME_HOST}:{GAME_PORT}")

    stable_label: str | None  = None
    stable_count: int         = 0
    last_sent_time: float     = 0.0
    last_sent_label: str | None = None
    frame_num = 0
    last_log_time = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("[inference] Camera read failed — exiting.")
            break

        frame_num += 1
        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        predicted_label: str | None = None
        confidence = 0.0
        n_hands    = len(result.hand_landmarks) if result.hand_landmarks else 0
        two_hands  = select_two_largest_hands(result)

        now = time.time()
        if now - last_log_time >= 1.0:
            if two_hands is not None:
                feats = normalize_two_hands(two_hands[0], two_hands[1]).reshape(1, -1)
                probs = model.predict_proba(feats)[0]
                best  = np.argmax(probs)
                print(f"[inference] frame={frame_num}  hands={n_hands}  "
                      f"pred={le.classes_[best]}  conf={probs[best]:.0%}  "
                      f"stable={stable_count}/{STABLE_FRAMES_REQUIRED}")
            else:
                print(f"[inference] frame={frame_num}  hands={n_hands}  (need 2 hands)")
            last_log_time = now

        if two_hands is not None:
            if not args.no_window:
                for hand_lm in two_hands:
                    drawing_utils.draw_landmarks(frame, hand_lm, hand_connections)
            feats      = normalize_two_hands(two_hands[0], two_hands[1]).reshape(1, -1)
            probs      = model.predict_proba(feats)[0]
            best_idx   = np.argmax(probs)
            confidence = float(probs[best_idx])
            if confidence >= threshold:
                predicted_label = le.classes_[best_idx]
        elif not args.no_window and result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_lm, hand_connections)

        if predicted_label == stable_label and predicted_label is not None:
            stable_count += 1
        else:
            stable_label = predicted_label
            stable_count = 0

        if (stable_count >= STABLE_FRAMES_REQUIRED
                and stable_label is not None
                and stable_label in ACTION_BYTE
                and (stable_label != last_sent_label
                     or now - last_sent_time >= COOLDOWN_SECONDS)):
            sock.sendto(prefix + ACTION_BYTE[stable_label], (GAME_HOST, GAME_PORT))
            last_sent_label = stable_label
            last_sent_time  = now
            stable_count    = 0
            print(f"[inference] >>> SENT P{args.player} → {ACTION_DISPLAY[stable_label]} ({confidence:.0%})")

        if not args.no_window:
            if predicted_label:
                # Show the raw class name as-is when it's not a game-routable label.
                display = ACTION_DISPLAY.get(predicted_label, str(predicted_label).upper())
                suffix  = "" if predicted_label in ACTION_BYTE else "  [unmapped]"
                label_text = f"{display} ({confidence:.0%}){suffix}"
                color = (0, 255, 100) if predicted_label in ACTION_BYTE else (0, 200, 255)
            elif two_hands is None:
                label_text, color = "Show BOTH hands", (200, 200, 200)
            else:
                label_text = f"Low confidence ({confidence:.0%})"
                color = (0, 140, 255)

            cv2.putText(frame, f"P{args.player} [landmark]: {label_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bar_pct = min(stable_count / STABLE_FRAMES_REQUIRED, 1.0)
            cv2.rectangle(frame, (10, 45), (10 + int(300 * bar_pct), 60), (0, 255, 100), -1)
            cv2.rectangle(frame, (10, 45), (310, 60), (200, 200, 200), 2)
            cv2.imshow(f"SIGIL STRIKE — P{args.player} Landmark", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    detector.close()
    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
