"""
camera.py
─────────
Standalone gesture-prediction process.

Run alongside the game:
    python camera.py --player 1      # webcam index 0
    python camera.py --player 2 --cam 1

How it communicates with the game
──────────────────────────────────
This script sends predicted Action values to game.py via a UDP socket.
game.py listens on localhost:5001.  camera.py sends one-byte messages:

    b'1T'  →  Player 1, Move1
    b'1D'  →  Player 1, Move2
    b'1S'  →  Player 1, Move3
    b'1H'  →  Player 1, Move4
    b'1P'  →  Player 1, Move5
    (same pattern with b'2' prefix for Player 2)

game.py integration
───────────────────
Add these lines to Game.__init__():

    import socket, threading
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.bind(("127.0.0.1", 5001))
    self._sock.setblocking(False)

And call self._poll_camera_socket() each frame from _update():

    def _poll_camera_socket(self):
        from moves import Action
        ACTION_BYTE = {b'T': Action.MOVE1, b'D': Action.MOVE2,
                       b'S': Action.MOVE3, b'H': Action.MOVE4,
                       b'P': Action.MOVE5}
        try:
            while True:
                data, _ = self._sock.recvfrom(8)
                if len(data) >= 2:
                    pid    = int(chr(data[0]))
                    player = self.p1 if pid == 1 else self.p2
                    action = ACTION_BYTE.get(data[1:2])
                    if action:
                        self._register_action(player, action)
        except BlockingIOError:
            pass

Gesture → Action mapping
─────────────────────────
We use MediaPipe Hands (lighter than full OpenPose, ships as a pip package)
to count extended fingers, then map finger counts to actions:

    1 finger  → Move1  (index only = pointing = "sharp")
    2 fingers → Move2  (index + middle = peace sign shape)
    3 fingers → Move3  (index + middle + ring = half open)
    4 fingers → Move4  (all fingers except thumb)
    5 fingers → Move5  (open palm)

To use the full OpenPose skeleton instead, replace the MediaPipe block with
your OpenPose inference call and map the resulting keypoints to one of the
five Action values however you prefer.

Requirements
────────────
    pip install mediapipe opencv-python

OpenPose alternative:
    pip install openpose   (or build from source: github.com/CMU-Perceptual-Computing-Lab/openpose)
"""

from __future__ import annotations

import argparse
import socket
import time
import sys

# ── Optional imports — degrade gracefully if not installed ────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[camera] opencv-python not found.  Install with:  pip install opencv-python")

try:
    import mediapipe as mp
    MP_OK = True
except ImportError:
    MP_OK = False
    print("[camera] mediapipe not found.  Install with:  pip install mediapipe")


# ── Constants ─────────────────────────────────────────────────────────────────

GAME_HOST = "127.0.0.1"
GAME_PORT = 5001

# Minimum frames a gesture must be stable before it is sent
STABLE_FRAMES_REQUIRED = 8

# How long to wait (seconds) before the same gesture can be sent again
COOLDOWN_SECONDS = 1.2

# Finger-count → Action byte
FINGER_TO_ACTION: dict[int, bytes] = {
    1: b'T',   # Move1
    2: b'D',   # Move2
    3: b'S',   # Move3
    4: b'H',   # Move4
    5: b'P',   # Move5
}

# Action byte → display name (for the overlay)
ACTION_NAMES: dict[bytes, str] = {
    b'T': "MOVE1  (1 finger)",
    b'D': "MOVE2  (2 fingers)",
    b'S': "MOVE3  (3 fingers)",
    b'H': "MOVE4  (4 fingers)",
    b'P': "MOVE5  (open palm)",
}


# ── Finger counting ───────────────────────────────────────────────────────────

def count_extended_fingers(hand_landmarks, handedness_label: str) -> int:
    """
    Count how many fingers are extended using MediaPipe landmark indices.
    Thumb uses a left/right-aware comparison; other fingers compare tip to PIP.
    """
    lm = hand_landmarks.landmark

    # Landmark indices
    THUMB_TIP, THUMB_IP   = 4,  3
    INDEX_TIP, INDEX_PIP  = 8,  6
    MIDDLE_TIP,MIDDLE_PIP = 12, 10
    RING_TIP,  RING_PIP   = 16, 14
    PINKY_TIP, PINKY_PIP  = 20, 18

    count = 0

    # Thumb: compare x coordinates (mirror for left hand)
    if handedness_label == "Right":
        if lm[THUMB_TIP].x < lm[THUMB_IP].x:
            count += 1
    else:
        if lm[THUMB_TIP].x > lm[THUMB_IP].x:
            count += 1

    # Other four fingers: tip y < pip y → extended (image coords, y grows down)
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP),   (PINKY_TIP, PINKY_PIP)]:
        if lm[tip].y < lm[pip].y:
            count += 1

    return count


# ── Camera loop ───────────────────────────────────────────────────────────────

def run(player_id: int, cam_index: int, show_window: bool) -> None:
    if not CV2_OK or not MP_OK:
        print("[camera] Required packages missing — see messages above.")
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    prefix = str(player_id).encode()

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[camera] Cannot open camera index {cam_index}")
        sys.exit(1)

    print(f"[camera] Player {player_id} — camera {cam_index} — sending to "
          f"{GAME_HOST}:{GAME_PORT}")
    print("[camera] Press Q in the camera window to quit.")

    stable_gesture: bytes | None = None
    stable_count:   int          = 0
    last_sent_time: float        = 0.0
    last_sent_gesture: bytes | None = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # Flip for mirror view, convert to RGB for MediaPipe
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected_gesture: bytes | None = None

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_lm    = result.multi_hand_landmarks[0]
                handedness = result.multi_handedness[0].classification[0].label

                if show_window:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                n_fingers = count_extended_fingers(hand_lm, handedness)
                detected_gesture = FINGER_TO_ACTION.get(n_fingers)

            # Stability filter: only register after STABLE_FRAMES_REQUIRED
            # consecutive frames showing the same gesture
            if detected_gesture == stable_gesture and detected_gesture is not None:
                stable_count += 1
            else:
                stable_gesture = detected_gesture
                stable_count   = 0

            now = time.time()
            if (stable_count >= STABLE_FRAMES_REQUIRED
                    and detected_gesture is not None
                    and (detected_gesture != last_sent_gesture
                         or now - last_sent_time >= COOLDOWN_SECONDS)):

                msg = prefix + detected_gesture
                sock.sendto(msg, (GAME_HOST, GAME_PORT))
                last_sent_gesture = detected_gesture
                last_sent_time    = now
                stable_count      = 0   # reset so next send needs re-stability

                name = ACTION_NAMES.get(detected_gesture, "?")
                print(f"[camera] P{player_id} → {name}")

            # ── Overlay ───────────────────────────────────────────────────────
            if show_window:
                label = (ACTION_NAMES.get(stable_gesture, "—")
                         if stable_gesture else "Show your hand")
                cv2.putText(frame, f"P{player_id}: {label}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 180), 2, cv2.LINE_AA)
                bar_pct = min(stable_count / STABLE_FRAMES_REQUIRED, 1.0)
                cv2.rectangle(frame, (10, 55),
                              (10 + int(300 * bar_pct), 70), (0, 255, 100), -1)
                cv2.rectangle(frame, (10, 55), (310, 70), (200, 200, 200), 2)

                cv2.imshow(f"DUEL.EXE — Player {player_id} Cam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    sock.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Camera gesture predictor for DUEL.EXE")
    p.add_argument("--player", type=int, choices=[1, 2], default=1,
                   help="Which player this camera controls (default: 1)")
    p.add_argument("--cam", type=int, default=0,
                   help="OpenCV camera device index (default: 0)")
    p.add_argument("--no-window", action="store_true",
                   help="Suppress the camera preview window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        player_id=args.player,
        cam_index=args.cam,
        show_window=not args.no_window,
    )
