"""
webcam_test.py
──────────────
Standalone diagnostic for verifying that a USB / built-in webcam works with
OpenCV's DirectShow backend on Windows. Not part of the game runtime.

Usage:
    python webcam_test.py

Controls (inside the preview window):
    Q      Quit
    S      Save a JPEG snapshot to the current directory
    +/-    Nudge the lens focus up / down (cameras that expose CAP_PROP_FOCUS)
"""

import sys
import cv2


def find_cameras(max_index: int = 5) -> list[int]:
    """Probe device indices [0, max_index) and return those that open successfully."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def main() -> None:
    """Open the first detected camera and run a preview loop with key controls."""
    print("Scanning for cameras...")
    cameras = find_cameras()

    if not cameras:
        print("No cameras found.")
        sys.exit(1)

    print(f"Found cameras at indices: {cameras}")
    index = cameras[0]
    print(f"Opening camera {index}... Press 'q' to quit, 's' to save snapshot.")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {w}x{h} @ {fps:.1f} FPS")

    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.putText(
            frame,
            f"Camera {index}  |  {w}x{h}  |  Q=quit  S=snapshot  +/-=focus",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow("Webcam Test", frame)

        # Single waitKey per frame — handles every control. The previous version
        # called waitKey twice, which split keypress consumption between two
        # branches and made focus +/- and quit/snapshot register on alternate
        # frames.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('s'):
            filename = f"snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            snapshot_count += 1
        elif key == ord('+'):
            f = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_FOCUS, min(f + 5, 255))
            print(f"Focus: {cap.get(cv2.CAP_PROP_FOCUS)}")
        elif key == ord('-'):
            f = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_FOCUS, max(f - 5, 0))
            print(f"Focus: {cap.get(cv2.CAP_PROP_FOCUS)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
