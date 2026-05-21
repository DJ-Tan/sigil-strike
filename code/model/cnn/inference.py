"""
inference.py  (CNN model — PyTorch)
────────────────────────────────────
Real-time hand-sign classification using MobileNetV2 — no landmarks.
The full two-hand frame is fed directly to the CNN.

Usage:
    python inference.py --team 1 --player 1
    python inference.py --team 2 --player 2 --cam 1 --threshold 0.7

GPU notes:
    By default the model uses the GPU if one is available.  PyTorch's
    default memory behaviour already lets two processes share a single
    GPU — no extra setup needed.  If you still need to cap memory:
        python inference.py --team 1 --player 1 --gpu-memory-mb 1500
    Use --no-gpu to force CPU.
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
import numpy as np

SCRIPT_DIR    = pathlib.Path(__file__).parent
TEAMS_ENV_DIR = SCRIPT_DIR.parent.parent.parent / "Teams"

GAME_HOST = "127.0.0.1"
GAME_PORT = 5001

STABLE_FRAMES_REQUIRED = 8
COOLDOWN_SECONDS       = 1.2
CAM_SCAN_RANGE         = 10

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
            print(f"[inference_cnn] Found camera at index {idx}")
            return idx
        cap.release()
    return None


def _load_team_arch(team_dir: pathlib.Path):
    """Try to import Teams/TeamN/model_arch.py.  Returns its build_model
    callable if found and valid, else None.

    Teams trained with a customised architecture (different hidden dim,
    extra layers, etc.) ship a `model_arch.py` alongside their weights so
    inference can rebuild the exact same architecture before loading the
    state_dict.  Without this file we fall back to the default below.
    """
    arch_path = team_dir / "model_arch.py"
    if not arch_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"team_arch_{team_dir.name}", arch_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[inference_cnn] Failed to import {arch_path}: {e}  (using default)")
        return None
    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        print(f"[inference_cnn] {arch_path} has no build_model() — using default.")
        return None
    print(f"[inference_cnn] Using team's custom architecture: {arch_path}")
    return mod.build_model


def build_model(num_classes: int, team_dir: pathlib.Path | None = None):
    """Recreate the architecture used during training before loading weights.

    If `team_dir/model_arch.py` exists, that team's `build_model()` is used;
    otherwise we fall back to the default architecture (must match
    `train_model.py`'s default).
    """
    if team_dir is not None:
        custom = _load_team_arch(team_dir)
        if custom is not None:
            return custom(num_classes=num_classes)

    import torch.nn as nn
    from torchvision import models
    model = models.mobilenet_v2()
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


def preprocess(frame: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    """BGR uint8 frame → normalized float32 tensor in NCHW layout."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, img_size).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)              # HWC → CHW
    return np.expand_dims(img, 0)             # add batch dim → NCHW


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN inference for SIGIL STRIKE (PyTorch)")
    parser.add_argument("--team",          type=int, required=True, choices=range(1, 7))
    parser.add_argument("--player",        type=int, choices=[1, 2], default=1)
    parser.add_argument("--cam",           type=int, default=None)
    parser.add_argument("--threshold",     type=float, default=None)
    parser.add_argument("--no-window",     action="store_true")
    parser.add_argument("--no-gpu",        action="store_true",
                        help="Force CPU even if a GPU is available")
    parser.add_argument("--gpu-memory-mb", type=int, default=0,
                        help="Hard cap on GPU memory per process in MB "
                             "(0 = grow as needed). Useful if 2 players don't fit on 1 GPU.")
    args = parser.parse_args()

    team_models = TEAMS_ENV_DIR / f"Team{args.team}" / "models"
    model_path  = team_models / "hand_sign_cnn.pth"

    if not model_path.exists():
        print(f"[inference_cnn] Missing checkpoint: {model_path}")
        print("  Run train_model.py --team", args.team, "first.")
        sys.exit(1)

    team_env  = _parse_env(TEAMS_ENV_DIR / f"Team{args.team}" / "team.env")
    threshold = args.threshold
    if threshold is None:
        try:
            threshold = float(team_env.get("CONFIDENCE_THRESHOLD", "0.6"))
        except ValueError:
            threshold = 0.6

    # Configure device BEFORE building / loading the model
    sys.path.insert(0, str(SCRIPT_DIR))
    from gpu import configure_gpu
    device, device_status = configure_gpu(
        memory_limit_mb=args.gpu_memory_mb or None,
        prefer_gpu=not args.no_gpu,
    )
    print(f"[inference_cnn] Device: {device_status}")

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("[inference_cnn] PyTorch not found.  pip install torch torchvision")
        sys.exit(1)

    print("[inference_cnn] Loading model ...")
    checkpoint   = torch.load(str(model_path), map_location=device, weights_only=False)
    class_names  = checkpoint["class_names"]
    img_size     = tuple(checkpoint.get("img_size", (224, 224)))

    team_dir = TEAMS_ENV_DIR / f"Team{args.team}"
    model = build_model(num_classes=len(class_names), team_dir=team_dir)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    print(f"[inference_cnn] Classes: {class_names}")

    # Warm up the model so the first real frame doesn't stall
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *img_size, device=device)
        _ = model(dummy)

    cam_index = args.cam if args.cam is not None else find_camera()
    if cam_index is None:
        print("[inference_cnn] No camera found. Use --cam <index>.")
        sys.exit(1)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[inference_cnn] Cannot open camera {cam_index}")
        sys.exit(1)

    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    prefix = str(args.player).encode()

    print(f"[inference_cnn] Player {args.player} | Team {args.team} | cam {cam_index} | "
          f"threshold {threshold:.0%} | → {GAME_HOST}:{GAME_PORT}")

    stable_label: str | None  = None
    stable_count: int         = 0
    last_sent_time: float     = 0.0
    last_sent_label: str | None = None
    frame_num     = 0
    last_log_time = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("[inference_cnn] Camera read failed — exiting.")
            break

        frame_num += 1
        frame      = cv2.flip(frame, 1)

        with torch.no_grad():
            tensor = torch.from_numpy(preprocess(frame, img_size)).to(device)
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        best_idx   = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        predicted_label: str | None = (
            class_names[best_idx] if confidence >= threshold else None
        )

        now = time.time()
        if now - last_log_time >= 1.0:
            print(f"[inference_cnn] frame={frame_num}  "
                  f"pred={class_names[best_idx]}  conf={confidence:.0%}  "
                  f"stable={stable_count}/{STABLE_FRAMES_REQUIRED}")
            last_log_time = now

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
            print(f"[inference_cnn] >>> SENT P{args.player} → {ACTION_DISPLAY[stable_label]} ({confidence:.0%})")

        if not args.no_window:
            if predicted_label:
                label_text = f"{ACTION_DISPLAY[predicted_label]} ({confidence:.0%})"
                color = (0, 255, 100)
            else:
                label_text = f"{class_names[best_idx]}? ({confidence:.0%}) — below threshold"
                color = (0, 140, 255)

            cv2.putText(frame, f"P{args.player} [CNN]: {label_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bar_pct = min(stable_count / STABLE_FRAMES_REQUIRED, 1.0)
            cv2.rectangle(frame, (10, 45), (10 + int(300 * bar_pct), 60), (0, 255, 100), -1)
            cv2.rectangle(frame, (10, 45), (310, 60), (200, 200, 200), 2)
            cv2.imshow(f"SIGIL STRIKE — P{args.player} CNN", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
