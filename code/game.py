"""
game.py
───────
Core game loop.  Owns the pygame window, the two Players, and the
timing / resolution logic.  Delegates all drawing to Renderer and all
visual effects to the effects subsystems.

To integrate a real camera / OpenPose model, see camera.py.
The only change needed here is to call:

    self._register_action(player, action)   # action is a moves.Action value

from your camera callback instead of (or in addition to) the keyboard path.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable, Optional

import numpy as np
import pygame

from constants import (
    WIDTH, HEIGHT, FPS, TITLE,
    P1_COLOR, P2_COLOR,
    GREEN, RED, GOLD, GRAY, WHITE,
    SHAPE_COLORS, P1_KEYS, P2_KEYS,
)
from configs.time_config import (
    RESOLVE_INTERVAL, COMBO_DISPLAY_TIME, EVENT_MESSAGE_DURATION,
    ACTION_COOLDOWN,
    DEATHMATCH_START_SEC, DEATHMATCH_HP_PER_SEC,
)
from configs.move_config import DEATHMATCH_HEAL_MULT
import audio
from moves import Action, ATTACK_MOVES, Move, MOVE_COLORS, RoundContext, resolve_moves
from player import Player
from effects import ParticleSystem, FloatingTextSystem, ScreenEffect
from renderer import Renderer, draw_text, draw_rounded_rect
from paths import external_dir, resource_dir

try:
    import mediapipe as mp
    from mediapipe import tasks
    import joblib
    _ML_OK = True
except ImportError:
    _ML_OK = False


class Game:
    def __init__(
        self,
        screen: Optional[pygame.Surface] = None,
        p1_name: str = "PLAYER 1", p1_color: tuple = P1_COLOR,
        p2_name: str = "PLAYER 2", p2_color: tuple = P2_COLOR,
        tournament_mode: bool = False,
        on_match_start: Optional[Callable[[], None]] = None,
    ):
        if screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption(TITLE)
            self._owns_display = True
        else:
            self.screen = screen
            self._owns_display = False
        self._fullscreen = False
        self.clock  = pygame.time.Clock()

        # Pre-allocated logical rendering surfaces (never recreated each frame)
        self._base    = pygame.Surface((WIDTH, HEIGHT))
        self._logical = pygame.Surface((WIDTH, HEIGHT))

        self._p1_name  = p1_name;  self._p1_color = p1_color
        self._p2_name  = p2_name;  self._p2_color = p2_color
        self._tournament_mode = tournament_mode
        self._on_match_start  = on_match_start

        self.renderer  = Renderer()
        self.particles = ParticleSystem()
        self.floats    = FloatingTextSystem()
        self.fx        = ScreenEffect()

        audio.init()
        self._init_hand_detector()
        self._init_round()
        self._waiting = True
        self._start_btn_rect: Optional[pygame.Rect] = None

    # ── Hand-sign detection (in-process, uses renderer's camera frames) ───

    _LABEL_TO_ACTION = {
        "move1": Action.MOVE1,
        "move2": Action.MOVE2,
        "move3": Action.MOVE3,
        "move4": Action.MOVE4,
        "move5": Action.MOVE5,
    }

    @staticmethod
    def _load_env() -> None:
        # Optional .env next to the .exe (or in the project root in dev mode).
        env_path = external_dir() / ".env"
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
        except FileNotFoundError:
            pass

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        val = os.environ.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        val = os.environ.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    _MOVE_NAMES = ["move1", "move2", "move3", "move4", "move5"]
    _TEAMS_DIR  = str(external_dir() / "Teams")

    @staticmethod
    def _parse_env_file(path: str) -> dict[str, str]:
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

    def _find_team_dir(self, team_name: str) -> Optional[str]:
        for i in range(1, 7):
            team_dir = os.path.join(self._TEAMS_DIR, f"Team{i}")
            env = self._parse_env_file(os.path.join(team_dir, "team.env"))
            if env.get("NAME", "").upper() == team_name.upper():
                return team_dir
        return None

    def _find_team_num(self, team_name: str) -> Optional[int]:
        for i in range(1, 7):
            team_dir = os.path.join(self._TEAMS_DIR, f"Team{i}")
            env = self._parse_env_file(os.path.join(team_dir, "team.env"))
            if env.get("NAME", "").upper() == team_name.upper():
                return i
        return None

    def _init_hand_detector(self) -> None:
        self._load_env()
        self._stable_frames = self._env_int("STABLE_FRAMES", 4)
        self._action_cooldown = self._env_float("ACTION_COOLDOWN", ACTION_COOLDOWN)
        self._combo_display_time = self._env_float("COMBO_DISPLAY_TIME", COMBO_DISPLAY_TIME)
        self._confidence_threshold = self._env_float("CONFIDENCE_THRESHOLD", 0.6)

        # Load per-team thresholds from Teams/<TeamN>/team.env
        self._thresholds: list[dict[str, float]] = [{}, {}]
        for pi, name in enumerate([self._p1_name, self._p2_name]):
            team_dir = self._find_team_dir(name)
            if team_dir is not None:
                env = self._parse_env_file(os.path.join(team_dir, "team.env"))
                for move in self._MOVE_NAMES:
                    raw = env.get(f"THRESHOLD_{move.upper()}")
                    try:
                        self._thresholds[pi][move] = float(raw) if raw else self._confidence_threshold
                    except ValueError:
                        self._thresholds[pi][move] = self._confidence_threshold
            else:
                for move in self._MOVE_NAMES:
                    self._thresholds[pi][move] = self._confidence_threshold

        # Per-player detection state. `_hand_modes[pi]` is 'landmark', 'cnn', or None.
        # The model objects in `_hand_models` are heterogeneous (sklearn classifier
        # vs torch nn.Module) and selected via the matching mode.
        self._hand_detector = None
        self._hand_modes:    list = [None, None]
        self._hand_models:   list = [None, None]
        self._hand_encoders: list = [None, None]   # landmark mode only
        self._hand_cnn_meta: list = [None, None]   # cnn mode only — class_names, img_size
        self._hand_stable:   list = [None, None]
        self._hand_stable_count: list = [0, 0]
        self._hand_last_sent:    list = [None, None]
        self._hand_last_time:    list = [0.0, 0.0]
        self._torch = None
        self._torch_F = None
        self._torch_device = None

        print(f"[game] Hand-sign config: stable_frames={self._stable_frames}, "
              f"cooldown={self._action_cooldown}s, "
              f"display={self._combo_display_time}s, "
              f"global_threshold={self._confidence_threshold}")
        for pi in range(2):
            print(f"[game] P{pi + 1} thresholds: {self._thresholds[pi]}")

        # Pre-pass: read each player's MODEL_TYPE so we know which runtimes
        # (mediapipe vs torch) to bring up below.
        for pi, name in enumerate([self._p1_name, self._p2_name]):
            team_dir = self._find_team_dir(name)
            if team_dir is None:
                continue
            env = self._parse_env_file(os.path.join(team_dir, "team.env"))
            mode = (env.get("MODEL_TYPE", "landmark") or "landmark").strip().lower()
            if mode not in ("landmark", "cnn"):
                print(f"[game] P{pi + 1} ({name}) unknown MODEL_TYPE={mode!r}; defaulting to landmark.")
                mode = "landmark"
            self._hand_modes[pi] = mode

        needs_landmark = "landmark" in self._hand_modes
        needs_cnn      = "cnn"      in self._hand_modes

        # ── Landmark runtime (MediaPipe) ────────────────────────────────────
        landmarker_path = ""
        if needs_landmark:
            if not _ML_OK:
                print("[game] mediapipe/joblib not installed — landmark detection disabled")
            else:
                # Frozen build bundles the task file at <MEIPASS>/model/landmark/;
                # source mode has it at code/model/landmark/. Try both.
                candidates = [
                    resource_dir() / "model" / "landmark" / "hand_landmarker.task",
                    resource_dir() / "code"  / "model" / "landmark" / "hand_landmarker.task",
                ]
                for cand in candidates:
                    if cand.exists():
                        landmarker_path = str(cand)
                        break
                if not landmarker_path:
                    print(f"[game] hand_landmarker.task not found in {candidates} "
                          "— landmark detection disabled")

        # ── CNN runtime (PyTorch) ──────────────────────────────────────────
        if needs_cnn:
            try:
                import torch
                import torch.nn.functional as F  # noqa: N812
                self._torch = torch
                self._torch_F = F
                self._torch_device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
                print(f"[game] CNN device: {self._torch_device}")
            except ImportError:
                print("[game] PyTorch not installed — CNN detection disabled")

        # ── Per-player model loading ───────────────────────────────────────
        for pi, name in enumerate([self._p1_name, self._p2_name]):
            mode = self._hand_modes[pi]
            if mode is None:
                print(f"[game] P{pi + 1} ({name}) no team match — detection disabled")
                continue

            team_num = self._find_team_num(name)
            team_root   = os.path.join(self._TEAMS_DIR, f"Team{team_num}")
            team_models = os.path.join(team_root, "models")

            if mode == "landmark":
                if not landmarker_path:
                    continue
                clf_path = os.path.join(team_models, "hand_sign_classifier.pkl")
                enc_path = os.path.join(team_models, "label_encoder.pkl")
                if os.path.exists(clf_path) and os.path.exists(enc_path):
                    self._hand_models[pi]   = joblib.load(clf_path)
                    self._hand_encoders[pi] = joblib.load(enc_path)
                    print(f"[game] P{pi + 1} ({name}) landmark model loaded — "
                          f"classes: {list(self._hand_encoders[pi].classes_)}")
                else:
                    print(f"[game] P{pi + 1} ({name}) landmark model files not found "
                          "— detection disabled")
            else:  # mode == 'cnn'
                if self._torch is None:
                    continue
                ckpt_path = os.path.join(team_models, "hand_sign_cnn.pth")
                if not os.path.exists(ckpt_path):
                    print(f"[game] P{pi + 1} ({name}) CNN checkpoint not found "
                          f"({ckpt_path}) — detection disabled")
                    continue
                try:
                    model, meta = self._load_cnn_model(ckpt_path, team_root)
                except Exception as e:
                    print(f"[game] P{pi + 1} ({name}) failed to load CNN: {e}")
                    continue
                self._hand_models[pi]   = model
                self._hand_cnn_meta[pi] = meta
                print(f"[game] P{pi + 1} ({name}) CNN model loaded — "
                      f"classes: {meta['class_names']}")

        if not any(m is not None for m in self._hand_models):
            print("[game] No models loaded — hand detection disabled")
            return

        # The HandLandmarker is shared by both players' landmark-mode detection.
        # Skip creating it if no player uses landmark mode (saves ~100 MB RAM).
        if landmarker_path and any(m == "landmark" for m, hm in zip(self._hand_modes, self._hand_models) if hm is not None):
            opts = tasks.vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=landmarker_path),
                num_hands=4,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self._hand_detector = tasks.vision.HandLandmarker.create_from_options(opts)

    # ImageNet normalization constants — must match CNN training preprocessing.
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _detect_hand_signs(self) -> None:
        # Bail early only if neither runtime is available.
        if self._hand_detector is None and self._torch is None:
            return

        for i, player in enumerate([self.p1, self.p2]):
            if player.combo_locked:
                continue
            if self._hand_models[i] is None:
                continue

            frame = self.renderer._cam_rgb_frames[i]
            if frame is None:
                continue

            mode = self._hand_modes[i]
            predicted: Optional[str] = None

            if mode == "landmark":
                predicted = self._predict_landmark(i, frame)
            elif mode == "cnn":
                predicted = self._predict_cnn(i, frame)

            if predicted == self._hand_stable[i] and predicted is not None:
                self._hand_stable_count[i] += 1
            else:
                self._hand_stable[i] = predicted
                self._hand_stable_count[i] = 0

            now = time.time()
            if (
                self._hand_stable_count[i] >= self._stable_frames
                and self._hand_stable[i] is not None
                and self._hand_stable[i] in self._LABEL_TO_ACTION
                and (self._hand_stable[i] != self._hand_last_sent[i]
                     or now - self._hand_last_time[i] >= self._action_cooldown)
            ):
                action = self._LABEL_TO_ACTION[self._hand_stable[i]]
                self._register_action(player, action)
                print(f"[game] P{player.pid} hand-sign -> {self._hand_stable[i].upper()}")
                self._hand_last_sent[i] = self._hand_stable[i]
                self._hand_last_time[i] = now
                self._hand_stable_count[i] = 0

    def _predict_landmark(self, i: int, frame_rgb: np.ndarray) -> Optional[str]:
        if self._hand_detector is None:
            return None
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._hand_detector.detect(mp_img)
        if not result.hand_landmarks or len(result.hand_landmarks) < 2:
            return None
        top2 = sorted(result.hand_landmarks,
                      key=lambda lms: self._bbox_area(lms), reverse=True)[:2]
        top2.sort(key=lambda lms: lms[0].x)
        features = np.concatenate([
            self._normalize_hand(top2[0]),
            self._normalize_hand(top2[1]),
        ]).reshape(1, -1)
        encoder = self._hand_encoders[i]
        probs = self._hand_models[i].predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        label = str(encoder.classes_[best_idx])
        threshold = self._thresholds[i].get(label, self._confidence_threshold)
        return label if probs[best_idx] >= threshold else None

    def _predict_cnn(self, i: int, frame_rgb: np.ndarray) -> Optional[str]:
        torch = self._torch
        if torch is None:
            return None
        meta = self._hand_cnn_meta[i]
        tensor = self._preprocess_cnn(frame_rgb, meta["img_size"])
        with torch.no_grad():
            logits = self._hand_models[i](
                torch.from_numpy(tensor).to(self._torch_device))
            probs = self._torch_F.softmax(logits, dim=1).cpu().numpy()[0]
        best_idx = int(np.argmax(probs))
        label = str(meta["class_names"][best_idx])
        threshold = self._thresholds[i].get(label, self._confidence_threshold)
        return label if probs[best_idx] >= threshold else None

    @staticmethod
    def _preprocess_cnn(frame_rgb: np.ndarray, img_size: tuple) -> np.ndarray:
        """RGB uint8 frame → normalized float32 NCHW tensor matching training."""
        import cv2  # renderer already requires cv2; safe to use here.
        img = cv2.resize(frame_rgb, tuple(img_size)).astype(np.float32) / 255.0
        img = (img - Game._IMAGENET_MEAN) / Game._IMAGENET_STD
        img = img.transpose(2, 0, 1)              # HWC → CHW
        return np.expand_dims(img, 0)             # add batch dim → NCHW

    def _load_cnn_model(self, ckpt_path: str, team_dir: str):
        """Load a CNN checkpoint, returning (model, meta) where
        meta = {'class_names': [...], 'img_size': (H, W)}.
        Architecture is rebuilt from the team's `model_arch.py` if present,
        else from the default MobileNetV2 + small head (must match training)."""
        torch  = self._torch
        device = self._torch_device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        class_names = list(ckpt["class_names"])
        img_size    = tuple(ckpt.get("img_size", (224, 224)))
        model = self._build_cnn_arch(num_classes=len(class_names), team_dir=team_dir)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        # Warm up so the first real frame doesn't stall during a round.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *img_size, device=device)
            _ = model(dummy)
        return model, {"class_names": class_names, "img_size": img_size}

    def _build_cnn_arch(self, num_classes: int, team_dir: str):
        # Per-team override: Teams/Team<N>/model_arch.py with build_model().
        arch_path = os.path.join(team_dir, "model_arch.py")
        if os.path.exists(arch_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"team_arch_{os.path.basename(team_dir)}", arch_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                build = getattr(mod, "build_model", None)
                if callable(build):
                    print(f"[game] Using team's custom architecture: {arch_path}")
                    return build(num_classes=num_classes)
            except Exception as e:
                print(f"[game] Failed to import {arch_path}: {e} — using default arch.")

        # Default architecture — must match code/model/cnn/train_model.py defaults.
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

    @staticmethod
    def _bbox_area(landmarks: list) -> float:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    @staticmethod
    def _normalize_hand(landmarks: list) -> np.ndarray:
        raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        wrist = raw[0]
        centered = raw - wrist
        scale = np.linalg.norm(centered[9])
        if scale > 1e-6:
            centered /= scale
        return centered.flatten()

    # ── Fullscreen toggle ─────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    # ── Round / restart ───────────────────────────────────────────────────────

    def _init_round(self) -> None:
        self.p1 = Player(1, self._p1_name, self._p1_color)
        self.p2 = Player(2, self._p2_name, self._p2_color)

        self.resolve_timer: float = RESOLVE_INTERVAL
        self.elapsed:       float = 0.0
        self._decay_accumulator: float = 0.0

        self.event_msg:       str   = ""
        self.event_msg_color: tuple = GRAY
        self.event_msg_life:  float = 0.0

        self.game_over:   bool             = False
        self.winner:      Optional[Player] = None
        self.tiebreaker:  bool             = False

        self.particles.clear()
        self.floats.clear()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            if not self.game_over and not self._waiting:
                self._update(dt)
            self._draw(dt)

    def _begin_match(self) -> None:
        """Dismiss the READY overlay and fire the on_match_start callback once."""
        if not self._waiting:
            return
        self._waiting = False
        if self._on_match_start is not None:
            cb = self._on_match_start
            self._on_match_start = None   # fire at most once per Game instance
            cb()

    # ── Tournament-mode single-game run ──────────────────────────────────────

    def run_once(self) -> tuple[Optional[int], float]:
        """
        Run one game to completion for tournament use.
        Returns (winner_pid, elapsed_seconds), or (None, 0.0) if abandoned.
        The result screen is shown until any key is pressed or 3 s elapse.
        """
        elapsed      = 0.0
        result_timer = 0.0
        SHOW_FOR     = 3.0

        while True:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, 0.0
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                        continue
                    if self._waiting:
                        if event.key == pygame.K_RETURN:
                            self._begin_match()
                        elif event.key == pygame.K_ESCAPE:
                            return None, 0.0
                        continue
                    if self.game_over:
                        return self.winner.pid if self.winner else None, elapsed
                    if event.key == pygame.K_ESCAPE:
                        return None, 0.0
                    if event.key in P1_KEYS:
                        self._register_action(self.p1, Action(P1_KEYS[event.key]))
                    if event.key in P2_KEYS:
                        self._register_action(self.p2, Action(P2_KEYS[event.key]))
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._waiting and self._start_btn_rect is not None:
                        lpos = self._to_logical(event.pos)
                        if self._start_btn_rect.collidepoint(lpos):
                            self._begin_match()

            if not self.game_over and not self._waiting:
                self._update(dt)
                elapsed += dt
            elif self.game_over:
                result_timer += dt
                if result_timer >= SHOW_FOR:
                    return self.winner.pid if self.winner else None, elapsed

            self._draw(dt)

    # ── Event handling ────────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.renderer.close(); pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                    continue

                if event.key == pygame.K_ESCAPE:
                    self.renderer.close(); pygame.quit(); sys.exit()

                if self._waiting:
                    if event.key == pygame.K_RETURN:
                        self._begin_match()
                    continue

                if self.game_over:
                    if event.key == pygame.K_RETURN:
                        self._init_round()
                        self._waiting = True
                    return

                # ── Keyboard test-mode input ──────────────────────────────
                if event.key in P1_KEYS:
                    action = Action(P1_KEYS[event.key])
                    self._register_action(self.p1, action)

                if event.key in P2_KEYS:
                    action = Action(P2_KEYS[event.key])
                    self._register_action(self.p2, action)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._waiting and self._start_btn_rect is not None:
                    lpos = self._to_logical(event.pos)
                    if self._start_btn_rect.collidepoint(lpos):
                        self._begin_match()

    # ── Action registration (used by keyboard AND camera pipeline) ────────────

    def _register_action(self, player: Player, action: Action) -> None:
        """
        Feed one action into a player's combo buffer.
        Emits particles, and if a complete Move is formed, shows a floating
        label and emits a larger burst.

        This is the single entry-point for both keyboard input (above) and
        camera-driven input (see camera.py).
        """
        cam_cx = 165 if player.pid == 1 else WIDTH - 165
        cam_cy = 370   # approximate arena centre for the player side

        self.particles.emit(cam_cx, cam_cy - 30,
                            SHAPE_COLORS[action.value], n=10, speed=3)

        move = player.add_action(action, display_time=self._combo_display_time)

        if move is not None:
            player.moves_hit += 1
            color = MOVE_COLORS[move]
            self.floats.add(move.value, cam_cx, cam_cy - 60, color, font_size=22)
            self.particles.emit(cam_cx, cam_cy - 30, color, n=30, speed=5)
        elif move is None and len(player.action_buffer) == 0:
            # Buffer was just cleared with no match → miss
            player.moves_miss += 1
            self.floats.add("✗", cam_cx, cam_cy - 50, (160, 60, 60), font_size=24)

    # ── Update ────────────────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        self._detect_hand_signs()
        self.elapsed += dt
        self.p1.update(dt)
        self.p2.update(dt)
        self.particles.update(dt)
        self.fx.update(dt)
        self.floats.update(dt)

        if self.event_msg_life > 0:
            self.event_msg_life -= dt

        self._apply_deathmatch_decay(dt)

        self.resolve_timer -= dt
        if self.resolve_timer <= 0:
            self.resolve_timer = RESOLVE_INTERVAL
            self._resolve_round()

    def _resolve_round(self) -> None:
        m1 = self.p1.pop_move()
        m2 = self.p2.pop_move()

        if m1 is None and m2 is None:
            self._set_event("No moves queued — round skipped!", GRAY)
            return

        ctx = RoundContext(
            heal_multiplier=DEATHMATCH_HEAL_MULT if self._deathmatch_active else 1.0,
        )
        p1d, p2d, dmg_by_p1, dmg_by_p2, events = resolve_moves(m1, m2, ctx)
        self.p1.apply_delta(p1d)
        self.p2.apply_delta(p2d)

        self._emit_resolution_vfx(p1d, p2d)
        self._emit_event_text(events)
        self._emit_sfx(m1, m2, events)

        n1 = m1.value if m1 else "IDLE"
        n2 = m2.value if m2 else "IDLE"
        self._set_event(f"P1: {n1}   vs   P2: {n2}", GOLD)

        self._check_game_over()

    @property
    def _deathmatch_active(self) -> bool:
        return self.elapsed >= DEATHMATCH_START_SEC

    def _apply_deathmatch_decay(self, dt: float) -> None:
        if not self._deathmatch_active:
            return
        self._decay_accumulator += dt * DEATHMATCH_HP_PER_SEC
        whole = int(self._decay_accumulator)
        if whole > 0:
            self._decay_accumulator -= whole
            self.p1.apply_delta(-whole)
            self.p2.apply_delta(-whole)
            self._check_game_over()

    def _check_game_over(self) -> None:
        if self.game_over:
            return
        if not (self.p1.is_defeated or self.p2.is_defeated):
            return
        self.game_over = True
        if self.p1.is_defeated and self.p2.is_defeated:
            # Both KO simultaneously — tiebreaker on move score
            p1_score = self.p1.moves_hit - self.p1.moves_miss
            p2_score = self.p2.moves_hit - self.p2.moves_miss
            if p1_score > p2_score:
                self.winner     = self.p1
                self.tiebreaker = True
            elif p2_score > p1_score:
                self.winner     = self.p2
                self.tiebreaker = True
            else:
                self.winner = None   # true draw even after tiebreaker
        elif self.p1.is_defeated:
            self.winner = self.p2
        else:
            self.winner = self.p1
        self.fx.flash(WHITE, 1.0)
        self.fx.shake(14, 0.8)

    def _emit_resolution_vfx(self, p1d: int, p2d: int) -> None:
        cx1, cx2 = WIDTH // 4, 3 * WIDTH // 4
        arena_y  = 430

        def burst(cx, delta):
            if delta < 0:
                self.particles.emit(cx, arena_y, RED,   n=40, speed=6)
                self.fx.flash(RED, 0.22)
                self.fx.shake(6,   0.30)
                self.floats.add(f"−{abs(delta)} HP", cx, arena_y - 60, RED, 32)
            elif delta > 0:
                self.particles.emit(cx, arena_y, GREEN, n=30, speed=4)
                self.floats.add(f"+{delta} HP",   cx, arena_y - 60, GREEN, 32)

        burst(cx1, p1d)
        burst(cx2, p2d)

    def _emit_event_text(self, events: list[tuple[str, int]]) -> None:
        cx1, cx2 = WIDTH // 4, 3 * WIDTH // 4
        arena_y  = 430
        labels = {
            "dodge_success": ("DODGED!",   GREEN),
            "dodge_fail":    ("HIT!",      RED),
            "reflect":       ("REFLECTED!", GOLD),
        }
        for name, defender in events:
            if name not in labels:
                continue
            text, color = labels[name]
            cx = cx1 if defender == 1 else cx2
            self.floats.add(text, cx, arena_y - 100, color, 28)

    def _emit_sfx(self, m1, m2, events: list[tuple[str, int]]) -> None:
        """One SFX per player per round, chosen from their move + resolve event.

        Defensive moves (Shield Wall / Dodge Roll) pick between two SFX based
        on the event the resolver produced — see moves._apply_defense for the
        full table.  Idle moves are silent.
        """
        event_set = set(events)
        for own, other, pid in ((m1, m2, 1), (m2, m1, 2)):
            if own == Move.POWER_STRIKE:
                audio.play("power_strike")
            elif own == Move.COMBO_BLAST:
                audio.play("combo_blast")
            elif own == Move.SHIELD_WALL:
                if ("reflect", pid) in event_set:
                    audio.play("shield_reflect")
                else:
                    audio.play("shield_block")
            elif own == Move.DODGE_ROLL:
                if ("dodge_fail", pid) in event_set:
                    audio.play("dodge_fail")
                else:
                    audio.play("dodge_success")
            elif own == Move.MEND:
                # Heal is cancelled if the opponent attacked — keep audio in sync.
                if other not in ATTACK_MOVES:
                    audio.play("mend")

    def _set_event(self, msg: str, color: tuple) -> None:
        self.event_msg       = msg
        self.event_msg_color = color
        self.event_msg_life  = EVENT_MESSAGE_DURATION

    # ── Coordinate mapping ─────────────────────────────────────────────────────

    def _to_logical(self, pos: tuple) -> tuple:
        sw, sh = self.screen.get_size()
        scale  = min(sw / WIDTH, sh / HEIGHT)
        ox     = (sw - int(WIDTH  * scale)) // 2
        oy     = (sh - int(HEIGHT * scale)) // 2
        return ((pos[0] - ox) / scale, (pos[1] - oy) / scale)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw_waiting_overlay(self, surf: pygame.Surface) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, 0))

        mx, my = WIDTH // 2, HEIGHT // 2

        draw_text(surf, "READY?", self.renderer.f(64), GOLD, mx, my - 60)

        BTN_W, BTN_H = 240, 64
        btn = pygame.Rect(mx - BTN_W // 2, my + 10, BTN_W, BTN_H)
        self._start_btn_rect = btn

        lmouse = self._to_logical(pygame.mouse.get_pos())
        hover  = btn.collidepoint(lmouse)
        bg     = (30, 100, 55) if hover else (20, 65, 35)
        border = GOLD if hover else (40, 110, 55)
        draw_rounded_rect(surf, bg, btn, radius=12, border=2, border_color=border)
        draw_text(surf, "START", self.renderer.sym(28), GREEN if not hover else GOLD,
                  btn.centerx, btn.centery)

        draw_text(surf, "or press ENTER", self.renderer.f(16), GRAY, mx, my + 100)

    def _draw(self, dt: float) -> None:
        self.renderer.update(dt)

        ox, oy = self.fx.offset

        # 1. Render the scene into the base surface
        self.renderer.draw_frame(
            self._base,
            self.p1, self.p2,
            self.resolve_timer, RESOLVE_INTERVAL,
            self.event_msg, self.event_msg_color, self.event_msg_life,
            self.elapsed,
            waiting=self._waiting,
        )

        # 2. Composite everything onto the logical surface (shake offset here)
        self._logical.fill((0, 0, 0))
        self._logical.blit(self._base, (ox, oy))
        self.fx.draw_flash(self._logical)
        self.particles.draw(self._logical)
        self.floats.draw(self._logical, self.renderer.fonts)

        if self._waiting:
            self._draw_waiting_overlay(self._logical)
        elif self.game_over:
            hint = ("Any key → return to bracket" if self._tournament_mode
                    else "ENTER → play again     ESC → quit")
            self.renderer.draw_game_over(self._logical, self.winner, hint,
                                         self.tiebreaker)

        # 3. Scale the logical surface to fill the actual screen (letterboxed)
        sw, sh  = self.screen.get_size()
        scale   = min(sw / WIDTH, sh / HEIGHT)
        sw_out  = int(WIDTH  * scale)
        sh_out  = int(HEIGHT * scale)
        scaled  = pygame.transform.smoothscale(self._logical, (sw_out, sh_out))
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, ((sw - sw_out) // 2, (sh - sh_out) // 2))
        pygame.display.flip()
