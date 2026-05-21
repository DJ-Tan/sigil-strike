"""
conftest.py — shared pytest setup for all test-cases.

Run from the project root or from code/:
    python -m pytest code/test-cases/

Requires: pip install pytest
"""
import os
import sys
import pathlib

# Use headless SDL so pygame can initialize without a real display or audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Add code/ to sys.path so game modules are importable from test-cases/.
CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pygame
pygame.init()
