"""
effects.py
──────────
Visual-only subsystems.  Nothing here touches game state.

  ParticleSystem   – burst particles that drift and fade
  FloatingTextSystem – damage / heal numbers that float upward
  ScreenEffect     – screen flash and camera shake
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame


# ── Particles ─────────────────────────────────────────────────────────────────

@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float; max_life: float
    color: tuple; size: float


class ParticleSystem:
    def __init__(self):
        self.particles: list[Particle] = []

    def emit(self, x: float, y: float, color: tuple,
             n: int = 20, speed: float = 4.0) -> None:
        for _ in range(n):
            angle = random.uniform(0, math.tau)
            spd   = random.uniform(1.0, speed)
            life  = random.uniform(0.4, 1.0)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                life=life, max_life=life,
                color=color,
                size=random.uniform(3.0, 8.0),
            ))

    def update(self, dt: float) -> None:
        for p in self.particles:
            p.x  += p.vx
            p.y  += p.vy
            p.vy += 0.12      # gravity
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]

    def draw(self, surf: pygame.Surface) -> None:
        for p in self.particles:
            t     = p.life / p.max_life
            alpha = int(255 * t)
            size  = max(1, int(p.size * t))
            s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p.color, alpha), (size, size), size)
            surf.blit(s, (int(p.x) - size, int(p.y) - size))

    def clear(self) -> None:
        self.particles.clear()


# ── Floating text ─────────────────────────────────────────────────────────────

@dataclass
class FloatingText:
    text: str
    x: float; y: float
    vy: float
    life: float; max_life: float
    color: tuple
    font_size: int = 28


class FloatingTextSystem:
    def __init__(self):
        self.texts: list[FloatingText] = []

    def add(self, text: str, x: float, y: float,
            color: tuple, font_size: int = 28) -> None:
        self.texts.append(FloatingText(
            text=text, x=x, y=y,
            vy=-2.5,
            life=1.2, max_life=1.2,
            color=color, font_size=font_size,
        ))

    def update(self, dt: float) -> None:
        for t in self.texts:
            t.y    += t.vy
            t.life -= dt
        self.texts = [t for t in self.texts if t.life > 0]

    def draw(self, surf: pygame.Surface, fonts: dict[int, pygame.font.Font]) -> None:
        for ft in self.texts:
            t     = ft.life / ft.max_life
            alpha = int(255 * min(t * 2.0, 1.0))
            fnt   = fonts.get(ft.font_size) or fonts[28]
            img   = fnt.render(ft.text, True, ft.color)
            img.set_alpha(alpha)
            surf.blit(img, (int(ft.x) - img.get_width() // 2, int(ft.y)))

    def clear(self) -> None:
        self.texts.clear()


# ── Screen effects ────────────────────────────────────────────────────────────

class ScreenEffect:
    def __init__(self):
        self.flash_color: tuple | None = None
        self.flash_life:  float        = 0.0
        self.shake_mag:   float        = 0.0
        self.shake_life:  float        = 0.0
        self.offset:      tuple[int, int] = (0, 0)

    def flash(self, color: tuple, duration: float = 0.3) -> None:
        self.flash_color = color
        self.flash_life  = duration

    def shake(self, magnitude: float = 8.0, duration: float = 0.4) -> None:
        self.shake_mag  = magnitude
        self.shake_life = duration

    def update(self, dt: float) -> None:
        if self.flash_life > 0:
            self.flash_life -= dt

        if self.shake_life > 0:
            self.shake_life -= dt
            m = self.shake_mag * max(self.shake_life / 0.4, 0.0)
            self.offset = (
                int(random.uniform(-m, m)),
                int(random.uniform(-m, m)),
            )
        else:
            self.offset = (0, 0)

    def draw_flash(self, surf: pygame.Surface) -> None:
        if self.flash_life > 0 and self.flash_color:
            alpha = int(180 * min(self.flash_life / 0.3, 1.0))
            s = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            s.fill((*self.flash_color, alpha))
            surf.blit(s, (0, 0))
