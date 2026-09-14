"""
AETHER — VFX Effect Classes
Reusable VFX building blocks: shockwaves, lightning arcs, energy orbs,
orbit rings, vortex fields, etc.  All render directly to a pygame.Surface.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import pygame

from core.particles import ParticlePool, lerp_color


# ─────────────────────────────────────────────────────────────────────────────
# Shockwave
# ─────────────────────────────────────────────────────────────────────────────

class Shockwave:
    """Expanding ring shockwave with fade-out."""

    def __init__(self, x: float, y: float, max_radius: float = 200.0,
                 duration: float = 0.6,
                 color: Tuple[int,int,int] = (255, 255, 255),
                 width: int = 3, glow_rings: int = 2):
        self.x = x
        self.y = y
        self.max_radius = max_radius
        self.duration = duration
        self.life = duration
        self.color = color
        self.width = width
        self.glow_rings = glow_rings
        self.alive = True

    def update(self, dt: float):
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False

    @property
    def t(self) -> float:
        """Progress 0→1 over lifetime."""
        return 1.0 - max(0.0, self.life / self.duration)

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        t = self.t
        radius = int(self.max_radius * t)
        alpha_f = 1.0 - t
        ix, iy = int(self.x), int(self.y)

        for ring in range(self.glow_rings, -1, -1):
            r_offset = ring * 4
            r = radius - r_offset
            if r <= 0:
                continue
            ring_alpha = alpha_f * (1.0 - ring * 0.25)
            if ring_alpha < 0.01:
                continue
            dim = int(ring_alpha * 200)
            col = (
                min(255, self.color[0] * dim // 200),
                min(255, self.color[1] * dim // 200),
                min(255, self.color[2] * dim // 200),
            )
            w = self.width + ring
            try:
                pygame.draw.circle(surface, col, (ix, iy), r, w)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Arc
# ─────────────────────────────────────────────────────────────────────────────

class LightningArc:
    """
    Branching lightning bolt between two points, regenerated each frame.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 color: Tuple[int,int,int] = (180, 180, 255),
                 duration: float = 0.18, branches: int = 3,
                 jaggedness: float = 25.0, width: int = 2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.color = color
        self.duration = duration
        self.life = duration
        self.branches = branches
        self.jaggedness = jaggedness
        self.width = width
        self.alive = True
        self._segments: List[List[Tuple[float, float]]] = []
        self._rebuild()

    def _build_zigzag(self, x1: float, y1: float, x2: float, y2: float,
                      steps: int = 8, jag: float = 25.0) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = [(x1, y1)]
        dx = x2 - x1
        dy = y2 - y1
        for i in range(1, steps):
            t = i / steps
            px = x1 + dx * t + (random.random() - 0.5) * jag
            py = y1 + dy * t + (random.random() - 0.5) * jag
            pts.append((px, py))
        pts.append((x2, y2))
        return pts

    def _rebuild(self):
        self._segments = [self._build_zigzag(self.x1, self.y1, self.x2, self.y2, jag=self.jaggedness)]
        # Add branches
        for _ in range(self.branches):
            main = self._segments[0]
            if len(main) < 3:
                break
            branch_start_idx = random.randint(1, len(main) - 2)
            bx, by = main[branch_start_idx]
            ex = bx + (random.random() - 0.5) * 80
            ey = by + (random.random() - 0.5) * 80
            self._segments.append(self._build_zigzag(bx, by, ex, ey, steps=5, jag=self.jaggedness * 0.6))

    def update(self, dt: float):
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False
            return
        # Regenerate each frame for flickering effect
        self._rebuild()

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        alpha_f = self.life / self.duration
        for si, seg in enumerate(self._segments):
            w = self.width if si == 0 else max(1, self.width - 1)
            bright = int(alpha_f * 255)
            col = (
                min(255, self.color[0] * bright // 255 + 30),
                min(255, self.color[1] * bright // 255 + 30),
                min(255, self.color[2] * bright // 255),
            )
            for i in range(len(seg) - 1):
                try:
                    pygame.draw.line(surface, col,
                                     (int(seg[i][0]), int(seg[i][1])),
                                     (int(seg[i+1][0]), int(seg[i+1][1])),
                                     w)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Energy Orb
# ─────────────────────────────────────────────────────────────────────────────

class EnergyOrb:
    """
    Glowing orb with pulsing core, inner glow, and outer halo.
    Rendered with layered circles for a bloom-like effect.
    """

    def __init__(self, x: float, y: float,
                 radius: float = 30.0,
                 color: Tuple[int,int,int] = (100, 200, 255),
                 pulse_speed: float = 3.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.pulse_speed = pulse_speed
        self._phase = random.uniform(0, math.tau)
        self.alive = True
        self.opacity = 1.0

    def update(self, dt: float):
        self._phase += self.pulse_speed * dt

    def draw(self, surface: pygame.Surface):
        if not self.alive or self.opacity < 0.02:
            return

        pulse = 0.85 + 0.15 * math.sin(self._phase)
        r = max(2, int(self.radius * pulse))
        ix, iy = int(self.x), int(self.y)
        alpha_scale = self.opacity

        # Outer glow halos
        for layer, (radius_mul, alpha_mul) in enumerate([(3.5, 0.07), (2.5, 0.12), (1.8, 0.20), (1.3, 0.35)]):
            lr = max(1, int(r * radius_mul))
            la = int(alpha_mul * alpha_scale * 255)
            col = (
                min(255, int(self.color[0] * 0.6 + la)),
                min(255, int(self.color[1] * 0.6 + la)),
                min(255, int(self.color[2] * 0.6 + la)),
            )
            try:
                pygame.draw.circle(surface, col, (ix, iy), lr, max(1, lr // 3))
            except Exception:
                pass

        # Core
        for layer, (radius_mul, bright_add) in enumerate([(1.0, 80), (0.6, 160), (0.3, 220)]):
            lr = max(1, int(r * radius_mul))
            col = (
                min(255, self.color[0] + bright_add),
                min(255, self.color[1] + bright_add),
                min(255, self.color[2] + bright_add),
            )
            try:
                pygame.draw.circle(surface, col, (ix, iy), lr)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# OrbitRing
# ─────────────────────────────────────────────────────────────────────────────

class OrbitRing:
    """Rotational ring with evenly spaced nodes."""

    def __init__(self, cx: float, cy: float, radius: float,
                 color: Tuple[int,int,int] = (100, 200, 255),
                 node_count: int = 8, node_size: int = 4,
                 speed: float = 1.5, tilt: float = 0.0):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.color = color
        self.node_count = node_count
        self.node_size = node_size
        self.speed = speed
        self.tilt = tilt           # vertical scale (0=flat, 1=circle)
        self._angle = 0.0
        self.alive = True
        self.opacity = 1.0

    def update(self, dt: float):
        self._angle += self.speed * dt

    def draw(self, surface: pygame.Surface):
        if not self.alive or self.opacity < 0.02:
            return
        step = math.tau / max(1, self.node_count)
        for i in range(self.node_count):
            a = self._angle + i * step
            nx = self.cx + math.cos(a) * self.radius
            ny = self.cy + math.sin(a) * self.radius * (0.35 + 0.65 * self.tilt)
            # Depth cue — nodes "behind" are dimmer
            depth = 0.5 + 0.5 * math.sin(a)
            sz = max(1, int(self.node_size * (0.5 + 0.5 * depth)))
            col = lerp_color((30, 30, 60), self.color, depth * self.opacity)
            try:
                pygame.draw.circle(surface, col, (int(nx), int(ny)), sz)
            except Exception:
                pass
        # Draw ring arc (dashed feel via many small dots)
        segs = self.node_count * 4
        for i in range(segs):
            a = self._angle + (i / segs) * math.tau
            nx = self.cx + math.cos(a) * self.radius
            ny = self.cy + math.sin(a) * self.radius * (0.35 + 0.65 * self.tilt)
            depth = 0.5 + 0.5 * math.sin(a)
            col = lerp_color((10, 15, 30), self.color, depth * 0.4 * self.opacity)
            try:
                pygame.draw.circle(surface, col, (int(nx), int(ny)), 1)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Vortex
# ─────────────────────────────────────────────────────────────────────────────

class Vortex:
    """
    Swirling vortex visualised as rotating spiral arms.
    """

    def __init__(self, cx: float, cy: float, radius: float = 80.0,
                 color: Tuple[int,int,int] = (180, 80, 255),
                 arms: int = 3, speed: float = 2.5):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.color = color
        self.arms = arms
        self.speed = speed
        self._phase = 0.0
        self.alive = True
        self.opacity = 1.0

    def update(self, dt: float):
        self._phase += self.speed * dt

    def draw(self, surface: pygame.Surface):
        if not self.alive or self.opacity < 0.02:
            return
        steps = 40
        for arm in range(self.arms):
            arm_offset = (arm / self.arms) * math.tau
            prev = None
            for s in range(steps):
                frac = s / (steps - 1)
                angle = self._phase + arm_offset + frac * math.pi * 3.5
                r = frac * self.radius
                nx = self.cx + math.cos(angle) * r
                ny = self.cy + math.sin(angle) * r * 0.55
                alpha_f = (1.0 - frac) * self.opacity
                col = lerp_color((10, 5, 20), self.color, alpha_f)
                if prev is not None and alpha_f > 0.05:
                    try:
                        pygame.draw.line(surface, col,
                                         (int(prev[0]), int(prev[1])),
                                         (int(nx), int(ny)), 2)
                    except Exception:
                        pass
                prev = (nx, ny)


# ─────────────────────────────────────────────────────────────────────────────
# EnergyTrail
# ─────────────────────────────────────────────────────────────────────────────

class EnergyTrail:
    """
    Smooth fading trail following a moving point.
    Stores recent positions and renders a tapered spline-like glow.
    """

    def __init__(self, color: Tuple[int,int,int] = (100, 200, 255),
                 max_len: int = 24, base_width: int = 6, glow: bool = True):
        self.color = color
        self.max_len = max_len
        self.base_width = base_width
        self.glow = glow
        self._points: list = []   # list of (x, y, t_age)
        self._age: float = 0.0
        self.alive = True

    def push(self, x: float, y: float):
        """Add a new position sample."""
        self._points.append((x, y, 0.0))
        if len(self._points) > self.max_len:
            self._points.pop(0)

    def update(self, dt: float):
        # Age each point
        self._points = [(x, y, a + dt) for (x, y, a) in self._points]
        # Remove very old points
        self._points = [(x, y, a) for (x, y, a) in self._points if a < 0.6]

    def draw(self, surface: pygame.Surface):
        if not self.alive or len(self._points) < 2:
            return
        max_age = max(a for (_, _, a) in self._points) + 0.001
        for i in range(1, len(self._points)):
            x0, y0, a0 = self._points[i - 1]
            x1, y1, a1 = self._points[i]
            frac0 = 1.0 - a0 / max_age
            frac1 = 1.0 - a1 / max_age
            frac = (frac0 + frac1) * 0.5
            w = max(1, int(self.base_width * frac))
            col = lerp_color((10, 10, 20), self.color, frac * 0.9)
            try:
                pygame.draw.line(surface, col, (int(x0), int(y0)), (int(x1), int(y1)), w)
            except Exception:
                pass
            if self.glow and w > 2:
                gw = w + 4
                gcol = lerp_color((10, 10, 20), self.color, frac * 0.3)
                try:
                    pygame.draw.line(surface, gcol, (int(x0), int(y0)), (int(x1), int(y1)), gw)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Projectile
# ─────────────────────────────────────────────────────────────────────────────

class Projectile:
    """
    Fast-moving elemental projectile with tail trail.
    """

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 color: Tuple[int,int,int] = (255, 180, 0),
                 radius: float = 12.0, lifetime: float = 2.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.life = lifetime
        self.max_life = lifetime
        self.alive = True
        self._trail: List[Tuple[float, float]] = []

    def update(self, dt: float):
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False
            return
        self._trail.append((self.x, self.y))
        if len(self._trail) > 16:
            self._trail.pop(0)
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        alpha_f = self.life / self.max_life
        # Draw trail
        for i in range(1, len(self._trail)):
            tf = i / len(self._trail) * alpha_f
            w = max(1, int(self.radius * 0.6 * tf))
            col = lerp_color((10, 10, 20), self.color, tf)
            try:
                pygame.draw.line(surface, col,
                                 (int(self._trail[i-1][0]), int(self._trail[i-1][1])),
                                 (int(self._trail[i][0]), int(self._trail[i][1])),
                                 w)
            except Exception:
                pass
        # Core orb
        ix, iy = int(self.x), int(self.y)
        r = max(2, int(self.radius))
        for lr, ba in [(r*2, 40), (r, 140), (int(r*0.5), 255)]:
            bright = int(alpha_f * ba)
            col = (min(255, self.color[0] * bright // 255 + bright // 3),
                   min(255, self.color[1] * bright // 255 + bright // 3),
                   min(255, self.color[2] * bright // 255 + bright // 3))
            try:
                pygame.draw.circle(surface, col, (ix, iy), lr)
            except Exception:
                pass
