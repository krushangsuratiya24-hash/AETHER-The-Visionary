"""
AETHER — VFX Effect Classes  (Phase 2 Quality Rebuild)
Reusable VFX building blocks: shockwaves, lightning arcs, energy orbs,
orbit rings, vortex fields, hand trails, and elemental projectiles.
All render directly to a pygame.Surface.

Design principles:
- Every major visual has multiple layers (core, aura, glow, sparks).
- No expensive surface allocations every frame — all rendering is procedural.
- Shapes are element-specific: fire uses flame geometry, ice uses shards, etc.
- Cached gradient surfaces are pre-built once and reused.
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
    """Expanding ring shockwave with fade-out and optional secondary rings."""

    def __init__(self, x: float, y: float, max_radius: float = 200.0,
                 duration: float = 0.6,
                 color: Tuple[int, int, int] = (255, 255, 255),
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
        return 1.0 - max(0.0, self.life / self.duration)

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        t = self.t
        radius = int(self.max_radius * t)
        alpha_f = 1.0 - t
        ix, iy = int(self.x), int(self.y)

        for ring in range(self.glow_rings, -1, -1):
            r_offset = ring * 5
            r = radius - r_offset
            if r <= 0:
                continue
            ring_alpha = alpha_f * (1.0 - ring * 0.25)
            if ring_alpha < 0.01:
                continue
            dim = int(ring_alpha * 220)
            col = (
                min(255, self.color[0] * dim // 220),
                min(255, self.color[1] * dim // 220),
                min(255, self.color[2] * dim // 220),
            )
            w = self.width + ring
            try:
                pygame.draw.circle(surface, col, (ix, iy), r, w)
            except Exception:
                pass
        # Inner flash fill at t=0 start
        if t < 0.15:
            flash_a = int((1.0 - t / 0.15) * 80)
            fr = max(1, int(radius * 0.4))
            fcol = (min(255, self.color[0] + 60), min(255, self.color[1] + 60),
                    min(255, self.color[2] + 60))
            try:
                pygame.draw.circle(surface, fcol, (ix, iy), fr)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Arc  (improved: brighter core, glow pass)
# ─────────────────────────────────────────────────────────────────────────────

class LightningArc:
    """Branching jagged lightning bolt between two points, regenerated each frame."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 color: Tuple[int, int, int] = (180, 180, 255),
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
                      steps: int = 10, jag: float = 25.0) -> List[Tuple[float, float]]:
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
        self._segments = [self._build_zigzag(self.x1, self.y1, self.x2, self.y2,
                                              jag=self.jaggedness)]
        for _ in range(self.branches):
            main = self._segments[0]
            if len(main) < 3:
                break
            bi = random.randint(1, len(main) - 2)
            bx, by = main[bi]
            ex = bx + (random.random() - 0.5) * 90
            ey = by + (random.random() - 0.5) * 90
            self._segments.append(self._build_zigzag(bx, by, ex, ey, steps=6,
                                                      jag=self.jaggedness * 0.55))

    def update(self, dt: float):
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False
            return
        self._rebuild()  # flicker every frame

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        alpha_f = self.life / self.duration
        for si, seg in enumerate(self._segments):
            is_main = (si == 0)
            w_main = max(1, self.width)
            w_glow = w_main + 3 if is_main else w_main + 1

            bright = int(alpha_f * 255)
            # Glow pass (wider, dimmer)
            gcol = (
                min(255, self.color[0] * bright // 300 + 20),
                min(255, self.color[1] * bright // 300 + 20),
                min(255, self.color[2] * bright // 300),
            )
            for i in range(len(seg) - 1):
                try:
                    pygame.draw.line(surface, gcol,
                                     (int(seg[i][0]), int(seg[i][1])),
                                     (int(seg[i + 1][0]), int(seg[i + 1][1])),
                                     w_glow)
                except Exception:
                    pass
            # Core pass (bright white-ish core)
            ccol = (
                min(255, self.color[0] * bright // 255 + 60),
                min(255, self.color[1] * bright // 255 + 60),
                min(255, self.color[2] * bright // 255 + 20),
            )
            for i in range(len(seg) - 1):
                try:
                    pygame.draw.line(surface, ccol,
                                     (int(seg[i][0]), int(seg[i][1])),
                                     (int(seg[i + 1][0]), int(seg[i + 1][1])),
                                     max(1, w_main - 1))
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Energy Orb  (completely redesigned per-element)
# ─────────────────────────────────────────────────────────────────────────────

class EnergyOrb:
    """
    Layered energy orb with element-specific visual style.
    element_style: 'fire' | 'solar' | 'frost' | 'thunder' | 'earth' | 'wind' | 'void'
    """

    def __init__(self, x: float, y: float,
                 radius: float = 30.0,
                 color: Tuple[int, int, int] = (100, 200, 255),
                 pulse_speed: float = 3.0,
                 element_style: str = 'default'):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.pulse_speed = pulse_speed
        self.element_style = element_style
        self._phase = random.uniform(0, math.tau)
        self._rot   = 0.0
        self.alive = True
        self.opacity = 1.0

    def update(self, dt: float):
        self._phase += self.pulse_speed * dt
        self._rot   += dt * 1.8

    def draw(self, surface: pygame.Surface):
        if not self.alive or self.opacity < 0.02:
            return

        pulse = 0.88 + 0.12 * math.sin(self._phase)
        r = max(3, int(self.radius * pulse))
        ix, iy = int(self.x), int(self.y)

        style = self.element_style

        if style == 'fire':
            self._draw_fire_orb(surface, ix, iy, r)
        elif style == 'solar':
            self._draw_solar_orb(surface, ix, iy, r)
        elif style == 'frost':
            self._draw_frost_orb(surface, ix, iy, r)
        elif style == 'thunder':
            self._draw_thunder_orb(surface, ix, iy, r)
        elif style == 'earth':
            self._draw_earth_orb(surface, ix, iy, r)
        elif style == 'wind':
            self._draw_wind_orb(surface, ix, iy, r)
        elif style == 'void':
            self._draw_void_orb(surface, ix, iy, r)
        else:
            self._draw_default_orb(surface, ix, iy, r)

    def _draw_default_orb(self, surface, ix, iy, r):
        """Generic layered orb (fallback)."""
        for (rm, bm) in [(3.2, 25), (2.2, 60), (1.5, 110), (1.0, 180), (0.5, 255)]:
            lr = max(1, int(r * rm))
            col = (min(255, self.color[0] * bm // 255 + bm // 4),
                   min(255, self.color[1] * bm // 255 + bm // 4),
                   min(255, self.color[2] * bm // 255 + bm // 4))
            try:
                pygame.draw.circle(surface, col, (ix, iy), lr,
                                   max(1, lr // 4) if rm > 1.0 else 0)
            except Exception:
                pass

    def _draw_fire_orb(self, surface, ix, iy, r):
        """Irregular flame-shaped orb with hot core and flame tongues."""
        flicker = 0.85 + 0.15 * math.sin(self._phase * 3.7)
        # Outer glow (wide, red-orange, hollow ring)
        for gr, gcol in [(int(r * 3.0), (80, 15, 0)),
                         (int(r * 2.0), (150, 30, 0)),
                         (int(r * 1.5), (220, 60, 0))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, gcol, (ix, iy), gr, max(1, gr // 4))
                except Exception:
                    pass
        # Flame tongues (irregular spikes outward)
        n_tongues = 7
        for i in range(n_tongues):
            a = self._rot * 0.3 + (i / n_tongues) * math.tau
            tongue_r = r * (1.2 + 0.6 * math.sin(self._phase * 2 + i * 1.3)) * flicker
            ex = ix + math.cos(a) * tongue_r
            ey = iy + math.sin(a) * tongue_r * 0.6  # squash vertically (flames go up)
            tip_col = (255, int(180 * flicker), 0)
            try:
                pygame.draw.line(surface, tip_col, (ix, iy), (int(ex), int(ey)), 2)
            except Exception:
                pass
        # Hot core
        for rm, bc in [(1.0, (255, 120, 20)), (0.6, (255, 200, 60)), (0.25, (255, 255, 160))]:
            lr = max(1, int(r * rm))
            try:
                pygame.draw.circle(surface, bc, (ix, iy), lr)
            except Exception:
                pass

    def _draw_solar_orb(self, surface, ix, iy, r):
        """Radiant solar sphere with concentric rings and radial rays."""
        # Outer glow halos
        for gr, ac in [(int(r * 3.5), (60, 40, 0)),
                       (int(r * 2.5), (120, 80, 0)),
                       (int(r * 1.8), (200, 140, 0))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, ac, (ix, iy), gr, max(2, gr // 5))
                except Exception:
                    pass
        # Rotating rings
        for i, (ring_r, ring_col) in enumerate([
            (int(r * 1.4), (255, 200, 50)),
            (int(r * 1.1), (255, 220, 80)),
        ]):
            if ring_r > 1:
                try:
                    pygame.draw.circle(surface, ring_col, (ix, iy), ring_r, 2)
                except Exception:
                    pass
        # Radial rays
        n_rays = 12
        for i in range(n_rays):
            angle = self._rot * 0.5 + (i / n_rays) * math.tau
            rlen = r * (1.6 + 0.4 * math.sin(self._phase * 2 + i))
            ex = ix + math.cos(angle) * rlen
            ey = iy + math.sin(angle) * rlen
            try:
                pygame.draw.line(surface, (255, 220, 100), (ix, iy), (int(ex), int(ey)), 1)
            except Exception:
                pass
        # Core: bright white-gold
        for rm, bc in [(1.0, (255, 200, 40)), (0.6, (255, 230, 120)), (0.3, (255, 255, 220))]:
            lr = max(1, int(r * rm))
            try:
                pygame.draw.circle(surface, bc, (ix, iy), lr)
            except Exception:
                pass

    def _draw_frost_orb(self, surface, ix, iy, r):
        """Six-sided crystalline orb with ice shard geometry."""
        # Outer icy glow
        for gr, ac in [(int(r * 3.0), (20, 40, 80)),
                       (int(r * 2.0), (60, 120, 180)),
                       (int(r * 1.6), (120, 200, 240))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, ac, (ix, iy), gr, max(2, gr // 4))
                except Exception:
                    pass
        # Six-sided crystal outline
        n = 6
        pts = []
        for i in range(n):
            a = self._rot * 0.2 + (i / n) * math.tau
            pr = r * (1.0 + 0.2 * math.sin(self._phase * 2 + i))
            pts.append((int(ix + math.cos(a) * pr), int(iy + math.sin(a) * pr)))
        if len(pts) >= 3:
            try:
                pygame.draw.polygon(surface, (180, 230, 255), pts, 2)
            except Exception:
                pass
        # Rotating inner shard lines
        for i in range(3):
            a1 = self._rot * 0.4 + (i / 3) * math.tau
            a2 = a1 + math.pi
            p1 = (int(ix + math.cos(a1) * r), int(iy + math.sin(a1) * r))
            p2 = (int(ix + math.cos(a2) * r), int(iy + math.sin(a2) * r))
            try:
                pygame.draw.line(surface, (200, 240, 255), p1, p2, 1)
            except Exception:
                pass
        # Core: icy blue-white
        for rm, bc in [(1.0, (140, 220, 255)), (0.5, (200, 245, 255)), (0.2, (240, 255, 255))]:
            lr = max(1, int(r * rm))
            try:
                pygame.draw.circle(surface, bc, (ix, iy), lr)
            except Exception:
                pass

    def _draw_thunder_orb(self, surface, ix, iy, r):
        """Unstable electrical sphere with flickering surface."""
        flicker = random.uniform(0.7, 1.0)
        # Electric glow
        for gr, ac in [(int(r * 3.0), (30, 0, 80)),
                       (int(r * 2.2), (80, 30, 180)),
                       (int(r * 1.6), (150, 80, 255))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, ac, (ix, iy), gr, max(2, gr // 4))
                except Exception:
                    pass
        # Surface arcs
        n_arcs = int(4 + self._phase % 3)
        for i in range(n_arcs):
            a = (i / n_arcs) * math.tau + self._phase
            sr = r * (0.9 + 0.3 * random.random())
            sx = int(ix + math.cos(a) * sr)
            sy = int(iy + math.sin(a) * sr)
            ex = sx + random.randint(-int(r * 0.5), int(r * 0.5))
            ey = sy + random.randint(-int(r * 0.5), int(r * 0.5))
            arc_col = (int(200 * flicker), int(160 * flicker), 255)
            try:
                pygame.draw.line(surface, arc_col, (sx, sy), (ex, ey), 1)
            except Exception:
                pass
        # Core
        for rm, bc in [(1.0, (160, 80, 255)), (0.55, (200, 150, 255)), (0.25, (240, 220, 255))]:
            lr = max(1, int(r * rm * flicker))
            try:
                pygame.draw.circle(surface, bc, (ix, iy), lr)
            except Exception:
                pass

    def _draw_earth_orb(self, surface, ix, iy, r):
        """Rocky polygon shell with orbiting fragments."""
        # Dust halo
        for gr, ac in [(int(r * 2.8), (20, 15, 5)),
                       (int(r * 2.0), (50, 35, 15)),
                       (int(r * 1.5), (100, 70, 30))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, ac, (ix, iy), gr, max(2, gr // 4))
                except Exception:
                    pass
        # Irregular polygon shell
        n = 8
        pts = []
        for i in range(n):
            a = self._rot * 0.15 + (i / n) * math.tau
            pr = r * (0.85 + 0.25 * math.sin(self._phase + i * 0.8))
            pts.append((int(ix + math.cos(a) * pr), int(iy + math.sin(a) * pr)))
        if len(pts) >= 3:
            try:
                pygame.draw.polygon(surface, (100, 70, 35), pts, 0)
                pygame.draw.polygon(surface, (160, 120, 60), pts, 2)
            except Exception:
                pass
        # Core crevice lines
        for i in range(3):
            a = self._rot * 0.2 + i * 1.05
            p1 = (int(ix + math.cos(a) * r * 0.7), int(iy + math.sin(a) * r * 0.7))
            p2 = (int(ix + math.cos(a + 0.5) * r * 0.3), int(iy + math.sin(a + 0.5) * r * 0.3))
            try:
                pygame.draw.line(surface, (60, 40, 15), p1, p2, 1)
            except Exception:
                pass
        # Bright mineral core
        for rm, bc in [(0.45, (180, 140, 70)), (0.2, (220, 185, 110))]:
            lr = max(1, int(r * rm))
            try:
                pygame.draw.circle(surface, bc, (ix, iy), lr)
            except Exception:
                pass

    def _draw_wind_orb(self, surface, ix, iy, r):
        """Transparent vortex shell — mostly visible through spiral lines."""
        # Faint outer rings
        for gr, ac in [(int(r * 3.0), (10, 30, 20)),
                       (int(r * 2.0), (30, 80, 50)),
                       (int(r * 1.5), (80, 170, 110))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, ac, (ix, iy), gr, max(1, gr // 5))
                except Exception:
                    pass
        # Spiral vortex arms
        n_arms = 3
        steps = 24
        for arm in range(n_arms):
            arm_offset = (arm / n_arms) * math.tau
            prev = None
            for s in range(steps):
                frac = s / (steps - 1)
                angle = self._rot * 2.0 + arm_offset + frac * math.pi * 2.5
                sr = frac * r * 1.1
                nx = ix + math.cos(angle) * sr
                ny = iy + math.sin(angle) * sr * 0.6
                col_t = 1.0 - frac
                ac = lerp_color((10, 25, 15), (160, 235, 180), col_t)
                if prev is not None:
                    try:
                        pygame.draw.line(surface, ac, (int(prev[0]), int(prev[1])),
                                         (int(nx), int(ny)), 2)
                    except Exception:
                        pass
                prev = (nx, ny)
        # Hollow core
        if r > 4:
            try:
                pygame.draw.circle(surface, (140, 220, 160), (ix, iy), max(2, int(r * 0.35)), 1)
            except Exception:
                pass

    def _draw_void_orb(self, surface, ix, iy, r):
        """Dark gravitational orb — black core with distorted outer ring."""
        # Outer void rings (dark purple)
        for gr, ac in [(int(r * 3.5), (20, 5, 40)),
                       (int(r * 2.5), (50, 15, 100)),
                       (int(r * 1.8), (100, 30, 180))]:
            if gr > 1:
                try:
                    pygame.draw.circle(surface, ac, (ix, iy), gr, max(2, gr // 4))
                except Exception:
                    pass
        # Distorted outer ring
        n = 16
        pts = []
        for i in range(n):
            a = self._rot * -0.6 + (i / n) * math.tau
            pr = r * (1.0 + 0.35 * math.sin(self._phase * 2.3 + i * 0.7))
            pts.append((int(ix + math.cos(a) * pr), int(iy + math.sin(a) * pr)))
        if len(pts) >= 3:
            try:
                pygame.draw.polygon(surface, (80, 20, 140), pts, 2)
            except Exception:
                pass
        # Gravitational in-fall lines
        for i in range(6):
            a = self._rot * -1.2 + (i / 6) * math.tau
            sr = r * (1.2 + 0.3 * math.sin(self._phase + i))
            sx = int(ix + math.cos(a) * sr)
            sy = int(iy + math.sin(a) * sr)
            try:
                pygame.draw.line(surface, (120, 40, 200), (sx, sy), (ix, iy), 1)
            except Exception:
                pass
        # Dark centre (overdraw with near-black)
        for rm, bc in [(1.0, (15, 5, 30)), (0.5, (5, 0, 15))]:
            lr = max(1, int(r * rm))
            try:
                pygame.draw.circle(surface, bc, (ix, iy), lr)
            except Exception:
                pass
        # Bright violet accent ring
        try:
            pygame.draw.circle(surface, (160, 60, 255), (ix, iy), max(2, int(r * 0.95)), 2)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# OrbitRing  (improved with connected arc and depth shading)
# ─────────────────────────────────────────────────────────────────────────────

class OrbitRing:
    """Rotational ring with evenly spaced nodes, depth-shaded arc."""

    def __init__(self, cx: float, cy: float, radius: float,
                 color: Tuple[int, int, int] = (100, 200, 255),
                 node_count: int = 8, node_size: int = 4,
                 speed: float = 1.5, tilt: float = 0.0):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.color = color
        self.node_count = node_count
        self.node_size = node_size
        self.speed = speed
        self.tilt = tilt
        self._angle = 0.0
        self.alive = True
        self.opacity = 1.0

    def update(self, dt: float):
        self._angle += self.speed * dt

    def draw(self, surface: pygame.Surface):
        if not self.alive or self.opacity < 0.02:
            return
        step = math.tau / max(1, self.node_count)
        # Arc segments
        segs = max(self.node_count * 4, 40)
        prev = None
        for i in range(segs + 1):
            a = self._angle + (i / segs) * math.tau
            nx = self.cx + math.cos(a) * self.radius
            ny = self.cy + math.sin(a) * self.radius * (0.30 + 0.70 * self.tilt)
            depth = 0.5 + 0.5 * math.sin(a)
            col = lerp_color((8, 12, 25), self.color, depth * 0.5 * self.opacity)
            if prev is not None:
                try:
                    pygame.draw.line(surface, col,
                                     (int(prev[0]), int(prev[1])),
                                     (int(nx), int(ny)), 1)
                except Exception:
                    pass
            prev = (nx, ny)
        # Nodes
        for i in range(self.node_count):
            a = self._angle + i * step
            nx = self.cx + math.cos(a) * self.radius
            ny = self.cy + math.sin(a) * self.radius * (0.30 + 0.70 * self.tilt)
            depth = 0.5 + 0.5 * math.sin(a)
            sz = max(2, int(self.node_size * (0.5 + 0.5 * depth)))
            col = lerp_color((30, 30, 60), self.color, depth * self.opacity)
            try:
                pygame.draw.circle(surface, col, (int(nx), int(ny)), sz)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Vortex  (improved arms with better tapering)
# ─────────────────────────────────────────────────────────────────────────────

class Vortex:
    """Swirling vortex as rotating spiral arms with variable width."""

    def __init__(self, cx: float, cy: float, radius: float = 80.0,
                 color: Tuple[int, int, int] = (180, 80, 255),
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
        steps = 45
        for arm in range(self.arms):
            arm_offset = (arm / self.arms) * math.tau
            prev = None
            for s in range(steps):
                frac = s / (steps - 1)
                angle = self._phase + arm_offset + frac * math.pi * 3.8
                r = frac * self.radius
                nx = self.cx + math.cos(angle) * r
                ny = self.cy + math.sin(angle) * r * 0.52
                alpha_f = (1.0 - frac) * self.opacity
                col = lerp_color((8, 4, 18), self.color, alpha_f)
                if prev is not None and alpha_f > 0.04:
                    w = max(1, int(2.5 * (1.0 - frac)))
                    try:
                        pygame.draw.line(surface, col,
                                         (int(prev[0]), int(prev[1])),
                                         (int(nx), int(ny)), w)
                    except Exception:
                        pass
                prev = (nx, ny)


# ─────────────────────────────────────────────────────────────────────────────
# EnergyTrail  (elemental flavours — element_style param)
# ─────────────────────────────────────────────────────────────────────────────

class EnergyTrail:
    """
    Smooth fading trail that changes appearance based on element_style.
    """

    TRAIL_FADE = 0.55  # seconds until a point fades completely

    def __init__(self, color: Tuple[int, int, int] = (100, 200, 255),
                 max_len: int = 28, base_width: int = 7, glow: bool = True,
                 element_style: str = 'default'):
        self.color = color
        self.max_len = max_len
        self.base_width = base_width
        self.glow = glow
        self.element_style = element_style
        self._points: list = []  # (x, y, age)
        self.alive = True

    def push(self, x: float, y: float):
        self._points.append((x, y, 0.0))
        if len(self._points) > self.max_len:
            self._points.pop(0)

    def update(self, dt: float):
        self._points = [(x, y, a + dt) for (x, y, a) in self._points]
        self._points = [(x, y, a) for (x, y, a) in self._points if a < self.TRAIL_FADE]

    def draw(self, surface: pygame.Surface):
        if not self.alive or len(self._points) < 2:
            return
        max_age = self.TRAIL_FADE
        style = self.element_style

        for i in range(1, len(self._points)):
            x0, y0, a0 = self._points[i - 1]
            x1, y1, a1 = self._points[i]
            frac = 1.0 - ((a0 + a1) * 0.5) / max_age
            if frac <= 0:
                continue
            w = max(1, int(self.base_width * frac))

            if style == 'fire':
                col = lerp_color((20, 5, 0), (255, 140, 20), frac)
            elif style == 'thunder':
                col = lerp_color((20, 0, 50), (180, 100, 255), frac)
                # Occasional spark jitter
                if random.random() < 0.15 * frac:
                    sx = int(x1 + random.uniform(-6, 6))
                    sy = int(y1 + random.uniform(-6, 6))
                    try:
                        pygame.draw.circle(surface, (220, 200, 255), (sx, sy), 2)
                    except Exception:
                        pass
            elif style == 'frost':
                col = lerp_color((10, 20, 50), (160, 230, 255), frac)
            elif style == 'void':
                col = lerp_color((5, 0, 15), (120, 40, 200), frac)
            elif style == 'wind':
                col = lerp_color((5, 20, 10), (160, 240, 180), frac)
            elif style == 'earth':
                col = lerp_color((15, 10, 3), (140, 100, 50), frac)
            elif style == 'solar':
                col = lerp_color((30, 20, 0), (255, 200, 40), frac)
            else:
                col = lerp_color((10, 10, 20), self.color, frac * 0.9)

            try:
                pygame.draw.line(surface, col, (int(x0), int(y0)), (int(x1), int(y1)), w)
            except Exception:
                pass

            if self.glow and w > 1:
                gw = w + 3
                gcol = lerp_color((5, 5, 10), col, 0.4)
                try:
                    pygame.draw.line(surface, gcol, (int(x0), int(y0)), (int(x1), int(y1)), gw)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Projectile  (element-aware, direction-driven, multi-layer)
# ─────────────────────────────────────────────────────────────────────────────

class Projectile:
    """
    Fast-moving elemental projectile with directional trail and glow.
    vx/vy drive the actual direction (from swipe velocity).
    element_style selects the visual language.
    """

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 color: Tuple[int, int, int] = (255, 180, 0),
                 radius: float = 12.0, lifetime: float = 2.0,
                 element_style: str = 'default'):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.life = lifetime
        self.max_life = lifetime
        self.element_style = element_style
        self.alive = True
        self._trail: List[Tuple[float, float]] = []
        self._phase = random.uniform(0, math.tau)

    def update(self, dt: float):
        self.life -= dt
        if self.life <= 0.0:
            self.alive = False
            return
        self._phase += dt * 8
        self._trail.append((self.x, self.y))
        if len(self._trail) > 22:
            self._trail.pop(0)
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        alpha_f = self.life / self.max_life
        style = self.element_style

        # Draw element-specific trail
        for i in range(1, len(self._trail)):
            tf = (i / len(self._trail)) * alpha_f
            if style == 'fire':
                col = lerp_color((15, 5, 0), (255, 120, 0), tf)
                w = max(1, int(self.radius * 0.8 * tf))
                # Add turbulence: small side offset for flame trail
                jx = int((random.random() - 0.5) * self.radius * 0.3 * tf)
                try:
                    pygame.draw.line(surface, col,
                                     (int(self._trail[i - 1][0]) + jx, int(self._trail[i - 1][1])),
                                     (int(self._trail[i][0]) + jx, int(self._trail[i][1])), w)
                except Exception:
                    pass
            elif style == 'thunder':
                col = lerp_color((10, 0, 30), (160, 80, 255), tf)
                w = max(1, int(self.radius * 0.6 * tf))
                try:
                    pygame.draw.line(surface, col,
                                     (int(self._trail[i - 1][0]), int(self._trail[i - 1][1])),
                                     (int(self._trail[i][0]), int(self._trail[i][1])), w)
                except Exception:
                    pass
                # Electric sparks along trail
                if random.random() < 0.25:
                    sx = int(self._trail[i][0] + random.uniform(-self.radius, self.radius))
                    sy = int(self._trail[i][1] + random.uniform(-self.radius * 0.5, self.radius * 0.5))
                    try:
                        pygame.draw.line(surface, (200, 180, 255),
                                         (int(self._trail[i][0]), int(self._trail[i][1])),
                                         (sx, sy), 1)
                    except Exception:
                        pass
            elif style == 'frost':
                col = lerp_color((5, 15, 40), (140, 220, 255), tf)
                w = max(1, int(self.radius * 0.7 * tf))
                try:
                    pygame.draw.line(surface, col,
                                     (int(self._trail[i - 1][0]), int(self._trail[i - 1][1])),
                                     (int(self._trail[i][0]), int(self._trail[i][1])), w)
                except Exception:
                    pass
            else:
                col = lerp_color((8, 8, 18), self.color, tf)
                w = max(1, int(self.radius * 0.6 * tf))
                try:
                    pygame.draw.line(surface, col,
                                     (int(self._trail[i - 1][0]), int(self._trail[i - 1][1])),
                                     (int(self._trail[i][0]), int(self._trail[i][1])), w)
                except Exception:
                    pass

        # Core orb (element-specific)
        ix, iy = int(self.x), int(self.y)
        r = max(3, int(self.radius * alpha_f))

        if style == 'fire':
            # Flame burst at leading edge
            for rm, fc in [(2.5, (100, 20, 0)), (1.8, (220, 60, 0)),
                           (1.0, (255, 140, 20)), (0.5, (255, 230, 80))]:
                lr = max(1, int(r * rm))
                try:
                    pygame.draw.circle(surface, fc, (ix, iy), lr,
                                       max(1, lr // 3) if rm > 1.0 else 0)
                except Exception:
                    pass
        elif style == 'thunder':
            # Electric core
            flicker = random.uniform(0.7, 1.0)
            for rm, fc in [(2.0, (40, 0, 100)), (1.2, (120, 60, 220)),
                           (0.6, (200, 160, 255))]:
                lr = max(1, int(r * rm * flicker))
                try:
                    pygame.draw.circle(surface, fc, (ix, iy), lr,
                                       max(1, lr // 3) if rm > 1.0 else 0)
                except Exception:
                    pass
        elif style == 'frost':
            # Ice shard front
            n = 6
            for j in range(n):
                a = self._phase + (j / n) * math.tau
                tip = (int(ix + math.cos(a) * r * 1.4), int(iy + math.sin(a) * r * 1.4))
                base1 = (int(ix + math.cos(a + 0.5) * r * 0.5),
                         int(iy + math.sin(a + 0.5) * r * 0.5))
                base2 = (int(ix + math.cos(a - 0.5) * r * 0.5),
                         int(iy + math.sin(a - 0.5) * r * 0.5))
                try:
                    pygame.draw.polygon(surface, (180, 230, 255), [tip, base1, base2])
                except Exception:
                    pass
            try:
                pygame.draw.circle(surface, (220, 248, 255), (ix, iy), max(2, int(r * 0.5)))
            except Exception:
                pass
        else:
            # Generic layered core
            for rm, ba in [(2.2, 35), (1.4, 100), (0.7, 200)]:
                lr = max(1, int(r * rm))
                bright = int(alpha_f * ba)
                col = (min(255, self.color[0] * bright // 200 + bright // 3),
                       min(255, self.color[1] * bright // 200 + bright // 3),
                       min(255, self.color[2] * bright // 200 + bright // 3))
                try:
                    pygame.draw.circle(surface, col, (ix, iy), lr,
                                       max(1, lr // 3) if rm > 1.0 else 0)
                except Exception:
                    pass
