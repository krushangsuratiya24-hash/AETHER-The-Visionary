"""
AETHER — Core Particle System
High-performance particle engine with object pooling and rich property support.
Designed for smooth 60 FPS rendering on Pygame without per-frame surface allocation.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple
import pygame


# ─────────────────────────────────────────────────────────────────────────────
# Colour Utilities
# ─────────────────────────────────────────────────────────────────────────────

def lerp_color(c1: Tuple[int,int,int], c2: Tuple[int,int,int], t: float) -> Tuple[int,int,int]:
    """Linear interpolate between two RGB colours."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def alpha_color(rgb: Tuple[int,int,int], alpha: float) -> Tuple[int,int,int,int]:
    return (rgb[0], rgb[1], rgb[2], max(0, min(255, int(alpha * 255))))


# ─────────────────────────────────────────────────────────────────────────────
# Particle
# ─────────────────────────────────────────────────────────────────────────────

class Particle:
    """
    Single particle with full physics and rendering properties.
    Reused via the pool — reset() is called instead of __init__.
    """
    __slots__ = (
        'x', 'y', 'vx', 'vy', 'ax', 'ay',
        'life', 'max_life',
        'size', 'min_size', 'max_size',
        'opacity', 'min_opacity', 'max_opacity',
        'rotation', 'angular_velocity',
        'color', 'color_end',
        'gravity', 'drag',
        'glow', 'trail', 'trail_points',
        'shape',       # 'circle' | 'square' | 'shard' | 'ring'
        'attract_x', 'attract_y', 'attract_strength',
        'orbit_cx', 'orbit_cy', 'orbit_radius', 'orbit_speed', 'orbit_angle',
        'turbulence',
        'alive',
    )

    def __init__(self):
        self.alive = False
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.ax = self.ay = 0.0
        self.life = self.max_life = 1.0
        self.size = self.min_size = self.max_size = 4.0
        self.opacity = self.min_opacity = 1.0
        self.max_opacity = 1.0
        self.rotation = 0.0
        self.angular_velocity = 0.0
        self.color = (255, 255, 255)
        self.color_end = (255, 255, 255)
        self.gravity = 0.0
        self.drag = 0.98
        self.glow = False
        self.trail = False
        self.trail_points: list = []
        self.shape = 'circle'
        self.attract_x = self.attract_y = 0.0
        self.attract_strength = 0.0
        self.orbit_cx = self.orbit_cy = 0.0
        self.orbit_radius = 0.0
        self.orbit_speed = 0.0
        self.orbit_angle = 0.0
        self.turbulence = 0.0

    def reset(self,
              x: float, y: float,
              vx: float = 0.0, vy: float = 0.0,
              ax: float = 0.0, ay: float = 0.0,
              life: float = 1.0,
              size: float = 4.0, min_size: float = 0.5,
              opacity: float = 1.0, min_opacity: float = 0.0,
              rotation: float = 0.0, angular_velocity: float = 0.0,
              color: Tuple[int,int,int] = (255, 255, 255),
              color_end: Optional[Tuple[int,int,int]] = None,
              gravity: float = 0.0,
              drag: float = 0.97,
              glow: bool = False,
              trail: bool = False,
              shape: str = 'circle',
              turbulence: float = 0.0,
              orbit_cx: float = 0.0, orbit_cy: float = 0.0,
              orbit_radius: float = 0.0, orbit_speed: float = 0.0, orbit_angle: float = 0.0,
              attract_x: float = 0.0, attract_y: float = 0.0, attract_strength: float = 0.0,
              ):
        self.alive = True
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.ax, self.ay = ax, ay
        self.life = self.max_life = max(0.001, life)
        self.size = self.max_size = size
        self.min_size = min_size
        self.opacity = self.max_opacity = opacity
        self.min_opacity = min_opacity
        self.rotation = rotation
        self.angular_velocity = angular_velocity
        self.color = color
        self.color_end = color_end if color_end is not None else color
        self.gravity = gravity
        self.drag = drag
        self.glow = glow
        self.trail = trail
        self.trail_points = []
        self.shape = shape
        self.turbulence = turbulence
        self.orbit_cx, self.orbit_cy = orbit_cx, orbit_cy
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
        self.orbit_angle = orbit_angle
        self.attract_x, self.attract_y = attract_x, attract_y
        self.attract_strength = attract_strength

    def update(self, dt: float):
        if not self.alive:
            return

        self.life -= dt
        if self.life <= 0.0:
            self.alive = False
            return

        t = 1.0 - (self.life / self.max_life)  # 0 → 1 over lifetime

        # Turbulence
        if self.turbulence > 0.0:
            self.vx += (random.random() - 0.5) * self.turbulence * dt * 60
            self.vy += (random.random() - 0.5) * self.turbulence * dt * 60

        # Attraction
        if self.attract_strength != 0.0:
            dx = self.attract_x - self.x
            dy = self.attract_y - self.y
            dist = math.sqrt(dx*dx + dy*dy) + 1.0
            f = self.attract_strength / dist
            self.vx += dx * f * dt * 60
            self.vy += dy * f * dt * 60

        # Orbit
        if self.orbit_radius > 0.0:
            self.orbit_angle += self.orbit_speed * dt
            self.x = self.orbit_cx + math.cos(self.orbit_angle) * self.orbit_radius
            self.y = self.orbit_cy + math.sin(self.orbit_angle) * self.orbit_radius
        else:
            # Normal physics
            self.vx += (self.ax) * dt * 60
            self.vy += (self.ay + self.gravity) * dt * 60
            self.vx *= self.drag
            self.vy *= self.drag
            self.x += self.vx * dt * 60
            self.y += self.vy * dt * 60

        # Rotation
        self.rotation += self.angular_velocity * dt * 60

        # Update size and opacity based on t
        size_t = 1.0 - t
        self.size = self.min_size + (self.max_size - self.min_size) * size_t
        self.opacity = self.min_opacity + (self.max_opacity - self.min_opacity) * (1.0 - t)

        # Trail history
        if self.trail:
            self.trail_points.append((self.x, self.y))
            if len(self.trail_points) > 12:
                self.trail_points.pop(0)

    def get_current_color(self) -> Tuple[int,int,int]:
        t = 1.0 - (self.life / self.max_life)
        return lerp_color(self.color, self.color_end, t)

    def draw(self, surface: pygame.Surface):
        if not self.alive or self.size < 0.3 or self.opacity < 0.01:
            return

        col = self.get_current_color()
        alpha = int(self.opacity * 255)
        ix, iy = int(self.x), int(self.y)
        isize = max(1, int(self.size))

        # Draw trail
        if self.trail and len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                ta = alpha * (i / len(self.trail_points)) * 0.5
                ts = max(1, int(isize * (i / len(self.trail_points)) * 0.7))
                if ta > 3:
                    try:
                        pygame.draw.circle(surface, col, (int(self.trail_points[i][0]), int(self.trail_points[i][1])), ts)
                    except Exception:
                        pass

        if self.shape == 'ring':
            if isize > 1:
                try:
                    pygame.draw.circle(surface, col, (ix, iy), isize, max(1, isize // 4))
                except Exception:
                    pass
        elif self.shape == 'square':
            half = isize // 2
            try:
                pygame.draw.rect(surface, col, (ix - half, iy - half, isize, isize))
            except Exception:
                pass
        elif self.shape == 'shard':
            # Small elongated triangle oriented by rotation
            angle = self.rotation
            length = isize * 2.5
            width = isize * 0.6
            pts = [
                (ix + math.cos(angle) * length, iy + math.sin(angle) * length),
                (ix + math.cos(angle + 2.3) * width, iy + math.sin(angle + 2.3) * width),
                (ix + math.cos(angle - 2.3) * width, iy + math.sin(angle - 2.3) * width),
            ]
            try:
                pygame.draw.polygon(surface, col, [(int(p[0]), int(p[1])) for p in pts])
            except Exception:
                pass
        else:  # 'circle'
            try:
                pygame.draw.circle(surface, col, (ix, iy), isize)
            except Exception:
                pass

        # Glow halo — larger semi-transparent circle
        if self.glow and isize >= 2:
            glow_size = isize * 2
            dim_col = (max(0, col[0] - 40), max(0, col[1] - 40), max(0, col[2] - 40))
            try:
                pygame.draw.circle(surface, dim_col, (ix, iy), glow_size, max(1, glow_size // 3))
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Particle Pool
# ─────────────────────────────────────────────────────────────────────────────

class ParticlePool:
    """
    Object pool of Particle instances to avoid per-frame allocation.
    Pre-allocates a fixed number of particles; excess emissions are silently dropped.
    """

    def __init__(self, capacity: int = 2000):
        self._pool: List[Particle] = [Particle() for _ in range(capacity)]
        self._capacity = capacity

    @property
    def active_count(self) -> int:
        return sum(1 for p in self._pool if p.alive)

    def acquire(self) -> Optional[Particle]:
        """Return a free (dead) particle from the pool, or None if full."""
        for p in self._pool:
            if not p.alive:
                return p
        return None  # pool exhausted — drop emission silently

    def update(self, dt: float):
        for p in self._pool:
            if p.alive:
                p.update(dt)

    def draw(self, surface: pygame.Surface):
        for p in self._pool:
            if p.alive:
                p.draw(surface)

    def clear(self):
        for p in self._pool:
            p.alive = False

    # Convenience emitters ────────────────────────────────────────────────────

    def emit(self, **kwargs) -> Optional[Particle]:
        """Acquire a particle and reset it with the given keyword args."""
        p = self.acquire()
        if p is not None:
            p.reset(**kwargs)
        return p

    def emit_burst(self, count: int, **kwargs):
        """Emit multiple particles with the same base params (spread is caller's job)."""
        for _ in range(count):
            self.emit(**kwargs)
