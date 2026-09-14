"""
AETHER — Effect Composer
High-level effect factory and batch manager.
Provides easy spawn/update/render of composite effects.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import pygame

from core.particles import ParticlePool
from core.vfx import (
    Shockwave, LightningArc, EnergyOrb, OrbitRing,
    Vortex, EnergyTrail, Projectile, lerp_color
)


class EffectComposer:
    """
    Owns and manages all live VFX objects for an experience.
    Single update/draw call drives the whole visual layer.
    """

    def __init__(self, pool: ParticlePool):
        self.pool = pool
        self.shockwaves: List[Shockwave]    = []
        self.lightnings: List[LightningArc] = []
        self.orbs: List[EnergyOrb]          = []
        self.rings: List[OrbitRing]         = []
        self.vortices: List[Vortex]         = []
        self.trails: List[EnergyTrail]      = []
        self.projectiles: List[Projectile]  = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def update(self, dt: float):
        self.pool.update(dt)
        for lst in (self.shockwaves, self.lightnings, self.orbs,
                    self.rings, self.vortices, self.trails, self.projectiles):
            for obj in lst:
                obj.update(dt)
        # Prune dead objects
        self.shockwaves   = [o for o in self.shockwaves   if o.alive]
        self.lightnings   = [o for o in self.lightnings   if o.alive]
        self.orbs         = [o for o in self.orbs         if o.alive]
        self.rings        = [o for o in self.rings        if o.alive]
        self.vortices     = [o for o in self.vortices     if o.alive]
        self.trails       = [o for o in self.trails       if o.alive]
        self.projectiles  = [o for o in self.projectiles  if o.alive]

    def draw(self, surface: pygame.Surface):
        # Draw order: particles behind, then effects on top
        self.pool.draw(surface)
        for obj in self.vortices:
            obj.draw(surface)
        for obj in self.rings:
            obj.draw(surface)
        for obj in self.trails:
            obj.draw(surface)
        for obj in self.shockwaves:
            obj.draw(surface)
        for obj in self.lightnings:
            obj.draw(surface)
        for obj in self.orbs:
            obj.draw(surface)
        for obj in self.projectiles:
            obj.draw(surface)

    def clear(self):
        self.pool.clear()
        self.shockwaves.clear()
        self.lightnings.clear()
        self.orbs.clear()
        self.rings.clear()
        self.vortices.clear()
        self.trails.clear()
        self.projectiles.clear()

    # ── Particle Emitters ─────────────────────────────────────────────────────

    def emit_radial_burst(self, x: float, y: float, count: int,
                          speed: float, color_a, color_b=None,
                          life: float = 0.8, size: float = 4.0,
                          gravity: float = 0.0, glow: bool = False,
                          shape: str = 'circle', trail: bool = False,
                          drag: float = 0.95, turbulence: float = 0.0):
        color_b = color_b or color_a
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            spd = speed * random.uniform(0.6, 1.4)
            self.pool.emit(
                x=x + random.uniform(-4, 4),
                y=y + random.uniform(-4, 4),
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                life=life * random.uniform(0.7, 1.3),
                size=size * random.uniform(0.6, 1.4),
                min_size=0.5,
                color=color_a,
                color_end=color_b,
                gravity=gravity,
                drag=drag,
                glow=glow,
                trail=trail,
                shape=shape,
                turbulence=turbulence,
            )

    def emit_directed_stream(self, x: float, y: float, angle: float,
                             spread: float, speed: float, count: int,
                             color_a, color_b=None,
                             life: float = 0.6, size: float = 3.5,
                             gravity: float = 0.0, drag: float = 0.96,
                             glow: bool = False, shape: str = 'circle'):
        color_b = color_b or color_a
        for _ in range(count):
            a = angle + random.uniform(-spread, spread)
            spd = speed * random.uniform(0.7, 1.3)
            self.pool.emit(
                x=x, y=y,
                vx=math.cos(a) * spd,
                vy=math.sin(a) * spd,
                life=life * random.uniform(0.6, 1.4),
                size=size,
                min_size=0.5,
                color=color_a,
                color_end=color_b,
                gravity=gravity,
                drag=drag,
                glow=glow,
                shape=shape,
            )

    def emit_rising_flames(self, x: float, y: float, count: int,
                           color_hot, color_cool, intensity: float = 1.0):
        """Fire-specific upward-biased emitter."""
        for _ in range(count):
            offset_x = random.uniform(-30, 30) * intensity
            vx = random.uniform(-0.8, 0.8) * intensity
            vy = random.uniform(-3.5, -1.5) * intensity
            self.pool.emit(
                x=x + offset_x,
                y=y + random.uniform(-10, 10),
                vx=vx, vy=vy,
                life=random.uniform(0.4, 0.9) * intensity,
                size=random.uniform(4, 10) * intensity,
                min_size=1.0,
                color=color_hot,
                color_end=color_cool,
                gravity=-0.02,
                drag=0.97,
                glow=True,
                trail=random.random() < 0.3,
                turbulence=0.8,
            )

    def emit_sparks(self, x: float, y: float, count: int,
                    color, speed: float = 4.0):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            spd = speed * random.uniform(0.5, 2.0)
            self.pool.emit(
                x=x, y=y,
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                life=random.uniform(0.2, 0.5),
                size=random.uniform(1, 3),
                min_size=0.3,
                color=color,
                color_end=(20, 20, 40),
                gravity=0.05,
                drag=0.94,
                glow=True,
            )

    def emit_orbiting(self, cx: float, cy: float, count: int,
                      radius: float, color, life: float = 2.5,
                      speed: float = 2.0):
        for i in range(count):
            angle = (i / max(1, count)) * math.tau + random.uniform(-0.3, 0.3)
            self.pool.emit(
                x=cx + math.cos(angle) * radius,
                y=cy + math.sin(angle) * radius,
                orbit_cx=cx, orbit_cy=cy,
                orbit_radius=radius * random.uniform(0.85, 1.15),
                orbit_speed=speed * random.uniform(0.8, 1.2),
                orbit_angle=angle,
                life=life,
                size=random.uniform(2, 5),
                min_size=0.5,
                color=color,
                color_end=(10, 10, 30),
                glow=True,
            )

    def emit_crystals(self, x: float, y: float, count: int,
                      color, speed: float = 3.5):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            spd = speed * random.uniform(0.6, 1.4)
            self.pool.emit(
                x=x, y=y,
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd,
                life=random.uniform(0.5, 1.2),
                size=random.uniform(4, 10),
                min_size=1.0,
                rotation=random.uniform(0, math.tau),
                angular_velocity=random.uniform(-0.08, 0.08),
                color=color,
                color_end=(200, 240, 255),
                gravity=0.04,
                drag=0.96,
                glow=True,
                shape='shard',
            )

    def emit_rock_chunks(self, x: float, y: float, count: int,
                         color, speed: float = 3.0):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            spd = speed * random.uniform(0.5, 1.5)
            self.pool.emit(
                x=x + random.uniform(-20, 20),
                y=y + random.uniform(-10, 10),
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd - 1.5,
                life=random.uniform(0.5, 1.0),
                size=random.uniform(5, 14),
                min_size=2.0,
                rotation=random.uniform(0, math.tau),
                angular_velocity=random.uniform(-0.12, 0.12),
                color=color,
                color_end=(40, 30, 20),
                gravity=0.12,
                drag=0.95,
                shape='square',
            )

    def emit_void_fragments(self, cx: float, cy: float, count: int,
                            color, radius: float = 80.0):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            r = radius * random.uniform(0.5, 1.5)
            # Fragments that spiral inward
            orb_speed = random.uniform(-3.5, -1.5)
            self.pool.emit(
                x=cx + math.cos(angle) * r,
                y=cy + math.sin(angle) * r,
                orbit_cx=cx, orbit_cy=cy,
                orbit_radius=r,
                orbit_speed=orb_speed,
                orbit_angle=angle,
                life=random.uniform(1.0, 2.5),
                size=random.uniform(2, 6),
                min_size=0.3,
                color=color,
                color_end=(5, 0, 15),
                glow=True,
                shape='ring',
            )

    # ── High-Level Compound Effects ───────────────────────────────────────────

    def spawn_shockwave(self, x: float, y: float, **kwargs) -> Shockwave:
        sw = Shockwave(x, y, **kwargs)
        self.shockwaves.append(sw)
        return sw

    def spawn_lightning(self, x1, y1, x2, y2, **kwargs) -> LightningArc:
        arc = LightningArc(x1, y1, x2, y2, **kwargs)
        self.lightnings.append(arc)
        return arc

    def spawn_orb(self, x, y, **kwargs) -> EnergyOrb:
        orb = EnergyOrb(x, y, **kwargs)
        self.orbs.append(orb)
        return orb

    def spawn_ring(self, cx, cy, **kwargs) -> OrbitRing:
        ring = OrbitRing(cx, cy, **kwargs)
        self.rings.append(ring)
        return ring

    def spawn_vortex(self, cx, cy, **kwargs) -> Vortex:
        v = Vortex(cx, cy, **kwargs)
        self.vortices.append(v)
        return v

    def spawn_trail(self, **kwargs) -> EnergyTrail:
        t = EnergyTrail(**kwargs)
        self.trails.append(t)
        return t

    def spawn_projectile(self, x, y, vx, vy, **kwargs) -> Projectile:
        # Cap concurrent projectiles to keep performance bounded
        if len(self.projectiles) >= 8:
            return self.projectiles[0]  # return existing, don't spawn
        p = Projectile(x, y, vx, vy, **kwargs)
        self.projectiles.append(p)
        return p
