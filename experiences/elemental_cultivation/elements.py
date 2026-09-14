"""
AETHER — Elemental Cultivation: Seven Element Definitions  (Phase 2 Quality Rebuild)
Each element owns its own energy state machine and provides
update() / render() hooks that speak to the shared EffectComposer.

Changes from Phase 2 Quality Correction:
- All elements pass element_style to EnergyOrb / EnergyTrail / Projectile.
- Projectile direction now uses gs.velocity_x / gs.velocity_y from the actual
  swipe motion vector instead of a hardcoded center-based formula.
- Fire: tapered flame particles, ember sparks, heat shimmer expanded.
- Solar: radial rays on render pass, solar flare particles.
- Frost: ice shard geometry, frost ring burst.
- Thunder: visible charge buildup arcs, high-voltage projectile.
- Earth: irregular polygon rocks, gravity-affected debris.
- Wind: curved ribbon trails, vortex ring.
- Void: gravitational inward spiral, dark implosion.
"""
from __future__ import annotations

import math
import random
from enum import Enum
from typing import Tuple, Optional

import pygame

from core.effects import EffectComposer
from core.vfx import EnergyOrb, OrbitRing, Vortex, EnergyTrail, LightningArc
from core.particles import lerp_color
from experiences.elemental_cultivation.gestures import ElementalGesture, ElementalGestureState


# ─────────────────────────────────────────────────────────────────────────────
# Element Identities
# ─────────────────────────────────────────────────────────────────────────────

class ElementID(Enum):
    PHOENIX_FLAME = 1
    GOLDEN_SOLAR  = 2
    FROST         = 3
    THUNDER       = 4
    EARTH         = 5
    WIND          = 6
    VOID          = 7


ELEMENT_INFO = {
    ElementID.PHOENIX_FLAME: dict(
        name="PHOENIX FLAME",
        icon="🔥",
        color=(255, 90, 20),
        color_b=(255, 220, 60),
        key='1',
    ),
    ElementID.GOLDEN_SOLAR: dict(
        name="GOLDEN SOLAR",
        icon="☀️",
        color=(255, 215, 0),
        color_b=(255, 160, 20),
        key='2',
    ),
    ElementID.FROST: dict(
        name="FROST",
        icon="❄️",
        color=(140, 220, 255),
        color_b=(200, 245, 255),
        key='3',
    ),
    ElementID.THUNDER: dict(
        name="THUNDER",
        icon="⚡",
        color=(180, 120, 255),
        color_b=(220, 200, 255),
        key='4',
    ),
    ElementID.EARTH: dict(
        name="EARTH",
        icon="🪨",
        color=(140, 100, 50),
        color_b=(200, 160, 80),
        key='5',
    ),
    ElementID.WIND: dict(
        name="WIND",
        icon="🌪️",
        color=(180, 240, 200),
        color_b=(230, 255, 220),
        key='6',
    ),
    ElementID.VOID: dict(
        name="VOID",
        icon="🌑",
        color=(130, 40, 200),
        color_b=(60, 10, 120),
        key='7',
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Base Element
# ─────────────────────────────────────────────────────────────────────────────

class BaseElement:
    """
    Common energy state machine shared by all elements.
    Subclasses override _emit_ambient(), _emit_charge(), etc.
    """

    ENERGY_REGEN = 0.18
    ENERGY_DRAIN = 0.08
    POWER_REGEN  = 0.22
    POWER_DRAIN  = 0.12

    # Subclass overrides this to pass the right style
    ELEMENT_STYLE = 'default'

    def __init__(self, element_id: ElementID, composer: EffectComposer,
                 width: int, height: int):
        self.element_id = element_id
        self.composer = composer
        self.width = width
        self.height = height
        info = ELEMENT_INFO[element_id]
        self.name: str = info['name']
        self.icon: str = info['icon']
        self.color: Tuple[int, int, int] = info['color']
        self.color_b: Tuple[int, int, int] = info['color_b']

        self.energy: float = 0.5
        self.power:  float = 0.0
        self.flow:   float = 0.0

        self._orb: Optional[EnergyOrb] = None
        self._trail: Optional[EnergyTrail] = None
        self._ring: Optional[OrbitRing] = None
        self._ring2: Optional[OrbitRing] = None
        self._vortex: Optional[Vortex] = None

        self._time: float = 0.0
        self._emit_timer: float = 0.0
        self._charge_timer: float = 0.0
        self._release_cooldown: float = 0.0

    def enter(self):
        self._spawn_persistent_vfx()

    def exit(self):
        self._kill_persistent_vfx()

    def _spawn_persistent_vfx(self):
        pass

    def _kill_persistent_vfx(self):
        for attr in ('_orb', '_trail', '_ring', '_ring2', '_vortex'):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.alive = False
                setattr(self, attr, None)

    def update(self, dt: float, gs: ElementalGestureState):
        self._time += dt
        self._emit_timer       = max(0.0, self._emit_timer - dt)
        self._release_cooldown = max(0.0, self._release_cooldown - dt)

        px = gs.hand_pos[0] * self.width
        py = gs.hand_pos[1] * self.height

        if gs.gesture == ElementalGesture.OPEN_PALM:
            self.energy = min(1.0, self.energy + self.ENERGY_REGEN * dt)
        elif gs.gesture == ElementalGesture.FIST:
            self.power  = min(1.0, self.power  + self.POWER_REGEN  * dt)
            self.energy = max(0.0, self.energy - self.ENERGY_DRAIN * dt)
        else:
            self.power  = max(0.0, self.power  - self.POWER_DRAIN  * dt)

        speed = gs.speed
        if speed > 0.15:
            self.flow = min(1.0, self.flow + speed * dt * 0.8)
        else:
            self.flow = max(0.0, self.flow - dt * 0.3)

        self._update_element(dt, gs, px, py, speed)

    def _update_element(self, dt, gs, px, py, speed):
        pass

    def render(self, surface: pygame.Surface, gs: ElementalGestureState):
        pass

    def handle_key(self, event: pygame.event.Event) -> bool:
        return False

    def _swipe_velocity(self, gs: ElementalGestureState, base_speed: float = 7.0):
        """
        Compute projectile velocity from the actual swipe direction vector.
        Falls back to center-relative direction if swipe vector is near zero.
        """
        vx = gs.velocity_x
        vy = gs.velocity_y
        mag = math.sqrt(vx * vx + vy * vy)
        if mag > 0.01:
            # Scale so the projectile speed = base_speed pixels-per-frame-tick
            scale = base_speed / max(mag, 0.01)
            return vx * scale, vy * scale
        # Fallback: away from screen center
        fx = (gs.hand_pos[0] - 0.5)
        fy = (gs.hand_pos[1] - 0.5)
        fallback_mag = math.sqrt(fx * fx + fy * fy) + 0.001
        return (fx / fallback_mag) * base_speed, (fy / fallback_mag) * base_speed


# ─────────────────────────────────────────────────────────────────────────────
# 1 — Phoenix Flame
# ─────────────────────────────────────────────────────────────────────────────

class PhoenixFlame(BaseElement):
    ELEMENT_STYLE = 'fire'
    HOT   = (255, 100, 20)
    WARM  = (255, 220, 60)
    EMBER = (255, 60, 10)

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=28, color=self.HOT,
                                             element_style='fire')
        self._trail = self.composer.spawn_trail(color=self.HOT, max_len=24,
                                                 base_width=8, glow=True,
                                                 element_style='fire')
        self._ring = self.composer.spawn_ring(640, 360, radius=55, color=self.WARM,
                                               node_count=12, speed=2.4, tilt=0.5)

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            charge_scale = 0.6 + self.power * 0.8
            self._orb.radius = (22 + self.energy * 32) * charge_scale
            self._orb.color = lerp_color(self.HOT, self.WARM, self.power)

        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 42 + self.power * 65
            self._ring.speed  = 1.8 + self.power * 2.5

        if self._trail:
            self._trail.color = lerp_color(self.HOT, self.WARM, self.power)
            self._trail.push(px, py)
            self._trail.update(dt)

        # Continuous rising flame particles
        if self._emit_timer <= 0:
            intensity = 0.5 + self.energy * 0.7 + speed * 0.5
            count = int(4 + intensity * 5)
            self.composer.emit_rising_flames(px, py, count=count,
                                              color_hot=self.HOT, color_cool=(80, 10, 0),
                                              intensity=min(2.2, intensity))
            # Ember sparks scatter sideways
            if self.energy > 0.3:
                self.composer.emit_sparks(px, py, count=2, color=self.EMBER, speed=4.0)
            self._emit_timer = 0.035

        # Fist charge: more embers + spiral ring
        if gs.gesture == ElementalGesture.FIST and self.power > 0.3:
            self.composer.emit_sparks(px, py, count=4, color=self.EMBER, speed=4.5)

        # Pinch: dense fireball compression
        if gs.gesture == ElementalGesture.PINCH and self._emit_timer <= 0:
            pr = max(0.3, 1.0 - gs.pinch_ratio)
            self.composer.emit_radial_burst(px, py, count=6, speed=pr * 2.5,
                                             color_a=self.HOT, color_b=self.WARM,
                                             life=0.4, size=6, glow=True)

        # Swipe: launch fire projectile in actual swipe direction
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            vx, vy = self._swipe_velocity(gs, base_speed=8.0)
            self.composer.spawn_projectile(px, py, vx, vy,
                                            color=self.HOT, radius=18, lifetime=1.6,
                                            element_style='fire')
            self.composer.spawn_shockwave(px, py, max_radius=90, duration=0.38,
                                           color=self.WARM, width=3)
            self.composer.emit_radial_burst(px, py, count=20, speed=6.0,
                                             color_a=self.HOT, color_b=self.WARM,
                                             life=0.5, size=8, glow=True, gravity=0.03)
            self.energy = max(0.0, self.energy - 0.25)
            self._release_cooldown = 0.5

        # Fast movement: intense flame trail
        if speed > 0.35:
            burst = int(speed * 8)
            self.composer.emit_rising_flames(px, py, count=burst,
                                              color_hot=self.HOT, color_cool=(120, 20, 0),
                                              intensity=1.4)

    def render(self, surface, gs):
        if self._orb and self.energy > 0.25:
            px, py = int(self._orb.x), int(self._orb.y)
            # Heat-shimmer lines rising from flame
            for _ in range(int(self.energy * 8)):
                ox = px + random.randint(-40, 40)
                oy = py - random.randint(5, 50)
                col = lerp_color((40, 5, 0), self.HOT, random.random())
                try:
                    pygame.draw.line(surface, col,
                                     (ox, oy), (ox + random.randint(-5, 5), oy - 8), 1)
                except Exception:
                    pass
            # Combustion ring around fist charge
            if gs.gesture == ElementalGesture.FIST and self.power > 0.5:
                cr = int(40 + self.power * 50)
                ring_col = lerp_color(self.HOT, self.WARM, self.power)
                try:
                    pygame.draw.circle(surface, ring_col, (px, py), cr, 2)
                    pygame.draw.circle(surface, lerp_color((0, 0, 0), self.HOT, 0.4),
                                       (px, py), cr + 6, 1)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Golden Solar
# ─────────────────────────────────────────────────────────────────────────────

class GoldenSolar(BaseElement):
    ELEMENT_STYLE = 'solar'
    GOLD   = (255, 215, 0)
    ORANGE = (255, 150, 20)
    WHITE  = (255, 255, 200)

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=38, color=self.GOLD,
                                             pulse_speed=2.0, element_style='solar')
        self._ring  = self.composer.spawn_ring(640, 360, radius=75, color=self.GOLD,
                                                node_count=16, speed=1.1, tilt=0.6)
        self._ring2 = self.composer.spawn_ring(640, 360, radius=115, color=self.ORANGE,
                                                node_count=10, speed=-0.7, tilt=0.8)

    def _update_element(self, dt, gs, px, py, speed):
        for ring in [self._ring, self._ring2]:
            if ring:
                ring.cx, ring.cy = px, py

        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 28 + self.energy * 42 + self.power * 22
            self._orb.color = lerp_color(self.ORANGE, self.WHITE, self.power)

        # Two-hand solar field scaling
        if gs.two_hands:
            scale = 1.0 + gs.two_hand_distance * 2.8
            if self._ring:
                self._ring.radius = 75 * scale
            if self._ring2:
                self._ring2.radius = 115 * scale

        # Orbiting solar particles
        if self._emit_timer <= 0:
            count = int(3 + self.energy * 6)
            self.composer.emit_orbiting(px, py, count=count,
                                         radius=65 + self.energy * 55,
                                         color=self.GOLD, life=1.6, speed=2.0)
            # Outward energy pulse particles
            self.composer.emit_radial_burst(px, py, count=2, speed=1.5,
                                             color_a=self.GOLD, color_b=self.WHITE,
                                             life=0.5, size=3, glow=True)
            self._emit_timer = 0.10

        # Swipe: radial solar blast
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.15:
            for angle_deg in range(0, 360, 24):
                a = math.radians(angle_deg)
                self.composer.spawn_projectile(px, py, math.cos(a) * 6.5, math.sin(a) * 6.5,
                                                color=self.GOLD, radius=11, lifetime=0.8,
                                                element_style='solar')
            self.composer.spawn_shockwave(px, py, max_radius=130, duration=0.55,
                                           color=self.GOLD, width=4, glow_rings=3)
            self.composer.emit_radial_burst(px, py, count=35, speed=5.5,
                                             color_a=self.GOLD, color_b=self.WHITE,
                                             life=0.7, size=5, glow=True)
            self.energy = max(0.0, self.energy - 0.3)
            self._release_cooldown = 0.65

        # Pinch: compress solar core
        if gs.gesture == ElementalGesture.PINCH:
            self.composer.emit_radial_burst(px, py, count=8, speed=0.8,
                                             color_a=self.GOLD, color_b=self.WHITE,
                                             life=0.3, size=3, glow=True)

    def render(self, surface, gs):
        if self._orb and self.energy > 0.2:
            ix, iy = int(self._orb.x), int(self._orb.y)
            ray_count = int(10 + self.energy * 10)
            for i in range(ray_count):
                angle = (i / ray_count) * math.tau + self._time * 0.6
                length = (35 + self.energy * 70) * (0.8 + 0.2 * math.sin(self._time * 3 + i))
                ex = ix + math.cos(angle) * length
                ey = iy + math.sin(angle) * length
                col = (min(255, self.GOLD[0]), min(255, self.GOLD[1]),
                       max(0, self.GOLD[2] - 30))
                try:
                    pygame.draw.line(surface, col, (ix, iy), (int(ex), int(ey)), 1)
                except Exception:
                    pass
            # Two-hand solar field connector visualization
            if gs.two_hands and gs.hand2_pos:
                x2 = int(gs.hand2_pos[0] * self.width)
                y2 = int(gs.hand2_pos[1] * self.height)
                mid_col = lerp_color(self.GOLD, self.WHITE, 0.5)
                try:
                    pygame.draw.line(surface, mid_col, (ix, iy), (x2, y2), 1)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Frost
# ─────────────────────────────────────────────────────────────────────────────

class Frost(BaseElement):
    ELEMENT_STYLE = 'frost'
    ICE     = (140, 220, 255)
    SNOW    = (220, 248, 255)
    CRYSTAL = (180, 240, 255)

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=24, color=self.ICE,
                                             pulse_speed=1.2, element_style='frost')
        self._ring = self.composer.spawn_ring(640, 360, radius=58, color=self.ICE,
                                               node_count=6, speed=0.4, tilt=0.5)
        self._trail = self.composer.spawn_trail(color=self.ICE, max_len=20,
                                                 base_width=6, glow=True,
                                                 element_style='frost')

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 20 + self.energy * 28 + self.power * 16
            self._orb.color = lerp_color(self.ICE, self.SNOW, self.power)
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 48 + self.power * 42
            self._ring.node_count = max(3, int(5 + self.power * 7))
        if self._trail:
            self._trail.push(px, py)
            self._trail.update(dt)

        # Ice crystal ambient
        if self._emit_timer <= 0:
            count = int(1 + self.energy * 4)
            for _ in range(count):
                ox = px + random.uniform(-45, 45)
                oy = py + random.uniform(-22, 22)
                self.composer.pool.emit(
                    x=ox, y=oy,
                    vx=random.uniform(-0.5, 0.5),
                    vy=random.uniform(-1.5, 0.2),
                    life=random.uniform(0.7, 1.4),
                    size=random.uniform(3, 7),
                    min_size=0.5,
                    rotation=random.uniform(0, math.tau),
                    angular_velocity=random.uniform(-0.08, 0.08),
                    color=self.ICE,
                    color_end=self.SNOW,
                    gravity=0.025,
                    drag=0.97,
                    glow=True,
                    shape='shard',
                )
            self._emit_timer = 0.07

        # Fast movement: ice shard spray
        if speed > 0.3:
            self.composer.emit_crystals(px, py, count=int(speed * 5),
                                         color=self.ICE, speed=speed * 3.5)

        # Pinch: ice crystal forms between fingers
        if gs.gesture == ElementalGesture.PINCH and self._emit_timer <= 0:
            self.composer.emit_radial_burst(px, py, count=6, speed=1.5,
                                             color_a=self.ICE, color_b=self.SNOW,
                                             life=0.4, size=5, glow=True, shape='shard')

        # Swipe: frost burst + ice projectile
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.15:
            vx, vy = self._swipe_velocity(gs, base_speed=8.5)
            self.composer.spawn_projectile(px, py, vx, vy,
                                            color=self.CRYSTAL, radius=20, lifetime=1.4,
                                            element_style='frost')
            self.composer.emit_crystals(px, py, count=45, color=self.CRYSTAL, speed=6.0)
            self.composer.spawn_shockwave(px, py, max_radius=150, duration=0.6,
                                           color=self.ICE, width=3, glow_rings=2)
            self.energy = max(0.0, self.energy - 0.3)
            self._release_cooldown = 0.55

    def render(self, surface, gs):
        if self._orb and self.power > 0.4:
            px, py = int(self._orb.x), int(self._orb.y)
            # Frost ring burst during charge
            n = int(3 + self.power * 5)
            for i in range(n):
                a = (i / n) * math.tau + self._time * 0.3
                r = int(30 + self.power * 50 + 8 * math.sin(self._time * 2 + i))
                tip = (int(px + math.cos(a) * r), int(py + math.sin(a) * r * 0.7))
                b1 = (int(px + math.cos(a + 0.4) * r * 0.5),
                       int(py + math.sin(a + 0.4) * r * 0.35))
                b2 = (int(px + math.cos(a - 0.4) * r * 0.5),
                       int(py + math.sin(a - 0.4) * r * 0.35))
                try:
                    pygame.draw.polygon(surface, lerp_color(self.ICE, self.SNOW, self.power),
                                        [tip, b1, b2], 1)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Thunder
# ─────────────────────────────────────────────────────────────────────────────

class Thunder(BaseElement):
    ELEMENT_STYLE = 'thunder'
    PURPLE     = (180, 100, 255)
    VIOLET     = (220, 180, 255)
    WHITE      = (240, 230, 255)
    CHARGE_COL = (140, 60, 220)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arc_timer    = 0.0
        self._strike_timer = 0.0

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=22, color=self.PURPLE,
                                             pulse_speed=6.0, element_style='thunder')
        self._trail = self.composer.spawn_trail(color=self.PURPLE, max_len=18,
                                                 base_width=5, glow=True,
                                                 element_style='thunder')

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 16 + self.energy * 24 + self.power * 22
            self._orb.color = lerp_color(self.PURPLE, self.WHITE, self.power)

        if self._trail:
            self._trail.push(px, py)
            self._trail.update(dt)

        self._arc_timer    = max(0.0, self._arc_timer    - dt)
        self._strike_timer = max(0.0, self._strike_timer - dt)

        # Ambient electrical arcs from hand center
        if self._arc_timer <= 0 and self.energy > 0.1:
            num_arcs = int(2 + self.energy * 4 + self.power * 3)
            for _ in range(num_arcs):
                angle = random.uniform(0, math.tau)
                dist  = 25 + self.energy * 65 + self.power * 45
                x2 = px + math.cos(angle) * dist
                y2 = py + math.sin(angle) * dist
                self.composer.spawn_lightning(
                    px, py, x2, y2,
                    color=lerp_color(self.PURPLE, self.WHITE, self.power),
                    duration=0.08 + self.power * 0.07,
                    branches=int(1 + self.power * 4),
                    jaggedness=18 + self.power * 22,
                    width=1 + int(self.power * 2),
                )
            self._arc_timer = 0.05 + (1.0 - self.energy) * 0.08

        # Particle sparks during movement
        if speed > 0.2 and self._emit_timer <= 0:
            self.composer.emit_sparks(px, py, count=int(speed * 6),
                                       color=self.PURPLE, speed=4.5)
            self._emit_timer = 0.04

        # Fist: charge strike bolts
        if gs.gesture == ElementalGesture.FIST and self.power > 0.3:
            if self._strike_timer <= 0:
                for _ in range(2):
                    x2 = px + random.uniform(-70, 70)
                    y2 = py - random.uniform(25, 85)
                    self.composer.spawn_lightning(px, py, x2, y2,
                                                   color=self.WHITE, duration=0.14,
                                                   branches=5, jaggedness=32, width=3)
                self._strike_timer = 0.10

        # Swipe: high-voltage projectile
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            vx, vy = self._swipe_velocity(gs, base_speed=10.0)
            self.composer.spawn_projectile(px, py, vx, vy,
                                            color=self.PURPLE, radius=16, lifetime=1.3,
                                            element_style='thunder')
            for _ in range(8):
                a = random.uniform(0, math.tau)
                d = random.uniform(45, 130)
                self.composer.spawn_lightning(px, py,
                                               px + math.cos(a) * d,
                                               py + math.sin(a) * d,
                                               color=self.WHITE, duration=0.17,
                                               branches=3, jaggedness=28, width=2)
            self.composer.spawn_shockwave(px, py, max_radius=110, duration=0.42,
                                           color=self.PURPLE, width=3)
            self.energy = max(0.0, self.energy - 0.3)
            self._release_cooldown = 0.55

    def render(self, surface, gs):
        if self._orb and self.power > 0.4:
            px, py = int(self._orb.x), int(self._orb.y)
            # Visible charge ring
            cr = int(35 + self.power * 55)
            try:
                pygame.draw.circle(surface, lerp_color(self.PURPLE, self.WHITE, self.power),
                                   (px, py), cr, 2)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Earth
# ─────────────────────────────────────────────────────────────────────────────

class Earth(BaseElement):
    ELEMENT_STYLE = 'earth'
    STONE = (140, 100, 50)
    DIRT  = (100, 70, 35)
    SAND  = (200, 165, 80)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debris_timer = 0.0

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=26, color=self.STONE,
                                             pulse_speed=1.0, element_style='earth')
        self._ring = self.composer.spawn_ring(640, 360, radius=55, color=self.STONE,
                                               node_count=7, speed=0.8, tilt=0.35)

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 22 + self.energy * 38 + self.power * 28
            self._orb.color = lerp_color(self.DIRT, self.STONE, self.energy)

        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 44 + self.energy * 44 + self.power * 32
            self._ring.node_count = max(4, int(5 + self.power * 9))

        self._debris_timer = max(0.0, self._debris_timer - dt)

        # Orbiting rock debris (irregular polygon shapes)
        if self._debris_timer <= 0:
            count = int(1 + self.energy * 3)
            radius = 38 + self.energy * 48
            for i in range(count):
                angle = random.uniform(0, math.tau)
                spd   = random.uniform(0.7, 1.5)
                size  = random.uniform(5, 12)
                self.composer.pool.emit(
                    x=px + math.cos(angle) * radius,
                    y=py + math.sin(angle) * radius,
                    orbit_cx=px, orbit_cy=py,
                    orbit_radius=radius * random.uniform(0.85, 1.15),
                    orbit_speed=spd,
                    orbit_angle=angle,
                    life=random.uniform(1.2, 2.2),
                    size=size,
                    min_size=2.0,
                    color=lerp_color(self.DIRT, self.STONE, random.random()),
                    color_end=(45, 32, 12),
                    shape='square',
                )
            self._debris_timer = 0.12

        # Dust cloud ambient
        if self._emit_timer <= 0 and self.energy > 0.15:
            self.composer.pool.emit(
                x=px + random.uniform(-28, 28),
                y=py + random.uniform(-12, 12),
                vx=random.uniform(-0.7, 0.7),
                vy=random.uniform(-0.4, 0.4),
                life=random.uniform(0.4, 1.0),
                size=random.uniform(4, 9),
                min_size=0.5,
                color=self.SAND,
                color_end=(38, 28, 12),
                gravity=0.05,
                drag=0.97,
                turbulence=0.6,
            )
            self._emit_timer = 0.055

        # Pinch: compress stone fragments
        if gs.gesture == ElementalGesture.PINCH:
            self.composer.emit_radial_burst(px, py, count=5, speed=1.2,
                                             color_a=self.STONE, color_b=self.SAND,
                                             life=0.45, size=7, shape='square')

        # Swipe: shockwave + rock debris
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            vx, vy = self._swipe_velocity(gs, base_speed=7.0)
            self.composer.spawn_projectile(px, py, vx, vy,
                                            color=self.STONE, radius=22, lifetime=1.2,
                                            element_style='earth')
            self.composer.emit_rock_chunks(px, py, count=30, color=self.STONE, speed=5.5)
            self.composer.spawn_shockwave(px, py, max_radius=170, duration=0.68,
                                           color=self.STONE, width=5, glow_rings=2)
            self.energy = max(0.0, self.energy - 0.35)
            self._release_cooldown = 0.7

    def render(self, surface, gs):
        if self._orb and self.energy > 0.2:
            px, py = int(self._orb.x), int(self._orb.y)
            # Dust wisps
            for _ in range(int(self.energy * 4)):
                ox = px + random.randint(-30, 30)
                oy = py + random.randint(-15, 15)
                col = lerp_color(self.DIRT, self.SAND, random.random())
                try:
                    pygame.draw.circle(surface, col, (ox, oy), random.randint(2, 4))
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Wind
# ─────────────────────────────────────────────────────────────────────────────

class Wind(BaseElement):
    ELEMENT_STYLE = 'wind'
    TEAL  = (160, 235, 180)
    WHITE = (235, 255, 230)
    MIST  = (200, 250, 210)

    def _spawn_persistent_vfx(self):
        self._vortex = self.composer.spawn_vortex(640, 360, radius=75, color=self.TEAL,
                                                   arms=3, speed=3.2)
        self._ring = self.composer.spawn_ring(640, 360, radius=95, color=self.TEAL,
                                               node_count=22, speed=-1.3, tilt=0.65)
        self._trail = self.composer.spawn_trail(color=self.TEAL, max_len=22,
                                                 base_width=6, glow=True,
                                                 element_style='wind')

    def _update_element(self, dt, gs, px, py, speed):
        if self._vortex:
            self._vortex.cx, self._vortex.cy = px, py
            self._vortex.radius = 55 + self.energy * 65 + self.power * 32
            self._vortex.speed  = 2.5 + self.energy * 2.2 + speed * 2.0
            self._vortex.color  = lerp_color(self.TEAL, self.WHITE, self.power)
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 85 + self.energy * 42
            self._ring.speed  = -(1.2 + self.energy * 1.7)
        if self._trail:
            self._trail.push(px, py)
            self._trail.update(dt)

        # Curved ribbon spiral particles
        if self._emit_timer <= 0:
            count = int(3 + self.energy * 5 + speed * 3)
            angle_base = self._time * 3.8
            for i in range(count):
                a = angle_base + (i / max(1, count)) * math.tau
                r = 22 + self.energy * 55
                ex = px + math.cos(a) * r
                ey = py + math.sin(a) * r
                self.composer.pool.emit(
                    x=ex, y=ey,
                    vx=math.cos(a + math.pi * 0.5) * (2.0 + speed * 1.5),
                    vy=math.sin(a + math.pi * 0.5) * (2.0 + speed * 1.5),
                    life=random.uniform(0.5, 1.1),
                    size=random.uniform(2, 6),
                    min_size=0.5,
                    color=self.TEAL,
                    color_end=self.WHITE,
                    drag=0.97,
                    trail=True,
                    turbulence=0.35,
                )
            self._emit_timer = 0.045

        # Swipe: compressed wind blade
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.15:
            vx, vy = self._swipe_velocity(gs, base_speed=9.5)
            self.composer.spawn_projectile(px, py, vx, vy,
                                            color=self.TEAL, radius=20, lifetime=1.1,
                                            element_style='wind')
            self.composer.spawn_shockwave(px, py, max_radius=140, duration=0.52,
                                           color=self.TEAL, width=2, glow_rings=4)
            self.composer.emit_radial_burst(px, py, count=22, speed=5.5,
                                             color_a=self.TEAL, color_b=self.WHITE,
                                             life=0.65, size=4, trail=True, drag=0.97)
            self.energy = max(0.0, self.energy - 0.28)
            self._release_cooldown = 0.5

    def render(self, surface, gs):
        if self._orb:
            pass  # Wind uses the vortex as its primary visual


# ─────────────────────────────────────────────────────────────────────────────
# 7 — Void
# ─────────────────────────────────────────────────────────────────────────────

class Void(BaseElement):
    ELEMENT_STYLE = 'void'
    DARK_PURPLE = (80, 20, 140)
    VIOLET      = (160, 60, 255)
    BLACK_GLOW  = (40, 10, 80)
    DISTORT     = (200, 150, 255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._implode_state = False
        self._implode_timer = 0.0

    def _spawn_persistent_vfx(self):
        self._vortex = self.composer.spawn_vortex(640, 360, radius=65, color=self.VIOLET,
                                                   arms=4, speed=-3.0)
        self._ring = self.composer.spawn_ring(640, 360, radius=85, color=self.DARK_PURPLE,
                                               node_count=14, speed=-2.0, tilt=0.55)
        self._orb = self.composer.spawn_orb(640, 360, radius=20, color=self.DARK_PURPLE,
                                             pulse_speed=1.5, element_style='void')
        self._trail = self.composer.spawn_trail(color=self.VIOLET, max_len=20,
                                                 base_width=5, glow=True,
                                                 element_style='void')

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 17 + self.energy * 22 + self.power * 28
            self._orb.color = lerp_color(self.DARK_PURPLE, self.DISTORT, self.power)
        if self._vortex:
            self._vortex.cx, self._vortex.cy = px, py
            self._vortex.radius = 58 + self.energy * 58 + self.power * 42
            self._vortex.speed  = -(2.2 + self.power * 2.8)
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 75 + self.energy * 52
            self._ring.speed  = -(1.6 + self.power * 1.6)
        if self._trail:
            self._trail.push(px, py)
            self._trail.update(dt)

        self._implode_timer = max(0.0, self._implode_timer - dt)

        # Gravitational inward spiral
        if self._emit_timer <= 0:
            count = int(2 + self.energy * 6)
            self.composer.emit_void_fragments(px, py, count=count,
                                               color=self.VIOLET,
                                               radius=85 + self.energy * 55)
            self._emit_timer = 0.09

        # Fist — distortion burst
        if gs.gesture == ElementalGesture.FIST and self._emit_timer <= 0:
            self.composer.emit_radial_burst(px, py, count=7, speed=1.2,
                                             color_a=self.VIOLET, color_b=self.DARK_PURPLE,
                                             life=0.5, size=5, glow=True, shape='ring',
                                             drag=0.92)

        # Swipe: implosion → explosion
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            self._implode_state = True
            self._implode_timer = 0.4

        if self._implode_state and self._implode_timer <= 0:
            self._implode_state = False
            self.composer.emit_radial_burst(px, py, count=65, speed=7.5,
                                             color_a=self.VIOLET, color_b=self.DISTORT,
                                             life=1.1, size=6, glow=True, shape='ring',
                                             drag=0.93)
            self.composer.spawn_shockwave(px, py, max_radius=210, duration=0.75,
                                           color=self.VIOLET, width=4, glow_rings=3)
            # Add inward projectile "implosion" shards
            vx, vy = self._swipe_velocity(gs, base_speed=8.0)
            self.composer.spawn_projectile(px, py, vx, vy,
                                            color=self.VIOLET, radius=18, lifetime=1.0,
                                            element_style='void')
            self.energy = max(0.0, self.energy - 0.4)
            self._release_cooldown = 0.7

        if self._implode_state:
            # Particles flying inward
            if self._emit_timer <= 0:
                for _ in range(10):
                    a = random.uniform(0, math.tau)
                    r = random.uniform(65, 160)
                    self.composer.pool.emit(
                        x=px + math.cos(a) * r,
                        y=py + math.sin(a) * r,
                        vx=0.0, vy=0.0,
                        attract_x=px, attract_y=py,
                        attract_strength=0.10,
                        life=0.38,
                        size=random.uniform(4, 8),
                        min_size=0.5,
                        color=self.VIOLET,
                        color_end=self.DARK_PURPLE,
                        drag=0.94,
                        glow=True,
                    )
                self._emit_timer = 0.035

    def render(self, surface, gs):
        if self._orb and self.power > 0.25:
            px, py = int(self._orb.x), int(self._orb.y)
            rings = int(self.power * 5)
            for i in range(1, rings + 1):
                r = int(i * 22 * (0.8 + 0.2 * math.sin(self._time * 4.5 + i)))
                if r > 0:
                    col = lerp_color(self.DARK_PURPLE, self.DISTORT, i / (rings + 1))
                    try:
                        pygame.draw.circle(surface, col, (px, py), r, 1)
                    except Exception:
                        pass


# ─────────────────────────────────────────────────────────────────────────────
# Element Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_element(element_id: ElementID, composer: EffectComposer,
                   width: int, height: int) -> BaseElement:
    cls_map = {
        ElementID.PHOENIX_FLAME: PhoenixFlame,
        ElementID.GOLDEN_SOLAR:  GoldenSolar,
        ElementID.FROST:         Frost,
        ElementID.THUNDER:       Thunder,
        ElementID.EARTH:         Earth,
        ElementID.WIND:          Wind,
        ElementID.VOID:          Void,
    }
    return cls_map[element_id](element_id, composer, width, height)
