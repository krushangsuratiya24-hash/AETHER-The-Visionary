"""
AETHER — Elemental Cultivation: Seven Element Definitions
Each element owns its own energy state machine and provides
update() / render() hooks that speak to the shared EffectComposer.
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
    Subclasses override _emit_ambient(), _emit_charge(), _emit_release(),
    _emit_sustained_trail(), etc.
    """

    ENERGY_REGEN = 0.18   # units/s when palm open
    ENERGY_DRAIN = 0.08   # units/s during charge
    POWER_REGEN  = 0.22   # units/s while fist
    POWER_DRAIN  = 0.12   # units/s passive

    def __init__(self, element_id: ElementID, composer: EffectComposer,
                 width: int, height: int):
        self.element_id = element_id
        self.composer = composer
        self.width = width
        self.height = height
        info = ELEMENT_INFO[element_id]
        self.name: str = info['name']
        self.icon: str = info['icon']
        self.color: Tuple[int,int,int] = info['color']
        self.color_b: Tuple[int,int,int] = info['color_b']

        # Energy/Power bars [0..1]
        self.energy: float = 0.5
        self.power:  float = 0.0
        self.flow:   float = 0.0   # combo/flow accumulator

        # Persistent VFX objects
        self._orb: Optional[EnergyOrb] = None
        self._trail: Optional[EnergyTrail] = None
        self._ring: Optional[OrbitRing] = None
        self._ring2: Optional[OrbitRing] = None
        self._vortex: Optional[Vortex] = None

        # Timers
        self._time: float = 0.0
        self._emit_timer: float = 0.0
        self._charge_timer: float = 0.0
        self._release_cooldown: float = 0.0

    # ── Life cycle ────────────────────────────────────────────────────────────

    def enter(self):
        """Called when this element is selected."""
        self._spawn_persistent_vfx()

    def exit(self):
        """Called when switching away from this element."""
        self._kill_persistent_vfx()

    def _spawn_persistent_vfx(self):
        pass  # Subclasses override

    def _kill_persistent_vfx(self):
        if self._orb:
            self._orb.alive = False
            self._orb = None
        if self._trail:
            self._trail.alive = False
            self._trail = None
        if self._ring:
            self._ring.alive = False
            self._ring = None
        if self._ring2:
            self._ring2.alive = False
            self._ring2 = None
        if self._vortex:
            self._vortex.alive = False
            self._vortex = None

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float, gs: ElementalGestureState):
        self._time += dt
        self._emit_timer  = max(0.0, self._emit_timer  - dt)
        self._release_cooldown = max(0.0, self._release_cooldown - dt)

        px = gs.hand_pos[0] * self.width
        py = gs.hand_pos[1] * self.height

        gesture = gs.gesture
        speed   = gs.speed

        # Energy/Power bookkeeping
        if gesture == ElementalGesture.OPEN_PALM:
            self.energy = min(1.0, self.energy + self.ENERGY_REGEN * dt)
        elif gesture == ElementalGesture.FIST:
            self.power  = min(1.0, self.power  + self.POWER_REGEN  * dt)
            self.energy = max(0.0, self.energy - self.ENERGY_DRAIN * dt)
        else:
            self.power  = max(0.0, self.power  - self.POWER_DRAIN  * dt)

        # Flow/combo — builds when hand moving
        if speed > 0.15:
            self.flow = min(1.0, self.flow + speed * dt * 0.8)
        else:
            self.flow = max(0.0, self.flow - dt * 0.3)

        # Subclass logic
        self._update_element(dt, gs, px, py, speed)

    def _update_element(self, dt: float, gs: ElementalGestureState,
                        px: float, py: float, speed: float):
        pass  # Subclasses override

    def render(self, surface: pygame.Surface, gs: ElementalGestureState):
        pass  # Subclasses may draw extra geometry

    def handle_key(self, event: pygame.event.Event) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1 — Phoenix Flame
# ─────────────────────────────────────────────────────────────────────────────

class PhoenixFlame(BaseElement):
    HOT   = (255, 100, 20)
    WARM  = (255, 220, 60)
    EMBER = (255, 60, 10)

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=25, color=self.HOT)
        self._trail = self.composer.spawn_trail(color=self.HOT, max_len=20, base_width=8, glow=True)
        self._ring = self.composer.spawn_ring(640, 360, radius=50, color=self.WARM, node_count=12, speed=2.2)

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            charge_scale = 0.6 + self.power * 0.8
            self._orb.radius = (20 + self.energy * 30) * charge_scale
            self._orb.color = lerp_color(self.HOT, self.WARM, self.power)

        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 40 + self.power * 60
            self._ring.speed = 1.5 + self.power * 2.5

        if self._trail:
            self._trail.color = lerp_color(self.HOT, self.WARM, self.power)
            self._trail.push(px, py)
            self._trail.update(dt)

        # Continuous ambient flame
        if self._emit_timer <= 0:
            intensity = 0.4 + self.energy * 0.6 + speed * 0.4
            self.composer.emit_rising_flames(px, py, count=int(3 + intensity * 4),
                                              color_hot=self.HOT, color_cool=(80, 10, 0),
                                              intensity=min(2.0, intensity))
            self._emit_timer = 0.04

        # Embers during charge
        if gs.gesture == ElementalGesture.FIST and self.power > 0.3:
            self.composer.emit_sparks(px, py, count=3, color=self.EMBER, speed=3.5)

        # Pinch: fire sphere
        if gs.gesture == ElementalGesture.PINCH:
            self.composer.emit_radial_burst(px, py, count=4, speed=1.5,
                                             color_a=self.HOT, color_b=self.WARM,
                                             life=0.4, size=5, glow=True)

        # Swipe: launch projectile
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            if len(self.composer.composer.projectiles if hasattr(self.composer, 'composer') else self.composer.projectiles) < 5:
                vx = (0.5 - gs.hand_pos[0]) * -8.0 + random.uniform(-1, 1)
                vy = (0.5 - gs.hand_pos[1]) * -8.0 + random.uniform(-1, 1)
                self.composer.spawn_projectile(px, py, vx, vy,
                                               color=self.HOT, radius=16, lifetime=1.5)
                self.composer.spawn_shockwave(px, py, max_radius=80, duration=0.35,
                                              color=self.WARM, width=3)
                self.energy = max(0.0, self.energy - 0.25)
                self._release_cooldown = 0.5

        # Fast movement: extra fire trail
        if speed > 0.4 and self._emit_timer <= 0:
            burst = int(speed * 6)
            self.composer.emit_rising_flames(px, py, count=burst,
                                              color_hot=self.HOT, color_cool=(120, 20, 0),
                                              intensity=1.2)

    def render(self, surface, gs):
        # Heat-distortion illusion: flickery thin lines above orb
        if self._orb and self.energy > 0.3:
            px, py = int(self._orb.x), int(self._orb.y)
            for _ in range(int(self.energy * 6)):
                ox = px + random.randint(-30, 30)
                oy = py - random.randint(10, 50)
                col = lerp_color((60, 10, 0), self.HOT, random.random())
                pygame.draw.line(surface, col, (ox, oy), (ox + random.randint(-4, 4), oy - 6), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Golden Solar
# ─────────────────────────────────────────────────────────────────────────────

class GoldenSolar(BaseElement):
    GOLD   = (255, 215, 0)
    ORANGE = (255, 150, 20)
    WHITE  = (255, 255, 200)

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=35, color=self.GOLD, pulse_speed=2.0)
        self._ring  = self.composer.spawn_ring(640, 360, radius=70, color=self.GOLD,
                                                node_count=16, speed=1.0, tilt=0.5)
        self._ring2 = self.composer.spawn_ring(640, 360, radius=110, color=self.ORANGE,
                                                node_count=10, speed=-0.6, tilt=0.7)

    def _update_element(self, dt, gs, px, py, speed):
        for ring in [self._ring, self._ring2]:
            if ring:
                ring.cx, ring.cy = px, py
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 25 + self.energy * 40 + self.power * 20
            self._orb.color = lerp_color(self.ORANGE, self.WHITE, self.power)

        # Scale with two hands
        if gs.two_hands and self._ring:
            scale = 1.0 + gs.two_hand_distance * 2.5
            self._ring.radius  = 70  * scale
            self._ring2.radius = 110 * scale

        # Ambient solar particles
        if self._emit_timer <= 0:
            count = int(2 + self.energy * 5)
            self.composer.emit_orbiting(px, py, count=count, radius=60 + self.energy * 50,
                                         color=self.GOLD, life=1.5, speed=1.8)
            self._emit_timer = 0.12

        # Radial ray burst on swipe/release
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.15:
            for angle_deg in range(0, 360, 30):
                a = math.radians(angle_deg)
                vx = math.cos(a) * 6.0
                vy = math.sin(a) * 6.0
                self.composer.spawn_projectile(px, py, vx, vy,
                                               color=self.GOLD, radius=10, lifetime=0.7)
            self.composer.spawn_shockwave(px, py, max_radius=120, duration=0.5,
                                          color=self.GOLD, width=4, glow_rings=3)
            self.composer.emit_radial_burst(px, py, count=30, speed=5.0,
                                             color_a=self.GOLD, color_b=self.WHITE,
                                             life=0.6, size=4, glow=True)
            self.energy = max(0.0, self.energy - 0.3)
            self._release_cooldown = 0.6

        # Pinch: compress the core
        if gs.gesture == ElementalGesture.PINCH:
            self.composer.emit_radial_burst(px, py, count=6, speed=0.8,
                                             color_a=self.GOLD, color_b=self.WHITE,
                                             life=0.3, size=3, glow=True)

    def render(self, surface, gs):
        if self._orb and self.energy > 0.25:
            # Draw radial "ray" lines
            ix, iy = int(self._orb.x), int(self._orb.y)
            ray_count = int(8 + self.energy * 8)
            for i in range(ray_count):
                angle = (i / ray_count) * math.tau + self._time * 0.5
                length = (30 + self.energy * 60) * (0.8 + 0.2 * math.sin(self._time * 3 + i))
                ex = ix + math.cos(angle) * length
                ey = iy + math.sin(angle) * length
                alpha_f = int(120 * self.energy)
                col = (min(255, self.GOLD[0]), min(255, self.GOLD[1]), max(0, self.GOLD[2] - 50))
                try:
                    pygame.draw.line(surface, col, (ix, iy), (int(ex), int(ey)), 1)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Frost
# ─────────────────────────────────────────────────────────────────────────────

class Frost(BaseElement):
    ICE    = (140, 220, 255)
    SNOW   = (220, 248, 255)
    CRYSTAL = (180, 240, 255)

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=22, color=self.ICE, pulse_speed=1.2)
        self._ring = self.composer.spawn_ring(640, 360, radius=55, color=self.ICE,
                                               node_count=6, speed=0.4)

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 18 + self.energy * 25 + self.power * 15
            self._orb.color = lerp_color(self.ICE, self.SNOW, self.power)
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 45 + self.power * 40
            self._ring.node_count = max(3, int(5 + self.power * 8))

        # Falling crystal ambient
        if self._emit_timer <= 0:
            count = int(1 + self.energy * 3)
            for _ in range(count):
                ox = px + random.uniform(-40, 40)
                oy = py + random.uniform(-20, 20)
                self.composer.pool.emit(
                    x=ox, y=oy,
                    vx=random.uniform(-0.4, 0.4),
                    vy=random.uniform(-1.2, 0.3),
                    life=random.uniform(0.6, 1.3),
                    size=random.uniform(2, 6),
                    min_size=0.5,
                    rotation=random.uniform(0, math.tau),
                    angular_velocity=random.uniform(-0.06, 0.06),
                    color=self.ICE,
                    color_end=self.SNOW,
                    gravity=0.02,
                    drag=0.97,
                    glow=True,
                    shape='shard',
                )
            self._emit_timer = 0.08

        # Fast movement: ice shards
        if speed > 0.35:
            self.composer.emit_crystals(px, py, count=int(speed * 4),
                                         color=self.ICE, speed=speed * 3.0)

        # Pinch: ice orb compression
        if gs.gesture == ElementalGesture.PINCH:
            self.composer.emit_radial_burst(px, py, count=5, speed=1.2,
                                             color_a=self.ICE, color_b=self.SNOW,
                                             life=0.35, size=4, glow=True, shape='shard')

        # Release/swipe: frost burst
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.15:
            self.composer.emit_crystals(px, py, count=40, color=self.CRYSTAL, speed=5.5)
            self.composer.spawn_shockwave(px, py, max_radius=140, duration=0.55,
                                          color=self.ICE, width=3, glow_rings=2)
            self.energy = max(0.0, self.energy - 0.3)
            self._release_cooldown = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Thunder
# ─────────────────────────────────────────────────────────────────────────────

class Thunder(BaseElement):
    PURPLE = (180, 100, 255)
    VIOLET = (220, 180, 255)
    WHITE  = (240, 230, 255)
    CHARGE_COL = (140, 60, 220)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arc_timer = 0.0
        self._strike_timer = 0.0

    def _spawn_persistent_vfx(self):
        self._orb = self.composer.spawn_orb(640, 360, radius=20, color=self.PURPLE, pulse_speed=5.0)

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 15 + self.energy * 22 + self.power * 20
            self._orb.color = lerp_color(self.PURPLE, self.WHITE, self.power)

        self._arc_timer    = max(0.0, self._arc_timer    - dt)
        self._strike_timer = max(0.0, self._strike_timer - dt)

        # Ambient electrical arcs from fingertips
        if self._arc_timer <= 0 and self.energy > 0.1:
            num_arcs = int(1 + self.energy * 3 + self.power * 2)
            for _ in range(num_arcs):
                angle = random.uniform(0, math.tau)
                dist = 20 + self.energy * 60 + self.power * 40
                x2 = px + math.cos(angle) * dist
                y2 = py + math.sin(angle) * dist
                self.composer.spawn_lightning(
                    px, py, x2, y2,
                    color=lerp_color(self.PURPLE, self.WHITE, self.power),
                    duration=0.10 + self.power * 0.08,
                    branches=int(1 + self.power * 3),
                    jaggedness=15 + self.power * 20,
                    width=1 + int(self.power * 2),
                )
            self._arc_timer = 0.06 + (1.0 - self.energy) * 0.1

        # Particle sparks during movement
        if speed > 0.25 and self._emit_timer <= 0:
            self.composer.emit_sparks(px, py, count=int(speed * 5),
                                       color=self.PURPLE, speed=4.0)
            self._emit_timer = 0.05

        # Fist — charge strike
        if gs.gesture == ElementalGesture.FIST and self.power > 0.3:
            if self._strike_timer <= 0:
                x2 = px + random.uniform(-60, 60)
                y2 = py - random.uniform(30, 80)
                self.composer.spawn_lightning(px, py, x2, y2,
                                              color=self.WHITE, duration=0.12,
                                              branches=4, jaggedness=30, width=3)
                self._strike_timer = 0.12

        # Swipe — lightning strike projectile
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            vx = (0.5 - gs.hand_pos[0]) * -10.0
            vy = (0.5 - gs.hand_pos[1]) * -10.0
            self.composer.spawn_projectile(px, py, vx, vy,
                                           color=self.PURPLE, radius=14, lifetime=1.2)
            # Big lightning flash from hand
            for _ in range(6):
                a = random.uniform(0, math.tau)
                dist = random.uniform(40, 120)
                self.composer.spawn_lightning(px, py,
                                              px + math.cos(a) * dist,
                                              py + math.sin(a) * dist,
                                              color=self.WHITE, duration=0.15,
                                              branches=3, jaggedness=25, width=2)
            self.composer.spawn_shockwave(px, py, max_radius=100, duration=0.4,
                                          color=self.PURPLE, width=3)
            self.energy = max(0.0, self.energy - 0.3)
            self._release_cooldown = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Earth
# ─────────────────────────────────────────────────────────────────────────────

class Earth(BaseElement):
    STONE  = (140, 100, 50)
    DIRT   = (100, 70, 35)
    SAND   = (200, 165, 80)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debris_timer = 0.0

    def _spawn_persistent_vfx(self):
        self._ring = self.composer.spawn_ring(640, 360, radius=50, color=self.STONE,
                                               node_count=6, speed=0.8, tilt=0.3)

    def _update_element(self, dt, gs, px, py, speed):
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 40 + self.energy * 40 + self.power * 30
            self._ring.node_count = max(3, int(4 + self.power * 8))

        self._debris_timer = max(0.0, self._debris_timer - dt)

        # Orbiting rock debris
        if self._debris_timer <= 0:
            count = int(1 + self.energy * 2)
            radius = 35 + self.energy * 45
            for i in range(count):
                angle = random.uniform(0, math.tau)
                spd = random.uniform(0.7, 1.4)
                self.composer.pool.emit(
                    x=px + math.cos(angle) * radius,
                    y=py + math.sin(angle) * radius,
                    orbit_cx=px, orbit_cy=py,
                    orbit_radius=radius * random.uniform(0.85, 1.15),
                    orbit_speed=spd,
                    orbit_angle=angle,
                    life=random.uniform(1.2, 2.0),
                    size=random.uniform(4, 10),
                    min_size=1.5,
                    color=lerp_color(self.DIRT, self.STONE, random.random()),
                    color_end=(50, 35, 15),
                    shape='square',
                )
            self._debris_timer = 0.15

        # Dust cloud ambient
        if self._emit_timer <= 0 and self.energy > 0.15:
            self.composer.pool.emit(
                x=px + random.uniform(-25, 25),
                y=py + random.uniform(-10, 10),
                vx=random.uniform(-0.6, 0.6),
                vy=random.uniform(-0.3, 0.3),
                life=random.uniform(0.4, 0.9),
                size=random.uniform(3, 8),
                min_size=0.5,
                color=self.SAND,
                color_end=(40, 30, 15),
                gravity=0.04,
                drag=0.97,
                turbulence=0.5,
            )
            self._emit_timer = 0.06

        # Pinch: compress debris into fist
        if gs.gesture == ElementalGesture.PINCH:
            self.composer.emit_radial_burst(px, py, count=4, speed=1.0,
                                             color_a=self.STONE, color_b=self.SAND,
                                             life=0.4, size=6, shape='square')

        # Swipe: shockwave + debris burst
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            self.composer.emit_rock_chunks(px, py, count=25, color=self.STONE, speed=5.0)
            self.composer.spawn_shockwave(px, py, max_radius=160, duration=0.65,
                                          color=self.STONE, width=5, glow_rings=2)
            # Ground-crack visual: horizontal lines
            for yi in range(-3, 4):
                x1 = px - random.randint(40, 120)
                x2 = px + random.randint(40, 120)
                y  = py + yi * 6 + random.randint(-3, 3)
                crack_col = lerp_color(self.DIRT, self.STONE, random.random())
                pygame.draw.line(pygame.display.get_surface(), crack_col, (int(x1), int(y)), (int(x2), int(y)), 2)
            self.energy = max(0.0, self.energy - 0.35)
            self._release_cooldown = 0.7


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Wind
# ─────────────────────────────────────────────────────────────────────────────

class Wind(BaseElement):
    TEAL  = (160, 235, 180)
    WHITE = (235, 255, 230)
    MIST  = (200, 250, 210)

    def _spawn_persistent_vfx(self):
        self._vortex = self.composer.spawn_vortex(640, 360, radius=70, color=self.TEAL,
                                                   arms=3, speed=3.0)
        self._ring = self.composer.spawn_ring(640, 360, radius=90, color=self.TEAL,
                                               node_count=20, speed=-1.2, tilt=0.6)

    def _update_element(self, dt, gs, px, py, speed):
        if self._vortex:
            self._vortex.cx, self._vortex.cy = px, py
            self._vortex.radius = 50 + self.energy * 60 + self.power * 30
            self._vortex.speed  = 2.0 + self.energy * 2.0 + speed * 2.0
            self._vortex.color = lerp_color(self.TEAL, self.WHITE, self.power)
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 80 + self.energy * 40
            self._ring.speed  = -(1.0 + self.energy * 1.5)

        # Spiral trail particles
        if self._emit_timer <= 0:
            count = int(2 + self.energy * 4 + speed * 3)
            angle_base = self._time * 3.5
            for i in range(count):
                a = angle_base + (i / max(1, count)) * math.tau
                r = 20 + self.energy * 50
                ex = px + math.cos(a) * r
                ey = py + math.sin(a) * r
                self.composer.pool.emit(
                    x=ex, y=ey,
                    vx=math.cos(a + math.pi/2) * (1.5 + speed),
                    vy=math.sin(a + math.pi/2) * (1.5 + speed),
                    life=random.uniform(0.4, 0.9),
                    size=random.uniform(2, 5),
                    min_size=0.5,
                    color=self.TEAL,
                    color_end=self.WHITE,
                    drag=0.97,
                    trail=True,
                    turbulence=0.3,
                )
            self._emit_timer = 0.05

        # Swipe: launch wind pulse
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.15:
            vx = (0.5 - gs.hand_pos[0]) * -9.0
            vy = (0.5 - gs.hand_pos[1]) * -9.0
            self.composer.spawn_projectile(px, py, vx, vy,
                                           color=self.TEAL, radius=18, lifetime=1.0)
            self.composer.spawn_shockwave(px, py, max_radius=130, duration=0.5,
                                          color=self.TEAL, width=2, glow_rings=3)
            self.composer.emit_radial_burst(px, py, count=20, speed=5.0,
                                             color_a=self.TEAL, color_b=self.WHITE,
                                             life=0.6, size=3, trail=True, drag=0.97)
            self.energy = max(0.0, self.energy - 0.28)
            self._release_cooldown = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 7 — Void
# ─────────────────────────────────────────────────────────────────────────────

class Void(BaseElement):
    DARK_PURPLE = (80, 20, 140)
    VIOLET      = (160, 60, 255)
    BLACK_GLOW  = (40, 10, 80)
    DISTORT     = (200, 150, 255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._implode_state = False
        self._implode_timer = 0.0

    def _spawn_persistent_vfx(self):
        self._vortex = self.composer.spawn_vortex(640, 360, radius=60, color=self.VIOLET,
                                                   arms=4, speed=-2.8)
        self._ring = self.composer.spawn_ring(640, 360, radius=80, color=self.DARK_PURPLE,
                                               node_count=12, speed=-1.8, tilt=0.5)
        self._orb = self.composer.spawn_orb(640, 360, radius=18, color=self.DARK_PURPLE,
                                             pulse_speed=1.5)

    def _update_element(self, dt, gs, px, py, speed):
        if self._orb:
            self._orb.x, self._orb.y = px, py
            self._orb.radius = 15 + self.energy * 20 + self.power * 25
            self._orb.color = lerp_color(self.DARK_PURPLE, self.DISTORT, self.power)
        if self._vortex:
            self._vortex.cx, self._vortex.cy = px, py
            self._vortex.radius = 55 + self.energy * 55 + self.power * 40
            self._vortex.speed  = -(2.0 + self.power * 2.5)
        if self._ring:
            self._ring.cx, self._ring.cy = px, py
            self._ring.radius = 70 + self.energy * 50
            self._ring.speed  = -(1.5 + self.power * 1.5)

        self._implode_timer = max(0.0, self._implode_timer - dt)

        # Gravitational particle attraction
        if self._emit_timer <= 0:
            count = int(2 + self.energy * 5)
            self.composer.emit_void_fragments(px, py, count=count,
                                               color=self.VIOLET, radius=80 + self.energy * 50)
            self._emit_timer = 0.10

        # Fist — increase distortion
        if gs.gesture == ElementalGesture.FIST and self._emit_timer <= 0:
            self.composer.emit_radial_burst(px, py, count=6, speed=1.0,
                                             color_a=self.VIOLET, color_b=self.DARK_PURPLE,
                                             life=0.5, size=4, glow=True, shape='ring',
                                             drag=0.93)

        # Swipe: implosion → explosion
        if gs.swipe_fired and self._release_cooldown <= 0 and self.energy > 0.2:
            self._implode_state = True
            self._implode_timer = 0.4

        if self._implode_state and self._implode_timer <= 0:
            self._implode_state = False
            # Explosion burst
            self.composer.emit_radial_burst(px, py, count=60, speed=7.0,
                                             color_a=self.VIOLET, color_b=self.DISTORT,
                                             life=1.0, size=5, glow=True, shape='ring',
                                             drag=0.93)
            self.composer.spawn_shockwave(px, py, max_radius=200, duration=0.7,
                                          color=self.VIOLET, width=4, glow_rings=3)
            self.energy = max(0.0, self.energy - 0.4)
            self._release_cooldown = 0.7

        if self._implode_state:
            # Visual implosion: particles flying inward
            if self._emit_timer <= 0:
                for _ in range(8):
                    a = random.uniform(0, math.tau)
                    r = random.uniform(60, 150)
                    sx = px + math.cos(a) * r
                    sy = py + math.sin(a) * r
                    # Attract toward center
                    self.composer.pool.emit(
                        x=sx, y=sy,
                        vx=0.0, vy=0.0,
                        attract_x=px, attract_y=py,
                        attract_strength=0.08,
                        life=0.35,
                        size=random.uniform(3, 7),
                        min_size=0.5,
                        color=self.VIOLET,
                        color_end=self.DARK_PURPLE,
                        drag=0.95,
                        glow=True,
                    )
                self._emit_timer = 0.04

    def render(self, surface, gs):
        # Distortion ring illusion
        if self._orb and self.power > 0.3:
            px, py = int(self._orb.x), int(self._orb.y)
            rings = int(self.power * 4)
            for i in range(1, rings + 1):
                r = int(i * 20 * (0.8 + 0.2 * math.sin(self._time * 4 + i)))
                alpha = int((1.0 - i / (rings + 1)) * 80 * self.power)
                if r > 0 and alpha > 5:
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
