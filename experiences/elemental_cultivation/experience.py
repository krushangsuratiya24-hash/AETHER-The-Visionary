"""
AETHER — Elemental Cultivation Experience (Phase 2)
Implements the BaseExperience interface and orchestrates all Phase 2 systems.
"""
from __future__ import annotations

import math
import time
import random
from typing import Optional, List, Tuple

import pygame
import cv2
import numpy as np

from core.config import palette, display_config
from core.gestures import GestureType
from core.tracker import HandLandmarkData
from core.particles import ParticlePool, lerp_color
from core.effects import EffectComposer
from experiences.base import BaseExperience
from experiences.elemental_cultivation.gestures import (
    ElementalGestureProcessor, ElementalGesture, ElementalGestureState
)
from experiences.elemental_cultivation.elements import (
    ElementID, ELEMENT_INFO, create_element, BaseElement
)
from experiences.elemental_cultivation.vfx_presets import ELEMENT_ORDER, get_element_hud_config

# Try to set up audio — graceful degradation if not available
_audio_enabled = False
try:
    import pygame.mixer
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    _audio_enabled = pygame.mixer.get_init() is not None
except Exception:
    pass


def _make_tone(freq: float, duration: float = 0.12, volume: float = 0.3,
               channels: int = 2) -> Optional[pygame.mixer.Sound]:
    """Procedurally generate a simple sine tone as a pygame Sound."""
    if not _audio_enabled:
        return None
    try:
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
        envelope = np.minimum(1.0, np.linspace(0, 1, n_samples) * (sample_rate * 0.01))
        envelope *= np.minimum(1.0, np.linspace(1, 0, n_samples) * (sample_rate * 0.05))
        wave = (wave * envelope).astype(np.int16)
        if channels == 2:
            wave = np.column_stack([wave, wave])
        return pygame.sndarray.make_sound(wave)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Elemental Cultivation Experience
# ─────────────────────────────────────────────────────────────────────────────

class ElementalCultivationExperience(BaseExperience):
    """
    Phase 2: Elemental Cultivation
    Real-time, gesture-driven elemental VFX experience.
    Seven distinct elements with unique visual identities.
    """

    POOL_CAPACITY = 2500

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        # Particle pool — shared across all elements
        self._pool = ParticlePool(capacity=self.POOL_CAPACITY)

        # Effect composer
        self._composer = EffectComposer(self._pool)

        # Elemental gesture processor
        self._gest_proc = ElementalGestureProcessor()

        # Build all seven element instances
        self._elements: dict[ElementID, BaseElement] = {
            eid: create_element(eid, self._composer, width, height)
            for eid in ELEMENT_ORDER
        }

        # Currently selected element
        self._current_id: ElementID = ElementID.PHOENIX_FLAME
        self._active_element: BaseElement = self._elements[ElementID.PHOENIX_FLAME]

        # Gesture state (refreshed every frame from hands list)
        self._gesture_state = ElementalGestureState()

        # Multi-hand raw data (set by main loop via handle_gesture extended call)
        self._raw_hands: List[HandLandmarkData] = []

        # Fonts
        self._font_title   = pygame.font.SysFont('Consolas', 22, bold=True)
        self._font_label   = pygame.font.SysFont('Consolas', 15, bold=True)
        self._font_small   = pygame.font.SysFont('Consolas', 12)
        self._font_element = pygame.font.SysFont('Consolas', 28, bold=True)
        self._font_key     = pygame.font.SysFont('Consolas', 13)

        # HUD
        self._time = 0.0
        self._switch_announce_timer = 0.0
        self._switch_announce_name  = ''
        self._switch_announce_color = (255, 255, 255)
        # Debug overlay (toggled with D key)
        self._debug_overlay = False

        # Audio tones
        self._sound_switch  = _make_tone(440.0, 0.12)
        self._sound_charge  = _make_tone(220.0, 0.10)
        self._sound_release = _make_tone(660.0, 0.15)

        # Background layer surface (pre-rendered grid)
        self._bg_surf: Optional[pygame.Surface] = None
        self._bg_dirty = True

        self._active = False

    # ── BaseExperience Lifecycle ──────────────────────────────────────────────

    def enter(self):
        self._active = True
        self._gest_proc.reset()
        self._composer.clear()
        self._active_element = self._elements[self._current_id]
        self._active_element.enter()
        self._bg_dirty = True
        print('[PHASE2] Elemental Cultivation — entered.')

    def exit(self):
        self._active = False
        self._active_element.exit()
        self._composer.clear()
        print('[PHASE2] Elemental Cultivation — exited.')

    # ── Input: called from AetherApp main loop ────────────────────────────────

    def handle_gesture(self, gesture: GestureType, action: GestureType,
                       hand_pos: Tuple[float, float]):
        """
        Phase 1 gesture bridge — only used by main.py for compatibility.
        Real gesture data arrives via handle_hands() in the extended path.
        We also use norm_pos from the Phase 1 processor as a fallback.
        """
        # Will be supplemented by handle_hands() when available
        if not self._raw_hands:
            # No full landmark data — synthesise a minimal gesture state
            st = ElementalGestureState()
            st.hand_pos = hand_pos
            st.gesture  = ElementalGesture.OPEN_PALM if gesture != GestureType.NEUTRAL else ElementalGesture.NONE
            self._gesture_state = st

    def handle_hands(self, hands: List[HandLandmarkData]):
        """
        Called by the enhanced AetherApp loop with raw landmark data.
        Produces a rich ElementalGestureState.
        """
        self._raw_hands = hands
        self._gesture_state = self._gest_proc.process(hands)

    def handle_key(self, event: pygame.event.Event) -> bool:
        key = event.key
        if key == pygame.K_d:
            self._debug_overlay = not self._debug_overlay
            return True
        mapping = {
            pygame.K_1: ElementID.PHOENIX_FLAME,
            pygame.K_2: ElementID.GOLDEN_SOLAR,
            pygame.K_3: ElementID.FROST,
            pygame.K_4: ElementID.THUNDER,
            pygame.K_5: ElementID.EARTH,
            pygame.K_6: ElementID.WIND,
            pygame.K_7: ElementID.VOID,
        }
        if key in mapping:
            self._switch_element(mapping[key])
            return True
        return False

    def _switch_element(self, new_id: ElementID):
        if new_id == self._current_id:
            return
        self._active_element.exit()
        self._composer.clear()
        self._current_id = new_id
        self._active_element = self._elements[new_id]
        self._active_element.enter()
        info = ELEMENT_INFO[new_id]
        self._switch_announce_name  = info['name']
        self._switch_announce_color = info['color']
        self._switch_announce_timer = 2.2
        self._gest_proc.reset()
        if self._sound_switch:
            try:
                self._sound_switch.play()
            except Exception:
                pass
        print(f'[PHASE2] Switched to {info["name"]}')

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        if not self._active:
            return
        self._time += dt
        self._switch_announce_timer = max(0.0, self._switch_announce_timer - dt)

        # Update active element
        gs = self._gesture_state
        self._active_element.update(dt, gs)

        # Update composer (all VFX objects + pool)
        self._composer.update(dt)

        # Update orbit centers for persistent objects that follow the hand
        # (handled inside each element's _update_element)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface):
        if not self._active:
            return

        # Background
        self._draw_background(surface)

        # VFX layer
        self._composer.draw(surface)

        # Element-specific extra geometry (heat shimmer, crack lines, etc.)
        self._active_element.render(surface, self._gesture_state)

        # Two-hand energy field connector
        gs = self._gesture_state
        if gs.two_hands and gs.hand2_pos is not None:
            self._draw_two_hand_field(surface, gs)

        # HUD
        self._draw_hud(surface)

        # Debug overlay
        if self._debug_overlay:
            self._draw_debug_overlay(surface)

        # Announce banner
        if self._switch_announce_timer > 0:
            self._draw_announce(surface)

    # ── Background ────────────────────────────────────────────────────────────

    def _draw_background(self, surface: pygame.Surface):
        surface.fill((8, 10, 18))
        # Faint grid
        for gx in range(0, self.width, 80):
            pygame.draw.line(surface, (14, 18, 28), (gx, 0), (gx, self.height), 1)
        for gy in range(0, self.height, 80):
            pygame.draw.line(surface, (14, 18, 28), (0, gy), (self.width, gy), 1)

    # ── Two-Hand Energy Field ─────────────────────────────────────────────────

    def _draw_two_hand_field(self, surface: pygame.Surface, gs: ElementalGestureState):
        x1 = int(gs.hand_pos[0] * self.width)
        y1 = int(gs.hand_pos[1] * self.height)
        x2 = int(gs.hand2_pos[0] * self.width)
        y2 = int(gs.hand2_pos[1] * self.height)
        col = ELEMENT_INFO[self._current_id]['color']
        dist = gs.two_hand_distance

        # Draw energy arc between hands
        segments = 12
        for i in range(segments - 1):
            t0 = i / segments
            t1 = (i + 1) / segments
            mid_x = x1 + (x2 - x1) * ((t0 + t1) * 0.5) + math.sin((t0 + t1) * math.pi * 2) * 15 * dist * 200
            mid_y = y1 + (y2 - y1) * ((t0 + t1) * 0.5)
            px0 = int(x1 + (x2 - x1) * t0)
            py0 = int(y1 + (y2 - y1) * t0)
            px1 = int(x1 + (x2 - x1) * t1)
            py1 = int(y1 + (y2 - y1) * t1)
            bright = int(100 + 80 * math.sin(self._time * 4 + i))
            c = (min(255, col[0] * bright // 180),
                 min(255, col[1] * bright // 180),
                 min(255, col[2] * bright // 180))
            try:
                pygame.draw.line(surface, c, (px0, py0), (px1, py1), 2)
            except Exception:
                pass

        # Central midpoint power orb
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        scale = 1.0 + dist * 2.5
        for lr, bright_mul in [(int(30 * scale), 40), (int(15 * scale), 100), (int(7 * scale), 200)]:
            if lr <= 0:
                continue
            c = (min(255, col[0] * bright_mul // 200),
                 min(255, col[1] * bright_mul // 200),
                 min(255, col[2] * bright_mul // 200))
            try:
                pygame.draw.circle(surface, c, (mx, my), lr)
            except Exception:
                pass

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, surface: pygame.Surface):
        gs = self._gesture_state
        elem = self._active_element
        info = ELEMENT_INFO[self._current_id]

        w, h = self.width, self.height
        panel_x = 20
        panel_y = 10
        panel_w = 240
        panel_h = 260

        # Semi-transparent panel
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((8, 12, 22, 195))
        pygame.draw.rect(panel, info['color'], (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel, (panel_x, panel_y))

        py = panel_y + 10
        px = panel_x + 12

        # Title
        t = self._font_title.render('ELEMENTAL CULTIVATION', True, palette.CYAN_NEON)
        surface.blit(t, (px, py))
        py += 26

        # Current element
        elem_name = info['name']
        elem_color = info['color']
        elem_surf = self._font_element.render(elem_name, True, elem_color)
        # Scale to fit if necessary
        max_w = panel_w - 24
        if elem_surf.get_width() > max_w:
            elem_surf = self._font_label.render(elem_name, True, elem_color)
        surface.blit(elem_surf, (px, py))
        py += elem_surf.get_height() + 6

        # Divider
        pygame.draw.line(surface, elem_color, (px, py), (panel_x + panel_w - 12, py), 1)
        py += 8

        # Energy bar
        self._draw_bar(surface, px, py, panel_w - 24, 'ENERGY', elem.energy, (0, 200, 255), (0, 80, 140))
        py += 22
        self._draw_bar(surface, px, py, panel_w - 24, 'POWER ', elem.power, elem_color, (40, 20, 80))
        py += 22
        self._draw_bar(surface, px, py, panel_w - 24, 'FLOW  ', elem.flow, (0, 255, 100), (0, 60, 30))
        py += 28

        # Gesture status
        gest_name = gs.gesture.value.replace('_', ' ')
        gest_col  = (200, 200, 200)
        if gs.gesture == ElementalGesture.FIST:
            gest_col = (255, 100, 40)
        elif gs.gesture == ElementalGesture.PINCH:
            gest_col = (100, 200, 255)
        elif gs.gesture == ElementalGesture.OPEN_PALM:
            gest_col = (100, 255, 120)
        elif gs.gesture == ElementalGesture.SWIPE:
            gest_col = (255, 215, 0)
        elif gs.gesture == ElementalGesture.TWO_HAND:
            gest_col = (255, 100, 255)
        t = self._font_label.render(f'GESTURE: {gest_name}', True, gest_col)
        surface.blit(t, (px, py))
        py += 18

        hands_col = (0, 255, 100) if self._raw_hands else (255, 60, 60)
        h_count = len(self._raw_hands)
        two_str = '  ★ 2 HANDS ACTIVE ★' if gs.two_hands else ''
        t = self._font_small.render(f'HANDS: {h_count}{two_str}', True,
                                     (255, 100, 255) if gs.two_hands else hands_col)
        surface.blit(t, (px, py))
        py += 16

        # Pinch indicator
        if gs.pinch_ratio > 0.1:
            pinch_col = (100, 200, 255)
            pb_w = int((panel_w - 24) * gs.pinch_ratio)
            pygame.draw.rect(surface, (20, 50, 80), (px, py, panel_w - 24, 6), border_radius=3)
            pygame.draw.rect(surface, pinch_col, (px, py, pb_w, 6), border_radius=3)
            pt = self._font_small.render(f'PINCH: {gs.pinch_ratio:.2f}', True, pinch_col)
            surface.blit(pt, (px + panel_w - 24 - pt.get_width(), py - 14))
            py += 10

        # Debug toggle hint
        t = self._font_small.render('[D] DEBUG', True, (60, 60, 80))
        surface.blit(t, (px, py))
        py += 14

        # Element selector (right panel)
        sel_x = w - 165
        sel_y = 10
        sel_w = 155
        sel_h = 10 + len(ELEMENT_ORDER) * 24 + 10
        sel_surf = pygame.Surface((sel_w, sel_h), pygame.SRCALPHA)
        sel_surf.fill((8, 12, 22, 195))
        pygame.draw.rect(sel_surf, palette.BORDER_DIM, (0, 0, sel_w, sel_h), 1, border_radius=6)
        surface.blit(sel_surf, (sel_x, sel_y))

        for i, eid in enumerate(ELEMENT_ORDER):
            ei = ELEMENT_INFO[eid]
            is_active = (eid == self._current_id)
            color = ei['color'] if is_active else palette.TEXT_MUTED
            prefix = '►' if is_active else ' '
            label = f"{prefix} [{ei['key']}] {ei['name']}"
            # Truncate long names
            if len(label) > 22:
                label = label[:22]
            fs = self._font_key.render(label, True, color)
            surface.blit(fs, (sel_x + 8, sel_y + 10 + i * 24))

        # Bottom mini-help bar
        help_items = [
            'OPEN PALM=CHARGE ENERGY',
            'FIST=POWER UP',
            'PINCH=COMPRESS',
            'SWIPE=RELEASE',
            '1-7 SWITCH ELEMENT',
            'ESC=MENU',
        ]
        help_str = '  |  '.join(help_items)
        hs = self._font_small.render(help_str, True, palette.TEXT_MUTED)
        hy = h - 20
        hx = w // 2 - hs.get_width() // 2
        # Background
        bg = pygame.Surface((hs.get_width() + 16, 18), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 130))
        surface.blit(bg, (hx - 8, hy - 2))
        surface.blit(hs, (hx, hy))

        # Pool debug (FPS-style)
        active = self._pool.active_count
        pool_str = f'PARTICLES: {active}/{self._pool._capacity}'
        ps = self._font_small.render(pool_str, True, palette.TEXT_MUTED)
        surface.blit(ps, (w - ps.get_width() - 8, h - 20))

    def _draw_bar(self, surface: pygame.Surface,
                  x: int, y: int, bar_width: int,
                  label: str, value: float,
                  color_full, color_empty,
                  height: int = 10):
        label_surf = self._font_small.render(label, True, palette.TEXT_MUTED)
        surface.blit(label_surf, (x, y))
        bx = x + 55
        bw = bar_width - 55
        bh = height
        # Background
        pygame.draw.rect(surface, color_empty, (bx, y + 1, bw, bh - 2), border_radius=3)
        # Fill
        fill_w = max(2, int(bw * max(0.0, min(1.0, value))))
        pygame.draw.rect(surface, color_full, (bx, y + 1, fill_w, bh - 2), border_radius=3)
        # Border
        pygame.draw.rect(surface, palette.BORDER_DIM, (bx, y + 1, bw, bh - 2), 1, border_radius=3)


    # ── Debug Overlay (D-key toggle) ──────────────────────────────────────────

    def _draw_debug_overlay(self, surface: pygame.Surface):
        """Full gesture debug panel for tuning and verification."""
        gs = self._gesture_state
        w, h = self.width, self.height

        panel_w = 280
        panel_h = 295
        panel_x = w // 2 - panel_w // 2
        panel_y = h - panel_h - 30

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 210))
        pygame.draw.rect(panel, (80, 200, 255), (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel, (panel_x, panel_y))

        lines = [
            ('DEBUG GESTURE MONITOR', (80, 200, 255)),
            (f'HANDS: {len(self._raw_hands)}', (0, 255, 120) if self._raw_hands else (255, 60, 60)),
            (f'PRIMARY:  {gs.gesture.value}', (255, 215, 0)),
            (f'RIGHT:    {gs.right_gesture or "---"}', (180, 180, 255)),
            (f'LEFT:     {gs.left_gesture or "---"}', (180, 255, 180)),
            (f'PINCH:    {gs.pinch_ratio:.2f}', (100, 200, 255)),
            (f'VELOCITY: {gs.speed:.3f} n/s', (255, 200, 100)),
            (f'VEL_XY:   {gs.velocity_x:.2f}, {gs.velocity_y:.2f}', (200, 200, 100)),
            (f'SWIPE:    {gs.swipe_dir or "---"} @ {gs.swipe_velocity:.2f}', (255, 215, 0)),
            (f'2-HANDS:  {"YES" if gs.two_hands else "NO"}', (255, 100, 255) if gs.two_hands else (120, 120, 120)),
            (f'2H-DIST:  {gs.two_hand_distance:.3f}', (220, 150, 255)),
            (f'ENERGY:   {self._active_element.energy * 100:.0f}%', (0, 200, 255)),
            (f'POWER:    {self._active_element.power * 100:.0f}%', (255, 100, 40)),
            ('D=CLOSE DEBUG', (80, 80, 80)),
        ]

        py_off = panel_y + 8
        for text, col in lines:
            s = self._font_small.render(text, True, col)
            surface.blit(s, (panel_x + 10, py_off))
            py_off += 18


    # ── Announce Banner ───────────────────────────────────────────────────────

    def _draw_announce(self, surface: pygame.Surface):
        t = self._switch_announce_timer
        alpha_f = min(1.0, t / 0.4) * min(1.0, t / 0.3)
        alpha = int(255 * alpha_f)
        cx = self.width // 2
        cy = self.height // 2 - 80

        name = self._switch_announce_name
        col  = self._switch_announce_color

        big_font = pygame.font.SysFont('Consolas', 52, bold=True)
        sub_font = pygame.font.SysFont('Consolas', 22)

        title_s = big_font.render(name, True, col)
        sub_s   = sub_font.render('ELEMENT ACTIVATED', True, palette.TEXT_WHITE)

        # Glow aura lines
        for gw in [10, 6, 3]:
            gcol = (min(255, col[0] // 2), min(255, col[1] // 2), min(255, col[2] // 2))
            ts = big_font.render(name, True, gcol)
            surface.blit(ts, (cx - ts.get_width() // 2, cy - ts.get_height() // 2 - 1))

        surface.blit(title_s, (cx - title_s.get_width() // 2, cy - title_s.get_height() // 2))
        surface.blit(sub_s,   (cx - sub_s.get_width()   // 2, cy + 36))
