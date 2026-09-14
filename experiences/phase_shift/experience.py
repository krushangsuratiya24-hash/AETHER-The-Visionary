"""
AETHER — Phase Shift Experience  (Phase 3 — HAND POWER rework)
👻 PHASE SHIFT — HAND POWER

The user's hand controls their body transparency.

Architecture:
    PhaseShiftExperience orchestrates:
        • ThreadedCamera          — existing AETHER camera (reused)
        • PersonSegmenter         — MediaPipe selfie segmenter
        • BackgroundCompositor    — running background model + compositing
        • PhaseGestureClassifier  — hand-gesture to transparency level
        • AlphaController         — smooth lerp of blend alpha
        • PhaseShiftVFX           — all visual effects
        • Phase Shift HUD         — status display

Gesture → Transparency:
    ☝ ONE FINGER     → 25%  (mostly visible, subtle distortion)
    ✌ TWO FINGERS    → 50%  (clearly translucent, holographic)
    🖖 THREE FINGERS  → 75%  (strong phase-shift, silhouette breakup)
    🖐 FULL HAND      → 100% (pure invisibility via segmentation composite)

Controls:
    Hand gesture → phase level (primary interaction, no microphone required)
    [D]          → toggle debug overlay
    [ESC]        → return to AETHER launcher (handled by main.py)

NO microphone required.
NO clap detection required.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygame

from core.config import palette, display_config
from core.particles import ParticlePool
from core.segmentation import PersonSegmenter
from core.compositor import BackgroundCompositor
from core.tracker import HandLandmarkData
from experiences.base import BaseExperience
from experiences.phase_shift.gestures import (
    PhaseGesture, PhaseGestureClassifier, GestureResult, GESTURE_ALPHA,
)
from experiences.phase_shift.state import AlphaController
from experiences.phase_shift.vfx import PhaseShiftVFX, PHASE_CYAN, PHASE_VIOLET, PHASE_TEAL
from experiences.phase_shift.config import phase_shift_config


# ─────────────────────────────────────────────────────────────────────────────
# Gesture label strings for HUD
# ─────────────────────────────────────────────────────────────────────────────

_GESTURE_LABELS = {
    PhaseGesture.NONE:          'NONE',
    PhaseGesture.ONE_FINGER:    'ONE FINGER',
    PhaseGesture.TWO_FINGERS:   'TWO FINGERS',
    PhaseGesture.THREE_FINGERS: 'THREE FINGERS',
    PhaseGesture.FIVE_FINGERS:  'FULL HAND',
}

_LEVEL_COLORS = {
    PhaseGesture.NONE:          (120, 140, 160),
    PhaseGesture.ONE_FINGER:    (0,   220, 255),     # cyan
    PhaseGesture.TWO_FINGERS:   (0,   255, 180),     # teal
    PhaseGesture.THREE_FINGERS: (180,  60, 255),     # violet
    PhaseGesture.FIVE_FINGERS:  (255,  80, 200),     # magenta
}

# Threshold: alpha above this = "stable invisible" — suppress person-local VFX
_INVISIBLE_SUPPRESS_ALPHA = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# PhaseShiftExperience
# ─────────────────────────────────────────────────────────────────────────────

class PhaseShiftExperience(BaseExperience):
    """
    Phase 3 — PHASE SHIFT (Hand Power)
    Genuine real-time transparency controlled by hand gesture count.
    """

    POOL_CAPACITY = 2000

    def __init__(self, width: int, height: int):
        self._width  = width
        self._height = height
        self._cfg    = phase_shift_config

        # ── Sub-systems (created once, reused across enter/exit cycles) ──────
        self._segmenter   = PersonSegmenter(lazy=True)
        self._compositor  = BackgroundCompositor()
        self._gesture_clf = PhaseGestureClassifier()
        self._alpha_ctrl  = AlphaController(lerp_speed=self._cfg.alpha_lerp_speed)

        # Particle pool
        self._pool = ParticlePool(capacity=self.POOL_CAPACITY)
        self._vfx  = PhaseShiftVFX(width, height, self._pool)

        # ── State ─────────────────────────────────────────────────────────────
        self._active       = False
        self._debug_mode   = False
        self._time         = 0.0
        self._seg_frame_counter = 0

        # Last gesture result for HUD
        self._last_result: Optional[GestureResult] = None
        self._prev_gesture: PhaseGesture = PhaseGesture.NONE
        self._last_hands: List[HandLandmarkData] = []

        # Most recent composited frame (numpy BGR) ready to blit
        self._current_frame:  Optional[np.ndarray] = None
        self._person_mask:    Optional[np.ndarray] = None  # full-res binary

        # Fonts
        pygame.font.init()
        self._font_title  = pygame.font.SysFont('Consolas', 24, bold=True)
        self._font_label  = pygame.font.SysFont('Consolas', 16, bold=True)
        self._font_small  = pygame.font.SysFont('Consolas', 13)
        self._font_badge  = pygame.font.SysFont('Consolas', 12, bold=True)
        self._font_state  = pygame.font.SysFont('Consolas', 32, bold=True)
        self._font_level  = pygame.font.SysFont('Consolas', 40, bold=True)

        # Cache: last camera frame (BGR) — set externally by main loop
        self._camera_frame: Optional[np.ndarray] = None

        # FPS tracking
        self._fps = 0.0
        self._fps_timer = time.time()
        self._fps_counter = 0

    # ─────────────────────────────────────────────────────────────────────────
    # BaseExperience lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def enter(self):
        self._active = True
        self._gesture_clf.reset()
        self._alpha_ctrl.reset()
        self._compositor.reset()
        self._pool.clear()
        self._vfx._time = 0.0
        self._current_frame = None
        self._person_mask   = None
        self._time = 0.0
        self._prev_gesture  = PhaseGesture.NONE
        self._last_result   = None
        self._last_hands    = []

        # Start hardware subsystems (lazy — not started in __init__)
        self._segmenter.start()   # loads MediaPipe model
        print('[PHASE3] Phase Shift (Hand Power) — entered.')

    def exit(self):
        self._active = False
        self._segmenter.stop()
        self._pool.clear()
        print('[PHASE3] Phase Shift — exited.')

    # ─────────────────────────────────────────────────────────────────────────
    # Input — hand landmarks from main loop
    # ─────────────────────────────────────────────────────────────────────────

    def push_hands(self, hands: List[HandLandmarkData]) -> None:
        """
        Called by main.py with the list of detected hands each frame.
        Uses the first (highest-confidence) hand for gesture classification.
        """
        self._last_hands = hands
        hand = hands[0] if hands else None
        result = self._gesture_clf.update(hand)
        self._last_result = result

        # Update alpha controller when gesture changes
        if result.gesture != self._prev_gesture:
            self._alpha_ctrl.set_gesture(result.gesture)
            self._prev_gesture = result.gesture
            print(f'[PHASE3] Gesture → {result.gesture.value}  α={result.target_alpha:.2f}')

    def handle_gesture(self, gesture, action, hand_pos):
        pass   # handled via push_hands()

    def handle_key(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_d:
            self._debug_mode = not self._debug_mode
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Frame injection (called by main loop)
    # ─────────────────────────────────────────────────────────────────────────

    def push_camera_frame(self, frame: np.ndarray):
        """
        Called by main.py with the latest camera BGR frame.
        Runs segmentation at the configured sub-sampling rate.
        """
        self._camera_frame = frame

        # Sub-sample segmentation to save CPU
        self._seg_frame_counter += 1
        if self._seg_frame_counter >= self._cfg.seg_update_interval:
            self._seg_frame_counter = 0
            self._segmenter.push_frame(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        if not self._active:
            return

        self._time += dt
        self._fps_counter += 1
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_timer)
            self._fps_counter = 0
            self._fps_timer = now

        # Advance alpha interpolation
        self._alpha_ctrl.update(dt)

        # Camera + segmentation
        frame = self._camera_frame
        if frame is None:
            return

        # Get full-resolution person mask  (H, W) uint8
        person_mask = self._segmenter.get_mask_for_frame(frame)
        self._person_mask = person_mask   # always (H, W)

        blend_alpha = self._alpha_ctrl.alpha

        # Background model update: update when not fully invisible
        if blend_alpha < 0.95:
            self._compositor.update_background(frame, person_mask)
        elif not self._compositor.has_background:
            # Still warming up — update without mask
            self._compositor.update_background(frame, None)

        # VFX update
        self._vfx.update(dt)

        # Trigger particle burst on gesture change
        if self._alpha_ctrl.just_changed and person_mask is not None:
            if blend_alpha > 0.05:
                self._vfx.on_phase_out_start(person_mask, self._width, self._height)
            else:
                self._vfx.on_phase_in_start(person_mask, self._width, self._height)

        # Compositing
        composited = self._compositor.composite(frame, person_mask, blend_alpha)

        # Apply frame-level VFX (edge glow, channel shift)
        composited = self._vfx.apply_frame_effects(composited, person_mask, blend_alpha)
        self._current_frame = composited

    # ─────────────────────────────────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface):
        if not self._active:
            return

        blend_alpha = self._alpha_ctrl.alpha

        # ── Pure-invisible mode: suppress ALL person-local VFX ───────────────
        # When alpha is at or near 1.0, the person is genuinely gone.
        # We must not draw anything that would reveal their location.
        stable_invisible = (blend_alpha >= _INVISIBLE_SUPPRESS_ALPHA)

        # Draw composited camera frame (background compositor handles invisibility)
        if self._current_frame is not None:
            frame_rgb = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
            sw, sh = surface.get_size()
            fh, fw = frame_rgb.shape[:2]
            if (fw, fh) != (sw, sh):
                frame_rgb = cv2.resize(frame_rgb, (sw, sh))
            frame_surf = pygame.surfarray.make_surface(
                np.transpose(frame_rgb, (1, 0, 2))
            )
            surface.blit(frame_surf, (0, 0))
        else:
            surface.fill(palette.VOID_DARK)

        if stable_invisible:
            # ── Clean background — ONLY draw HUD and nothing over the person ──
            self._draw_hud(surface)
            if self._debug_mode:
                self._draw_debug_overlay(surface)
            return

        # ── Transition / partial transparency: draw VFX layer ────────────────
        if self._person_mask is not None:
            sw, sh = surface.get_size()
            pmask_disp = cv2.resize(self._person_mask, (sw, sh))
        else:
            pmask_disp = None

        if pmask_disp is not None:
            self._vfx.draw_pygame_effects(
                surface, pmask_disp, blend_alpha,
                phase_out=(blend_alpha > 0.0)
            )

        # Scanline overlay during active phase (transition only, not at full invisible)
        if 0.05 < blend_alpha < _INVISIBLE_SUPPRESS_ALPHA:
            self._draw_scanline_overlay(surface, blend_alpha)

        # HUD
        self._draw_hud(surface)

        # Debug overlay
        if self._debug_mode:
            self._draw_debug_overlay(surface)

    # ─────────────────────────────────────────────────────────────────────────
    # Scanline overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_scanline_overlay(self, surface: pygame.Surface, blend_alpha: float):
        """Subtle full-frame horizontal scanlines during active phase."""
        w, h = surface.get_size()
        spacing = 4
        offset = int((self._time * 40) % spacing)
        intensity = int(blend_alpha * 0.10 * 255)
        if intensity < 3:
            return
        col = (
            min(255, PHASE_CYAN[0] * intensity // 80),
            min(255, PHASE_CYAN[1] * intensity // 80),
            min(255, PHASE_CYAN[2] * intensity // 80),
        )
        for y in range(offset, h, spacing):
            try:
                pygame.draw.line(surface, col, (0, y), (w, y), 1)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # HUD
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_hud(self, surface: pygame.Surface):
        ctrl  = self._alpha_ctrl
        clf   = self._gesture_clf
        w, h  = surface.get_size()

        result = self._last_result

        # ── Top-left panel ────────────────────────────────────────────────
        panel_w, panel_h = 280, 230
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((8, 12, 22, 200))
        pygame.draw.rect(panel, PHASE_CYAN, (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel, (16, 60))

        px, py = 28, 68

        # Title
        title = self._font_title.render('AETHER', True, PHASE_CYAN)
        surface.blit(title, (px, py))
        py += 26

        sub = self._font_label.render('PHASE SHIFT', True, PHASE_CYAN)
        surface.blit(sub, (px, py))
        py += 22

        pygame.draw.line(surface, PHASE_CYAN, (px, py), (px + panel_w - 24, py), 1)
        py += 8

        # ── Phase level ───────────────────────────────────────────────────
        level_str = ctrl.phase_level_label()
        gesture   = ctrl.gesture if result is None else result.gesture
        lev_col   = _LEVEL_COLORS.get(gesture, PHASE_CYAN)

        # Pulse when transitioning
        if ctrl.is_transitioning:
            pulse = 0.7 + 0.3 * math.sin(self._time * 6.0 * math.tau)
            lev_col = tuple(int(c * pulse) for c in lev_col)

        lev_lbl = self._font_label.render('PHASE LEVEL', True, palette.TEXT_MUTED)
        surface.blit(lev_lbl, (px, py))
        py += 17

        lev_txt = self._font_label.render(level_str, True, lev_col)
        surface.blit(lev_txt, (px, py))
        py += 20

        # Alpha progress bar
        bar_w = panel_w - 44
        pygame.draw.rect(surface, (30, 40, 60), (px, py, bar_w, 7), border_radius=3)
        fill = max(2, int(bar_w * ctrl.alpha))
        pygame.draw.rect(surface, lev_col, (px, py, fill, 7), border_radius=3)
        py += 13

        pygame.draw.line(surface, (30, 40, 60), (px, py), (px + panel_w - 24, py), 1)
        py += 7

        # ── Current gesture ────────────────────────────────────────────────
        gest_str  = _GESTURE_LABELS.get(gesture, 'NONE')
        gest_col  = _LEVEL_COLORS.get(gesture, palette.TEXT_MUTED)

        gest_lbl = self._font_badge.render('GESTURE', True, palette.TEXT_MUTED)
        surface.blit(gest_lbl, (px, py))
        py += 15

        gest_txt = self._font_label.render(gest_str, True, gest_col)
        surface.blit(gest_txt, (px, py))
        py += 20

        # ── Segmentation status ────────────────────────────────────────────
        seg_ok  = self._segmenter.ready
        seg_col = palette.GREEN_MATRIX if seg_ok else palette.MAGENTA_LASER
        seg_str = '● READY' if seg_ok else '● UNAVAILABLE'
        seg_txt = self._font_badge.render(f'SEGMENTATION  {seg_str}', True, seg_col)
        surface.blit(seg_txt, (px, py))
        py += 16

        # ── FPS ───────────────────────────────────────────────────────────
        fps_col = palette.GREEN_MATRIX if self._fps >= 30 else palette.GOLD_ACCENT
        fps_txt = self._font_badge.render(f'FPS    {int(self._fps)}', True, fps_col)
        surface.blit(fps_txt, (px, py))

        # ── Large level banner during transition (NOT at stable invisible) ──
        if 0.05 < ctrl.alpha < _INVISIBLE_SUPPRESS_ALPHA:
            self._draw_level_banner(surface, ctrl, gesture)

        # ── Bottom help bar ────────────────────────────────────────────────
        help_str = '☝ 25%   ✌ 50%   🖖 75%   🖐 INVISIBLE   [D] Debug   [ESC] Menu'
        hs = self._font_small.render(help_str, True, palette.TEXT_MUTED)
        hx = w // 2 - hs.get_width() // 2
        bg = pygame.Surface((hs.get_width() + 16, 18), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surface.blit(bg, (hx - 8, h - 22))
        surface.blit(hs, (hx, h - 20))

    def _draw_level_banner(self, surface: pygame.Surface,
                           ctrl: AlphaController,
                           gesture: PhaseGesture):
        """Large centered banner showing current phase level."""
        w, h = surface.get_size()
        cx, cy = w // 2, h // 2

        col = _LEVEL_COLORS.get(gesture, PHASE_CYAN)
        pulse = 0.8 + 0.2 * math.sin(self._time * 3.0 * math.tau)
        col = tuple(int(c * pulse) for c in col)

        level_str = ctrl.phase_level_label()
        # Never show "PHASE SHIFTED" banner at stable invisible
        # (nothing should appear over the person's location)
        label_txt = f'PHASE  {level_str}'

        shadow = self._font_state.render(label_txt, True, (0, 0, 0))
        surface.blit(shadow, (cx - shadow.get_width() // 2 + 2, cy - 18 + 2))
        label = self._font_state.render(label_txt, True, col)
        surface.blit(label, (cx - label.get_width() // 2, cy - 18))

    # ─────────────────────────────────────────────────────────────────────────
    # Debug Overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_debug_overlay(self, surface: pygame.Surface):
        ctrl  = self._alpha_ctrl
        clf   = self._gesture_clf
        w, h  = surface.get_size()
        result = self._last_result

        panel_w, panel_h = 310, 370
        px, py = w - panel_w - 16, 60

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 220))
        pygame.draw.rect(panel, PHASE_CYAN, (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel, (px, py))

        # Mask dimensions
        mask_dims = (
            f'{self._person_mask.shape[0]}x{self._person_mask.shape[1]}'
            if self._person_mask is not None else 'N/A'
        )

        # Extended finger flags [thumb, index, middle, ring, pinky]
        ext = clf.extended_flags
        n_fingers = clf.finger_count
        if len(ext) == 5:
            ext_str = (f'T={int(ext[0])} I={int(ext[1])} M={int(ext[2])} '
                       f'R={int(ext[3])} P={int(ext[4])}')
        else:
            ext_str = 'N/A'

        gesture_label = result.gesture.value if result else 'N/A'
        target_pct = f'{int(ctrl.target_alpha * 100)}%'
        alpha_pct  = f'{int(ctrl.alpha * 100)}%'

        lines = [
            ('DEBUG  [D]', PHASE_CYAN),
            ('', palette.TEXT_MUTED),
            (f'FINGERS:       {n_fingers}', palette.TEXT_WHITE),
            (f'GESTURE:       {gesture_label}', PHASE_CYAN),
            (f'CONFIDENCE:    {(f"{result.confidence:.2f}" if result else "N/A")}',
             palette.GREEN_MATRIX),
            (f'FINGER FLAGS:  {ext_str}', palette.TEXT_WHITE),
            ('', palette.TEXT_MUTED),
            (f'TARGET:        {target_pct}', palette.GOLD_ACCENT),
            (f'ALPHA:         {alpha_pct}', palette.TEXT_WHITE),
            (f'INVISIBLE:     {"YES" if ctrl.alpha >= _INVISIBLE_SUPPRESS_ALPHA else "NO"}',
             PHASE_VIOLET if ctrl.alpha >= _INVISIBLE_SUPPRESS_ALPHA else palette.TEXT_MUTED),
            ('', palette.TEXT_MUTED),
            (f'SEGMENTATION:  {"READY" if self._segmenter.ready else "UNAVAIL"}',
             palette.GREEN_MATRIX if self._segmenter.ready else palette.MAGENTA_LASER),
            (f'MASK:          {mask_dims}', palette.TEXT_WHITE),
            (f'BG WARM:       {"YES" if self._compositor.has_background else "NO"}',
             palette.GREEN_MATRIX if self._compositor.has_background else palette.GOLD_ACCENT),
            ('', palette.TEXT_MUTED),
            (f'FPS:           {int(self._fps)}', palette.TEXT_WHITE),
            (f'HANDS:         {len(self._last_hands)}', palette.TEXT_WHITE),
            ('D=CLOSE DEBUG', (60, 60, 80)),
        ]

        y_off = py + 8
        for text, col in lines:
            s = self._font_small.render(text, True, col)
            surface.blit(s, (px + 10, y_off))
            y_off += 16
