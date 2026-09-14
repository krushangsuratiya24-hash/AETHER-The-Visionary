"""
AETHER — Phase Shift Experience  (Phase 3)
👻 QUANTUM DISSOLUTION / SPECTRAL RECONSTRUCTION

Architecture:
    PhaseShiftExperience orchestrates:
        • ThreadedCamera          — existing AETHER camera (reused)
        • PersonSegmenter         — MediaPipe selfie segmenter
        • BackgroundCompositor    — running background model + compositing
        • ClapDetector            — microphone audio pipeline
        • PhaseStateMachine       — VISIBLE / PHASING_OUT / INVISIBLE / PHASING_IN
        • PhaseShiftVFX           — all visual effects
        • Phase Shift HUD         — status display

Controls:
    Real clap  → toggle phase shift (primary exhibition interaction)
    [K]        → [DEV] keyboard clap fallback (clearly labeled in HUD)
    [D]        → toggle debug overlay (audio telemetry, mask, etc.)
    [ESC]      → return to AETHER launcher (handled by main.py)
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pygame

from core.config import palette, display_config
from core.gestures import GestureType
from core.particles import ParticlePool
from core.segmentation import PersonSegmenter
from core.audio import ClapDetector
from core.compositor import BackgroundCompositor
from experiences.base import BaseExperience
from experiences.phase_shift.state import PhaseStateMachine, PhaseState
from experiences.phase_shift.vfx import PhaseShiftVFX, PHASE_CYAN, PHASE_VIOLET, PHASE_TEAL
from experiences.phase_shift.config import phase_shift_config


# ─────────────────────────────────────────────────────────────────────────────
# PhaseShiftExperience
# ─────────────────────────────────────────────────────────────────────────────

class PhaseShiftExperience(BaseExperience):
    """
    Phase 3 — PHASE SHIFT
    Genuine real-time invisibility triggered by a hand clap.
    """

    POOL_CAPACITY = 2000

    def __init__(self, width: int, height: int):
        self._width  = width
        self._height = height
        self._cfg    = phase_shift_config

        # ── Sub-systems (created once, reused across enter/exit cycles) ──────
        # lazy=True: model is NOT loaded here — hardware starts in enter(),
        # so constructing the experience object never touches hardware.
        self._segmenter   = PersonSegmenter(lazy=True)
        self._compositor  = BackgroundCompositor()
        self._clap        = ClapDetector(sensitivity=self._cfg.clap_sensitivity)
        self._state_machine = PhaseStateMachine(
            phase_out_duration=self._cfg.phase_out_duration,
            phase_in_duration=self._cfg.phase_in_duration,
        )

        # Particle pool
        self._pool = ParticlePool(capacity=self.POOL_CAPACITY)
        self._vfx  = PhaseShiftVFX(width, height, self._pool)

        # ── State ─────────────────────────────────────────────────────────────
        self._active       = False
        self._debug_mode   = False
        self._time         = 0.0
        self._seg_frame_counter = 0
        self._phase_out_triggered = False
        self._phase_in_triggered  = False

        # Most recent composited frame (numpy BGR) ready to blit
        self._current_frame:  Optional[np.ndarray] = None
        self._person_mask:    Optional[np.ndarray] = None  # full-res binary

        # Fonts
        pygame.font.init()
        self._font_title  = pygame.font.SysFont('Consolas', 26, bold=True)
        self._font_label  = pygame.font.SysFont('Consolas', 16, bold=True)
        self._font_small  = pygame.font.SysFont('Consolas', 13)
        self._font_badge  = pygame.font.SysFont('Consolas', 12, bold=True)
        self._font_state  = pygame.font.SysFont('Consolas', 36, bold=True)

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
        self._state_machine.reset()
        self._compositor.reset()
        self._pool.clear()
        self._vfx._time = 0.0
        self._current_frame = None
        self._person_mask   = None
        self._time = 0.0
        self._phase_out_triggered = False
        self._phase_in_triggered  = False

        # Start hardware subsystems (lazy — not started in __init__)
        self._segmenter.start()   # loads MediaPipe model (creates executor thread)
        self._clap.start()        # opens microphone stream
        print('[PHASE3] Phase Shift — entered.')

    def exit(self):
        self._active = False
        self._clap.stop()
        self._segmenter.stop()
        self._pool.clear()
        print('[PHASE3] Phase Shift — exited.')

    # ─────────────────────────────────────────────────────────────────────────
    # Input
    # ─────────────────────────────────────────────────────────────────────────

    def handle_gesture(self, gesture: GestureType, action: GestureType,
                       hand_pos: Tuple[float, float]):
        pass   # Phase Shift does not use hand gesture input

    def handle_key(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_d:
            self._debug_mode = not self._debug_mode
            return True
        # [K] — DEV keyboard clap fallback
        if event.key == pygame.K_k:
            print('[PHASE3] [DEV] Keyboard fallback clap injected')
            self._clap.inject_clap()
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

        sm = self._state_machine

        # ── Poll clap detector ─────────────────────────────────────────────
        if self._clap.poll_clap():
            accepted = sm.trigger_clap()
            if accepted:
                print(f'[PHASE3] Clap accepted → state: {sm.state.value}')

        # ── Advance state machine ──────────────────────────────────────────
        sm.update(dt)

        # ── Camera + segmentation ──────────────────────────────────────────
        frame = self._camera_frame
        if frame is None:
            return

        # Get full-resolution person mask
        person_mask = self._segmenter.get_mask_for_frame(frame)
        self._person_mask = person_mask

        # ── Background model update (only during visible states) ───────────
        if sm.state in (PhaseState.VISIBLE, PhaseState.PHASING_OUT):
            # During phase-out we still want to keep the BG estimate fresh
            # for any newly visible regions
            self._compositor.update_background(frame, person_mask)
        elif not self._compositor.has_background:
            # Before first segmentation, prime the background model
            self._compositor.update_background(frame, None)

        # ── VFX update ──────────────────────────────────────────────────────
        self._vfx.update(dt)

        # ── One-shot VFX triggers ──────────────────────────────────────────
        if sm.state == PhaseState.PHASING_OUT and not self._phase_out_triggered:
            self._phase_out_triggered = True
            self._phase_in_triggered  = False
            self._vfx.on_phase_out_start(person_mask, self._width, self._height)

        if sm.state == PhaseState.PHASING_IN and not self._phase_in_triggered:
            self._phase_in_triggered  = True
            self._phase_out_triggered = False
            self._vfx.on_phase_in_start(person_mask, self._width, self._height)

        if sm.just_completed:
            cx = self._width  // 2
            cy = self._height // 2
            phase_out_completed = (sm.state == PhaseState.INVISIBLE)
            self._vfx.on_transition_complete(cx, cy, phase_out_completed)

        if sm.state == PhaseState.VISIBLE:
            self._phase_out_triggered = False
        if sm.state == PhaseState.INVISIBLE:
            self._phase_in_triggered = False

        # ── Compositing ─────────────────────────────────────────────────────
        blend_alpha = sm.blend_alpha
        composited = self._compositor.composite(frame, person_mask, blend_alpha)

        # Apply frame-level VFX (edge glow, glitch)
        composited = self._vfx.apply_frame_effects(composited, person_mask, blend_alpha)
        self._current_frame = composited

    # ─────────────────────────────────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface):
        if not self._active:
            return

        # ── Draw composited camera frame ────────────────────────────────────
        if self._current_frame is not None:
            frame_rgb = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
            # Resize to fill Pygame surface if needed
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

        # ── Draw Pygame VFX layer (particles, tiles, shockwaves) ──────────
        sm = self._state_machine
        blend_alpha = sm.blend_alpha
        phase_out_dir = sm.state in (PhaseState.PHASING_OUT, PhaseState.INVISIBLE)

        if self._person_mask is not None:
            # Resize mask to surface dimensions
            sw, sh = surface.get_size()
            pmask_disp = cv2.resize(self._person_mask, (sw, sh))
        else:
            pmask_disp = None

        if pmask_disp is not None:
            self._vfx.draw_pygame_effects(surface, pmask_disp, blend_alpha, phase_out_dir)

        # ── Scanline band overlay (across whole image during transition) ────
        if blend_alpha > 0.05:
            self._draw_scanline_overlay(surface, blend_alpha)

        # ── HUD ─────────────────────────────────────────────────────────────
        self._draw_hud(surface)

        # ── Debug overlay ────────────────────────────────────────────────────
        if self._debug_mode:
            self._draw_debug_overlay(surface)

    # ─────────────────────────────────────────────────────────────────────────
    # Scanline overlay (across full frame during transitions)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_scanline_overlay(self, surface: pygame.Surface, blend_alpha: float):
        """Subtle full-frame horizontal scanlines during transition."""
        w, h = surface.get_size()
        spacing = 4
        offset = int((self._time * 40) % spacing)
        intensity = int(blend_alpha * 0.12 * 255)
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
        sm    = self._state_machine
        w, h  = surface.get_size()

        # ── Top-left panel ────────────────────────────────────────────────
        panel_w, panel_h = 260, 180
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((8, 12, 22, 200))
        pygame.draw.rect(panel, PHASE_CYAN, (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel, (16, 60))

        px, py = 28, 70

        # Title
        title = self._font_title.render('PHASE SHIFT', True, PHASE_CYAN)
        surface.blit(title, (px, py))
        py += 28

        sub = self._font_label.render('AETHER  //  PHASE 3', True, palette.TEXT_MUTED)
        surface.blit(sub, (px, py))
        py += 24

        pygame.draw.line(surface, PHASE_CYAN, (px, py), (px + panel_w - 24, py), 1)
        py += 8

        # ── Phase state ───────────────────────────────────────────────────
        state_str = sm.state.value
        state_colors = {
            PhaseState.VISIBLE:     palette.GREEN_MATRIX,
            PhaseState.PHASING_OUT: palette.GOLD_ACCENT,
            PhaseState.INVISIBLE:   PHASE_VIOLET,
            PhaseState.PHASING_IN:  PHASE_TEAL,
        }
        state_col = state_colors.get(sm.state, palette.TEXT_WHITE)

        # Pulse during transition
        if sm.is_transitioning:
            pulse = 0.7 + 0.3 * math.sin(self._time * 6.0 * math.tau)
            state_col = tuple(int(c * pulse) for c in state_col)

        sstate = self._font_label.render(f'STATE:  {state_str}', True, state_col)
        surface.blit(sstate, (px, py))
        py += 20

        # Progress bar during transitions
        if sm.is_transitioning:
            bar_w = panel_w - 44
            t = sm.transition_t
            pygame.draw.rect(surface, (30, 40, 60), (px, py, bar_w, 6), border_radius=3)
            fill = max(2, int(bar_w * t))
            pygame.draw.rect(surface, state_col, (px, py, fill, 6), border_radius=3)
            py += 12

        py += 6

        # ── Mic status ────────────────────────────────────────────────────
        mic_ok  = self._clap.available
        mic_col = palette.GREEN_MATRIX if mic_ok else palette.MAGENTA_LASER
        mic_str = '● MIC READY' if mic_ok else '● MIC UNAVAILABLE'
        mic_txt = self._font_badge.render(f'MIC    {mic_str}', True, mic_col)
        surface.blit(mic_txt, (px, py))
        py += 16

        # ── Segmentation status ────────────────────────────────────────────
        seg_ok  = self._segmenter.ready
        seg_col = palette.GREEN_MATRIX if seg_ok else palette.MAGENTA_LASER
        seg_str = '● SEG READY' if seg_ok else '● SEG UNAVAILABLE'
        seg_txt = self._font_badge.render(f'SEG    {seg_str}', True, seg_col)
        surface.blit(seg_txt, (px, py))
        py += 16

        # ── FPS ───────────────────────────────────────────────────────────
        fps_col = palette.GREEN_MATRIX if self._fps >= 30 else palette.GOLD_ACCENT
        fps_txt = self._font_badge.render(f'FPS    {int(self._fps)}', True, fps_col)
        surface.blit(fps_txt, (px, py))

        # ── Large state banner (center, during transitions) ───────────────
        if sm.is_transitioning or sm.state == PhaseState.INVISIBLE:
            self._draw_state_banner(surface, sm.state, sm.blend_alpha)

        # ── Bottom help bar ────────────────────────────────────────────────
        if self._clap.available:
            help_str = 'CLAP to phase in/out   |  [K] DEV fallback  |  [D] Debug  |  [ESC] Menu'
        else:
            help_str = '[K] DEV keyboard clap (mic unavailable)   |  [D] Debug  |  [ESC] Menu'
        hs = self._font_small.render(help_str, True, palette.TEXT_MUTED)
        hx = w // 2 - hs.get_width() // 2
        bg = pygame.Surface((hs.get_width() + 16, 18), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surface.blit(bg, (hx - 8, h - 22))
        surface.blit(hs, (hx, h - 20))

    def _draw_state_banner(self, surface: pygame.Surface,
                           state: PhaseState, blend_alpha: float):
        """Large centered banner showing current state during transitions."""
        w, h = surface.get_size()
        cx, cy = w // 2, h // 2

        state_colors = {
            PhaseState.PHASING_OUT: palette.GOLD_ACCENT,
            PhaseState.INVISIBLE:   PHASE_VIOLET,
            PhaseState.PHASING_IN:  PHASE_TEAL,
        }
        col = state_colors.get(state, palette.TEXT_WHITE)
        pulse = 0.8 + 0.2 * math.sin(self._time * 4.0 * math.tau)
        col = tuple(int(c * pulse) for c in col)

        if state == PhaseState.INVISIBLE:
            txt = 'PHASE SHIFTED'
        else:
            txt = state.value

        # Shadow
        shadow = self._font_state.render(txt, True, (0, 0, 0))
        surface.blit(shadow, (cx - shadow.get_width() // 2 + 2, cy - 18 + 2))
        # Main text
        label = self._font_state.render(txt, True, col)
        surface.blit(label, (cx - label.get_width() // 2, cy - 18))

    # ─────────────────────────────────────────────────────────────────────────
    # Debug Overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_debug_overlay(self, surface: pygame.Surface):
        tel   = self._clap.telemetry
        sm    = self._state_machine
        w, h  = surface.get_size()

        panel_w, panel_h = 290, 340
        px, py = w - panel_w - 16, 60

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 220))
        pygame.draw.rect(panel, PHASE_CYAN, (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel, (px, py))

        lines = [
            ('DEBUG TELEMETRY [D]', PHASE_CYAN),
            ('', palette.TEXT_MUTED),
            # Audio
            (f'RMS:           {tel.rms:.4f}', palette.TEXT_WHITE),
            (f'NOISE FLOOR:   {tel.noise_floor:.4f}', palette.TEXT_WHITE),
            (f'RATIO:         {tel.ratio:.2f}x', palette.TEXT_WHITE),
            (f'CONFIDENCE:    {tel.confidence:.2f}', palette.GREEN_MATRIX if tel.confidence > 0.5 else palette.TEXT_WHITE),
            (f'COOLDOWN:      {tel.cooldown_remaining:.2f}s', palette.GOLD_ACCENT),
            (f'IN TRANSIENT:  {"YES" if tel.in_transient else "NO"}', PHASE_VIOLET if tel.in_transient else palette.TEXT_MUTED),
            (f'MIC AVAILABLE: {"YES" if tel.mic_available else "NO"}', palette.GREEN_MATRIX if tel.mic_available else palette.MAGENTA_LASER),
            ('', palette.TEXT_MUTED),
            # Segmentation
            (f'SEG READY:     {"YES" if self._segmenter.ready else "NO"}', palette.GREEN_MATRIX if self._segmenter.ready else palette.MAGENTA_LASER),
            (f'SEG FPS:       {self._segmenter.fps:.1f}', palette.TEXT_WHITE),
            (f'MASK AGE:      {self._segmenter.mask_age:.2f}s', palette.TEXT_WHITE),
            (f'BG WARM:       {"YES" if self._compositor.has_background else "NO"}', palette.GREEN_MATRIX if self._compositor.has_background else palette.GOLD_ACCENT),
            (f'BG COVERAGE:   {self._compositor.bg_coverage:.1%}', palette.TEXT_WHITE),
            ('', palette.TEXT_MUTED),
            # State machine
            (f'STATE:         {sm.state.value}', PHASE_CYAN),
            (f'BLEND ALPHA:   {sm.blend_alpha:.2f}', palette.TEXT_WHITE),
            (f'TRANSITION T:  {sm.transition_t:.2f}', palette.TEXT_WHITE),
            ('', palette.TEXT_MUTED),
            # Particles
            (f'PARTICLES:     {self._pool.active_count}/{self._pool._capacity}', palette.TEXT_MUTED),
            ('D=CLOSE DEBUG', (60, 60, 80)),
        ]

        y_off = py + 8
        for text, col in lines:
            s = self._font_small.render(text, True, col)
            surface.blit(s, (px + 10, y_off))
            y_off += 16

        # Mini mask preview
        if self._cfg.show_mask_debug and self._person_mask is not None:
            mask_small = cv2.resize(self._person_mask, (120, 68))
            mask_rgb   = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2RGB)
            mask_surf  = pygame.surfarray.make_surface(
                np.transpose(mask_rgb, (1, 0, 2))
            )
            surface.blit(mask_surf, (px + 10, py + panel_h - 85))
            pygame.draw.rect(surface, PHASE_CYAN, (px + 9, py + panel_h - 86, 122, 70), 1)
            lbl = self._font_small.render('MASK', True, PHASE_CYAN)
            surface.blit(lbl, (px + 12, py + panel_h - 95))
