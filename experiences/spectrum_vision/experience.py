"""
AETHER — Spectrum Vision Experience  (Phase 4)
🌀 SPECTRUM VISION — "See people beyond the ordinary."

Main orchestrator for the Spectrum Vision experience.

Architecture:
    SpectrumVisionExperience orchestrates:
        • PersonSegmenter      — MediaPipe selfie segmenter (reused from Phase 3)
        • detect_persons()     — connected-component multi-person decomposition
        • MultiPersonTracker   — stable temporary IDs across frames
        • SpectrumRenderer     — full-screen theme rendering
        • ThemeRegistry        — seven visual themes

Keyboard Controls:
    1–7   → Select theme
    R     → Reset tracking
    D     → Toggle debug overlay
    ESC   → Return to main menu (handled by main.py)
    Q     → Quit (handled by main.py)

Camera input:
    The webcam is used exclusively for people detection/segmentation.
    No microphone, no hand gestures as primary controls.
"""
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np
import pygame

from core.config import palette, display_config
from core.segmentation import PersonSegmenter
from core.gestures import GestureType
from experiences.base import BaseExperience
from experiences.spectrum_vision.people import DetectedPerson, detect_persons
from experiences.spectrum_vision.renderer import SpectrumRenderer
from experiences.spectrum_vision.themes import (
    ThemeDescriptor, ThemeID, THEME_REGISTRY,
)
from experiences.spectrum_vision.tracking import MultiPersonTracker, PersonTrack


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# How many main-loop frames to skip between full segmentation updates.
# 2 = run segmentation every other frame, halving CPU load.
_SEG_UPDATE_INTERVAL = 2

# Default theme on entry
_DEFAULT_THEME_ID = ThemeID.CYBER


# ─────────────────────────────────────────────────────────────────────────────
# SpectrumVisionExperience
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumVisionExperience(BaseExperience):
    """
    Phase 4 — SPECTRUM VISION
    Full-screen real-time people visualization with seven visual themes.
    """

    def __init__(self, width: int, height: int):
        self._width  = width
        self._height = height

        # Sub-systems (lazy init — model loads on enter())
        self._segmenter = PersonSegmenter(lazy=True)
        self._tracker   = MultiPersonTracker()
        self._renderer  = SpectrumRenderer(width, height)

        # State
        self._active       = False
        self._debug_mode   = False
        self._time         = 0.0
        self._seg_counter  = 0
        self._fps          = 0.0
        self._fps_timer    = time.time()
        self._fps_counter  = 0

        # Current theme
        self._current_theme: ThemeDescriptor = THEME_REGISTRY.get(_DEFAULT_THEME_ID)

        # Last known data
        self._camera_frame:  Optional[np.ndarray] = None
        self._persons:       List[DetectedPerson]  = []
        self._tracks:        List[PersonTrack]      = []

        # Fonts for debug
        pygame.font.init()
        self._font_debug = pygame.font.SysFont('Consolas', 13)

    # ─────────────────────────────────────────────────────────────────────────
    # BaseExperience lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def enter(self) -> None:
        self._active       = True
        self._time         = 0.0
        self._seg_counter  = 0
        self._fps          = 0.0
        self._fps_timer    = time.time()
        self._fps_counter  = 0
        self._persons      = []
        self._tracks       = []
        self._camera_frame = None

        self._tracker.reset()
        self._segmenter.start()
        print('[SPECTRUM] Spectrum Vision — entered.')

    def exit(self) -> None:
        self._active = False
        self._segmenter.stop()
        print('[SPECTRUM] Spectrum Vision — exited.')

    # ─────────────────────────────────────────────────────────────────────────
    # Input
    # ─────────────────────────────────────────────────────────────────────────

    def handle_gesture(
        self,
        gesture: GestureType,
        action: GestureType,
        hand_pos: tuple,
    ) -> None:
        pass  # Spectrum Vision uses keyboard, not hand gestures

    def handle_key(self, event: pygame.event.Event) -> bool:
        key = event.key

        # Theme switching: 1–7
        if pygame.K_1 <= key <= pygame.K_7:
            theme_key = key - pygame.K_0
            descriptor = THEME_REGISTRY.get_by_key(theme_key)
            if descriptor:
                self._current_theme = descriptor
                print(f'[SPECTRUM] Theme → {descriptor.name}')
                return True

        # Reset tracking
        if key == pygame.K_r:
            self._tracker.reset()
            self._persons = []
            self._tracks  = []
            print('[SPECTRUM] Tracking reset.')
            return True

        # Debug toggle
        if key == pygame.K_d:
            self._debug_mode = not self._debug_mode
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Frame injection (called by main loop)
    # ─────────────────────────────────────────────────────────────────────────

    def push_camera_frame(self, frame: np.ndarray) -> None:
        """Store the latest BGR camera frame and sub-sample segmentation."""
        self._camera_frame = frame

        self._seg_counter += 1
        if self._seg_counter >= _SEG_UPDATE_INTERVAL:
            self._seg_counter = 0
            self._segmenter.push_frame(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        if not self._active:
            return

        self._time += dt
        self._fps_counter += 1
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_timer)
            self._fps_counter = 0
            self._fps_timer   = now

        # Advance renderer animations
        self._renderer.update(dt)

        frame = self._camera_frame
        if frame is None:
            return

        fh, fw = frame.shape[:2]

        # Get full-resolution binary mask
        full_mask = self._segmenter.get_mask_for_frame(frame)

        # Decompose into individual person blobs
        self._persons = detect_persons(full_mask, fh, fw)

        # Build detection list for tracker:
        # [(cx, cy, (bx, by, bw, bh)), ...]
        detections = [
            (p.centroid[0], p.centroid[1], p.bbox)
            for p in self._persons
        ]

        # Update tracking
        self._tracks = self._tracker.update(detections, fw, fh)

    # ─────────────────────────────────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface) -> None:
        if not self._active:
            return

        frame = self._camera_frame

        if frame is None:
            # No frame yet — show placeholder
            surface.fill(palette.VOID_DARK)
            msg = self._font_debug.render(
                'INITIALIZING SPECTRUM VISION...', True, palette.CYAN_NEON
            )
            w, h = surface.get_size()
            surface.blit(msg, (w // 2 - msg.get_width() // 2, h // 2))
            return

        # Determine if mock camera
        # We can't access camera directly here; pass through from main if needed.
        # Check if frame looks synthetic (all uniform rows) — approximation.
        is_mock = False  # main.py sets this context

        self._renderer.render_frame(
            surface     = surface,
            bgr_frame   = frame,
            persons     = self._persons,
            tracks      = self._tracks,
            theme       = self._current_theme,
            fps         = self._fps,
            is_mock     = is_mock,
        )

        if self._debug_mode:
            self._draw_debug(surface, frame)

    def set_mock_flag(self, is_mock: bool) -> None:
        """Allow main.py to pass the mock camera flag."""
        self._is_mock = is_mock

    # ─────────────────────────────────────────────────────────────────────────
    # Debug overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_debug(self, surface: pygame.Surface, frame: np.ndarray) -> None:
        fh, fw = frame.shape[:2]
        sw, sh = surface.get_size()
        seg_ready = self._segmenter.ready

        lines = [
            ('SPECTRUM VISION DEBUG', palette.CYAN_NEON),
            (f'Theme:      {self._current_theme.name}', palette.TEXT_WHITE),
            (f'Persons:    {len(self._persons)}', palette.TEXT_WHITE),
            (f'Tracks:     {self._tracker.person_count}', palette.TEXT_WHITE),
            (f'Segmenter:  {"READY" if seg_ready else "UNAVAIL"}',
             palette.GREEN_MATRIX if seg_ready else palette.MAGENTA_LASER),
            (f'Frame:      {fw}x{fh}', palette.TEXT_WHITE),
            (f'FPS:        {int(self._fps)}', palette.TEXT_WHITE),
            (f'Time:       {self._time:.1f}s', palette.TEXT_MUTED),
        ]

        panel_w, panel_h = 300, len(lines) * 16 + 16
        dbg_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        dbg_surf.fill((0, 0, 0, 210))
        pygame.draw.rect(dbg_surf, palette.CYAN_NEON, (0, 0, panel_w, panel_h), 1)
        surface.blit(dbg_surf, (sw - panel_w - 10, 10))

        for i, (text, col) in enumerate(lines):
            s = self._font_debug.render(text, True, col)
            surface.blit(s, (sw - panel_w + 8, 18 + i * 16))
