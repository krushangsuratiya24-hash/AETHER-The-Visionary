"""
AETHER — Spectrum Vision: High-Performance Rendering Pipeline  (Phase 4)

Handles all seven theme rendering passes:
  1. THERMAL   — simulated thermal heat-map visualization
  2. XRAY      — simulated X-ray structural overlay
  3. NEON EDGE — glowing segmentation contour
  4. SILHOUETTE— holographic silhouette scanner
  5. RADAR     — targeting / tracking visualization
  6. ENERGY    — animated energy-field particles
  7. CYBER     — primary showcase: futuristic multi-layer cyber vision

All rendering is composited onto a full-screen Pygame surface.
No fake data is ever generated: every visual derives from actual camera input.
"""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygame

from core.config import palette
from core.particles import ParticlePool
from experiences.spectrum_vision.people import DetectedPerson
from experiences.spectrum_vision.themes import ThemeDescriptor, ThemeID
from experiences.spectrum_vision.tracking import PersonTrack


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _lerp_col(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _clamp255(v: float) -> int:
    return max(0, min(255, int(v)))


def _safe_draw(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Thermal palette (256-entry BGR lookup table, simulated from luminance)
# ─────────────────────────────────────────────────────────────────────────────

def _build_thermal_lut() -> np.ndarray:
    """
    Build a 256-entry thermal colour-map (cold→hot) in BGR.
    Simulates thermal camera false-colour from luminance — NOT real temperature.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Black → violet → blue → cyan → green → yellow → orange → red → white
        if t < 0.125:
            s = t / 0.125
            r, g, b = int(20 * s), 0, int(40 * s)
        elif t < 0.25:
            s = (t - 0.125) / 0.125
            r, g, b = int(20 + 40 * s), 0, int(40 + 80 * s)
        elif t < 0.375:
            s = (t - 0.25) / 0.125
            r, g, b = int(60 * (1 - s)), int(80 * s), int(120 + 135 * s)
        elif t < 0.5:
            s = (t - 0.375) / 0.125
            r, g, b = 0, int(80 + 175 * s), int(255 * (1 - s * 0.6))
        elif t < 0.625:
            s = (t - 0.5) / 0.125
            r, g, b = int(100 * s), 255, int(102 - 102 * s)
        elif t < 0.75:
            s = (t - 0.625) / 0.125
            r, g, b = int(100 + 155 * s), 255, 0
        elif t < 0.875:
            s = (t - 0.75) / 0.125
            r, g, b = 255, int(255 - 130 * s), 0
        else:
            s = (t - 0.875) / 0.125
            r, g, b = 255, int(125 + 130 * s), int(200 * s)

        # Store in BGR order for OpenCV
        lut[i] = [b, g, r]

    return lut


_THERMAL_LUT = _build_thermal_lut()


# ─────────────────────────────────────────────────────────────────────────────
# SpectrumRenderer
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumRenderer:
    """
    Full-screen theme renderer for Spectrum Vision.

    Call render_frame() each frame with the camera frame, detected persons,
    their tracks, the current theme, and the current time.
    """

    # Particle pool for Energy and Cyber themes
    POOL_CAPACITY = 3000

    def __init__(self, width: int, height: int):
        self._w = width
        self._h = height

        # Particle pool (shared across all themes)
        self._pool = ParticlePool(capacity=self.POOL_CAPACITY)

        # Animated state
        self._time:   float = 0.0
        self._scan_angle: float = 0.0   # RADAR sweep
        self._pulse:  float = 0.0

        # Pre-built surfaces (reused every frame)
        self._overlay_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        self._edge_surf     = pygame.Surface((width, height), pygame.SRCALPHA)

        # Font references
        pygame.font.init()
        self._font_hud     = pygame.font.SysFont('Consolas', 14, bold=True)
        self._font_id      = pygame.font.SysFont('Consolas', 16, bold=True)
        self._font_small   = pygame.font.SysFont('Consolas', 12)
        self._font_label   = pygame.font.SysFont('Consolas', 13)

        # Cached noise (refreshed every ~8 frames)
        self._noise_frame: Optional[np.ndarray] = None
        self._noise_age:   int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """Advance all animated state by dt seconds."""
        self._time   += dt
        self._scan_angle = (self._scan_angle + dt * 90.0) % 360.0
        self._pulse   = math.sin(self._time * math.tau * 1.5) * 0.5 + 0.5
        self._pool.update(dt)

        # Refresh noise occasionally
        self._noise_age += 1
        if self._noise_age >= 8:
            self._noise_age = 0
            self._noise_frame = None

    def render_frame(
        self,
        surface: pygame.Surface,
        bgr_frame: np.ndarray,
        persons: List[DetectedPerson],
        tracks:  List[PersonTrack],
        theme:   ThemeDescriptor,
        fps:     float = 0.0,
        is_mock: bool  = False,
    ) -> None:
        """
        Composite the full-frame themed visualization onto `surface`.

        Args:
            surface:   Target pygame surface (full screen).
            bgr_frame: Raw camera frame in BGR (uint8).
            persons:   Detected persons for this frame.
            tracks:    PersonTrack objects (matched by tracking.py).
            theme:     Current ThemeDescriptor.
            fps:       Current FPS for HUD.
            is_mock:   True if camera is mock (synthetic frame).
        """
        sw, sh = surface.get_size()
        frame_h, frame_w = bgr_frame.shape[:2]

        # Scale frame to surface if needed
        if (frame_w, frame_h) != (sw, sh):
            display_bgr = cv2.resize(bgr_frame, (sw, sh))
        else:
            display_bgr = bgr_frame

        # Dispatch to per-theme renderer
        if theme.theme_id == ThemeID.THERMAL:
            self._render_thermal(surface, display_bgr, persons, tracks, sw, sh)
        elif theme.theme_id == ThemeID.XRAY:
            self._render_xray(surface, display_bgr, persons, tracks, sw, sh)
        elif theme.theme_id == ThemeID.NEON_EDGE:
            self._render_neon_edge(surface, display_bgr, persons, tracks, sw, sh)
        elif theme.theme_id == ThemeID.SILHOUETTE:
            self._render_silhouette(surface, display_bgr, persons, tracks, sw, sh)
        elif theme.theme_id == ThemeID.RADAR:
            self._render_radar(surface, display_bgr, persons, tracks, sw, sh)
        elif theme.theme_id == ThemeID.ENERGY:
            self._render_energy(surface, display_bgr, persons, tracks, sw, sh)
        elif theme.theme_id == ThemeID.CYBER:
            self._render_cyber(surface, display_bgr, persons, tracks, sw, sh)

        # Draw particles on top
        self._pool.draw(surface)

        # HUD overlay
        self._draw_spectrum_hud(surface, theme, persons, tracks, fps, is_mock, sw, sh)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: BGR frame → pygame surface
    # ─────────────────────────────────────────────────────────────────────────

    def _bgr_to_surface(self, bgr: np.ndarray) -> pygame.Surface:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: noise texture
    # ─────────────────────────────────────────────────────────────────────────

    def _get_noise(self, h: int, w: int) -> np.ndarray:
        if self._noise_frame is None or self._noise_frame.shape[:2] != (h, w):
            self._noise_frame = np.random.randint(0, 40, (h, w), dtype=np.uint8)
        return self._noise_frame

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: build combined mask for all persons at surface resolution
    # ─────────────────────────────────────────────────────────────────────────

    def _combined_mask(
        self,
        persons: List[DetectedPerson],
        sw: int,
        sh: int,
    ) -> np.ndarray:
        """Return a uint8 binary mask (sh×sw) covering all detected persons."""
        combined = np.zeros((sh, sw), dtype=np.uint8)
        for p in persons:
            pmh, pmw = p.mask.shape[:2]
            if (pmw, pmh) != (sw, sh):
                m = cv2.resize(p.mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            else:
                m = p.mask
            combined = cv2.bitwise_or(combined, m)
        return combined

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: draw target brackets
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_target_brackets(
        self,
        surface: pygame.Surface,
        x: int, y: int, w: int, h: int,
        color: Tuple[int, int, int],
        arm: int = 18,
        thick: int = 2,
        pulse: float = 1.0,
    ) -> None:
        alpha_c = tuple(min(255, int(c * pulse)) for c in color)
        corners = [
            (x,     y,     arm, 0,   0,   arm),
            (x+w,   y,     -arm, 0,  0,   arm),
            (x,     y+h,   arm, 0,   0,   -arm),
            (x+w,   y+h,   -arm, 0,  0,   -arm),
        ]
        for cx, cy, dx1, dy1, dx2, dy2 in corners:
            _safe_draw(pygame.draw.line, surface, alpha_c,
                       (cx, cy), (cx + dx1, cy + dy1), thick)
            _safe_draw(pygame.draw.line, surface, alpha_c,
                       (cx, cy), (cx + dx2, cy + dy2), thick)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. THERMAL SIMULATION
    # ─────────────────────────────────────────────────────────────────────────

    def _render_thermal(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        """
        Thermal-style visual simulation derived from RGB luminance.
        NOT real temperature measurement.
        """
        # Darken background
        dark = (bgr.astype(np.float32) * 0.25).astype(np.uint8)
        base_surf = self._bgr_to_surface(dark)
        surface.blit(base_surf, (0, 0))

        if not persons:
            return

        combined = self._combined_mask(persons, sw, sh)
        person_region_exists = combined.any()

        if not person_region_exists:
            return

        # Convert to grayscale luminance
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Add animated shimmer noise to person region
        noise = self._get_noise(sh, sw)
        gray_noisy = gray.astype(np.int16) + noise.astype(np.int16) * 2
        gray_noisy = np.clip(gray_noisy, 0, 255).astype(np.uint8)

        # Apply thermal LUT via indexed lookup
        thermal_bgr = _THERMAL_LUT[gray_noisy]   # (H, W, 3) BGR

        # Add horizontal scanline effect
        scanline_mask = np.ones((sh, sw), dtype=np.float32)
        offset = int(self._time * 60) % 4
        for y in range(offset, sh, 4):
            if y < sh:
                scanline_mask[y, :] *= 0.75

        # Blend thermal into person area only
        person_float = (combined > 0).astype(np.float32)[:, :, np.newaxis]
        thermal_f = thermal_bgr.astype(np.float32) * scanline_mask[:, :, np.newaxis]
        dark_f = dark.astype(np.float32)
        result = dark_f * (1.0 - person_float) + thermal_f * person_float
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)

        surf = self._bgr_to_surface(result_uint8)
        surface.blit(surf, (0, 0))

        # Glow contour around person edges
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        glow_col = (255, 200, 50)
        for cnt in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            if len(pts) >= 2:
                _safe_draw(pygame.draw.polygon, surface, glow_col, pts, 2)

        # Thermal label
        lbl = self._font_small.render('[SIMULATED THERMAL]', True, (255, 180, 50))
        surface.blit(lbl, (sw - lbl.get_width() - 8, sh - 20))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. X-RAY STYLE
    # ─────────────────────────────────────────────────────────────────────────

    def _render_xray(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        """
        Simulated X-ray structural visualization.
        NOT actual X-ray imaging.
        """
        # Very dark background
        dark = (bgr.astype(np.float32) * 0.12).astype(np.uint8)
        base_surf = self._bgr_to_surface(dark)
        surface.blit(base_surf, (0, 0))

        if not persons:
            return

        # Scan line effect (animated sweep)
        scan_y = int((self._time * 0.4 % 1.0) * sh)
        for y in range(max(0, scan_y - 3), min(sh, scan_y + 3)):
            alpha = max(0, 80 - abs(y - scan_y) * 25)
            _safe_draw(pygame.draw.line, surface, (120, 180, 255),
                       (0, y), (sw, y), 1)

        for idx, person in enumerate(persons):
            pmh, pmw = person.mask.shape[:2]
            if (pmw, pmh) != (sw, sh):
                pm = cv2.resize(person.mask, (sw, sh),
                                interpolation=cv2.INTER_NEAREST)
            else:
                pm = person.mask

            if not pm.any():
                continue

            # Translucent dark blue body fill
            fill_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            pm_rgb = np.zeros((sh, sw, 4), dtype=np.uint8)
            pm_rgb[pm > 0] = [20, 60, 120, 100]
            fill_pg = pygame.surfarray.make_surface(
                np.transpose(pm_rgb[:, :, :3], (1, 0, 2))
            ).convert_alpha()
            fill_pg.set_alpha(100)
            surface.blit(fill_pg, (0, 0))

            # Glowing silhouette contour
            contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                if len(pts) >= 2:
                    _safe_draw(pygame.draw.polygon, surface,
                               (150, 200, 255), pts, 3)
                    _safe_draw(pygame.draw.polygon, surface,
                               (80, 120, 200), pts, 6)

            # Procedural skeleton lines derived from bounding box
            bx, by, bw, bh = person.bbox
            self._draw_procedural_skeleton(surface, bx, by, bw, bh, pm,
                                           (180, 220, 255))

            # Glowing joint nodes (key structural points)
            cx, cy = int(person.centroid[0]), int(person.centroid[1])
            joints = self._estimate_joints(bx, by, bw, bh)
            for jx, jy in joints:
                pulse = 0.6 + 0.4 * math.sin(self._time * 4.0 + jx * 0.01)
                r = int(5 * pulse)
                c = (int(180 * pulse), int(220 * pulse), 255)
                _safe_draw(pygame.draw.circle, surface, c, (jx, jy), r)
                _safe_draw(pygame.draw.circle, surface, (80, 120, 200),
                           (jx, jy), r + 3, 1)

        # Disclaimer label
        lbl = self._font_small.render('[SIMULATED X-RAY]', True, (140, 180, 255))
        surface.blit(lbl, (sw - lbl.get_width() - 8, sh - 20))

    def _estimate_joints(
        self,
        bx: int, by: int, bw: int, bh: int,
    ) -> List[Tuple[int, int]]:
        """Estimate approximate body joint positions from a bounding box."""
        cx = bx + bw // 2
        # Anatomically proportioned joint positions (roughly)
        head_top = by + int(bh * 0.05)
        neck     = by + int(bh * 0.18)
        shoulder_l = bx + int(bw * 0.20)
        shoulder_r = bx + int(bw * 0.80)
        elbow_l   = bx + int(bw * 0.10)
        elbow_r   = bx + int(bw * 0.90)
        hip_l     = bx + int(bw * 0.30)
        hip_r     = bx + int(bw * 0.70)
        knee_l    = bx + int(bw * 0.32)
        knee_r    = bx + int(bw * 0.68)
        torso_mid = by + int(bh * 0.45)
        hip_y     = by + int(bh * 0.62)
        knee_y    = by + int(bh * 0.78)
        foot_y    = by + int(bh * 0.97)

        return [
            (cx, head_top),
            (cx, neck),
            (shoulder_l, neck + int(bh * 0.04)),
            (shoulder_r, neck + int(bh * 0.04)),
            (elbow_l, torso_mid - int(bh * 0.05)),
            (elbow_r, torso_mid - int(bh * 0.05)),
            (cx, torso_mid),
            (hip_l, hip_y),
            (hip_r, hip_y),
            (knee_l, knee_y),
            (knee_r, knee_y),
            (knee_l + int(bw * 0.02), foot_y),
            (knee_r - int(bw * 0.02), foot_y),
        ]

    def _draw_procedural_skeleton(
        self,
        surface: pygame.Surface,
        bx: int, by: int, bw: int, bh: int,
        mask: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw an approximate procedural skeleton from bounding box."""
        j = self._estimate_joints(bx, by, bw, bh)
        if len(j) < 13:
            return

        # Connections: [head, neck, shoulders, elbows, torso, hips, knees, feet]
        connections = [
            (0, 1),   # head → neck
            (1, 2),   # neck → left shoulder
            (1, 3),   # neck → right shoulder
            (2, 4),   # left shoulder → left elbow
            (3, 5),   # right shoulder → right elbow
            (1, 6),   # neck → torso mid
            (6, 7),   # torso → left hip
            (6, 8),   # torso → right hip
            (7, 9),   # left hip → left knee
            (8, 10),  # right hip → right knee
            (9, 11),  # left knee → left foot
            (10, 12), # right knee → right foot
        ]

        pulse = 0.7 + 0.3 * math.sin(self._time * 3.0)
        c = tuple(int(v * pulse) for v in color)
        for i1, i2 in connections:
            if i1 < len(j) and i2 < len(j):
                _safe_draw(pygame.draw.line, surface, c, j[i1], j[i2], 2)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. NEON EDGE
    # ─────────────────────────────────────────────────────────────────────────

    def _render_neon_edge(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        # Darkened live background
        dark = (bgr.astype(np.float32) * 0.30).astype(np.uint8)
        surface.blit(self._bgr_to_surface(dark), (0, 0))

        if not persons:
            return

        combined = self._combined_mask(persons, sw, sh)

        # Multiple glow layers on edges
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        pulse = 0.7 + 0.3 * math.sin(self._time * math.tau * 1.2)

        for cnt in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            if len(pts) < 3:
                continue

            # Outer glow (wide, dim)
            c_outer = (0, int(80 * pulse), int(120 * pulse))
            _safe_draw(pygame.draw.polygon, surface, c_outer, pts, 12)

            # Mid glow
            c_mid = (0, int(160 * pulse), int(230 * pulse))
            _safe_draw(pygame.draw.polygon, surface, c_mid, pts, 5)

            # Inner bright edge
            c_inner = (int(50 * pulse), int(255 * pulse), int(200 * pulse))
            _safe_draw(pygame.draw.polygon, surface, c_inner, pts, 2)

        # Animated contour particles
        t_now = self._time
        if contours and len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            step = max(1, len(cnt) // 30)
            for i in range(0, len(cnt), step):
                px_pt = cnt[i][0]
                px, py = int(px_pt[0]), int(px_pt[1])
                if random.random() < 0.3:
                    self._pool.emit(
                        x=float(px), y=float(py),
                        vx=(random.random() - 0.5) * 1.5,
                        vy=(random.random() - 0.5) * 1.5,
                        life=random.uniform(0.3, 0.8),
                        size=random.uniform(1.5, 3.5),
                        color=(0, 220, 200),
                        color_end=(0, 100, 255),
                        drag=0.95,
                        glow=True,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SILHOUETTE
    # ─────────────────────────────────────────────────────────────────────────

    def _render_silhouette(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        # Live background slightly dimmed
        dimmed = (bgr.astype(np.float32) * 0.35).astype(np.uint8)
        surface.blit(self._bgr_to_surface(dimmed), (0, 0))

        if not persons:
            return

        # Track dict by index
        track_map = {t.track_id: t for t in tracks}

        for idx, person in enumerate(persons):
            pmh, pmw = person.mask.shape[:2]
            if (pmw, pmh) != (sw, sh):
                pm = cv2.resize(person.mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            else:
                pm = person.mask

            if not pm.any():
                continue

            # Holographic dark fill
            fill_col = (10, 25, 60)
            fill_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            # Animated internal texture (subtle scanlines within person)
            scan_offset = int(self._time * 30) % 4
            for y in range(scan_offset, sh, 4):
                row_mask = (pm[y, :] > 0) if y < sh else np.array([False])
                if row_mask.any():
                    xs = np.where(row_mask)[0]
                    if len(xs) >= 2:
                        alpha_v = int(40 + 20 * math.sin(self._time * 2 + y * 0.05))
                        _safe_draw(pygame.draw.line, fill_surf,
                                   (30, 80, 160, alpha_v),
                                   (int(xs[0]), y), (int(xs[-1]), y), 1)

            surface.blit(fill_surf, (0, 0))

            # Edge glow
            pulse = 0.6 + 0.4 * math.sin(self._time * 2.5 + idx * 1.2)
            contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                if len(pts) < 3:
                    continue
                _safe_draw(pygame.draw.polygon, surface,
                           (0, int(60 * pulse), int(120 * pulse)), pts, 8)
                _safe_draw(pygame.draw.polygon, surface,
                           (0, int(220 * pulse), int(255 * pulse)), pts, 2)

            # Track ID
            track = tracks[idx] if idx < len(tracks) else None
            if track:
                cx_t, cy_t = int(track.centroid[0]), int(track.centroid[1])
                bx, by, bw, bh = track.bbox
                id_surf = self._font_id.render(track.label, True, (0, 220, 255))
                surface.blit(id_surf, (bx, max(0, by - 24)))

    # ─────────────────────────────────────────────────────────────────────────
    # 5. RADAR
    # ─────────────────────────────────────────────────────────────────────────

    def _render_radar(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        # Dark desaturated background
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tinted = (gray_3.astype(np.float32) * np.array([0.05, 0.25, 0.08])).astype(np.uint8)
        surface.blit(self._bgr_to_surface(tinted), (0, 0))

        # Grid overlay
        grid_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        g_col = (0, 60, 20, 40)
        for gx in range(0, sw, 80):
            pygame.draw.line(grid_surf, g_col, (gx, 0), (gx, sh), 1)
        for gy in range(0, sh, 80):
            pygame.draw.line(grid_surf, g_col, (0, gy), (sw, gy), 1)
        surface.blit(grid_surf, (0, 0))

        # Radar sweep
        cx_r, cy_r = sw // 2, sh // 2
        sweep_rad = math.radians(self._scan_angle)
        sweep_len = max(sw, sh)
        sx = int(cx_r + math.cos(sweep_rad) * sweep_len)
        sy = int(cy_r + math.sin(sweep_rad) * sweep_len)

        sweep_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for spread in range(0, 25, 3):
            a = math.radians(self._scan_angle - spread)
            ex = int(cx_r + math.cos(a) * sweep_len)
            ey = int(cy_r + math.sin(a) * sweep_len)
            alpha = max(5, 60 - spread * 2)
            pygame.draw.line(sweep_surf, (0, 255, 80, alpha), (cx_r, cy_r), (ex, ey), 1)
        surface.blit(sweep_surf, (0, 0))

        # Per-track rendering
        for track in tracks:
            if not track.is_active:
                continue
            bx, by, bw, bh = track.bbox

            # Ensure bbox within surface
            bx = max(0, bx); by = max(0, by)
            bx2 = min(sw, bx + bw); by2 = min(sh, by + bh)
            bw, bh = bx2 - bx, by2 - by
            if bw <= 0 or bh <= 0:
                continue

            pulse = 0.7 + 0.3 * math.sin(self._time * 3.0 + track.track_id)
            c_green = (0, int(255 * pulse), int(80 * pulse))

            # Target brackets
            self._draw_target_brackets(surface, bx, by, bw, bh,
                                       c_green, arm=20, thick=2, pulse=pulse)

            # Center cross
            tcx = bx + bw // 2
            tcy = by + bh // 2
            _safe_draw(pygame.draw.line, surface, c_green,
                       (tcx - 8, tcy), (tcx + 8, tcy), 1)
            _safe_draw(pygame.draw.line, surface, c_green,
                       (tcx, tcy - 8), (tcx, tcy + 8), 1)

            # ID label
            label_surf = self._font_id.render(track.label, True, c_green)
            state_surf = self._font_small.render('TRACKING', True, (0, 180, 60))
            surface.blit(label_surf, (bx, max(0, by - 26)))
            surface.blit(state_surf, (bx, max(0, by - 12)))

            # Trail
            if len(track.trail) > 2:
                for i in range(1, len(track.trail)):
                    ta = int(100 * (i / len(track.trail)))
                    tc = (0, min(255, int(200 * (i / len(track.trail)))), 40)
                    _safe_draw(pygame.draw.line, surface, tc,
                               (int(track.trail[i - 1][0]), int(track.trail[i - 1][1])),
                               (int(track.trail[i][0]),     int(track.trail[i][1])), 1)

        # Center reticle
        _safe_draw(pygame.draw.circle, surface, (0, 100, 40),
                   (cx_r, cy_r), 12, 1)
        _safe_draw(pygame.draw.line, surface, (0, 100, 40),
                   (cx_r - 18, cy_r), (cx_r + 18, cy_r), 1)
        _safe_draw(pygame.draw.line, surface, (0, 100, 40),
                   (cx_r, cy_r - 18), (cx_r, cy_r + 18), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. ENERGY
    # ─────────────────────────────────────────────────────────────────────────

    def _render_energy(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        # Background — tinted dark orange
        dark = (bgr.astype(np.float32) * 0.20).astype(np.uint8)
        surface.blit(self._bgr_to_surface(dark), (0, 0))

        if not persons:
            return

        for idx, person in enumerate(persons):
            pmh, pmw = person.mask.shape[:2]
            if (pmw, pmh) != (sw, sh):
                pm = cv2.resize(person.mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            else:
                pm = person.mask

            if not pm.any():
                continue

            cx_p, cy_p = person.centroid

            # Inner aura (body glow)
            aura_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            pulse = 0.6 + 0.4 * math.sin(self._time * 4.0 + idx * 1.5)
            aura_alpha = int(60 * pulse)
            aura_bgr = np.zeros((sh, sw, 4), dtype=np.uint8)
            aura_bgr[pm > 0] = [255, 200, 0, aura_alpha]
            aura_pg = pygame.surfarray.make_surface(
                np.transpose(aura_bgr[:, :, :3], (1, 0, 2))
            ).convert_alpha()
            aura_pg.set_alpha(aura_alpha)
            surface.blit(aura_pg, (0, 0))

            # Energy contour
            contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                if len(pts) < 3:
                    continue
                c1 = (255, int(220 * pulse), 0)
                _safe_draw(pygame.draw.polygon, surface, c1, pts, 2)

            # Energy arc strands — procedural lightning between person edges
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                n_strands = 6
                stride = max(1, len(cnt) // n_strands)
                pts_list = [cnt[i][0] for i in range(0, len(cnt), stride)]

                for i in range(0, len(pts_list) - 1, 2):
                    p1 = pts_list[i]
                    p2 = pts_list[min(i + 1, len(pts_list) - 1)]
                    mid_x = (p1[0] + p2[0]) // 2 + random.randint(-15, 15)
                    mid_y = (p1[1] + p2[1]) // 2 + random.randint(-15, 15)
                    arc_col = (255, int(180 * pulse), int(50 * pulse))
                    _safe_draw(pygame.draw.line, surface, arc_col,
                               (int(p1[0]), int(p1[1])), (mid_x, mid_y), 1)
                    _safe_draw(pygame.draw.line, surface, arc_col,
                               (mid_x, mid_y), (int(p2[0]), int(p2[1])), 1)

            # Emit energy particles from contour
            if contours and self._time % 0.04 < 0.04:
                cnt = max(contours, key=cv2.contourArea)
                step = max(1, len(cnt) // 12)
                for i in range(0, len(cnt), step):
                    px_pt = cnt[i][0]
                    angle = random.uniform(0, math.tau)
                    speed = random.uniform(0.8, 2.5)
                    self._pool.emit(
                        x=float(px_pt[0]), y=float(px_pt[1]),
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed,
                        life=random.uniform(0.4, 1.2),
                        size=random.uniform(2.0, 5.0),
                        color=(255, 200, 0),
                        color_end=(255, 80, 0),
                        drag=0.93,
                        glow=True,
                        turbulence=0.3,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. CYBER VISION (Primary Showcase)
    # ─────────────────────────────────────────────────────────────────────────

    def _render_cyber(
        self,
        surface: pygame.Surface,
        bgr: np.ndarray,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        sw: int, sh: int,
    ) -> None:
        """
        Primary showcase mode: futuristic holographic multi-layer cyber visualization.
        Each detected person receives their own independent visual treatment.
        """
        # ── Background: dark with moving grid ────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tinted = (gray_3.astype(np.float32) * np.array([0.04, 0.10, 0.15])).astype(np.uint8)
        surface.blit(self._bgr_to_surface(tinted), (0, 0))

        # Moving digital grid
        grid_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        grid_shift = int(self._time * 20) % 40
        for gx in range(-grid_shift, sw, 40):
            alpha_g = 25 + int(10 * math.sin(self._time + gx * 0.02))
            pygame.draw.line(grid_surf, (0, 80, 120, alpha_g), (gx, 0), (gx, sh), 1)
        for gy in range(-grid_shift, sh, 40):
            alpha_g = 25 + int(10 * math.sin(self._time + gy * 0.02))
            pygame.draw.line(grid_surf, (0, 80, 120, alpha_g), (0, gy), (sw, gy), 1)
        surface.blit(grid_surf, (0, 0))

        if not persons:
            # No-person scan line animation
            scan_y = int((self._time * 0.5 % 1.0) * sh)
            pulse = 0.5 + 0.5 * math.sin(self._time * 3.0)
            col = (0, int(120 * pulse), int(200 * pulse))
            _safe_draw(pygame.draw.line, surface, col, (0, scan_y), (sw, scan_y), 2)
            return

        # ── Per-person rendering ──────────────────────────────────────────────
        person_colors = [
            ((0, 240, 255), (100, 0, 255)),    # cyan / purple
            ((0, 255, 160), (0, 100, 255)),    # teal / blue
            ((180, 0, 255), (0, 200, 255)),    # violet / cyan
            ((255, 200, 0), (255, 60, 0)),     # gold / orange
        ]

        for idx, person in enumerate(persons):
            pmh, pmw = person.mask.shape[:2]
            if (pmw, pmh) != (sw, sh):
                pm = cv2.resize(person.mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            else:
                pm = person.mask

            if not pm.any():
                continue

            c_pri, c_sec = person_colors[idx % len(person_colors)]
            pulse_p = 0.7 + 0.3 * math.sin(self._time * 2.5 + idx * 1.1)

            # ── Holographic body fill ─────────────────────────────────────────
            holo_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            scan_offset = int(self._time * 50 + idx * 7) % 3
            for y in range(scan_offset, sh, 3):
                if y >= sh:
                    continue
                row_mask = pm[y, :]
                xs = np.where(row_mask > 0)[0]
                if len(xs) < 2:
                    continue
                base_alpha = int(40 + 20 * math.sin(self._time * 3.0 + y * 0.04))
                pygame.draw.line(
                    holo_surf,
                    (*c_pri[:2], min(255, base_alpha), min(255, base_alpha)),
                    (int(xs[0]), y), (int(xs[-1]), y), 1
                )
            surface.blit(holo_surf, (0, 0))

            # ── Animated scanlines through body ──────────────────────────────
            scan_y_body = int((self._time * 0.6 + idx * 0.3) % 1.0 * sh)
            scan_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            for dy in range(-4, 5):
                sy2 = scan_y_body + dy
                if 0 <= sy2 < sh:
                    row_mask2 = pm[sy2, :]
                    xs2 = np.where(row_mask2 > 0)[0]
                    if len(xs2) >= 2:
                        a_val = max(0, 120 - abs(dy) * 25)
                        pygame.draw.line(
                            scan_surf,
                            (min(255, c_pri[0]), min(255, c_pri[1]), min(255, c_pri[2]), a_val),
                            (int(xs2[0]), sy2), (int(xs2[-1]), sy2), 1,
                        )
            surface.blit(scan_surf, (0, 0))

            # ── Multi-layer edge glow ─────────────────────────────────────────
            contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                if len(pts) < 3:
                    continue
                # Deep outer glow
                _safe_draw(pygame.draw.polygon, surface,
                           (max(0, c_sec[0] // 4), max(0, c_sec[1] // 4),
                            max(0, c_sec[2] // 4)), pts, 14)
                # Mid glow
                c_mid = tuple(int(v * pulse_p * 0.6) for v in c_sec)
                _safe_draw(pygame.draw.polygon, surface, c_mid, pts, 6)
                # Inner bright
                c_bright = tuple(int(v * pulse_p) for v in c_pri)
                _safe_draw(pygame.draw.polygon, surface, c_bright, pts, 2)

            # ── Energy pulse traveling through body ───────────────────────────
            pulse_pos = (self._time * 0.8 + idx * 0.4) % 1.0
            pulse_y = int(person.bbox[1] + person.bbox[3] * pulse_pos)
            pulse_mask = pm[min(pulse_y, sh - 1), :]
            px_list = np.where(pulse_mask > 0)[0]
            if len(px_list) >= 2:
                pulse_col = tuple(min(255, int(v * (0.5 + pulse_pos))) for v in c_pri)
                _safe_draw(pygame.draw.line, surface, pulse_col,
                           (int(px_list[0]), pulse_y), (int(px_list[-1]), pulse_y), 3)

            # ── Edge distortion / glitch accents ─────────────────────────────
            glitch_t = math.sin(self._time * 7.3 + idx * 2.1)
            if glitch_t > 0.7 and contours:
                cnt_g = max(contours, key=cv2.contourArea)
                glitch_idx = int(len(cnt_g) * random.random())
                gp = cnt_g[glitch_idx % len(cnt_g)][0]
                gx_p, gy_p = int(gp[0]), int(gp[1])
                _safe_draw(pygame.draw.circle, surface,
                           (255, 255, 255), (gx_p, gy_p), 3)

            # ── Tracking brackets ─────────────────────────────────────────────
            track = tracks[idx] if idx < len(tracks) else None
            bx, by, bw, bh = person.bbox
            if track:
                bx, by, bw, bh = track.bbox
            bx = max(0, bx); by = max(0, by)
            bx = min(sw - 1, bx); by = min(sh - 1, by)
            bw = min(bw, sw - bx); bh = min(bh, sh - by)
            if bw > 0 and bh > 0:
                self._draw_target_brackets(surface, bx, by, bw, bh,
                                           c_pri, arm=22, thick=2, pulse=pulse_p)

            # ── Data labels ───────────────────────────────────────────────────
            if track:
                lbl_x = bx + bw + 8
                lbl_y = by

                label_str = track.label
                status_str = 'TRACKING'
                id_surf  = self._font_id.render(label_str, True, c_pri)
                st_surf  = self._font_label.render(status_str, True,
                                                   tuple(int(v * 0.7) for v in c_pri))
                # Area info derived from actual mask
                area_str = f'AREA: {person.area // 100}K'
                ar_surf  = self._font_small.render(area_str, True, c_sec)

                lbl_x_clamp = min(lbl_x, sw - id_surf.get_width() - 4)
                surface.blit(id_surf,  (lbl_x_clamp, lbl_y))
                surface.blit(st_surf,  (lbl_x_clamp, lbl_y + 18))
                surface.blit(ar_surf,  (lbl_x_clamp, lbl_y + 32))
            else:
                # No track yet — show detection index
                det_str = f'P{idx + 1}'
                det_s = self._font_id.render(det_str, True, c_pri)
                surface.blit(det_s, (bx, max(0, by - 22)))

            # ── Particles ────────────────────────────────────────────────────
            if contours and self._time % 0.05 < 0.05:
                cnt_pt = max(contours, key=cv2.contourArea)
                step_pt = max(1, len(cnt_pt) // 15)
                for pi in range(0, len(cnt_pt), step_pt):
                    pp = cnt_pt[pi][0]
                    angle = random.uniform(0, math.tau)
                    speed = random.uniform(0.3, 1.5)
                    self._pool.emit(
                        x=float(pp[0]), y=float(pp[1]),
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed,
                        life=random.uniform(0.5, 1.5),
                        size=random.uniform(1.5, 4.0),
                        color=c_pri,
                        color_end=c_sec,
                        drag=0.96,
                        glow=True,
                    )

        # ── Global chromatic overlay ──────────────────────────────────────────
        scan_full_y = int((self._time * 0.3) % 1.0 * sh)
        chrom_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for dy2 in range(-2, 3):
            sy3 = scan_full_y + dy2
            if 0 <= sy3 < sh:
                alpha_s = max(0, 40 - abs(dy2) * 15)
                pygame.draw.line(chrom_surf, (0, 200, 255, alpha_s),
                                 (0, sy3), (sw, sy3), 1)
        surface.blit(chrom_surf, (0, 0))

    # ─────────────────────────────────────────────────────────────────────────
    # HUD
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_spectrum_hud(
        self,
        surface: pygame.Surface,
        theme: ThemeDescriptor,
        persons: List[DetectedPerson],
        tracks: List[PersonTrack],
        fps: float,
        is_mock: bool,
        sw: int, sh: int,
    ) -> None:
        # ── Top-left panel ────────────────────────────────────────────────────
        panel_w, panel_h = 240, 130
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((8, 12, 22, 200))
        pygame.draw.rect(panel_surf, theme.primary_color,
                         (0, 0, panel_w, panel_h), 1, border_radius=6)
        surface.blit(panel_surf, (10, 10))

        px, py = 18, 16
        title = self._font_hud.render('SPECTRUM VISION', True, theme.primary_color)
        surface.blit(title, (px, py))
        py += 18

        mode_s = self._font_hud.render(f'MODE: {theme.short_name}', True,
                                       theme.secondary_color)
        surface.blit(mode_s, (px, py))
        py += 16

        n_people = len([t for t in tracks if t.is_active])
        ppl_s = self._font_hud.render(f'PEOPLE: {n_people:02d}', True,
                                      palette.TEXT_WHITE)
        surface.blit(ppl_s, (px, py))
        py += 16

        trk_state = 'ACTIVE' if n_people > 0 else 'SCANNING'
        trk_col = palette.GREEN_MATRIX if n_people > 0 else palette.TEXT_MUTED
        trk_s = self._font_hud.render(f'TRACKING: {trk_state}', True, trk_col)
        surface.blit(trk_s, (px, py))
        py += 16

        fps_col = (palette.GREEN_MATRIX if fps >= 30
                   else palette.GOLD_ACCENT if fps >= 20
                   else palette.MAGENTA_LASER)
        fps_s = self._font_hud.render(f'FPS: {int(fps):02d}', True, fps_col)
        surface.blit(fps_s, (px, py))

        # ── Theme selector (bottom-left) ──────────────────────────────────────
        sel_surf = pygame.Surface((220, 108), pygame.SRCALPHA)
        sel_surf.fill((8, 12, 22, 180))
        pygame.draw.rect(sel_surf, (30, 40, 60), (0, 0, 220, 108), 1, border_radius=4)
        surface.blit(sel_surf, (10, sh - 118))

        theme_names = [
            (1, 'THERMAL'), (2, 'X-RAY'), (3, 'NEON EDGE'), (4, 'SILHOUETTE'),
            (5, 'RADAR'),   (6, 'ENERGY'), (7, 'CYBER'),
        ]
        for row, (k, name) in enumerate(theme_names):
            is_active_theme = (k == theme.key)
            col = theme.primary_color if is_active_theme else palette.TEXT_MUTED
            txt = self._font_small.render(
                f'[{k}] {"▶ " if is_active_theme else "  "}{name}',
                True, col,
            )
            surface.blit(txt, (16, sh - 112 + row * 14))

        # ── Top-right: per-person tracking list ──────────────────────────────
        active = [t for t in tracks if t.is_active]
        if active:
            trk_panel_h = len(active) * 32 + 20
            trk_panel_w = 160
            trk_surf = pygame.Surface((trk_panel_w, trk_panel_h), pygame.SRCALPHA)
            trk_surf.fill((8, 12, 22, 190))
            pygame.draw.rect(trk_surf, theme.primary_color,
                             (0, 0, trk_panel_w, trk_panel_h), 1, border_radius=4)
            surface.blit(trk_surf, (sw - trk_panel_w - 10, 10))

            for ti, t in enumerate(active):
                id_s = self._font_hud.render(t.label, True, theme.primary_color)
                st_s = self._font_small.render('TRACKING', True, palette.GREEN_MATRIX)
                ty_off = 14 + ti * 32
                surface.blit(id_s, (sw - trk_panel_w + 8, 10 + ty_off))
                surface.blit(st_s, (sw - trk_panel_w + 8, 10 + ty_off + 15))

        # ── Bottom controls bar ───────────────────────────────────────────────
        hint = self._font_small.render(
            '[1-7] Theme  [R] Reset Tracking  [D] Debug  [ESC] Menu  [Q] Quit',
            True, palette.TEXT_MUTED,
        )
        bg_h = pygame.Surface((hint.get_width() + 16, 16), pygame.SRCALPHA)
        bg_h.fill((0, 0, 0, 140))
        hx = sw // 2 - hint.get_width() // 2
        surface.blit(bg_h, (hx - 8, sh - 18))
        surface.blit(hint, (hx, sh - 17))

        # ── Mock camera indicator ─────────────────────────────────────────────
        if is_mock:
            mock_s = self._font_small.render('MOCK CAM', True, palette.MAGENTA_LASER)
            surface.blit(mock_s, (sw - mock_s.get_width() - 10, sh - 32))
