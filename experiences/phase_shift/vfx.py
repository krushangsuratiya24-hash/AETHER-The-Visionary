"""
AETHER — Phase Shift Visual Effects  (Phase 3)

Original sci-fi visual identity: "Quantum Dissolution / Spectral Reconstruction"

Visual language:
  • Holographic scanlines pulsing across the body silhouette
  • Cyan-violet energy edge glow along the person's outline
  • Particle fragments exploding outward during phase-out
  • Digital pixel-grid dissolve (the body breaks into quantum tiles)
  • Reconstruction particles converging from edges during phase-in
  • Chromatic aberration / glitch distortion at transition peaks
  • Residual energy ring shockwave at transition completion
  • Subtle body outline ghost during invisible state

All rendering targets the pygame.Surface directly (no secondary windows).
No copyrighted references.
"""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygame

from core.particles import ParticlePool, lerp_color


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for Phase Shift
# ─────────────────────────────────────────────────────────────────────────────

PHASE_CYAN    = (0,   220, 255)
PHASE_VIOLET  = (180,  60, 255)
PHASE_TEAL    = (0,   255, 180)
PHASE_WHITE   = (220, 240, 255)
PHASE_GHOST   = (100, 200, 220)
PHASE_ORANGE  = (255, 140,  30)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blend_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _shimmer(t: float, freq: float = 6.0) -> float:
    """Returns 0..1 shimmer value."""
    return 0.5 + 0.5 * math.sin(t * freq * math.tau)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumScanlines — animated horizontal scanlines over the body silhouette
# ─────────────────────────────────────────────────────────────────────────────

class QuantumScanlines:
    """
    Draws moving horizontal scanline bands over the body region.
    Gives a holographic / data-projection feel.
    """

    def __init__(self, spacing: int = 4, alpha: float = 0.18):
        self._spacing = spacing
        self._alpha   = alpha
        self._phase   = 0.0

    def update(self, dt: float):
        self._phase += dt * 0.8   # slow crawl

    def draw(self, surface: pygame.Surface,
             mask_bgr: np.ndarray, blend_alpha: float,
             sx: int, sy: int):
        """
        Draw scanlines within the person's bounding region.

        mask_bgr: BGR overlay image (person region highlighted, bg transparent).
        sx, sy:   pixel offsets of mask_bgr in surface coordinates.
        """
        if blend_alpha < 0.05:
            return

        h, w = mask_bgr.shape[:2]
        intensity = int(blend_alpha * self._alpha * 255)
        if intensity < 4:
            return

        # Animated offset
        offset_y = int((self._phase % 1.0) * self._spacing)

        # Draw scanlines directly onto surface
        for y in range(offset_y, h, self._spacing):
            screen_y = sy + y
            if 0 <= screen_y < surface.get_height():
                col = (
                    min(255, PHASE_CYAN[0] * intensity // 255 + 10),
                    min(255, PHASE_CYAN[1] * intensity // 255 + 10),
                    min(255, PHASE_CYAN[2] * intensity // 255),
                )
                try:
                    pygame.draw.line(surface, col, (sx, screen_y), (sx + w, screen_y), 1)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# EdgeGlow — glowing silhouette outline
# ─────────────────────────────────────────────────────────────────────────────

class EdgeGlow:
    """
    Computes and draws a glowing energy edge along the person's mask boundary.
    """

    def __init__(self):
        self._phase = 0.0

    def update(self, dt: float):
        self._phase += dt * 2.0

    def draw_on_frame(self, frame: np.ndarray,
                      person_mask: np.ndarray,
                      blend_alpha: float) -> np.ndarray:
        """
        Overlays an energy edge glow onto frame (BGR numpy array).
        Returns modified frame.
        """
        if blend_alpha < 0.05:
            return frame

        h, w = frame.shape[:2]

        # Resize mask if needed
        if person_mask.shape[:2] != (h, w):
            mask = cv2.resize(person_mask, (w, h))
        else:
            mask = person_mask

        # Find edges of the mask via dilation - original
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(mask, kernel, iterations=2)
        edge_mask = cv2.subtract(dilated, mask)   # ring of pixels around person

        # Pulse the glow
        pulse = 0.7 + 0.3 * math.sin(self._phase * math.tau)
        alpha_f = blend_alpha * pulse

        # Colour cycles between cyan and violet with phase
        t_col = 0.5 + 0.5 * math.sin(self._phase * 0.8 * math.tau)
        glow_color = _blend_color(PHASE_CYAN, PHASE_VIOLET, t_col)

        # Apply glow: tint edge pixels toward glow_color
        edge_3ch = edge_mask[:, :, np.newaxis].astype(np.float32) / 255.0
        overlay = np.zeros_like(frame, dtype=np.float32)
        overlay[:] = glow_color

        result = frame.astype(np.float32)
        result = result + edge_3ch * overlay * alpha_f * 0.8
        return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# PixelDissolve — the body fragmenting into grid tiles
# ─────────────────────────────────────────────────────────────────────────────

class PixelDissolve:
    """
    Draws a grid-dissolve overlay on the person region.
    Random tiles vanish or appear depending on transition direction.
    """

    TILE_SIZE = 12   # pixels per quantum tile

    def __init__(self):
        self._tiles: List[Tuple[int,int,float]] = []   # (gx, gy, birth_t)
        self._rng = random.Random(42)

    def reset(self, w: int, h: int):
        self._tiles = []
        nx = w // self.TILE_SIZE + 1
        ny = h // self.TILE_SIZE + 1
        # Assign random stagger times [0, 1] for each tile
        self._tile_grid = [
            (gx * self.TILE_SIZE, gy * self.TILE_SIZE,
             self._rng.random())
            for gy in range(ny)
            for gx in range(nx)
        ]

    def draw(self, surface: pygame.Surface,
             person_mask: np.ndarray,
             blend_alpha: float,
             sx: int, sy: int,
             phase_out: bool = True):
        """
        Draw tile dissolve overlay on surface.

        phase_out=True  → tiles disappear as blend_alpha rises 0→1
        phase_out=False → tiles appear as blend_alpha falls 1→0
        """
        if blend_alpha < 0.03 or not hasattr(self, '_tile_grid'):
            return

        ts = self.TILE_SIZE
        mh, mw = person_mask.shape[:2]

        for (tx, ty, rnd_t) in self._tile_grid:
            # Is this tile inside the person region?
            mid_x = min(tx + ts // 2, mw - 1)
            mid_y = min(ty + ts // 2, mh - 1)
            if person_mask[mid_y, mid_x] == 0:
                continue   # background tile — skip

            # This tile fades at time: rnd_t  (0=early, 1=last)
            if phase_out:
                visible = rnd_t > blend_alpha
            else:
                visible = rnd_t > (1.0 - blend_alpha)

            if not visible:
                # Draw ghost tile
                brightness = int(blend_alpha * 90)
                t_col = 0.5 + 0.5 * math.sin(rnd_t * math.tau)
                col = _blend_color(PHASE_CYAN, PHASE_VIOLET, t_col)
                col = tuple(min(255, c * brightness // 80) for c in col)
                rx = sx + tx
                ry = sy + ty
                if (0 <= rx < surface.get_width() - ts
                        and 0 <= ry < surface.get_height() - ts):
                    try:
                        pygame.draw.rect(surface, col, (rx, ry, ts - 1, ts - 1), 1)
                    except Exception:
                        pass


# ─────────────────────────────────────────────────────────────────────────────
# PhaseParticles — fragment and reconstruction particles
# ─────────────────────────────────────────────────────────────────────────────

class PhaseParticles:
    """
    Manages the particle bursts for phase-out (fragmentation) and
    phase-in (reconstruction).
    """

    def __init__(self, pool: ParticlePool):
        self._pool = pool

    def burst_phase_out(self, person_mask: np.ndarray,
                        frame_w: int, frame_h: int,
                        count: int = 120):
        """
        Spawn fragment particles scattered around the body silhouette.
        """
        mh, mw = person_mask.shape[:2]
        scale_x = frame_w / mw
        scale_y = frame_h / mh

        # Sample random points on the person boundary
        ys, xs = np.where(person_mask > 127)
        if len(xs) == 0:
            return

        idxs = np.random.choice(len(xs), min(count, len(xs)), replace=False)
        for i in idxs:
            px = xs[i] * scale_x
            py = ys[i] * scale_y

            angle = random.uniform(0, math.tau)
            speed = random.uniform(2.0, 8.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            t_col = random.random()
            color = _blend_color(PHASE_CYAN, PHASE_VIOLET, t_col)

            self._pool.emit(
                x=px, y=py,
                vx=vx, vy=vy,
                life=random.uniform(0.5, 1.4),
                size=random.uniform(2.0, 6.0),
                min_size=0.3,
                color=color,
                color_end=PHASE_WHITE,
                gravity=-0.02,
                drag=0.94,
                glow=True,
                shape='shard' if random.random() < 0.4 else 'circle',
                turbulence=0.4,
            )

    def burst_phase_in(self, person_mask: np.ndarray,
                       frame_w: int, frame_h: int,
                       count: int = 90):
        """
        Spawn reconstruction particles converging inward toward the body center.
        """
        mh, mw = person_mask.shape[:2]
        scale_x = frame_w / mw
        scale_y = frame_h / mh

        ys, xs = np.where(person_mask > 127)
        if len(xs) == 0:
            return

        # Center of mass
        cx = float(np.mean(xs)) * scale_x
        cy = float(np.mean(ys)) * scale_y

        idxs = np.random.choice(len(xs), min(count, len(xs)), replace=False)
        for i in idxs:
            # Start far from body, converge toward body
            px = xs[i] * scale_x + random.uniform(-80, 80)
            py = ys[i] * scale_y + random.uniform(-80, 80)

            dx = xs[i] * scale_x - px
            dy = ys[i] * scale_y - py
            dist = math.sqrt(dx**2 + dy**2) + 1.0
            speed = random.uniform(3.0, 7.0)
            vx = (dx / dist) * speed
            vy = (dy / dist) * speed

            t_col = random.random()
            color = _blend_color(PHASE_TEAL, PHASE_VIOLET, t_col)

            self._pool.emit(
                x=px, y=py,
                vx=vx, vy=vy,
                life=random.uniform(0.4, 1.0),
                size=random.uniform(1.5, 5.0),
                min_size=0.2,
                color=color,
                color_end=PHASE_WHITE,
                gravity=0.0,
                drag=0.96,
                glow=True,
                shape='circle',
            )


# ─────────────────────────────────────────────────────────────────────────────
# GlitchDistortion — chromatic aberration + pixel shift
# ─────────────────────────────────────────────────────────────────────────────

class GlitchDistortion:
    """
    Applies a chromatic-aberration / horizontal-glitch distortion to the
    camera frame near the transition peak.
    """

    def apply(self, frame: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply glitch to frame.  strength 0..1.
        Returns distorted frame (BGR uint8).
        """
        if strength < 0.05:
            return frame

        h, w = frame.shape[:2]
        result = frame.copy()

        # Chromatic aberration: shift R and B channels laterally
        shift_px = int(strength * 12)
        if shift_px > 0:
            # R channel shift right
            result[:, shift_px:, 2] = frame[:, :-shift_px, 2]
            # B channel shift left
            result[:, :-shift_px, 0] = frame[:, shift_px:, 0]

        # Random horizontal glitch lines (only at high strength)
        if strength > 0.4:
            n_lines = int(strength * 8)
            for _ in range(n_lines):
                gy = random.randint(0, h - 1)
                gx_shift = random.randint(-int(strength * 20), int(strength * 20))
                if gx_shift > 0:
                    result[gy, gx_shift:] = frame[gy, :-gx_shift] if gx_shift < w else frame[gy]
                elif gx_shift < 0:
                    result[gy, :gx_shift] = frame[gy, -gx_shift:] if -gx_shift < w else frame[gy]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# ShockwaveRing — energy ring at transition completion
# ─────────────────────────────────────────────────────────────────────────────

class PhaseShockwave:
    """Single expanding ring fired at transition completion."""

    def __init__(self, cx: int, cy: int, color: Tuple[int,int,int]):
        self.cx = cx
        self.cy = cy
        self.color = color
        self.life = 0.7
        self.duration = 0.7
        self.alive = True

    def update(self, dt: float):
        self.life -= dt
        if self.life <= 0:
            self.alive = False

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        t = 1.0 - (self.life / self.duration)
        r = int(280 * t)
        alpha_f = 1.0 - t
        if r < 2:
            return
        for width, offset in [(5, 0), (3, 8), (1, 15)]:
            ro = r - offset
            if ro <= 0:
                continue
            bright = int(alpha_f * (1.0 - offset / 20.0) * 200)
            col = tuple(min(255, c * bright // 200) for c in self.color)
            try:
                pygame.draw.circle(surface, col, (self.cx, self.cy), ro, width)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# PhaseShiftVFX — orchestrator for all Phase Shift visual effects
# ─────────────────────────────────────────────────────────────────────────────

class PhaseShiftVFX:
    """
    Orchestrates all VFX for the Phase Shift experience.

    Call in the main loop::

        vfx.update(dt, state, blend_alpha, person_mask)
        vfx.on_phase_out_start(person_mask, frame_w, frame_h)   # once
        vfx.on_phase_in_start(person_mask, frame_w, frame_h)    # once
        vfx.on_transition_complete(cx, cy, phase_out)           # once

        # In render pipeline:
        composited_frame = vfx.apply_frame_effects(composited_frame, person_mask, blend_alpha)
        vfx.draw_pygame_effects(surface, person_mask_resized, blend_alpha, state)
    """

    def __init__(self, width: int, height: int, pool: ParticlePool):
        self._width  = width
        self._height = height

        self._scanlines   = QuantumScanlines()
        self._edge_glow   = EdgeGlow()
        self._tile_dissolve = PixelDissolve()
        self._particles   = PhaseParticles(pool)
        self._glitch      = GlitchDistortion()
        self._shockwaves: List[PhaseShockwave] = []

        self._pool = pool
        self._time = 0.0
        self._phase_out_active = False

        # Pre-build tile grid
        self._tile_dissolve.reset(width, height)

    def update(self, dt: float):
        self._time += dt
        self._scanlines.update(dt)
        self._edge_glow.update(dt)
        self._pool.update(dt)
        self._shockwaves = [sw for sw in self._shockwaves if sw.alive]
        for sw in self._shockwaves:
            sw.update(dt)

    def on_phase_out_start(self, person_mask: np.ndarray,
                           frame_w: int, frame_h: int):
        """Called once when PHASING_OUT transition begins."""
        self._phase_out_active = True
        self._particles.burst_phase_out(person_mask, frame_w, frame_h, count=120)

    def on_phase_in_start(self, person_mask: np.ndarray,
                          frame_w: int, frame_h: int):
        """Called once when PHASING_IN transition begins."""
        self._phase_out_active = False
        self._particles.burst_phase_in(person_mask, frame_w, frame_h, count=90)

    def on_transition_complete(self, cx: int, cy: int, phase_out: bool):
        """Called once when a transition finishes (for shockwave)."""
        color = PHASE_CYAN if phase_out else PHASE_TEAL
        self._shockwaves.append(PhaseShockwave(cx, cy, color))

    def apply_frame_effects(self,
                            frame: np.ndarray,
                            person_mask: np.ndarray,
                            blend_alpha: float) -> np.ndarray:
        """
        Apply OpenCV-based frame effects (edge glow, glitch).
        Returns modified BGR frame.
        """
        # Edge glow along silhouette
        frame = self._edge_glow.draw_on_frame(frame, person_mask, blend_alpha)

        # Chromatic glitch — peak near mid-transition
        glitch_strength = math.sin(blend_alpha * math.pi) * 0.6
        if glitch_strength > 0.05:
            frame = self._glitch.apply(frame, glitch_strength)

        return frame

    def draw_pygame_effects(self, surface: pygame.Surface,
                            person_mask: np.ndarray,
                            blend_alpha: float,
                            phase_out: bool = True):
        """
        Draw Pygame-layer effects (particles, scanlines, tile dissolve, shockwaves).
        """
        # Particles
        self._pool.draw(surface)

        # Scanlines over person region (visible during transitions + invisible)
        if blend_alpha > 0.1:
            self._scanlines.draw(surface, np.zeros_like(person_mask), blend_alpha, 0, 0)

        # Pixel-grid dissolve tiles
        if 0.05 < blend_alpha < 0.98:
            self._tile_dissolve.draw(
                surface, person_mask, blend_alpha, 0, 0,
                phase_out=phase_out
            )

        # Ghost silhouette outline when invisible (subtle)
        if blend_alpha > 0.9 and person_mask is not None:
            self._draw_ghost_outline(surface, person_mask)

        # Shockwaves
        for sw in self._shockwaves:
            sw.draw(surface)

    def _draw_ghost_outline(self, surface: pygame.Surface,
                            person_mask: np.ndarray):
        """Draw very faint body outline when fully invisible."""
        h, w = person_mask.shape[:2]
        pulse = 0.15 + 0.05 * math.sin(self._time * 2.0 * math.tau)

        # Find contours via numpy diff (avoid extra cv2 call every frame)
        contour_count = 0
        step = 4  # sample every N rows for speed
        for y in range(1, h - 1, step):
            for x in range(1, w - 1):
                if (person_mask[y, x] > 127
                        and (person_mask[y - 1, x] == 0
                             or person_mask[y + 1, x] == 0
                             or person_mask[y, x - 1] == 0
                             or person_mask[y, x + 1] == 0)):
                    col = tuple(int(c * pulse) for c in PHASE_GHOST)
                    try:
                        surface.set_at((x, y), col)
                    except Exception:
                        pass
                    contour_count += 1
                    if contour_count > 3000:
                        return
