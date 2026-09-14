"""
AETHER — Background Modeler & Invisibility Compositor  (Phase 3)
Maintains a live background estimate and composites invisible frames.

Background modeling strategy:
    While the person is VISIBLE the compositor receives a person mask each
    frame.  Pixels marked as BACKGROUND (mask==0) are used to update the
    running background estimate via EMA:

        bg[px] = alpha * frame[px]  +  (1-alpha)  * bg[px]    if mask[px]==0

    Pixels currently occupied by the person are NOT updated, so the previous
    background estimate "shows through" those regions.

    When the person becomes INVISIBLE the compositor replaces every person
    pixel with the background estimate, producing genuine see-through compositing:

        output[px] = bg[px]         if mask[px] == person
        output[px] = frame[px]      if mask[px] == background

Transition blending:
    During PHASING_OUT / PHASING_IN a smooth alpha blend mixes between the
    original frame and the composited output:

        result[px] = lerp(frame[px], composite[px], blend_alpha)

        blend_alpha 0→1 = fading out
        blend_alpha 1→0 = fading in

Design notes:
    - The background model is initialised from the first few frames before
      any segmentation is available (warm-up period).
    - A slowly-decaying memory coefficient prevents ghost trails from
      objects that moved permanently (e.g. the person walking off-screen).
    - All compositing is done in float32 for accuracy, then cast to uint8.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

# EMA alpha for background update (lower = more temporal averaging = smoother)
_BG_ALPHA    = 0.04     # ~25 frames half-life at 30 FPS → tolerates bg motion

# Minimum frames to collect before background is considered "warm"
_WARMUP_FRAMES = 10

# Alpha for background update during warmup (fast convergence)
_BG_ALPHA_WARMUP = 0.4


class BackgroundCompositor:
    """
    Maintains a running background estimate and composites invisibility frames.

    Usage::

        comp = BackgroundCompositor()

        # Each frame (visible state or warmup):
        comp.update_background(frame, person_mask)

        # When rendering (invisible or transitioning):
        output = comp.composite(
            frame,
            person_mask,
            blend_alpha=1.0   # 0.0=original frame, 1.0=fully composited
        )

    Attributes:
        has_background (bool):  True once the background model is warm.
        bg_frame (ndarray):     Current background estimate (BGR, uint8).
    """

    def __init__(self):
        self._bg:        Optional[np.ndarray] = None   # float32 BGR
        self._frame_count = 0
        self.has_background = False

        # Diagnostics
        self.bg_coverage: float = 0.0   # fraction of pixels with bg data

    # ─────────────────────────────────────────────────────────────────────────
    # Background update
    # ─────────────────────────────────────────────────────────────────────────

    def update_background(self,
                          frame: np.ndarray,
                          person_mask: Optional[np.ndarray] = None) -> None:
        """
        Update the background model using pixels not occupied by the person.

        Args:
            frame:        BGR camera frame (uint8, any resolution).
            person_mask:  Binary person mask (uint8, 0=bg, 255=person).
                          If None, every pixel updates the background
                          (used during warmup before segmentation is ready).
        """
        frame_f = frame.astype(np.float32)
        h, w = frame.shape[:2]

        if self._bg is None:
            # First frame — initialise from full frame
            self._bg = frame_f.copy()
            self._frame_count = 1
            return

        # Resize background if frame resolution changed
        if self._bg.shape[:2] != (h, w):
            self._bg = cv2.resize(self._bg, (w, h)).astype(np.float32)

        self._frame_count += 1
        alpha = _BG_ALPHA_WARMUP if self._frame_count < _WARMUP_FRAMES else _BG_ALPHA

        if person_mask is None:
            # No mask available — update everything (warmup mode)
            self._bg = alpha * frame_f + (1.0 - alpha) * self._bg
        else:
            # Resize mask to match frame if needed
            if person_mask.shape[:2] != (h, w):
                pmask = cv2.resize(person_mask, (w, h))
            else:
                pmask = person_mask

            # bg_pixel = True where mask is 0 (background)
            bg_region = (pmask == 0)

            # EMA update only for background-labelled pixels
            # Using boolean indexing on float32 arrays (fast)
            if bg_region.any():
                self._bg[bg_region] = (
                    alpha * frame_f[bg_region]
                    + (1.0 - alpha) * self._bg[bg_region]
                )

            # Track how much background we have
            self.bg_coverage = float(bg_region.sum()) / (h * w)

        if self._frame_count >= _WARMUP_FRAMES:
            self.has_background = True

    # ─────────────────────────────────────────────────────────────────────────
    # Compositing
    # ─────────────────────────────────────────────────────────────────────────

    def composite(self,
                  frame: np.ndarray,
                  person_mask: np.ndarray,
                  blend_alpha: float = 1.0) -> np.ndarray:
        """
        Produce the composited output frame.

        When blend_alpha=1.0 (fully invisible):
            Person pixels → background estimate
            Non-person pixels → live camera

        When blend_alpha=0.0:
            Returns the raw camera frame unchanged.

        When 0 < blend_alpha < 1:
            Smooth transition between visible and invisible.

        Args:
            frame:        BGR camera frame (uint8).
            person_mask:  Binary person mask (uint8, 0=bg, 255=person).
            blend_alpha:  0.0 = visible, 1.0 = invisible.

        Returns:
            Composited BGR frame (uint8, same resolution as input).
        """
        if blend_alpha <= 0.001:
            return frame.copy()

        h, w = frame.shape[:2]
        frame_f = frame.astype(np.float32)

        # Resize mask to match frame
        if person_mask.shape[:2] != (h, w):
            pmask = cv2.resize(person_mask, (w, h))
        else:
            pmask = person_mask.copy()

        # Soft-edge the mask with Gaussian blur for anti-aliased transitions
        pmask_soft = cv2.GaussianBlur(pmask.astype(np.float32), (11, 11), 0) / 255.0
        pmask_soft = pmask_soft[:, :, np.newaxis]   # (H, W, 1) for broadcasting

        # Background layer
        if self._bg is not None:
            bg_f = self._bg.copy()
            if bg_f.shape[:2] != (h, w):
                bg_f = cv2.resize(bg_f, (w, h)).astype(np.float32)
        else:
            # No background yet — use blurred/darkened version of current frame
            bg_f = cv2.GaussianBlur(frame_f, (21, 21), 0)

        # Invisibility composite:
        #   composite = bg where person, frame where not person
        composited = (1.0 - pmask_soft) * frame_f + pmask_soft * bg_f

        # Blend between original frame and composited based on blend_alpha
        result_f = (1.0 - blend_alpha) * frame_f + blend_alpha * composited

        return np.clip(result_f, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # Background access
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def bg_frame(self) -> Optional[np.ndarray]:
        """Current background estimate as uint8 BGR, or None if not warmed up."""
        if self._bg is None:
            return None
        return np.clip(self._bg, 0, 255).astype(np.uint8)

    def reset(self):
        """Reset background model (called when re-entering the experience)."""
        self._bg = None
        self._frame_count = 0
        self.has_background = False
        self.bg_coverage = 0.0
        print('[AETHER-COMP] Background model reset.')
