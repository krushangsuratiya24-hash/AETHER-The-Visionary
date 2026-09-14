"""
AETHER — Person Segmentation Engine  (Phase 3)
Uses MediaPipe ImageSegmenter (selfie_segmenter model) to separate a person from
the background in real time.

Pipeline per frame:
  raw BGR → resize to processing resolution → MediaPipe → confidence mask →
  temporal EMA smoothing → morphological cleanup → hole-fill → refined binary mask

Designed to run on the main thread or a dedicated worker.  The public API is
thread-safe: call update(bgr_frame) to push a new frame, then read .person_mask
and .confidence_mask at any time.

Graceful degradation: if the model cannot be initialised the segmenter enters
UNAVAILABLE mode and always returns a zero mask (person never obscures bg).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Processing resolution — smaller = faster inference, lower quality
_PROC_W, _PROC_H = 256, 144        # 16:9 thumbnail

# Temporal EMA alpha: how fast the smoothed mask tracks the raw mask.
# Lower = smoother but more lag.  Higher = sharper but flickery edges.
_TEMPORAL_ALPHA = 0.55

# Binary threshold on the smoothed confidence mask (0-1 float)
_THRESHOLD = 0.50

# Morphology kernel sizes
_ERODE_K  = 3   # remove small isolated foreground specks
_DILATE_K = 5   # fill small holes in body, expand contour slightly

# How long (seconds) to keep the last valid mask when inference is slow
_MASK_MAX_AGE = 0.3


class PersonSegmenter:
    """
    Real-time person segmentation backed by MediaPipe selfie segmenter.

    Usage::

        seg = PersonSegmenter()
        seg.start()

        # Per frame (main thread or any thread):
        seg.push_frame(bgr_frame)

        # Read results (always returns a valid mask, even before first inference):
        mask = seg.person_mask    # uint8 binary [0/255], full camera resolution
        conf = seg.confidence_mask  # float32 [0..1], full camera resolution

        seg.stop()

    Attributes:
        ready (bool):   True when the model loaded successfully.
        status (str):   Human-readable status string for HUD.
        fps (float):    Measured segmentation updates per second.
    """

    def __init__(self, model_path: Optional[str] = None, lazy: bool = False):
        """
        Args:
            model_path: Path to selfie_segmenter.tflite.  None = default location.
            lazy:       If True, skip model initialisation in __init__.
                        Call start() explicitly when ready to use hardware.
                        Use lazy=True in unit tests and when the experience
                        is constructed but not yet entered.
        """
        if model_path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root, 'assets', 'models', 'selfie_segmenter.tflite')

        self._model_path = model_path
        self._segmenter  = None

        # Output masks (full-resolution, ready to use)
        self._person_mask:     np.ndarray = np.zeros((_PROC_H, _PROC_W), dtype=np.uint8)
        self._confidence_mask: np.ndarray = np.zeros((_PROC_H, _PROC_W), dtype=np.float32)
        self._smooth_mask:     np.ndarray = np.zeros((_PROC_H, _PROC_W), dtype=np.float32)

        # Full-resolution versions (resized on demand to match input frame)
        self._out_mask_full: Optional[np.ndarray] = None
        self._out_conf_full: Optional[np.ndarray] = None
        self._last_frame_shape: Tuple[int, int, int] = (0, 0, 3)

        self._lock = threading.Lock()
        self._last_update = 0.0

        # Performance
        self._fps_counter = 0
        self._fps_timer   = time.time()
        self.fps: float   = 0.0

        # Status
        self.ready  = False
        self.status = 'NOT STARTED' if lazy else 'INITIALIZING'

        # Morphology kernels (pre-built once)
        self._k_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ERODE_K,  _ERODE_K))
        self._k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_DILATE_K, _DILATE_K))

        if not lazy:
            self._init_segmenter()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> 'PersonSegmenter':
        """
        Explicitly initialise the segmentation model.
        Call this from enter() / when hardware is needed.
        Safe to call multiple times — re-initialises if previously stopped.
        Returns self for chaining.
        """
        if self._segmenter is None:
            self._init_segmenter()
        return self

    def _init_segmenter(self):
        """Load the MediaPipe segmentation model.  Sets self.ready."""
        try:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f'Selfie segmenter model not found: {self._model_path}\n'
                    'Run: python -c "import urllib.request; '
                    'urllib.request.urlretrieve('
                    '\'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite\','
                    ' \'assets/models/selfie_segmenter.tflite\')"'
                )

            from mediapipe.tasks.python.vision import (
                ImageSegmenter, ImageSegmenterOptions,
            )
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )

            opts = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=self._model_path),
                running_mode=VisionTaskRunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self._segmenter = ImageSegmenter.create_from_options(opts)
            self.ready  = True
            self.status = 'READY'
            print('[AETHER-SEG] Selfie segmenter initialised successfully.')

        except Exception as exc:
            self._segmenter = None
            self.ready  = False
            self.status = f'UNAVAILABLE: {exc}'
            print(f'[AETHER-SEG] Segmenter init failed — degraded mode: {exc}')

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def push_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Process a single camera frame synchronously.
        This should be called from whichever thread owns the heavy work.
        Typically called at a sub-sampled rate (every 2-3 main loop frames)
        to save CPU while keeping the mask temporally smooth.
        """
        if not self.ready or self._segmenter is None:
            # Return zero mask — person never disappears (safe fallback)
            return

        h, w = bgr_frame.shape[:2]
        try:
            # 1. Resize to processing resolution
            small = cv2.resize(bgr_frame, (_PROC_W, _PROC_H), interpolation=cv2.INTER_LINEAR)

            # 2. Convert BGR → RGB (MediaPipe expects RGB)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # 3. Run inference
            from mediapipe import Image, ImageFormat
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_small)
            result = self._segmenter.segment(mp_image)

            if result.confidence_masks and len(result.confidence_masks) > 0:
                # numpy_view() returns (H, W, 1) on this MediaPipe version — squeeze to (H, W)
                raw_conf = np.array(result.confidence_masks[0].numpy_view(), dtype=np.float32)
                if raw_conf.ndim == 3 and raw_conf.shape[2] == 1:
                    raw_conf = raw_conf[:, :, 0]

                # 4. Temporal EMA smoothing
                self._smooth_mask = (
                    _TEMPORAL_ALPHA * raw_conf
                    + (1.0 - _TEMPORAL_ALPHA) * self._smooth_mask
                )

                # 5. Threshold → binary
                binary = (self._smooth_mask > _THRESHOLD).astype(np.uint8) * 255

                # 6. Morphological cleanup: erode small noise, dilate to fill holes
                binary = cv2.erode(binary, self._k_erode, iterations=1)
                binary = cv2.dilate(binary, self._k_dilate, iterations=2)

                # 7. Flood-fill to close internal holes (fill from corners = background fill)
                flooded = binary.copy()
                mask_fill = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8)
                cv2.floodFill(flooded, mask_fill, (0, 0), 255)
                cv2.floodFill(flooded, mask_fill, (_PROC_W - 1, 0), 255)
                cv2.floodFill(flooded, mask_fill, (0, _PROC_H - 1), 255)
                cv2.floodFill(flooded, mask_fill, (_PROC_W - 1, _PROC_H - 1), 255)
                # Inverted flood = holes inside person
                bg_mask = cv2.bitwise_not(flooded)
                binary = cv2.bitwise_or(binary, bg_mask)

                # 8. One more dilate for edge smoothness
                binary = cv2.dilate(binary, self._k_dilate, iterations=1)

                with self._lock:
                    self._person_mask     = binary
                    self._confidence_mask = self._smooth_mask.copy()
                    self._last_frame_shape = bgr_frame.shape
                    self._last_update = time.time()
                    # Invalidate cached full-res versions
                    self._out_mask_full = None
                    self._out_conf_full = None

            # Measure FPS
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.fps = self._fps_counter / (now - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer = now

        except Exception as exc:
            print(f'[AETHER-SEG] push_frame error: {exc}')

    @property
    def person_mask(self) -> np.ndarray:
        """
        Binary person mask at processing resolution (_PROC_W × _PROC_H).
        Values: 0 = background, 255 = person.
        Always returns a valid array (zeros if segmenter unavailable).
        """
        with self._lock:
            return self._person_mask.copy()

    @property
    def confidence_mask(self) -> np.ndarray:
        """Float32 confidence mask [0..1] at processing resolution."""
        with self._lock:
            return self._confidence_mask.copy()

    def get_mask_for_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the person mask resized to match `frame`'s dimensions (H×W, uint8).
        Cached — only re-computed when the internal mask or frame size changes.
        """
        fh, fw = frame.shape[:2]
        with self._lock:
            # Rebuild cache if needed
            if (self._out_mask_full is None
                    or self._out_mask_full.shape[:2] != (fh, fw)):
                self._out_mask_full = cv2.resize(
                    self._person_mask, (fw, fh),
                    interpolation=cv2.INTER_LINEAR
                )
                # Re-threshold after up-sampling to keep clean edges
                _, self._out_mask_full = cv2.threshold(
                    self._out_mask_full, 127, 255, cv2.THRESH_BINARY
                )
            return self._out_mask_full.copy()

    def get_confidence_for_frame(self, frame: np.ndarray) -> np.ndarray:
        """Float32 confidence mask resized to match `frame`."""
        fh, fw = frame.shape[:2]
        with self._lock:
            if (self._out_conf_full is None
                    or self._out_conf_full.shape[:2] != (fh, fw)):
                self._out_conf_full = cv2.resize(
                    self._confidence_mask, (fw, fh),
                    interpolation=cv2.INTER_LINEAR
                )
            return self._out_conf_full.copy()

    @property
    def mask_age(self) -> float:
        """Seconds since last successful mask update."""
        return time.time() - self._last_update if self._last_update > 0 else 999.0

    def stop(self):
        """Release the underlying MediaPipe segmenter."""
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass
            self._segmenter = None
        print('[AETHER-SEG] Segmenter stopped.')

    def __del__(self):
        self.stop()
