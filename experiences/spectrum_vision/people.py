"""
AETHER — Spectrum Vision: Person Representation  (Phase 4)

Handles per-person mask extraction, bounding boxes, and metadata.
Bridges the global segmentation mask → individual person blobs.

Multi-person detection strategy:
    MediaPipe selfie segmenter produces a single binary mask for the
    "foreground" (person area) of the frame.  We detect individual people
    by running connected-component analysis on the binary mask and filtering
    by area.  Each connected component becomes one "detected person".

    This approach is honest: it depends entirely on actual camera input.
    No fake persons are ever synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum area (fraction of total frame pixels) for a connected component to
# be treated as a real person detection.  Filters noise.
_MIN_AREA_FRAC = 0.003   # 0.3 % of frame ≈ 2764 px at 1280×720

# Maximum persons to report per frame (computational budget)
_MAX_PERSONS = 6

# Kernel for mask cleanup
_CLEANUP_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ─────────────────────────────────────────────────────────────────────────────
# DetectedPerson
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedPerson:
    """
    A single person detected in one frame.

    Attributes:
        mask:       Binary uint8 mask (H×W, values 0 or 255) for this person only.
        bbox:       (x, y, w, h) bounding box in pixels.
        centroid:   (cx, cy) centroid in pixels.
        area:       Number of foreground pixels.
        index:      Detection index within this frame (0-based, largest first).
    """
    mask:     np.ndarray                     # H × W uint8
    bbox:     Tuple[int, int, int, int]      # x, y, w, h
    centroid: Tuple[float, float]            # cx, cy (pixels)
    area:     int
    index:    int = 0

    def __post_init__(self):
        assert self.mask.ndim == 2, "Mask must be 2D (H×W)"
        assert self.mask.dtype == np.uint8, "Mask must be uint8"

    @property
    def bbox_center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Person Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_persons(
    full_mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    min_area_frac: float = _MIN_AREA_FRAC,
    max_persons: int = _MAX_PERSONS,
) -> List[DetectedPerson]:
    """
    Decompose a full-frame binary segmentation mask into individual person blobs.

    Args:
        full_mask:     Binary uint8 mask at any resolution (0=bg, 255=person).
        frame_h:       Target frame height (for area threshold calculation).
        frame_w:       Target frame width.
        min_area_frac: Minimum fraction of frame area to count as a person.
        max_persons:   Maximum persons to return.

    Returns:
        List of DetectedPerson, sorted by area descending (largest first).
        Empty list if no persons detected.
    """
    if full_mask is None or full_mask.size == 0:
        return []

    # Ensure mask is at frame resolution
    mh, mw = full_mask.shape[:2]
    if (mh, mw) != (frame_h, frame_w):
        mask = cv2.resize(full_mask, (frame_w, frame_h),
                          interpolation=cv2.INTER_NEAREST)
    else:
        mask = full_mask.copy()

    # Ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Light cleanup to separate touching blobs
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _CLEANUP_K, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return []

    min_area = int(frame_h * frame_w * min_area_frac)

    persons = []
    # Label 0 = background, skip it
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        # Per-person mask
        person_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        person_mask[labels == label_idx] = 255

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        cx = float(centroids[label_idx][0])
        cy = float(centroids[label_idx][1])

        persons.append(DetectedPerson(
            mask=person_mask,
            bbox=(x, y, w, h),
            centroid=(cx, cy),
            area=area,
        ))

    # Sort largest first, cap count
    persons.sort(key=lambda p: p.area, reverse=True)
    persons = persons[:max_persons]

    for i, p in enumerate(persons):
        p.index = i

    return persons


def validate_mask(mask: np.ndarray) -> bool:
    """Return True if mask is a valid 2D uint8 numpy array."""
    return (
        isinstance(mask, np.ndarray)
        and mask.ndim == 2
        and mask.dtype == np.uint8
    )
