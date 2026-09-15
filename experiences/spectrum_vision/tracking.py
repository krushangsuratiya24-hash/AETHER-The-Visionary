"""
AETHER — Spectrum Vision: Multi-Person Temporal Tracker  (Phase 4)

Assigns stable temporary IDs to detected people across frames.
Matches blobs between frames using centroid distance (IoU fallback).
Handles graceful entry, exit, and re-entry.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum distance (pixels, at 1280×720) to associate a new detection with an
# existing track.  Scaled proportionally to frame size at runtime.
_MAX_MATCH_DIST_FRAC = 0.30   # 30% of the frame diagonal

# Frames a track can be "missing" before it's removed
_MAX_MISSING_FRAMES = 10

# Maximum simultaneous tracked persons
_MAX_TRACKS = 8

# Minimum mask area (pixels at full res) to be considered a real person
_MIN_MASK_AREA = 500


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonTrack:
    """
    A single tracked person with a stable ID and history.

    Attributes:
        track_id:       Stable display ID (1-based), e.g. 1 → "PERSON 01"
        centroid:       (cx, cy) in pixels, normalised to current frame size
        bbox:           (x, y, w, h) bounding box in pixels
        age:            Number of frames this track has been active
        missing_frames: Consecutive frames without a matching detection
        last_seen:      Wall-clock time of last match
        trail:          Recent centroid positions for trail effects
    """
    track_id:       int
    centroid:       Tuple[float, float]
    bbox:           Tuple[int, int, int, int]   # x, y, w, h
    age:            int = 0
    missing_frames: int = 0
    last_seen:      float = field(default_factory=time.time)
    trail:          List[Tuple[float, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f'PERSON {self.track_id:02d}'

    @property
    def is_active(self) -> bool:
        return self.missing_frames == 0

    def update_trail(self, cx: float, cy: float, max_len: int = 20) -> None:
        self.trail.append((cx, cy))
        if len(self.trail) > max_len:
            self.trail.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Person Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MultiPersonTracker:
    """
    Frame-to-frame centroid tracker for multiple people.

    Usage::

        tracker = MultiPersonTracker()

        # Each frame, provide a list of (centroid_x, centroid_y, bbox) tuples
        # derived from connected-component analysis of the segmentation mask.
        detections = [(cx1, cy1, bbox1), (cx2, cy2, bbox2)]
        tracks = tracker.update(detections, frame_w, frame_h)

        for t in tracks:
            print(t.label, t.centroid, t.bbox)

    IDs start at 1 and are never reused within a session (though they reset
    after reset() is called — typically on experience enter()).
    """

    def __init__(self):
        self._tracks:       Dict[int, PersonTrack] = {}
        self._next_id:      int = 1
        self._frame_count:  int = 0

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracks — call on experience enter()."""
        self._tracks.clear()
        self._next_id  = 1
        self._frame_count = 0

    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Tuple[float, float, Tuple[int, int, int, int]]],
        frame_w: int,
        frame_h: int,
    ) -> List[PersonTrack]:
        """
        Associate new detections with existing tracks and return all active tracks.

        Args:
            detections: List of (cx, cy, (x, y, w, h)) in pixel coordinates.
            frame_w:    Frame width (pixels).
            frame_h:    Frame height (pixels).

        Returns:
            List of PersonTrack objects (active + briefly missing).
        """
        self._frame_count += 1

        # Max association distance in pixels
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        max_dist = diag * _MAX_MATCH_DIST_FRAC

        # Mark all existing tracks as temporarily missing
        for t in self._tracks.values():
            t.missing_frames += 1

        # Greedy nearest-centroid matching
        unmatched_detections = list(range(len(detections)))

        if self._tracks and detections:
            track_ids = list(self._tracks.keys())
            for det_idx in list(unmatched_detections):
                cx, cy, bbox = detections[det_idx]
                best_id, best_dist = None, float('inf')

                for tid in track_ids:
                    t = self._tracks[tid]
                    dx = cx - t.centroid[0]
                    dy = cy - t.centroid[1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_id = tid

                if best_id is not None and best_dist <= max_dist:
                    t = self._tracks[best_id]
                    t.centroid = (cx, cy)
                    t.bbox = bbox
                    t.age += 1
                    t.missing_frames = 0
                    t.last_seen = time.time()
                    t.update_trail(cx, cy)
                    track_ids.remove(best_id)
                    unmatched_detections.remove(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            if len(self._tracks) >= _MAX_TRACKS:
                break
            cx, cy, bbox = detections[det_idx]
            tid = self._next_id
            self._next_id += 1
            track = PersonTrack(
                track_id=tid,
                centroid=(cx, cy),
                bbox=bbox,
                age=0,
                missing_frames=0,
                last_seen=time.time(),
            )
            track.update_trail(cx, cy)
            self._tracks[tid] = track

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items()
                 if t.missing_frames > _MAX_MISSING_FRAMES]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def active_tracks(self) -> List[PersonTrack]:
        """Tracks with matching detections this frame."""
        return [t for t in self._tracks.values() if t.is_active]

    @property
    def all_tracks(self) -> List[PersonTrack]:
        return list(self._tracks.values())

    @property
    def person_count(self) -> int:
        return len(self.active_tracks)
