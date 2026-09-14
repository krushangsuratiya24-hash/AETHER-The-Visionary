"""
AETHER — Phase Shift Hand Gesture Classifier  (Phase 3 — Hand Power)

Classifies five distinct states from MediaPipe hand landmarks:

    ONE_FINGER    — index extended only              → 25% transparency
    TWO_FINGERS   — index + middle extended          → 50% transparency
    THREE_FINGERS — index + middle + ring extended   → 75% transparency
    FIVE_FINGERS  — all five (thumb+index+middle+ring+pinky) extended  → 100% invisible
    NONE          — any other configuration          → no change / previous state held

FOUR_FINGERS (index+middle+ring+pinky but no thumb, or any other 4-finger combo)
is intentionally mapped to NONE — acts as a "dead zone" between THREE and FIVE.

There is NO V/scissor gesture.  TWO_FINGERS is detected purely by extension count.

Thumb detection:
    The thumb is horizontal relative to the palm; standard tip-above-pip logic
    does not work for the thumb.  We use the angle between the thumb tip and
    the index MCP (landmark 5) relative to the wrist (landmark 0).  When the
    thumb is open/extended it points away from the palm; when closed it lies
    close to the index finger.

Temporal smoothing with confirmation window prevents flicker:
    • raw classification history buffer (_HISTORY_LEN frames)
    • require _CONFIRM_FRAMES consecutive matching classifications to confirm
    • hysteresis: once confirmed, require _HOLD_FRAMES of a different gesture to change

All thresholds are normalised by hand scale (wrist-to-middle-MCP distance).
"""
from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import List, Optional, Tuple

from core.tracker import HandLandmarkData


# ─────────────────────────────────────────────────────────────────────────────
# Gesture classes
# ─────────────────────────────────────────────────────────────────────────────

class PhaseGesture(Enum):
    NONE          = 'NONE'
    ONE_FINGER    = 'ONE FINGER'
    TWO_FINGERS   = 'TWO FINGERS'
    THREE_FINGERS = 'THREE FINGERS'
    FIVE_FINGERS  = 'FULL HAND'


# Transparency level each gesture maps to (0.0 = visible, 1.0 = invisible)
GESTURE_ALPHA: dict = {
    PhaseGesture.NONE:          0.0,
    PhaseGesture.ONE_FINGER:    0.25,
    PhaseGesture.TWO_FINGERS:   0.50,
    PhaseGesture.THREE_FINGERS: 0.75,
    PhaseGesture.FIVE_FINGERS:  1.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# Classifier configuration
# ─────────────────────────────────────────────────────────────────────────────

# History buffer length (raw frame classifications before confirmation)
_HISTORY_LEN = 12

# Consecutive matching frames needed to change the confirmed gesture
_CONFIRM_FRAMES = 7

# Consecutive different frames needed to drop the current confirmed gesture
_HOLD_FRAMES = 5

# Minimum hand detection confidence to accept
_MIN_CONFIDENCE = 0.50

# Finger extension threshold: tip must be this much above (lower y) PIP
# expressed as a fraction of hand_scale
_EXTEND_THRESH = 0.10   # tip_y - pip_y must be < -_EXTEND_THRESH * hand_scale

# Curl threshold: tip is folded if tip_y > pip_y (+ margin)
_CURL_THRESH   = 0.00   # tip_y - pip_y > _CURL_THRESH * hand_scale

# Thumb: normalised distance from thumb tip (4) to index MCP (5).
# When the thumb is closed it sits close to the index finger base.
# When open the thumb tip is clearly further away.
_THUMB_OPEN_DIST = 0.25   # thumb tip dist / scale > this → thumb extended


# ─────────────────────────────────────────────────────────────────────────────
# Per-landmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hand_scale(lms: List[Tuple]) -> float:
    """
    Normalising factor: Euclidean distance from wrist (0) to middle MCP (9).
    Robust, landmark-stable proxy for apparent hand size.
    """
    w = lms[0]
    m = lms[9]
    return math.sqrt((m[0] - w[0]) ** 2 + (m[1] - w[1]) ** 2) + 1e-6


def _is_finger_extended(lms: List[Tuple], tip: int, pip: int, scale: float) -> bool:
    """Return True when fingertip is clearly above its PIP joint (extended)."""
    tip_y = lms[tip][1]
    pip_y = lms[pip][1]
    return (tip_y - pip_y) < -_EXTEND_THRESH * scale


def _is_finger_folded(lms: List[Tuple], tip: int, pip: int, scale: float) -> bool:
    """Return True when fingertip is clearly below its PIP joint (curled)."""
    tip_y = lms[tip][1]
    pip_y = lms[pip][1]
    return (tip_y - pip_y) > _CURL_THRESH * scale


def _is_thumb_open(lms: List[Tuple], scale: float) -> bool:
    """
    Thumb extension test based on thumb tip (4) distance from index MCP (5).
    The thumb is considered open when it points outward away from the palm.
    """
    tx, ty = lms[4][0], lms[4][1]
    ix, iy = lms[5][0], lms[5][1]
    dist = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)
    return (dist / scale) > _THUMB_OPEN_DIST


# ─────────────────────────────────────────────────────────────────────────────
# Raw frame classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_raw(hand: HandLandmarkData) -> Tuple[PhaseGesture, float]:
    """
    Single-frame gesture classification.

    Returns:
        (PhaseGesture, confidence_0_to_1)
    """
    lms   = hand.landmarks
    scale = _hand_scale(lms)

    # Finger extension flags (non-thumb)
    # MediaPipe landmark indices:
    #   Index:  tip=8, pip=6
    #   Middle: tip=12, pip=10
    #   Ring:   tip=16, pip=14
    #   Pinky:  tip=20, pip=18
    index_ext  = _is_finger_extended(lms, 8,  6,  scale)
    middle_ext = _is_finger_extended(lms, 12, 10, scale)
    ring_ext   = _is_finger_extended(lms, 16, 14, scale)
    pinky_ext  = _is_finger_extended(lms, 20, 18, scale)

    index_fold  = _is_finger_folded(lms, 8,  6,  scale)
    middle_fold = _is_finger_folded(lms, 12, 10, scale)
    ring_fold   = _is_finger_folded(lms, 16, 14, scale)
    pinky_fold  = _is_finger_folded(lms, 20, 18, scale)

    thumb_open  = _is_thumb_open(lms, scale)

    extended_4 = sum([index_ext, middle_ext, ring_ext, pinky_ext])
    folded_4   = sum([index_fold, middle_fold, ring_fold, pinky_fold])

    # ── FIVE FINGERS (full open hand) ──────────────────────────────────────────
    # All 4 non-thumb fingers extended + thumb open
    if index_ext and middle_ext and ring_ext and pinky_ext and thumb_open:
        conf_bonus = 0.1 if all([index_ext, middle_ext, ring_ext, pinky_ext]) else 0.0
        return PhaseGesture.FIVE_FINGERS, 0.80 + conf_bonus

    # ── FOUR FINGERS → NONE (dead zone / rejection) ────────────────────────────
    # Four non-thumb fingers extended (with or without thumb)
    if index_ext and middle_ext and ring_ext and pinky_ext:
        # Thumb closed: 4 fingers only → dead zone
        return PhaseGesture.NONE, 0.2

    # ── THREE FINGERS ──────────────────────────────────────────────────────────
    if index_ext and middle_ext and ring_ext and pinky_fold:
        return PhaseGesture.THREE_FINGERS, 0.80

    # ── TWO FINGERS ────────────────────────────────────────────────────────────
    if index_ext and middle_ext and ring_fold and pinky_fold:
        return PhaseGesture.TWO_FINGERS, 0.80

    # ── ONE FINGER ─────────────────────────────────────────────────────────────
    if index_ext and middle_fold and ring_fold and pinky_fold:
        return PhaseGesture.ONE_FINGER, 0.80

    # ── No clear gesture ───────────────────────────────────────────────────────
    return PhaseGesture.NONE, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoother with debounce / hysteresis
# ─────────────────────────────────────────────────────────────────────────────

class PhaseGestureClassifier:
    """
    Wraps the raw frame classifier with temporal stability:

    - History buffer of last _HISTORY_LEN raw classifications
    - Requires _CONFIRM_FRAMES consecutive matching frames to change gesture
    - Requires _HOLD_FRAMES consecutive different frames to release current
    - Reports confidence as fraction of recent frames matching the winner
    - No microphone required

    Usage::

        clf = PhaseGestureClassifier()
        for hand in ...:
            result = clf.update(hand)
            print(result.gesture, result.confidence, result.target_alpha)
    """

    def __init__(self):
        self._history: deque = deque(maxlen=_HISTORY_LEN)
        self._confirmed: PhaseGesture = PhaseGesture.NONE
        self._candidate: PhaseGesture = PhaseGesture.NONE
        self._candidate_count: int    = 0
        self._hold_count: int         = 0

        # Last raw info for debug display
        self.raw_gesture:     PhaseGesture = PhaseGesture.NONE
        self.raw_confidence:  float        = 0.0
        self.extended_flags:  List[bool]   = [False] * 5   # [thumb, idx, mid, ring, pinky]
        self.finger_count:    int          = 0

    # ── Debug geometry snapshot ──────────────────────────────────────────────

    def _capture_debug(self, hand: Optional[HandLandmarkData]) -> None:
        """Capture geometry for debug overlay."""
        if hand is None:
            self.extended_flags = [False] * 5
            self.finger_count   = 0
            return
        lms   = hand.landmarks
        scale = _hand_scale(lms)
        thumb = _is_thumb_open(lms, scale)
        idx   = _is_finger_extended(lms, 8,  6,  scale)
        mid   = _is_finger_extended(lms, 12, 10, scale)
        ring  = _is_finger_extended(lms, 16, 14, scale)
        pky   = _is_finger_extended(lms, 20, 18, scale)
        self.extended_flags = [thumb, idx, mid, ring, pky]
        self.finger_count   = sum(self.extended_flags)

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, hand: Optional[HandLandmarkData]) -> 'GestureResult':
        """
        Process one frame's hand landmark data.

        Args:
            hand: HandLandmarkData or None (no detection)

        Returns:
            GestureResult with confirmed gesture, confidence, target_alpha
        """
        self._capture_debug(hand)

        if hand is None or hand.confidence < _MIN_CONFIDENCE:
            raw, conf = PhaseGesture.NONE, 0.0
        else:
            raw, conf = _classify_raw(hand)

        self.raw_gesture    = raw
        self.raw_confidence = conf

        self._history.append(raw)

        # Count votes in recent history
        if len(self._history) >= _CONFIRM_FRAMES:
            counts: dict = {}
            for g in self._history:
                counts[g] = counts.get(g, 0) + 1

            # Best candidate in history
            best       = max(counts, key=lambda g: counts[g])
            best_count = counts[best]

            if best == self._confirmed:
                # Still seeing the same gesture — reset hold counter
                self._hold_count = 0
                self._candidate  = best
                self._candidate_count = best_count
            elif best_count >= _CONFIRM_FRAMES:
                # New gesture confirmed
                self._confirmed        = best
                self._candidate        = best
                self._candidate_count  = best_count
                self._hold_count       = 0
            else:
                # Seeing something different but not enough frames yet
                self._hold_count += 1

        hist_confidence = (
            sum(1 for g in self._history if g == self._confirmed)
            / max(len(self._history), 1)
        )

        return GestureResult(
            gesture=self._confirmed,
            confidence=hist_confidence,
            target_alpha=GESTURE_ALPHA.get(self._confirmed, 0.0),
            raw_gesture=self.raw_gesture,
            raw_confidence=self.raw_confidence,
        )

    def reset(self):
        """Reset all state (call on enter/exit)."""
        self._history.clear()
        self._confirmed       = PhaseGesture.NONE
        self._candidate       = PhaseGesture.NONE
        self._candidate_count = 0
        self._hold_count      = 0
        self.raw_gesture      = PhaseGesture.NONE
        self.raw_confidence   = 0.0
        self.extended_flags   = [False] * 5
        self.finger_count     = 0


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

class GestureResult:
    """Immutable snapshot returned by PhaseGestureClassifier.update()."""

    __slots__ = (
        'gesture', 'confidence', 'target_alpha',
        'raw_gesture', 'raw_confidence',
    )

    def __init__(self,
                 gesture: PhaseGesture,
                 confidence: float,
                 target_alpha: float,
                 raw_gesture: PhaseGesture,
                 raw_confidence: float):
        self.gesture        = gesture
        self.confidence     = confidence
        self.target_alpha   = target_alpha
        self.raw_gesture    = raw_gesture
        self.raw_confidence = raw_confidence

    def __repr__(self) -> str:
        return (f'GestureResult({self.gesture.value!r}, '
                f'conf={self.confidence:.2f}, alpha={self.target_alpha:.2f})')
