"""
AETHER — Elemental Cultivation: Gesture Processor
Interprets raw MediaPipe HandLandmarkData for the elemental experience.
Detects: OPEN_PALM, PINCH, FIST, POINT, SWIPE, TWO_HAND, CLAP
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from core.tracker import HandLandmarkData


class ElementalGesture(Enum):
    NONE       = "NONE"
    OPEN_PALM  = "OPEN_PALM"
    PINCH      = "PINCH"
    FIST       = "FIST"
    POINT      = "POINT"
    SWIPE      = "SWIPE"
    TWO_HAND   = "TWO_HAND"
    CLAP       = "CLAP"


@dataclass
class ElementalGestureState:
    gesture: ElementalGesture = ElementalGesture.NONE
    # Primary hand (palm) position in normalised coords [0..1]
    hand_pos: Tuple[float, float] = (0.5, 0.5)
    # Second hand position (if two hands detected)
    hand2_pos: Optional[Tuple[float, float]] = None
    # Distance between hands normalised (0 = together, 1 = screen width)
    two_hand_distance: float = 0.0
    # Instantaneous palm speed (norm units/s)
    speed: float = 0.0
    # Pinch ratio: 0 = open, 1 = fully pinched
    pinch_ratio: float = 0.0
    # Fist ratio: 0 = open, 1 = full fist
    fist_ratio: float = 0.0
    # True if this tick a SWIPE event fired
    swipe_fired: bool = False
    # True if this tick a CLAP event fired
    clap_fired: bool = False
    # True if two hands are currently detected
    two_hands: bool = False


def _finger_extended(lms: list, tip: int, pip: int, mcp: int) -> bool:
    """Return True when the finger tip is above (smaller y) than the PIP joint."""
    return lms[tip][1] < lms[pip][1]


def _landmark_distance(lms: list, a: int, b: int) -> float:
    dx = lms[a][0] - lms[b][0]
    dy = lms[a][1] - lms[b][1]
    return math.sqrt(dx*dx + dy*dy)


class ElementalGestureProcessor:
    """
    Stateful gesture processor for the Elemental Cultivation experience.
    Smoothed, hysteresis-based, cooldown-gated.
    """

    SMOOTHING = 0.30    # EMA alpha for position
    SWIPE_SPEED = 0.90  # normalised units/s threshold
    SWIPE_COOLDOWN = 0.45
    CLAP_DISTANCE = 0.18
    CLAP_COOLDOWN = 0.8
    PINCH_THRESHOLD_CLOSE = 0.05
    PINCH_THRESHOLD_OPEN  = 0.10

    def __init__(self):
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None
        self._history: deque = deque(maxlen=12)  # (time, x, y)
        self._last_swipe: float = 0.0
        self._last_clap:  float = 0.0
        self._pinched = False
        self._fisted  = False
        self._prev_two_hand_dist: Optional[float] = None
        self.state = ElementalGestureState()

    def reset(self):
        self._sx = self._sy = None
        self._history.clear()
        self._pinched = self._fisted = False
        self._prev_two_hand_dist = None
        self.state = ElementalGestureState()

    def process(self, hands: List[HandLandmarkData]) -> ElementalGestureState:
        now = time.time()
        st = ElementalGestureState()
        st.swipe_fired = False
        st.clap_fired  = False

        if not hands:
            self.state = st
            return st

        h0 = hands[0]
        lms = h0.landmarks

        # ── Position (smoothed) ──────────────────────────────────────────────
        rx, ry = h0.palm_center[0], h0.palm_center[1]
        if self._sx is None:
            self._sx, self._sy = rx, ry
        else:
            a = self.SMOOTHING
            self._sx = a * rx + (1 - a) * self._sx
            self._sy = a * ry + (1 - a) * self._sy
        st.hand_pos = (self._sx, self._sy)
        self._history.append((now, self._sx, self._sy))

        # ── Speed ────────────────────────────────────────────────────────────
        if len(self._history) >= 4:
            t0, x0, y0 = self._history[0]
            t1, x1, y1 = self._history[-1]
            dt = max(0.001, t1 - t0)
            st.speed = math.sqrt((x1-x0)**2 + (y1-y0)**2) / dt
        else:
            st.speed = 0.0

        # ── Two-hand detection ───────────────────────────────────────────────
        if len(hands) >= 2:
            h1 = hands[1]
            st.two_hands = True
            st.hand2_pos = (h1.palm_center[0], h1.palm_center[1])
            dx = h1.palm_center[0] - h0.palm_center[0]
            dy = h1.palm_center[1] - h0.palm_center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            st.two_hand_distance = dist

            # Clap detection: hands coming quickly close together
            if (self._prev_two_hand_dist is not None
                    and self._prev_two_hand_dist > self.CLAP_DISTANCE * 1.5
                    and dist < self.CLAP_DISTANCE
                    and now - self._last_clap > self.CLAP_COOLDOWN):
                st.clap_fired = True
                self._last_clap = now

            self._prev_two_hand_dist = dist
        else:
            self._prev_two_hand_dist = None

        # ── Pinch ratio ──────────────────────────────────────────────────────
        # Distance between thumb tip (4) and index tip (8)
        pinch_d = _landmark_distance(lms, 4, 8)
        # Normalise by hand size (wrist-to-middle-mcp distance)
        hand_scale = _landmark_distance(lms, 0, 9) + 0.001
        pinch_norm = pinch_d / hand_scale

        if self._pinched:
            if pinch_norm > self.PINCH_THRESHOLD_OPEN:
                self._pinched = False
        else:
            if pinch_norm < self.PINCH_THRESHOLD_CLOSE:
                self._pinched = True
        st.pinch_ratio = max(0.0, min(1.0, 1.0 - (pinch_norm / self.PINCH_THRESHOLD_OPEN)))

        # ── Fist detection ───────────────────────────────────────────────────
        # Count curled fingers (tip y > mcp y = curled)
        finger_specs = [
            (8, 6, 5),   # index
            (12, 10, 9), # middle
            (16, 14, 13),# ring
            (20, 18, 17),# pinky
        ]
        curled = sum(1 for (tip, pip, mcp) in finger_specs
                     if not _finger_extended(lms, tip, pip, mcp))
        fist_ratio = curled / 4.0
        st.fist_ratio = fist_ratio
        self._fisted = fist_ratio >= 0.75

        # ── Swipe (fast horizontal) ──────────────────────────────────────────
        if (st.speed > self.SWIPE_SPEED
                and not st.two_hands
                and now - self._last_swipe > self.SWIPE_COOLDOWN):
            st.swipe_fired = True
            self._last_swipe = now

        # ── Classify gesture ─────────────────────────────────────────────────
        if st.two_hands:
            g = ElementalGesture.TWO_HAND
        elif self._fisted:
            g = ElementalGesture.FIST
        elif self._pinched:
            g = ElementalGesture.PINCH
        else:
            # Count extended fingers for OPEN_PALM / POINT
            extended = sum(1 for (tip, pip, mcp) in finger_specs
                           if _finger_extended(lms, tip, pip, mcp))
            if extended >= 3:
                g = ElementalGesture.OPEN_PALM
            elif extended == 1:
                g = ElementalGesture.POINT
            else:
                g = ElementalGesture.NONE

        if st.swipe_fired:
            g = ElementalGesture.SWIPE

        st.gesture = g
        self.state = st
        return st
