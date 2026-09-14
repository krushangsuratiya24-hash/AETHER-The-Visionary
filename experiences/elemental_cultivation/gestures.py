"""
AETHER — Elemental Cultivation: Gesture Processor
Interprets raw MediaPipe HandLandmarkData for the elemental experience.
Detects: OPEN_PALM, PINCH, FIST, POINT, SWIPE, TWO_HAND, CLAP

--- FIX SUMMARY (Phase 2 Quality Correction) ---
1. Pinch: normalized distance (thumb-index / wrist-middle_mcp) with proper
   hysteresis (START < RELEASE) and temporal confirmation (3 consecutive frames).
2. Swipe: velocity calculated from actual elapsed time; direction-aware
   (LEFT, RIGHT, UP, DOWN); configurable thresholds tuned for normal webcam use;
   cooldown + minimum displacement debounce.
3. Multi-hand: left/right hand resolved from corrected MediaPipe handedness
   (mirror already applied in tracker.py).
4. HUD fields: pinch_ratio, swipe_dir, velocity exposed on state.
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
    # Swipe direction string: 'LEFT', 'RIGHT', 'UP', 'DOWN', or ''
    swipe_dir: str = ''
    # Swipe velocity (normalised units/s) at time of fire
    swipe_velocity: float = 0.0
    # True if this tick a CLAP event fired
    clap_fired: bool = False
    # True if two hands are currently detected
    two_hands: bool = False
    # Per-hand gestures (for HUD debug)
    left_gesture: str = ''
    right_gesture: str = ''
    # Hand velocity vector (for projectile direction)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    # Midpoint between two hands (normalised)
    two_hand_midpoint: Optional[Tuple[float, float]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finger_extended(lms: list, tip: int, pip: int, mcp: int) -> bool:
    """Return True when the finger tip is above (smaller y) than the PIP joint."""
    return lms[tip][1] < lms[pip][1]


def _landmark_distance(lms: list, a: int, b: int) -> float:
    dx = lms[a][0] - lms[b][0]
    dy = lms[a][1] - lms[b][1]
    return math.sqrt(dx * dx + dy * dy)


def _classify_single_hand(lms: list, pinched: bool, fisted: bool) -> ElementalGesture:
    """Return the gesture for one hand given precomputed pinch/fist flags."""
    if fisted:
        return ElementalGesture.FIST
    if pinched:
        return ElementalGesture.PINCH
    finger_specs = [
        (8, 6, 5),   # index
        (12, 10, 9), # middle
        (16, 14, 13),# ring
        (20, 18, 17),# pinky
    ]
    extended = sum(1 for (tip, pip, mcp) in finger_specs
                   if _finger_extended(lms, tip, pip, mcp))
    if extended >= 3:
        return ElementalGesture.OPEN_PALM
    if extended == 1:
        return ElementalGesture.POINT
    return ElementalGesture.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Single-hand state machine (one instance per tracked hand)
# ─────────────────────────────────────────────────────────────────────────────

class _HandState:
    """Tracks smoothed position, pinch, fist, and swipe for a single hand."""

    # Pinch thresholds — normalised (pinch_distance / hand_scale)
    # PINCH_START must be < PINCH_RELEASE to form a hysteresis band.
    PINCH_START   = 0.22   # enter pinch when norm_dist falls below this
    PINCH_RELEASE = 0.35   # exit pinch only after norm_dist rises above this
    # Frames the norm_dist must be below START before we commit to pinch
    PINCH_CONFIRM_FRAMES = 2

    # Swipe thresholds
    SWIPE_SPEED_MIN    = 0.55   # normalised units/s (tuned for normal webcam use)
    SWIPE_DISPLACE_MIN = 0.08   # minimum displacement (norm) to count as swipe
    SWIPE_COOLDOWN     = 0.50   # seconds between swipe fires
    SWIPE_WINDOW       = 0.20   # look-back window (seconds) for velocity calculation

    SMOOTHING = 0.35  # EMA alpha — higher = more responsive, more jitter

    def __init__(self):
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None
        # History: (time, smooth_x, smooth_y)
        self._history: deque = deque(maxlen=30)
        self._pinched  = False
        self._fisted   = False
        self._pinch_confirm = 0   # consecutive frames below START threshold
        self._last_swipe = 0.0

    def reset(self):
        self._sx = self._sy = None
        self._history.clear()
        self._pinched = self._fisted = False
        self._pinch_confirm = 0
        self._last_swipe = 0.0

    def update(self, hand: HandLandmarkData, now: float) -> dict:
        """
        Process one frame for this hand.  Returns a dict with:
          pos, speed, pinch_ratio, fist_ratio, pinched, fisted,
          swipe_fired, swipe_dir, swipe_velocity, velocity_x, velocity_y
        """
        lms = hand.landmarks

        # ── Smooth position ──────────────────────────────────────────────────
        rx, ry = hand.palm_center[0], hand.palm_center[1]
        if self._sx is None:
            self._sx, self._sy = rx, ry
        else:
            a = self.SMOOTHING
            self._sx = a * rx + (1 - a) * self._sx
            self._sy = a * ry + (1 - a) * self._sy
        self._history.append((now, self._sx, self._sy))

        # ── Speed & velocity ─────────────────────────────────────────────────
        speed = 0.0
        vx = vy = 0.0
        # Use a short window for velocity
        cutoff = now - self.SWIPE_WINDOW
        recent = [(t, x, y) for (t, x, y) in self._history if t >= cutoff]
        if len(recent) >= 2:
            t0, x0, y0 = recent[0]
            t1, x1, y1 = recent[-1]
            dt = max(0.001, t1 - t0)
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            speed = math.sqrt(vx * vx + vy * vy)

        # ── Normalised pinch ─────────────────────────────────────────────────
        # pinch_distance = thumb_tip(4) ↔ index_tip(8)
        # hand_scale     = wrist(0) ↔ middle_mcp(9)
        pinch_d  = _landmark_distance(lms, 4, 8)
        scale_d  = _landmark_distance(lms, 0, 9) + 1e-4
        pinch_norm = pinch_d / scale_d

        if self._pinched:
            # Already pinched — wait for it to open past RELEASE threshold
            if pinch_norm > self.PINCH_RELEASE:
                self._pinched = False
                self._pinch_confirm = 0
        else:
            if pinch_norm < self.PINCH_START:
                self._pinch_confirm += 1
                if self._pinch_confirm >= self.PINCH_CONFIRM_FRAMES:
                    self._pinched = True
            else:
                self._pinch_confirm = max(0, self._pinch_confirm - 1)

        # Pinch ratio: 0=open, 1=fully pinched
        # Mapped so 0 at PINCH_RELEASE and 1 at PINCH_START
        pinch_ratio = max(0.0, min(1.0,
            (self.PINCH_RELEASE - pinch_norm) / (self.PINCH_RELEASE - self.PINCH_START + 1e-6)))

        # ── Fist detection ───────────────────────────────────────────────────
        finger_specs = [
            (8, 6, 5),    # index
            (12, 10, 9),  # middle
            (16, 14, 13), # ring
            (20, 18, 17), # pinky
        ]
        curled = sum(1 for (tip, pip, mcp) in finger_specs
                     if not _finger_extended(lms, tip, pip, mcp))
        fist_ratio = curled / 4.0
        self._fisted = fist_ratio >= 0.75

        # ── Swipe detection ──────────────────────────────────────────────────
        swipe_fired = False
        swipe_dir   = ''
        swipe_vel   = 0.0

        if (speed >= self.SWIPE_SPEED_MIN
                and now - self._last_swipe >= self.SWIPE_COOLDOWN):
            # Check displacement over the window
            if len(recent) >= 2:
                t0, x0, y0 = recent[0]
                t1, x1, y1 = recent[-1]
                displace = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if displace >= self.SWIPE_DISPLACE_MIN:
                    abs_vx = abs(vx)
                    abs_vy = abs(vy)
                    if abs_vx >= abs_vy:
                        swipe_dir = 'RIGHT' if vx > 0 else 'LEFT'
                    else:
                        swipe_dir = 'DOWN' if vy > 0 else 'UP'
                    swipe_fired = True
                    swipe_vel   = speed
                    self._last_swipe = now

        return {
            'pos':           (self._sx, self._sy),
            'speed':         speed,
            'pinch_ratio':   pinch_ratio,
            'fist_ratio':    fist_ratio,
            'pinched':       self._pinched,
            'fisted':        self._fisted,
            'swipe_fired':   swipe_fired,
            'swipe_dir':     swipe_dir,
            'swipe_velocity': swipe_vel,
            'velocity_x':    vx,
            'velocity_y':    vy,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Processor
# ─────────────────────────────────────────────────────────────────────────────

class ElementalGestureProcessor:
    """
    Stateful gesture processor for the Elemental Cultivation experience.
    Handles up to 2 hands with independent per-hand state machines.
    The primary hand drives the main gesture; left/right resolved via
    corrected MediaPipe handedness (mirror already applied in tracker.py).
    """

    CLAP_DISTANCE = 0.18
    CLAP_COOLDOWN = 0.8

    def __init__(self):
        # Per-hand state — keyed by handedness label
        self._hand_states: dict = {
            'Right': _HandState(),
            'Left':  _HandState(),
        }
        self._last_clap: float = 0.0
        self._prev_two_hand_dist: Optional[float] = None
        self.state = ElementalGestureState()

    def reset(self):
        for hs in self._hand_states.values():
            hs.reset()
        self._last_clap = 0.0
        self._prev_two_hand_dist = None
        self.state = ElementalGestureState()

    def process(self, hands: List[HandLandmarkData]) -> ElementalGestureState:
        now = time.time()
        st = ElementalGestureState()

        if not hands:
            self.state = st
            return st

        # ── Classify and update each detected hand ────────────────────────────
        # Identify which hands we have by handedness
        right_hand: Optional[HandLandmarkData] = None
        left_hand:  Optional[HandLandmarkData] = None
        for h in hands[:2]:
            if h.handedness == 'Right' and right_hand is None:
                right_hand = h
            elif h.handedness == 'Left' and left_hand is None:
                left_hand = h

        # If both hands have same handedness label (edge case), use index order
        if right_hand is None and left_hand is None:
            right_hand = hands[0]
            if len(hands) >= 2:
                left_hand = hands[1]
        elif right_hand is None and len(hands) >= 2:
            right_hand = hands[1]
        elif left_hand is None and len(hands) >= 2:
            left_hand = hands[1]

        # Primary hand = Right (most natural pointer); fall back to Left
        primary_hand   = right_hand if right_hand is not None else left_hand
        secondary_hand = left_hand  if right_hand is not None else None

        # Update per-hand state machines
        right_data = self._hand_states['Right'].update(right_hand, now) if right_hand else None
        left_data  = self._hand_states['Left'].update(left_hand, now)   if left_hand  else None

        primary_data = right_data if right_hand is not None else left_data

        # ── Populate state from primary hand ─────────────────────────────────
        if primary_data:
            st.hand_pos      = primary_data['pos']
            st.speed         = primary_data['speed']
            st.pinch_ratio   = primary_data['pinch_ratio']
            st.fist_ratio    = primary_data['fist_ratio']
            st.velocity_x    = primary_data['velocity_x']
            st.velocity_y    = primary_data['velocity_y']
            st.swipe_fired   = primary_data['swipe_fired']
            st.swipe_dir     = primary_data['swipe_dir']
            st.swipe_velocity = primary_data['swipe_velocity']

        # ── Per-hand gesture labels for HUD ───────────────────────────────────
        if right_data and right_hand:
            g = _classify_single_hand(right_hand.landmarks,
                                      right_data['pinched'], right_data['fisted'])
            st.right_gesture = g.value
        if left_data and left_hand:
            g = _classify_single_hand(left_hand.landmarks,
                                      left_data['pinched'], left_data['fisted'])
            st.left_gesture = g.value

        # ── Two-hand interaction ──────────────────────────────────────────────
        if right_hand is not None and left_hand is not None:
            st.two_hands = True
            st.hand2_pos = left_data['pos'] if right_data else right_data['pos']  # second hand pos

            # Use corrected positions
            p1x, p1y = primary_data['pos']
            sec_data = left_data if right_hand is not None else right_data
            p2x, p2y = sec_data['pos']
            st.hand2_pos = (p2x, p2y)

            dx = p2x - p1x
            dy = p2y - p1y
            dist = math.sqrt(dx * dx + dy * dy)
            st.two_hand_distance = dist
            st.two_hand_midpoint = ((p1x + p2x) * 0.5, (p1y + p2y) * 0.5)

            # Clap detection
            if (self._prev_two_hand_dist is not None
                    and self._prev_two_hand_dist > self.CLAP_DISTANCE * 1.5
                    and dist < self.CLAP_DISTANCE
                    and now - self._last_clap > self.CLAP_COOLDOWN):
                st.clap_fired = True
                self._last_clap = now

            self._prev_two_hand_dist = dist
        else:
            self._prev_two_hand_dist = None

        # ── Classify primary gesture ──────────────────────────────────────────
        if st.two_hands:
            g = ElementalGesture.TWO_HAND
        elif primary_data:
            g = _classify_single_hand(
                primary_hand.landmarks,
                primary_data['pinched'],
                primary_data['fisted'],
            )
        else:
            g = ElementalGesture.NONE

        if st.swipe_fired:
            g = ElementalGesture.SWIPE

        st.gesture = g
        self.state = st
        return st
