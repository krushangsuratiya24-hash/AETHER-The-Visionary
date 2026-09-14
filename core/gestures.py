import time
from enum import Enum
from typing import Optional, Tuple
from collections import deque
from core.config import gesture_config
from core.tracker import HandLandmarkData

class GestureType(Enum):
    NEUTRAL = "NEUTRAL"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    JUMP = "JUMP"
    SLIDE = "SLIDE"

class GestureProcessor:
    """
    Exhibition-grade gesture engine with temporal smoothing, hysteresis thresholding,
    and debounce cooldowns. Prevents jitter and accidental triggers.
    """
    def __init__(self):
        # Smoothing state
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.raw_x: Optional[float] = None
        self.raw_y: Optional[float] = None
        
        # Motion history for velocity/transient detection
        self.history = deque(maxlen=10) # (time, x, y)
        
        # State tracking
        self.current_gesture: GestureType = GestureType.NEUTRAL
        self.active_action: GestureType = GestureType.NEUTRAL
        self.last_jump_time: float = 0.0
        self.last_slide_time: float = 0.0
        self.last_lane_change_time: float = 0.0
        
        # Confidence and presence
        self.is_hand_present: bool = False
        self.hand_confidence: float = 0.0

    def reset(self):
        self.smooth_x = None
        self.smooth_y = None
        self.history.clear()
        self.current_gesture = GestureType.NEUTRAL
        self.active_action = GestureType.NEUTRAL
        self.is_hand_present = False

    def process(self, hand: Optional[HandLandmarkData]) -> Tuple[GestureType, GestureType, Tuple[float, float]]:
        """
        Process a hand detection and return:
        (current_gesture, triggered_action, (smooth_x, smooth_y))
        
        - current_gesture: The ongoing sustained state (NEUTRAL, LEFT, RIGHT, JUMP, SLIDE)
        - triggered_action: An edge-triggered action event (fires only once per motion/cooldown)
        - (smooth_x, smooth_y): Normalized smoothed hand position (0.0 to 1.0)
        """
        now = time.time()
        
        if hand is None:
            self.is_hand_present = False
            self.hand_confidence = 0.0
            self.current_gesture = GestureType.NEUTRAL
            self.active_action = GestureType.NEUTRAL
            pos = (0.5, 0.5) if self.smooth_x is None else (self.smooth_x, self.smooth_y)
            return self.current_gesture, self.active_action, pos

        self.is_hand_present = True
        self.hand_confidence = hand.confidence
        
        # Primary anchor is palm center (stable and robust against finger fidgeting)
        rx, ry = hand.palm_center[0], hand.palm_center[1]
        self.raw_x, self.raw_y = rx, ry
        
        # Apply Exponential Moving Average (EMA) smoothing
        alpha = gesture_config.smoothing_factor
        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = rx, ry
        else:
            self.smooth_x = alpha * rx + (1.0 - alpha) * self.smooth_x
            self.smooth_y = alpha * ry + (1.0 - alpha) * self.smooth_y

        sx, sy = self.smooth_x, self.smooth_y
        self.history.append((now, sx, sy))

        # Check for vertical swipe velocity (upward or downward surge)
        vy = 0.0
        if len(self.history) >= 4:
            dt = self.history[-1][0] - self.history[0][0]
            if dt > 0.02:
                # dy negative = upward movement
                vy = (self.history[-1][2] - self.history[0][2]) / dt

        # Evaluate Gestures using Hysteresis & Boundaries
        triggered_action = GestureType.NEUTRAL
        detected_gesture = GestureType.NEUTRAL

        # 1. Check Vertical gestures (JUMP & SLIDE)
        is_jump_pos = sy < gesture_config.jump_threshold
        is_jump_velocity = vy < -1.8 and sy < 0.45
        
        is_slide_pos = sy > gesture_config.slide_threshold
        is_slide_velocity = vy > 1.8 and sy > 0.55

        if is_jump_pos or is_jump_velocity:
            detected_gesture = GestureType.JUMP
            if now - self.last_jump_time > gesture_config.jump_cooldown:
                triggered_action = GestureType.JUMP
                self.last_jump_time = now
        elif is_slide_pos or is_slide_velocity:
            detected_gesture = GestureType.SLIDE
            if now - self.last_slide_time > gesture_config.slide_cooldown:
                triggered_action = GestureType.SLIDE
                self.last_slide_time = now
        else:
            # 2. Check Horizontal gestures (LEFT & RIGHT) with Hysteresis
            # When currently NEUTRAL or returning from jump/slide:
            if self.current_gesture in (GestureType.NEUTRAL, GestureType.JUMP, GestureType.SLIDE):
                if sx < (gesture_config.left_threshold - gesture_config.hysteresis_x):
                    detected_gesture = GestureType.LEFT
                    if now - self.last_lane_change_time > gesture_config.lane_change_cooldown:
                        triggered_action = GestureType.LEFT
                        self.last_lane_change_time = now
                elif sx > (gesture_config.right_threshold + gesture_config.hysteresis_x):
                    detected_gesture = GestureType.RIGHT
                    if now - self.last_lane_change_time > gesture_config.lane_change_cooldown:
                        triggered_action = GestureType.RIGHT
                        self.last_lane_change_time = now
                else:
                    detected_gesture = GestureType.NEUTRAL

            # When currently in LEFT state (requires moving back past hysteresis margin to exit):
            elif self.current_gesture == GestureType.LEFT:
                if sx > (gesture_config.left_threshold + gesture_config.hysteresis_x):
                    detected_gesture = GestureType.NEUTRAL
                else:
                    detected_gesture = GestureType.LEFT

            # When currently in RIGHT state:
            elif self.current_gesture == GestureType.RIGHT:
                if sx < (gesture_config.right_threshold - gesture_config.hysteresis_x):
                    detected_gesture = GestureType.NEUTRAL
                else:
                    detected_gesture = GestureType.RIGHT

        self.current_gesture = detected_gesture
        self.active_action = triggered_action
        
        return self.current_gesture, self.active_action, (sx, sy)
