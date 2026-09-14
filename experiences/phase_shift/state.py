"""
AETHER — Phase Shift Alpha Controller  (Phase 3 — Hand Power)

Controls the current transparency alpha from 0.0 (fully visible) to
1.0 (fully invisible) based on the confirmed hand gesture.

Gesture → target alpha mapping:
    NONE          → 0.0  (visible)
    ONE_FINGER    → 0.25 (25% transparent)
    TWO_FINGERS   → 0.50 (50% transparent)
    THREE_FINGERS → 0.75 (75% transparent)
    FIVE_FINGERS  → 1.00 (fully invisible)

The alpha smoothly interpolates from its current value toward the target
using a lerp per dt, giving fluid holographic transitions between levels.

This replaces the binary clap-triggered VISIBLE/PHASING_OUT/INVISIBLE/PHASING_IN
state machine from the old Phase 3 design.
"""
from __future__ import annotations

from experiences.phase_shift.gestures import PhaseGesture, GESTURE_ALPHA


# Lerp speed: larger = faster convergence
# At speed=4.0 and 60fps, half-life ≈ 0.17s (fast but perceptually smooth)
_DEFAULT_LERP_SPEED = 4.0


class AlphaController:
    """
    Smooth alpha interpolator for Phase Shift.

    Usage::

        ctrl = AlphaController(lerp_speed=4.0)
        ctrl.set_gesture(PhaseGesture.TWO_FINGERS)   # called when gesture changes
        ctrl.update(dt)                               # called every frame
        blend_alpha = ctrl.alpha                      # current interpolated value
    """

    def __init__(self, lerp_speed: float = _DEFAULT_LERP_SPEED):
        self._lerp_speed    = lerp_speed
        self._target_alpha  = 0.0
        self._alpha         = 0.0

        self._gesture       = PhaseGesture.NONE
        self._prev_gesture  = PhaseGesture.NONE

        # Flag: was there a transition this frame?
        self.just_changed: bool = False

    # ── Public API ───────────────────────────────────────────────────────────

    def set_gesture(self, gesture: PhaseGesture) -> None:
        """
        Update the target gesture.  The alpha will smoothly lerp toward
        the new target on subsequent update() calls.
        """
        self._prev_gesture = self._gesture
        self._gesture      = gesture
        new_target = GESTURE_ALPHA.get(gesture, 0.0)

        self.just_changed = (new_target != self._target_alpha)
        self._target_alpha = new_target

    def update(self, dt: float) -> None:
        """
        Advance interpolation by dt seconds.
        Call once per main loop frame.
        """
        speed = self._lerp_speed * dt
        diff  = self._target_alpha - self._alpha
        if abs(diff) < 0.001:
            self._alpha = self._target_alpha
        else:
            self._alpha += diff * min(1.0, speed)

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Current interpolated transparency alpha [0.0, 1.0]."""
        return self._alpha

    @property
    def target_alpha(self) -> float:
        """Target transparency alpha [0.0, 1.0]."""
        return self._target_alpha

    @property
    def gesture(self) -> PhaseGesture:
        """Currently active (confirmed) gesture."""
        return self._gesture

    @property
    def is_active(self) -> bool:
        """True when any transparency is active (alpha > 0)."""
        return self._alpha > 0.01

    @property
    def is_fully_invisible(self) -> bool:
        """True when fully at 100% invisible."""
        return self._alpha > 0.98

    @property
    def is_transitioning(self) -> bool:
        """True when alpha is still converging toward target."""
        return abs(self._target_alpha - self._alpha) > 0.005

    def phase_level_label(self) -> str:
        """Human-readable level label for HUD."""
        alpha = self._target_alpha
        if alpha == 0.0:
            return 'VISIBLE'
        elif alpha <= 0.25:
            return '25%'
        elif alpha <= 0.50:
            return '50%'
        elif alpha <= 0.75:
            return '75%'
        else:
            return 'INVISIBLE'

    def reset(self):
        """Reset to fully visible (called on enter/exit)."""
        self._alpha         = 0.0
        self._target_alpha  = 0.0
        self._gesture       = PhaseGesture.NONE
        self._prev_gesture  = PhaseGesture.NONE
        self.just_changed   = False
