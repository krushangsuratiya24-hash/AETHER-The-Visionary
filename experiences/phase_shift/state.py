"""
AETHER — Phase Shift State Machine  (Phase 3)

States:
    VISIBLE       — person is fully visible; background model is being updated.
    PHASING_OUT   — transition in progress (visible → invisible).
    INVISIBLE     — person is hidden; background composite is active.
    PHASING_IN    — transition in progress (invisible → visible).

Transitions (driven exclusively by confirmed clap events):
    VISIBLE     + clap → PHASING_OUT
    PHASING_OUT + transition_complete → INVISIBLE
    INVISIBLE   + clap → PHASING_IN
    PHASING_IN  + transition_complete → VISIBLE

The state machine does NOT accept clap events while a transition is in progress,
preventing double-triggers.
"""
from __future__ import annotations

import time
from enum import Enum


class PhaseState(Enum):
    VISIBLE     = 'VISIBLE'
    PHASING_OUT = 'PHASING OUT'
    INVISIBLE   = 'INVISIBLE'
    PHASING_IN  = 'PHASING IN'


class PhaseStateMachine:
    """
    Explicit state machine for Phase Shift transitions.

    Usage::

        sm = PhaseStateMachine(phase_out_duration=1.8, phase_in_duration=1.8)
        sm.trigger_clap()        # request a transition
        sm.update(dt)            # advance timers (call every frame)

        current_state  = sm.state
        blend_alpha    = sm.blend_alpha    # 0=visible, 1=invisible
        transition_t   = sm.transition_t   # 0→1 progress of active transition
        just_completed = sm.just_completed # True for one frame when state settles

    Attributes:
        state (PhaseState):     Current state.
        blend_alpha (float):    0.0 = fully visible, 1.0 = fully invisible.
        transition_t (float):   0.0 → 1.0 progress through current transition.
        just_completed (bool):  True for exactly one update() cycle when a
                                transition (OUT or IN) completes.
    """

    def __init__(self, phase_out_duration: float = 1.8,
                 phase_in_duration: float = 1.8):
        self._phase_out_dur = phase_out_duration
        self._phase_in_dur  = phase_in_duration

        self.state:          PhaseState = PhaseState.VISIBLE
        self.blend_alpha:    float      = 0.0
        self.transition_t:   float      = 0.0
        self.just_completed: bool       = False

        self._transition_elapsed: float = 0.0
        self._clap_queued:        bool  = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def trigger_clap(self) -> bool:
        """
        Signal that a confirmed clap was detected.
        Returns True if the clap was accepted (transition started or queued).
        Returns False if ignored (transition already in progress).
        """
        if self.state == PhaseState.VISIBLE:
            self._enter_phasing_out()
            return True
        elif self.state == PhaseState.INVISIBLE:
            self._enter_phasing_in()
            return True
        # Ignore clap during active transition
        return False

    def update(self, dt: float):
        """Advance the state machine by dt seconds."""
        self.just_completed = False

        if self.state == PhaseState.PHASING_OUT:
            self._transition_elapsed += dt
            t = min(1.0, self._transition_elapsed / self._phase_out_dur)
            self.transition_t = t
            # Smooth easing curve for blend
            self.blend_alpha = _ease_in_out(t)

            if t >= 1.0:
                self.state       = PhaseState.INVISIBLE
                self.blend_alpha = 1.0
                self.transition_t = 1.0
                self.just_completed = True
                print('[PHASE-SM] → INVISIBLE')

        elif self.state == PhaseState.PHASING_IN:
            self._transition_elapsed += dt
            t = min(1.0, self._transition_elapsed / self._phase_in_dur)
            self.transition_t = t
            # Blend goes from 1.0 → 0.0
            self.blend_alpha = _ease_in_out(1.0 - t)

            if t >= 1.0:
                self.state       = PhaseState.VISIBLE
                self.blend_alpha = 0.0
                self.transition_t = 1.0
                self.just_completed = True
                print('[PHASE-SM] → VISIBLE')

    # ─────────────────────────────────────────────────────────────────────────
    # Internal transitions
    # ─────────────────────────────────────────────────────────────────────────

    def _enter_phasing_out(self):
        self.state = PhaseState.PHASING_OUT
        self._transition_elapsed = 0.0
        self.transition_t = 0.0
        print('[PHASE-SM] VISIBLE → PHASING_OUT')

    def _enter_phasing_in(self):
        self.state = PhaseState.PHASING_IN
        self._transition_elapsed = 0.0
        self.transition_t = 0.0
        print('[PHASE-SM] INVISIBLE → PHASING_IN')

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_transitioning(self) -> bool:
        return self.state in (PhaseState.PHASING_OUT, PhaseState.PHASING_IN)

    @property
    def is_invisible(self) -> bool:
        return self.state == PhaseState.INVISIBLE

    @property
    def is_visible(self) -> bool:
        return self.state == PhaseState.VISIBLE

    def reset(self):
        """Return to VISIBLE state (called on enter/exit experience)."""
        self.state               = PhaseState.VISIBLE
        self.blend_alpha         = 0.0
        self.transition_t        = 0.0
        self.just_completed      = False
        self._transition_elapsed = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Easing function
# ─────────────────────────────────────────────────────────────────────────────

def _ease_in_out(t: float) -> float:
    """Smooth S-curve: starts slow, accelerates, decelerates at end."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
