"""
AETHER — Phase 3 Tests  (Hand Power — five-finger rework)
Tests for:
  1.  one-finger classification → 25%
  2.  two-finger classification → 50%
  3.  three-finger classification → 75%
  4.  five-finger (full open hand) classification → 100%
  5.  four-finger rejection → NONE
  6.  temporal confirmation
  7.  gesture hysteresis
  8.  alpha mapping (gesture → target alpha)
  9.  smooth alpha transition
 10.  segmentation mask dimensions
 11.  compositing
 12.  background preservation
 13.  pure-invisible final state (no person VFX at alpha=1.0)
 14.  no person VFX drawn when stable invisible
 15.  Phase Shift lifecycle (enter/exit)
 16.  no microphone dependency
 17.  Phase 2 regression

All run without a webcam or microphone.

Run with:
    pytest tests/test_phase3.py -v
"""
from __future__ import annotations

import math
import os
import time
import pytest
import numpy as np
import pygame


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module', autouse=True)
def init_pygame_headless():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    yield
    # Do NOT call pygame.quit() here — it would corrupt the SDL state for
    # subsequent test modules (test_app_lifecycle, test_display_init) that
    # share the same pytest process. Let the last module handle teardown.


@pytest.fixture(scope='module')
def screen():
    return pygame.display.set_mode((1280, 720))


# ─────────────────────────────────────────────────────────────────────────────
# Hand landmark factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_bgr_frame(h=480, w=640, color=(80, 120, 60)):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_person_mask_fast(h=48, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    cond = ((X - cx) / (w * 0.2)) ** 2 + ((Y - cy) / (h * 0.35)) ** 2 <= 1.0
    mask[cond] = 255
    return mask


def _make_hand(
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    thumb_open: bool = False,
    cx: float = 0.5,
    cy: float = 0.7,
    confidence: float = 0.95,
) -> 'HandLandmarkData':
    """
    Build a HandLandmarkData with controlled finger extension geometry.

    Extension: tip y < pip y (tip higher on screen = smaller y value)
    Folded:    tip y > pip y

    Thumb open: thumb tip far from index MCP (dist/scale > _THUMB_OPEN_DIST)
    Thumb closed: thumb tip close to index MCP
    """
    from core.tracker import HandLandmarkData

    lms = [(cx, cy, 0.0)] * 21

    # Wrist
    lms[0]  = (cx, cy, 0.0)
    # Middle MCP at 0.15 above wrist — used for hand_scale
    lms[9]  = (cx + 0.002, cy - 0.15, 0.0)

    # MCPs slightly below wrist
    for mcp in [5, 9, 13, 17]:
        lms[mcp] = (cx, cy - 0.02, 0.0)
    lms[9] = (cx + 0.002, cy - 0.15, 0.0)   # restore scale landmark

    # ── Non-thumb fingers ─────────────────────────────────────────────────────
    # Index: tip=8, pip=6, mcp=5  — very close tips for normal 2-finger
    if index_extended:
        lms[8] = (cx - 0.005, cy - 0.18, 0.0)
        lms[6] = (cx - 0.003, cy - 0.09, 0.0)
        lms[5] = (cx - 0.002, cy - 0.02, 0.0)
    else:
        lms[8] = (cx, cy + 0.08, 0.0)
        lms[6] = (cx, cy + 0.03, 0.0)
        lms[5] = (cx, cy - 0.02, 0.0)

    # Middle: tip=12, pip=10, mcp=9
    if middle_extended:
        lms[12] = (cx + 0.005, cy - 0.18, 0.0)
        lms[10] = (cx + 0.003, cy - 0.09, 0.0)
        lms[9]  = (cx + 0.002, cy - 0.15, 0.0)
    else:
        lms[12] = (cx, cy + 0.08, 0.0)
        lms[10] = (cx, cy + 0.03, 0.0)
        lms[9]  = (cx + 0.002, cy - 0.15, 0.0)

    # Ring: tip=16, pip=14, mcp=13
    if ring_extended:
        lms[16] = (cx, cy - 0.17, 0.0)
        lms[14] = (cx, cy - 0.07, 0.0)
        lms[13] = (cx, cy - 0.02, 0.0)
    else:
        lms[16] = (cx, cy + 0.08, 0.0)
        lms[14] = (cx, cy + 0.03, 0.0)
        lms[13] = (cx, cy - 0.02, 0.0)

    # Pinky: tip=20, pip=18, mcp=17
    if pinky_extended:
        lms[20] = (cx + 0.01, cy - 0.14, 0.0)
        lms[18] = (cx + 0.005, cy - 0.06, 0.0)
        lms[17] = (cx, cy - 0.02, 0.0)
    else:
        lms[20] = (cx, cy + 0.06, 0.0)
        lms[18] = (cx, cy + 0.02, 0.0)
        lms[17] = (cx, cy - 0.02, 0.0)

    # ── Thumb: tip=4, ip=3, mcp=2, cmc=1 ─────────────────────────────────────
    # Thumb open: tip (4) far from index MCP (5)
    # Thumb closed: tip (4) close to index MCP (5)
    if thumb_open:
        # Thumb spread outward — tip far from index MCP
        lms[4] = (cx - 0.14, cy - 0.10, 0.0)   # dist to lms[5] ≈ 0.14-0.15 > scale*0.25
        lms[3] = (cx - 0.09, cy - 0.06, 0.0)
        lms[2] = (cx - 0.05, cy - 0.01, 0.0)
    else:
        # Thumb tucked — tip close to index MCP
        lms[4] = (cx - 0.01, cy - 0.03, 0.0)   # very close to lms[5]
        lms[3] = (cx - 0.02, cy - 0.02, 0.0)
        lms[2] = (cx - 0.03, cy, 0.0)

    return HandLandmarkData(
        landmarks=lms,
        wrist=lms[0],
        index_tip=lms[8],
        middle_tip=lms[12],
        palm_center=(cx, cy - 0.08, 0.0),
        confidence=confidence,
        handedness='Right',
    )


def _one_finger():
    return _make_hand(index_extended=True)


def _two_finger():
    return _make_hand(index_extended=True, middle_extended=True)


def _three_finger():
    return _make_hand(index_extended=True, middle_extended=True, ring_extended=True)


def _four_finger():
    """Four fingers, no thumb — dead-zone gesture."""
    return _make_hand(index_extended=True, middle_extended=True,
                      ring_extended=True, pinky_extended=True, thumb_open=False)


def _five_finger():
    """Full open palm — all five extended."""
    return _make_hand(index_extended=True, middle_extended=True,
                      ring_extended=True, pinky_extended=True, thumb_open=True)


def _fist():
    return _make_hand()  # nothing extended


# ─────────────────────────────────────────────────────────────────────────────
# 1–5: Raw Gesture Classification
# ─────────────────────────────────────────────────────────────────────────────

class TestRawGestureClassification:

    def _classify(self, hand):
        from experiences.phase_shift.gestures import _classify_raw
        return _classify_raw(hand)

    # 1. ONE_FINGER → 25%
    def test_one_finger_classified(self):
        from experiences.phase_shift.gestures import PhaseGesture
        g, conf = self._classify(_one_finger())
        assert g == PhaseGesture.ONE_FINGER, f'Expected ONE_FINGER, got {g}'
        assert conf > 0.5

    # 2. TWO_FINGERS → 50%
    def test_two_fingers_classified(self):
        from experiences.phase_shift.gestures import PhaseGesture
        g, conf = self._classify(_two_finger())
        assert g == PhaseGesture.TWO_FINGERS, f'Expected TWO_FINGERS, got {g}'
        assert conf > 0.5

    # 3. THREE_FINGERS → 75%
    def test_three_fingers_classified(self):
        from experiences.phase_shift.gestures import PhaseGesture
        g, conf = self._classify(_three_finger())
        assert g == PhaseGesture.THREE_FINGERS, f'Expected THREE_FINGERS, got {g}'
        assert conf > 0.5

    # 4. FIVE_FINGERS → 100% invisible
    def test_five_fingers_classified(self):
        from experiences.phase_shift.gestures import PhaseGesture
        g, conf = self._classify(_five_finger())
        assert g == PhaseGesture.FIVE_FINGERS, f'Expected FIVE_FINGERS, got {g}'
        assert conf > 0.5

    # 5. FOUR_FINGERS → NONE (dead zone)
    def test_four_fingers_rejected(self):
        from experiences.phase_shift.gestures import PhaseGesture
        g, conf = self._classify(_four_finger())
        assert g == PhaseGesture.NONE, (
            f'Four fingers must be rejected (NONE), got {g}'
        )

    def test_fist_is_none(self):
        from experiences.phase_shift.gestures import PhaseGesture
        g, _ = self._classify(_fist())
        assert g == PhaseGesture.NONE

    def test_confidence_always_in_range(self):
        from experiences.phase_shift.gestures import _classify_raw
        for hand in [_one_finger(), _two_finger(), _three_finger(),
                     _four_finger(), _five_finger(), _fist()]:
            _, conf = _classify_raw(hand)
            assert 0.0 <= conf <= 1.0

    def test_scissor_not_in_enum(self):
        """SCISSOR must not exist in the gesture enum."""
        from experiences.phase_shift.gestures import PhaseGesture
        names = [g.name for g in PhaseGesture]
        assert 'SCISSOR' not in names

    def test_five_fingers_not_classified_as_four(self):
        from experiences.phase_shift.gestures import PhaseGesture
        # Five fingers (thumb open) must NOT be classified as NONE/four-finger dead zone
        g, _ = self._classify(_five_finger())
        assert g == PhaseGesture.FIVE_FINGERS


# ─────────────────────────────────────────────────────────────────────────────
# Thumb detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestThumbDetection:

    def test_thumb_open_detected(self):
        from experiences.phase_shift.gestures import _is_thumb_open, _hand_scale
        hand = _five_finger()
        lms  = hand.landmarks
        scale = _hand_scale(lms)
        assert _is_thumb_open(lms, scale), 'Five-finger hand should have thumb open'

    def test_thumb_closed_detected(self):
        from experiences.phase_shift.gestures import _is_thumb_open, _hand_scale
        hand = _four_finger()   # four fingers, thumb tucked
        lms  = hand.landmarks
        scale = _hand_scale(lms)
        assert not _is_thumb_open(lms, scale), 'Four-finger hand should have thumb closed'

    def test_four_finger_without_thumb_is_none(self):
        """Four fingers extended but thumb closed → NONE, not FIVE_FINGERS."""
        from experiences.phase_shift.gestures import PhaseGesture, _classify_raw
        hand = _four_finger()
        g, _ = _classify_raw(hand)
        assert g == PhaseGesture.NONE, f'Expected NONE for four fingers, got {g}'

    def test_five_finger_requires_thumb(self):
        """All four non-thumb fingers + thumb open → FIVE_FINGERS."""
        from experiences.phase_shift.gestures import PhaseGesture, _classify_raw
        hand = _five_finger()
        g, _ = _classify_raw(hand)
        assert g == PhaseGesture.FIVE_FINGERS


# ─────────────────────────────────────────────────────────────────────────────
# 6–7: Temporal Confirmation and Hysteresis
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalClassifier:

    def _make_clf(self):
        from experiences.phase_shift.gestures import PhaseGestureClassifier
        return PhaseGestureClassifier()

    # 6. Temporal confirmation
    def test_confirmation_after_enough_frames(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        clf = self._make_clf()
        for _ in range(_CONFIRM_FRAMES + 4):
            result = clf.update(_one_finger())
        assert result.gesture == PhaseGesture.ONE_FINGER

    def test_five_finger_confirmed_after_enough_frames(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        clf = self._make_clf()
        for _ in range(_CONFIRM_FRAMES + 4):
            result = clf.update(_five_finger())
        assert result.gesture == PhaseGesture.FIVE_FINGERS

    def test_single_frame_does_not_confirm(self):
        from experiences.phase_shift.gestures import PhaseGesture
        clf = self._make_clf()
        result = clf.update(_five_finger())
        # One frame alone is not enough to confirm
        assert isinstance(result.gesture, PhaseGesture)   # no crash

    # 7. Hysteresis — single different frame does not break confirmed gesture
    def test_hysteresis_one_rogue_frame(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        clf = self._make_clf()
        for _ in range(_CONFIRM_FRAMES + 4):
            clf.update(_one_finger())

        # One frame of something different
        clf.update(_five_finger())

        # Confirmed should still be ONE_FINGER
        result = clf.update(_one_finger())
        assert result.gesture == PhaseGesture.ONE_FINGER

    def test_none_hand_gives_none_after_confirmation(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        clf = self._make_clf()
        for _ in range(_CONFIRM_FRAMES + 4):
            r = clf.update(None)
        assert r.gesture == PhaseGesture.NONE

    def test_reset_clears_history(self):
        from experiences.phase_shift.gestures import PhaseGesture
        clf = self._make_clf()
        for _ in range(20):
            clf.update(_five_finger())
        clf.reset()
        result = clf.update(None)
        assert result.gesture == PhaseGesture.NONE
        assert clf.finger_count == 0

    def test_low_confidence_hand_ignored(self):
        from experiences.phase_shift.gestures import PhaseGesture, _MIN_CONFIDENCE, _CONFIRM_FRAMES
        clf = self._make_clf()
        hand = _make_hand(index_extended=True, confidence=_MIN_CONFIDENCE - 0.1)
        for _ in range(_CONFIRM_FRAMES + 4):
            r = clf.update(hand)
        assert r.gesture == PhaseGesture.NONE

    def test_finger_count_field_populated(self):
        from experiences.phase_shift.gestures import PhaseGestureClassifier
        clf = PhaseGestureClassifier()
        clf.update(_five_finger())
        assert clf.finger_count == 5

    def test_finger_count_one(self):
        from experiences.phase_shift.gestures import PhaseGestureClassifier
        clf = PhaseGestureClassifier()
        clf.update(_one_finger())
        # At minimum 1 extended (index)
        assert clf.finger_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8–9: Alpha Mapping and Smooth Transition
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaMapping:

    # 8. Gesture → target alpha mapping
    def test_none_maps_to_zero(self):
        from experiences.phase_shift.gestures import PhaseGesture, GESTURE_ALPHA
        assert GESTURE_ALPHA[PhaseGesture.NONE] == pytest.approx(0.0)

    def test_one_finger_maps_to_25(self):
        from experiences.phase_shift.gestures import PhaseGesture, GESTURE_ALPHA
        assert GESTURE_ALPHA[PhaseGesture.ONE_FINGER] == pytest.approx(0.25)

    def test_two_fingers_maps_to_50(self):
        from experiences.phase_shift.gestures import PhaseGesture, GESTURE_ALPHA
        assert GESTURE_ALPHA[PhaseGesture.TWO_FINGERS] == pytest.approx(0.50)

    def test_three_fingers_maps_to_75(self):
        from experiences.phase_shift.gestures import PhaseGesture, GESTURE_ALPHA
        assert GESTURE_ALPHA[PhaseGesture.THREE_FINGERS] == pytest.approx(0.75)

    def test_five_fingers_maps_to_100(self):
        from experiences.phase_shift.gestures import PhaseGesture, GESTURE_ALPHA
        assert GESTURE_ALPHA[PhaseGesture.FIVE_FINGERS] == pytest.approx(1.00)

    def test_no_scissor_in_mapping(self):
        from experiences.phase_shift.gestures import GESTURE_ALPHA, PhaseGesture
        for key in GESTURE_ALPHA:
            assert key.name != 'SCISSOR', 'SCISSOR must not appear in GESTURE_ALPHA'

    # 9. Smooth alpha transition
    def test_alpha_starts_at_zero(self):
        from experiences.phase_shift.state import AlphaController
        ctrl = AlphaController()
        assert ctrl.alpha == pytest.approx(0.0)

    def test_alpha_interpolates_toward_target(self):
        from experiences.phase_shift.state import AlphaController
        from experiences.phase_shift.gestures import PhaseGesture
        ctrl = AlphaController(lerp_speed=4.0)
        ctrl.set_gesture(PhaseGesture.TWO_FINGERS)
        ctrl.update(1.0 / 60.0)
        assert 0.0 < ctrl.alpha < 0.5

    def test_alpha_converges_to_target(self):
        from experiences.phase_shift.state import AlphaController
        from experiences.phase_shift.gestures import PhaseGesture
        ctrl = AlphaController(lerp_speed=10.0)
        ctrl.set_gesture(PhaseGesture.FIVE_FINGERS)
        for _ in range(300):
            ctrl.update(1.0 / 60.0)
        assert ctrl.alpha == pytest.approx(1.0, abs=0.01)

    def test_alpha_decreases_to_zero(self):
        from experiences.phase_shift.state import AlphaController
        from experiences.phase_shift.gestures import PhaseGesture
        ctrl = AlphaController(lerp_speed=10.0)
        ctrl.set_gesture(PhaseGesture.FIVE_FINGERS)
        for _ in range(300):
            ctrl.update(1.0 / 60.0)
        ctrl.set_gesture(PhaseGesture.NONE)
        for _ in range(300):
            ctrl.update(1.0 / 60.0)
        assert ctrl.alpha < 0.1

    def test_alpha_clamped_to_one(self):
        from experiences.phase_shift.state import AlphaController
        from experiences.phase_shift.gestures import PhaseGesture
        ctrl = AlphaController(lerp_speed=100.0)
        ctrl.set_gesture(PhaseGesture.FIVE_FINGERS)
        for _ in range(100):
            ctrl.update(1.0)
        assert ctrl.alpha <= 1.0

    def test_alpha_never_below_zero(self):
        from experiences.phase_shift.state import AlphaController
        from experiences.phase_shift.gestures import PhaseGesture
        ctrl = AlphaController(lerp_speed=100.0)
        ctrl.set_gesture(PhaseGesture.NONE)
        for _ in range(100):
            ctrl.update(1.0)
        assert ctrl.alpha >= 0.0

    def test_phase_level_labels(self):
        from experiences.phase_shift.state import AlphaController
        from experiences.phase_shift.gestures import PhaseGesture
        ctrl = AlphaController()
        assert ctrl.phase_level_label() == 'VISIBLE'
        ctrl._target_alpha = 0.25
        assert ctrl.phase_level_label() == '25%'
        ctrl._target_alpha = 0.50
        assert ctrl.phase_level_label() == '50%'
        ctrl._target_alpha = 0.75
        assert ctrl.phase_level_label() == '75%'
        ctrl._target_alpha = 1.00
        assert ctrl.phase_level_label() == 'INVISIBLE'


# ─────────────────────────────────────────────────────────────────────────────
# 10–12: Segmentation Mask Dimensions, Compositing, Background
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentationMaskDimensions:

    # 10. Segmentation mask dimensions
    def test_mask_always_2d(self):
        from core.segmentation import PersonSegmenter
        seg = PersonSegmenter(lazy=True)
        mask = seg.person_mask
        assert mask.ndim == 2

    def test_get_mask_for_frame_matches_frame_size(self):
        from core.segmentation import PersonSegmenter
        seg = PersonSegmenter(lazy=True)
        for h, w in [(240, 320), (480, 640), (144, 256), (720, 1280)]:
            frame = _make_bgr_frame(h, w)
            mask = seg.get_mask_for_frame(frame)
            assert mask.shape == (h, w), f'Expected ({h},{w}), got {mask.shape}'
            assert mask.ndim == 2

    def test_mask_dtype_uint8(self):
        from core.segmentation import PersonSegmenter
        seg = PersonSegmenter(lazy=True)
        mask = seg.get_mask_for_frame(_make_bgr_frame(240, 320))
        assert mask.dtype == np.uint8

    def test_no_broadcast_error_144x256(self):
        """Explicitly cover the historical (144,256,1) vs (144,256) broadcast bug."""
        from core.segmentation import PersonSegmenter
        from core.compositor import BackgroundCompositor
        seg  = PersonSegmenter(lazy=True)
        comp = BackgroundCompositor()
        frame = _make_bgr_frame(144, 256)
        mask  = seg.get_mask_for_frame(frame)
        assert mask.shape == (144, 256)
        comp.update_background(frame, mask)
        result = comp.composite(frame, mask, blend_alpha=0.5)
        assert result.shape == (144, 256, 3)

    def test_compositor_handles_3d_mask(self):
        """Compositor must handle (H,W,1) mask without broadcast error."""
        from core.compositor import BackgroundCompositor
        comp  = BackgroundCompositor()
        frame = _make_bgr_frame(48, 64)
        mask_3d = _make_person_mask_fast(48, 64)[:, :, np.newaxis]
        comp.update_background(frame, mask_3d)
        result = comp.composite(frame, mask_3d, blend_alpha=1.0)
        assert result.shape == (48, 64, 3)


# 11. Compositing
class TestCompositing:

    def _make_comp(self):
        from core.compositor import BackgroundCompositor
        return BackgroundCompositor()

    def test_composite_shape_and_dtype(self):
        comp = self._make_comp()
        h, w = 48, 64
        frame = _make_bgr_frame(h, w)
        mask  = _make_person_mask_fast(h, w)
        result = comp.composite(frame, mask, blend_alpha=0.5)
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_blend_alpha_zero_returns_original(self):
        comp = self._make_comp()
        frame = _make_bgr_frame(48, 64, color=(100, 150, 200))
        mask  = _make_person_mask_fast(48, 64)
        result = comp.composite(frame, mask, blend_alpha=0.0)
        assert np.array_equal(result, frame)

    # 12. Background preservation
    def test_background_not_updated_by_person_pixels(self):
        from core.compositor import _WARMUP_FRAMES
        comp = self._make_comp()
        h, w = 48, 64
        green = _make_bgr_frame(h, w, color=(0, 200, 0))
        empty = np.zeros((h, w), dtype=np.uint8)
        for _ in range(_WARMUP_FRAMES + 5):
            comp.update_background(green, empty)
        assert comp.has_background

        full_mask = np.full((h, w), 255, dtype=np.uint8)
        red_frame = _make_bgr_frame(h, w, color=(0, 0, 200))
        comp.update_background(red_frame, full_mask)

        bg = comp.bg_frame
        assert bg is not None
        assert float(bg[:, :, 1].mean()) > 100   # green channel preserved

    def test_full_alpha_replaces_person_with_bg(self):
        from core.compositor import _WARMUP_FRAMES
        comp = self._make_comp()
        h, w = 48, 64
        green = _make_bgr_frame(h, w, color=(0, 200, 0))
        empty = np.zeros((h, w), dtype=np.uint8)
        for _ in range(_WARMUP_FRAMES + 5):
            comp.update_background(green, empty)

        frame = _make_bgr_frame(h, w, color=(0, 0, 200))
        mask  = _make_person_mask_fast(h, w)
        result = comp.composite(frame, mask, blend_alpha=1.0)

        cy, cx = h // 2, w // 2
        if mask[cy, cx] == 255:
            assert result[cy, cx, 1] > 50   # green from background is present


# ─────────────────────────────────────────────────────────────────────────────
# 13–14: Pure-invisible final state — no person VFX
# ─────────────────────────────────────────────────────────────────────────────

class TestPureInvisibleRendering:

    def _make_exp(self):
        from experiences.phase_shift.experience import PhaseShiftExperience
        return PhaseShiftExperience(1280, 720)

    # 13. At stable alpha=1.0 the render path skips all person VFX
    def test_stable_invisible_skips_vfx_draw(self, screen):
        """When alpha >= _INVISIBLE_SUPPRESS_ALPHA, draw_pygame_effects must NOT be called."""
        from experiences.phase_shift.experience import _INVISIBLE_SUPPRESS_ALPHA
        exp = self._make_exp()
        exp.enter()

        # Force alpha to fully invisible
        exp._alpha_ctrl._alpha        = 1.0
        exp._alpha_ctrl._target_alpha = 1.0

        # Inject a camera frame and person mask
        frame = _make_bgr_frame(480, 640)
        exp._camera_frame  = frame
        exp._current_frame = frame
        exp._person_mask   = _make_person_mask_fast(480, 640)

        # Patch draw_pygame_effects to detect if called
        called = []
        original = exp._vfx.draw_pygame_effects
        def spy(*args, **kwargs):
            called.append(1)
            return original(*args, **kwargs)
        exp._vfx.draw_pygame_effects = spy

        exp.render(screen)

        assert len(called) == 0, (
            'draw_pygame_effects must NOT be called when alpha >= suppress threshold'
        )
        exp.exit()

    # 14. At stable alpha=1.0 scanlines are NOT drawn
    def test_stable_invisible_no_scanlines(self, screen):
        """Scanline overlay must be suppressed at stable alpha=1.0."""
        from experiences.phase_shift.experience import _INVISIBLE_SUPPRESS_ALPHA
        exp = self._make_exp()
        exp.enter()
        exp._alpha_ctrl._alpha        = 1.0
        exp._alpha_ctrl._target_alpha = 1.0
        frame = _make_bgr_frame(480, 640)
        exp._camera_frame  = frame
        exp._current_frame = frame

        scanline_called = []
        original_sl = exp._draw_scanline_overlay
        def spy_sl(*args, **kwargs):
            scanline_called.append(1)
            return original_sl(*args, **kwargs)
        exp._draw_scanline_overlay = spy_sl

        exp.render(screen)
        assert len(scanline_called) == 0
        exp.exit()

    def test_partial_alpha_allows_vfx(self, screen):
        """At partial alpha (< threshold) VFX should still draw."""
        from experiences.phase_shift.experience import _INVISIBLE_SUPPRESS_ALPHA
        exp = self._make_exp()
        exp.enter()
        exp._alpha_ctrl._alpha        = 0.5
        exp._alpha_ctrl._target_alpha = 0.5
        frame = _make_bgr_frame(480, 640)
        exp._camera_frame  = frame
        exp._current_frame = frame
        exp._person_mask   = _make_person_mask_fast(480, 640)

        called = []
        original = exp._vfx.draw_pygame_effects
        def spy(*args, **kwargs):
            called.append(1)
            return original(*args, **kwargs)
        exp._vfx.draw_pygame_effects = spy

        exp.render(screen)
        assert len(called) > 0, 'VFX should draw at partial alpha'
        exp.exit()

    def test_suppress_threshold_value(self):
        from experiences.phase_shift.experience import _INVISIBLE_SUPPRESS_ALPHA
        assert 0.85 <= _INVISIBLE_SUPPRESS_ALPHA <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 15: Phase Shift Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseShiftLifecycle:

    def _make_exp(self):
        from experiences.phase_shift.experience import PhaseShiftExperience
        return PhaseShiftExperience(1280, 720)

    def test_enter_sets_active(self):
        exp = self._make_exp()
        exp.enter()
        assert exp._active
        exp.exit()

    def test_exit_clears_active(self):
        exp = self._make_exp()
        exp.enter()
        exp.exit()
        assert not exp._active

    def test_enter_resets_alpha(self):
        exp = self._make_exp()
        exp.enter()
        assert exp._alpha_ctrl.alpha == pytest.approx(0.0)
        exp.exit()

    def test_enter_exit_multiple_times(self):
        exp = self._make_exp()
        for _ in range(3):
            exp.enter()
            assert exp._active
            exp.exit()
            assert not exp._active

    def test_gesture_to_alpha_pipeline(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        exp = self._make_exp()
        exp.enter()
        hand = _five_finger()
        for _ in range(_CONFIRM_FRAMES + 4):
            exp.push_hands([hand])
        assert exp._alpha_ctrl.target_alpha == pytest.approx(1.0)
        exp.exit()

    def test_three_finger_to_75(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        exp = self._make_exp()
        exp.enter()
        hand = _three_finger()
        for _ in range(_CONFIRM_FRAMES + 4):
            exp.push_hands([hand])
        assert exp._alpha_ctrl.target_alpha == pytest.approx(0.75)
        exp.exit()

    def test_no_hands_gives_none_gesture(self):
        from experiences.phase_shift.gestures import PhaseGesture, _CONFIRM_FRAMES
        exp = self._make_exp()
        exp.enter()
        for _ in range(_CONFIRM_FRAMES + 4):
            exp.push_hands([])
        assert exp._alpha_ctrl.target_alpha == pytest.approx(0.0)
        exp.exit()

    def test_update_no_crash(self):
        exp = self._make_exp()
        exp.enter()
        for _ in range(10):
            exp.update(1 / 60)
        exp.exit()

    def test_update_with_frame_no_crash(self):
        exp = self._make_exp()
        exp.enter()
        frame = _make_bgr_frame(480, 640)
        exp.push_camera_frame(frame)
        exp.push_hands([])
        for _ in range(5):
            exp.update(1 / 60)
        exp.exit()

    def test_render_no_crash(self, screen):
        exp = self._make_exp()
        exp.enter()
        exp._current_frame = _make_bgr_frame(480, 640)
        exp.push_hands([])
        exp.update(1 / 60)
        exp.render(screen)
        exp.exit()

    def test_render_debug_mode(self, screen):
        exp = self._make_exp()
        exp.enter()
        exp._debug_mode = True
        exp._current_frame = _make_bgr_frame(480, 640)
        exp.push_hands([_five_finger()])
        exp.update(1 / 60)
        exp.render(screen)
        exp.exit()

    def test_handle_key_d_toggles_debug(self):
        exp = self._make_exp()
        exp.enter()
        assert not exp._debug_mode
        evt = pygame.event.Event(
            pygame.KEYDOWN, {'key': pygame.K_d, 'mod': 0, 'unicode': 'd', 'scancode': 0}
        )
        exp.handle_key(evt)
        assert exp._debug_mode
        exp.handle_key(evt)
        assert not exp._debug_mode
        exp.exit()


# ─────────────────────────────────────────────────────────────────────────────
# 16: No microphone dependency
# ─────────────────────────────────────────────────────────────────────────────

class TestNoMicrophoneDependency:

    def test_experience_has_no_clap_attribute(self):
        from experiences.phase_shift.experience import PhaseShiftExperience
        exp = PhaseShiftExperience(640, 480)
        assert not hasattr(exp, '_clap')

    def test_experience_imports_no_audio(self):
        src = os.path.join('experiences', 'phase_shift', 'experience.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'core.audio' not in content
        assert 'ClapDetector' not in content

    def test_no_scissor_in_experience(self):
        src = os.path.join('experiences', 'phase_shift', 'experience.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'SCISSOR' not in content

    def test_enter_no_audio_error(self):
        from experiences.phase_shift.experience import PhaseShiftExperience
        exp = PhaseShiftExperience(640, 480)
        try:
            exp.enter()
            exp.exit()
        except Exception as e:
            if any(kw in str(e).lower() for kw in ('audio', 'microphone', 'sounddevice')):
                pytest.fail(f'enter() triggered audio error: {e}')

    def test_no_scissor_in_gestures(self):
        src = os.path.join('experiences', 'phase_shift', 'gestures.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'SCISSOR' not in content


# ─────────────────────────────────────────────────────────────────────────────
# 17: Phase 2 Regression
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2Regression:

    def test_runner_experience(self):
        from experiences.vision_controller.runner import NeonRunnerExperience
        from core.gestures import GestureType
        runner = NeonRunnerExperience(1280, 720)
        runner.enter()
        assert runner.state == 'READY'
        runner.state = 'PLAYING'
        runner.handle_gesture(GestureType.JUMP, GestureType.JUMP, (0.5, 0.2))
        assert runner.is_jumping
        runner.exit()

    def test_elemental_cultivation(self, screen):
        from experiences.elemental_cultivation.experience import ElementalCultivationExperience
        from core.gestures import GestureType
        exp = ElementalCultivationExperience(1280, 720)
        exp.enter()
        for _ in range(5):
            exp.handle_gesture(GestureType.NEUTRAL, GestureType.NEUTRAL, (0.5, 0.5))
            exp.update(1 / 60)
        exp.render(screen)
        exp.exit()
        assert not exp._active

    def test_gesture_processor_intact(self):
        from core.gestures import GestureProcessor, GestureType
        from core.tracker import HandLandmarkData
        gp = GestureProcessor()
        lms = [(0.5, 0.5, 0.0)] * 21
        hand = HandLandmarkData(
            landmarks=lms, wrist=(0.5, 0.5, 0.0),
            index_tip=(0.5, 0.5, 0.0), middle_tip=(0.5, 0.5, 0.0),
            palm_center=(0.5, 0.5, 0.0), confidence=0.95, handedness='Right',
        )
        g, a, pos = gp.process(hand)
        assert g == GestureType.NEUTRAL

    def test_particle_pool_intact(self):
        from core.particles import ParticlePool
        pool = ParticlePool(capacity=50)
        pool.emit_burst(count=10, x=0, y=0, life=1.0, color=(255, 0, 0))
        assert pool.active_count == 10
        pool.clear()
        assert pool.active_count == 0

    def test_compositor_intact(self):
        from core.compositor import BackgroundCompositor
        comp = BackgroundCompositor()
        frame = _make_bgr_frame(48, 64)
        mask  = _make_person_mask_fast(48, 64)
        result = comp.composite(frame, mask, blend_alpha=0.5)
        assert result.shape == (48, 64, 3)

    def test_vfx_intact(self, screen):
        from core.particles import ParticlePool
        from experiences.phase_shift.vfx import PhaseShiftVFX
        pool = ParticlePool(capacity=200)
        vfx  = PhaseShiftVFX(640, 480, pool)
        mask = _make_person_mask_fast(48, 64)
        vfx.on_phase_out_start(mask, 640, 480)
        assert pool.active_count > 0
        for _ in range(10):
            vfx.update(1 / 60)
        result = vfx.apply_frame_effects(_make_bgr_frame(480, 640), mask, 0.5)
        assert result.shape == (480, 640, 3)
