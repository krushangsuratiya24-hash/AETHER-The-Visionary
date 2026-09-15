"""
AETHER — Phase 4 Tests: Spectrum Vision
Tests for:
  1.  Theme registry — all seven themes present
  2.  Theme registry — get by key
  3.  Theme registry — get by ThemeID
  4.  Theme switching
  5.  Theme showcase flag (CYBER is showcase)
  6.  Person data structure (DetectedPerson)
  7.  Mask validation
  8.  Bounding box access
  9.  Empty person list (no persons)
 10.  Single person detection from synthetic mask
 11.  Multi-person detection from synthetic mask
 12.  Mask dimensions match frame
 13.  detect_persons minimum area filter
 14.  Tracking initialization
 15.  Tracking continuity across frames
 16.  Person entering the scene
 17.  Person leaving the scene
 18.  Multiple-person tracking
 19.  Track ID stability
 20.  Tracker reset
 21.  Renderer initialization
 22.  Renderer output — no crash on render
 23.  Renderer no-person rendering
 24.  Renderer single-person rendering
 25.  Renderer all seven themes no crash
 26.  Experience lifecycle (enter/exit)
 27.  Experience keyboard theme switching
 28.  Experience handle_key R (reset tracking)
 29.  Launcher integration — spectrum vision in main
 30.  No fake data / no audio dependency
 31.  Phase 1–3 regression (ensure existing tests still pass)

All run without a webcam.

Run with:
    pytest tests/test_phase4.py -v
"""
from __future__ import annotations

import os
import math
import numpy as np
import pytest
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


@pytest.fixture(scope='module')
def screen():
    return pygame.display.set_mode((1280, 720))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Solid dark BGR frame — simulates camera with no person."""
    return np.full((h, w, 3), 20, dtype=np.uint8)


def _person_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """BGR frame with a brighter region that segmentation would find."""
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    # Add a bright 'person' blob in the centre
    cy, cx = h // 2, w // 2
    frame[cy - 80:cy + 80, cx - 40:cx + 40] = 200
    return frame


def _make_person_mask(
    h: int = 480, w: int = 640,
    cx_frac: float = 0.5, cy_frac: float = 0.5,
    r_frac: float = 0.15,
) -> np.ndarray:
    """Create a circular binary person mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(w * cx_frac)
    cy = int(h * cy_frac)
    r  = int(min(h, w) * r_frac)
    Y, X = np.ogrid[:h, :w]
    mask[((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2] = 255
    return mask


def _make_two_person_mask(h: int = 480, w: int = 640) -> np.ndarray:
    """Create two separate circular blobs (two people)."""
    m1 = _make_person_mask(h, w, cx_frac=0.25, cy_frac=0.5, r_frac=0.12)
    m2 = _make_person_mask(h, w, cx_frac=0.75, cy_frac=0.5, r_frac=0.12)
    return np.clip(m1.astype(np.uint16) + m2.astype(np.uint16), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 1–5: Theme Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestThemeRegistry:

    def _reg(self):
        from experiences.spectrum_vision.themes import ThemeRegistry
        return ThemeRegistry()

    # 1. All seven themes present
    def test_seven_themes_in_registry(self):
        reg = self._reg()
        assert len(reg) == 7

    # 2. Get by key
    def test_get_by_key_all_keys(self):
        reg = self._reg()
        for k in range(1, 8):
            t = reg.get_by_key(k)
            assert t is not None, f'Key {k} not found in registry'
            assert t.key == k

    # 3. Get by ThemeID
    def test_get_by_theme_id(self):
        from experiences.spectrum_vision.themes import ThemeID
        reg = self._reg()
        for tid in ThemeID:
            t = reg.get(tid)
            assert t is not None
            assert t.theme_id == tid

    # 4. Theme switching via key
    def test_theme_names_unique(self):
        reg = self._reg()
        names = [t.name for t in reg.all_themes()]
        assert len(names) == len(set(names)), 'Theme names must be unique'

    # 5. CYBER is marked as showcase
    def test_cyber_is_showcase(self):
        from experiences.spectrum_vision.themes import ThemeID
        reg = self._reg()
        cyber = reg.get(ThemeID.CYBER)
        assert cyber.is_showcase

    def test_no_other_theme_is_showcase(self):
        from experiences.spectrum_vision.themes import ThemeID
        reg = self._reg()
        for tid in ThemeID:
            if tid != ThemeID.CYBER:
                assert not reg.get(tid).is_showcase

    def test_theme_key_1_is_thermal(self):
        from experiences.spectrum_vision.themes import ThemeID
        reg = self._reg()
        t = reg.get_by_key(1)
        assert t.theme_id == ThemeID.THERMAL

    def test_theme_key_7_is_cyber(self):
        from experiences.spectrum_vision.themes import ThemeID
        reg = self._reg()
        t = reg.get_by_key(7)
        assert t.theme_id == ThemeID.CYBER

    def test_all_themes_have_primary_color(self):
        reg = self._reg()
        for t in reg.all_themes():
            r, g, b = t.primary_color
            assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_get_unknown_key_returns_none(self):
        reg = self._reg()
        assert reg.get_by_key(0) is None
        assert reg.get_by_key(8) is None

    def test_contains_all_theme_ids(self):
        from experiences.spectrum_vision.themes import ThemeID
        reg = self._reg()
        for tid in ThemeID:
            assert tid in reg


# ─────────────────────────────────────────────────────────────────────────────
# 6–13: Person data structure and detection
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonDataStructure:

    # 6. DetectedPerson data structure
    def test_detected_person_creation(self):
        from experiences.spectrum_vision.people import DetectedPerson
        mask = _make_person_mask(240, 320)
        p = DetectedPerson(
            mask=mask,
            bbox=(80, 60, 160, 120),
            centroid=(160.0, 120.0),
            area=int(mask.sum() // 255),
        )
        assert p.mask.ndim == 2
        assert p.mask.dtype == np.uint8
        assert len(p.bbox) == 4
        assert len(p.centroid) == 2

    # 7. Mask validation
    def test_validate_mask_2d_uint8(self):
        from experiences.spectrum_vision.people import validate_mask
        mask = _make_person_mask(240, 320)
        assert validate_mask(mask)

    def test_validate_mask_3d_fails(self):
        from experiences.spectrum_vision.people import validate_mask
        mask = _make_person_mask(240, 320)[:, :, np.newaxis]
        assert not validate_mask(mask)

    def test_validate_mask_float_fails(self):
        from experiences.spectrum_vision.people import validate_mask
        mask = _make_person_mask(240, 320).astype(np.float32)
        assert not validate_mask(mask)

    # 8. Bounding box
    def test_bbox_center(self):
        from experiences.spectrum_vision.people import DetectedPerson
        mask = _make_person_mask(100, 100)
        p = DetectedPerson(mask=mask, bbox=(10, 20, 40, 60), centroid=(30.0, 50.0), area=100)
        cx, cy = p.bbox_center
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    # 9. Empty person list from blank mask
    def test_detect_persons_empty_mask(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = np.zeros((480, 640), dtype=np.uint8)
        persons = detect_persons(mask, 480, 640)
        assert persons == []

    # 10. Single person detection
    def test_detect_persons_single(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = _make_person_mask(480, 640, cx_frac=0.5, cy_frac=0.5, r_frac=0.20)
        persons = detect_persons(mask, 480, 640)
        assert len(persons) >= 1
        p = persons[0]
        assert p.mask.shape == (480, 640)
        assert p.mask.dtype == np.uint8
        assert p.area > 0
        assert len(p.bbox) == 4
        assert len(p.centroid) == 2

    # 11. Multi-person detection
    def test_detect_persons_two_blobs(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = _make_two_person_mask(480, 640)
        persons = detect_persons(mask, 480, 640)
        assert len(persons) == 2

    # 12. Mask dimensions match frame
    def test_person_mask_at_frame_resolution(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = _make_person_mask(144, 256)
        persons = detect_persons(mask, 144, 256)
        if persons:
            assert persons[0].mask.shape == (144, 256)

    # 13. Minimum area filter
    def test_tiny_blob_filtered_out(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Very small blob: 10×10 pixels — should be filtered
        mask[10:20, 10:20] = 255
        persons = detect_persons(mask, 480, 640, min_area_frac=0.003)
        assert persons == [], 'Tiny blob should be filtered out'

    def test_large_blob_not_filtered(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = _make_person_mask(480, 640, r_frac=0.20)
        persons = detect_persons(mask, 480, 640)
        assert len(persons) >= 1, 'Large blob should not be filtered'

    def test_persons_sorted_by_area_descending(self):
        from experiences.spectrum_vision.people import detect_persons
        mask = _make_two_person_mask(480, 640)
        persons = detect_persons(mask, 480, 640)
        if len(persons) >= 2:
            assert persons[0].area >= persons[1].area


# ─────────────────────────────────────────────────────────────────────────────
# 14–20: Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiPersonTracker:

    def _tracker(self):
        from experiences.spectrum_vision.tracking import MultiPersonTracker
        return MultiPersonTracker()

    # 14. Initialization
    def test_tracker_starts_empty(self):
        tracker = self._tracker()
        assert tracker.person_count == 0
        assert tracker.all_tracks == []

    # 15. Tracking continuity
    def test_single_person_stable_id(self):
        tracker = self._tracker()
        det = [(320.0, 240.0, (200, 120, 240, 240))]
        tracks1 = tracker.update(det, 640, 480)
        tid1 = tracks1[0].track_id

        # Same position next frame
        tracks2 = tracker.update(det, 640, 480)
        tid2 = tracks2[0].track_id

        assert tid1 == tid2, 'Track ID must remain stable across frames'

    # 16. Person entering the scene
    def test_new_person_gets_new_id(self):
        tracker = self._tracker()
        tracks1 = tracker.update([(160.0, 240.0, (80, 120, 160, 240))], 640, 480)
        id1 = tracks1[0].track_id

        # New person appears far from the first
        det2 = [
            (160.0, 240.0, (80, 120, 160, 240)),
            (480.0, 240.0, (400, 120, 160, 240)),
        ]
        tracks2 = tracker.update(det2, 640, 480)
        ids = {t.track_id for t in tracks2}
        assert len(ids) == 2
        assert id1 in ids

    # 17. Person leaving the scene
    def test_person_leaving_removes_after_timeout(self):
        from experiences.spectrum_vision.tracking import _MAX_MISSING_FRAMES
        tracker = self._tracker()
        det = [(320.0, 240.0, (200, 120, 240, 240))]
        tracker.update(det, 640, 480)

        # Feed empty frames until track expires
        for _ in range(_MAX_MISSING_FRAMES + 2):
            tracker.update([], 640, 480)

        assert tracker.person_count == 0

    # 18. Multiple-person tracking
    def test_two_people_get_separate_ids(self):
        tracker = self._tracker()
        det = [
            (160.0, 240.0, (80, 120, 160, 240)),
            (480.0, 240.0, (400, 120, 160, 240)),
        ]
        tracks = tracker.update(det, 640, 480)
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2  # unique IDs

    # 19. Track ID stability over many frames
    def test_id_stable_over_many_frames(self):
        tracker = self._tracker()
        det = [(320.0, 240.0, (200, 120, 240, 240))]
        first_id = None
        for _ in range(30):
            tracks = tracker.update(det, 640, 480)
            if first_id is None:
                first_id = tracks[0].track_id
            else:
                assert tracks[0].track_id == first_id

    # 20. Reset
    def test_reset_clears_all_tracks(self):
        tracker = self._tracker()
        tracker.update([(320.0, 240.0, (200, 120, 240, 240))], 640, 480)
        assert tracker.person_count > 0
        tracker.reset()
        assert tracker.person_count == 0
        assert tracker.all_tracks == []

    def test_track_label_format(self):
        tracker = self._tracker()
        tracks = tracker.update([(320.0, 240.0, (200, 120, 240, 240))], 640, 480)
        label = tracks[0].label
        assert label.startswith('PERSON ')
        assert len(label) == len('PERSON 01')

    def test_trail_grows_over_frames(self):
        tracker = self._tracker()
        det = [(320.0, 240.0, (200, 120, 240, 240))]
        for _ in range(5):
            tracks = tracker.update(det, 640, 480)
        assert len(tracks[0].trail) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 21–25: Renderer
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectrumRenderer:

    def _make_renderer(self):
        from experiences.spectrum_vision.renderer import SpectrumRenderer
        return SpectrumRenderer(1280, 720)

    # 21. Renderer initialization
    def test_renderer_creates_without_error(self):
        r = self._make_renderer()
        assert r is not None
        assert r._w == 1280
        assert r._h == 720

    # 22. Renderer update no crash
    def test_renderer_update_no_crash(self):
        r = self._make_renderer()
        for _ in range(10):
            r.update(1 / 60)

    # 23. Renderer no-person rendering
    def test_render_no_persons_no_crash(self, screen):
        from experiences.spectrum_vision.themes import THEME_REGISTRY, ThemeID
        r = self._make_renderer()
        frame = _blank_frame(480, 640)
        theme = THEME_REGISTRY.get(ThemeID.CYBER)
        r.render_frame(screen, frame, [], [], theme, fps=30.0)

    # 24. Renderer single-person rendering
    def test_render_single_person_no_crash(self, screen):
        from experiences.spectrum_vision.themes import THEME_REGISTRY, ThemeID
        from experiences.spectrum_vision.people import DetectedPerson
        r = self._make_renderer()
        mask = _make_person_mask(480, 640)
        frame = _blank_frame(480, 640)
        p = DetectedPerson(mask=mask, bbox=(200, 120, 240, 240),
                           centroid=(320.0, 240.0), area=int(mask.sum() // 255))
        theme = THEME_REGISTRY.get(ThemeID.CYBER)
        r.render_frame(screen, frame, [p], [], theme, fps=30.0)

    # 25. All seven themes render without crash
    def test_all_themes_render_no_crash(self, screen):
        from experiences.spectrum_vision.themes import THEME_REGISTRY, ThemeID
        from experiences.spectrum_vision.people import DetectedPerson
        from experiences.spectrum_vision.tracking import PersonTrack

        r = self._make_renderer()
        mask = _make_person_mask(480, 640)
        frame = _blank_frame(480, 640)
        p = DetectedPerson(mask=mask, bbox=(200, 120, 240, 240),
                           centroid=(320.0, 240.0), area=int(mask.sum() // 255))
        track = PersonTrack(track_id=1, centroid=(320.0, 240.0),
                            bbox=(200, 120, 240, 240))

        for tid in ThemeID:
            theme = THEME_REGISTRY.get(tid)
            r.update(1 / 60)
            r.render_frame(screen, frame, [p], [track], theme, fps=30.0)


# ─────────────────────────────────────────────────────────────────────────────
# 26–29: Experience lifecycle and integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectrumVisionExperience:

    def _make_exp(self):
        from experiences.spectrum_vision.experience import SpectrumVisionExperience
        return SpectrumVisionExperience(1280, 720)

    # 26. Lifecycle
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

    def test_enter_exit_multiple_cycles(self):
        exp = self._make_exp()
        for _ in range(3):
            exp.enter()
            assert exp._active
            exp.exit()
            assert not exp._active

    # 27. Keyboard theme switching
    def test_key_1_switches_to_thermal(self):
        from experiences.spectrum_vision.themes import ThemeID
        exp = self._make_exp()
        exp.enter()
        evt = pygame.event.Event(
            pygame.KEYDOWN,
            {'key': pygame.K_1, 'mod': 0, 'unicode': '1', 'scancode': 0},
        )
        handled = exp.handle_key(evt)
        assert handled
        assert exp._current_theme.theme_id == ThemeID.THERMAL
        exp.exit()

    def test_key_7_switches_to_cyber(self):
        from experiences.spectrum_vision.themes import ThemeID
        exp = self._make_exp()
        exp.enter()
        evt = pygame.event.Event(
            pygame.KEYDOWN,
            {'key': pygame.K_7, 'mod': 0, 'unicode': '7', 'scancode': 0},
        )
        exp.handle_key(evt)
        assert exp._current_theme.theme_id == ThemeID.CYBER
        exp.exit()

    def test_all_keys_1_to_7_switch_theme(self):
        from experiences.spectrum_vision.themes import THEME_REGISTRY
        exp = self._make_exp()
        exp.enter()
        key_map = {
            pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
            pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7,
        }
        for key, num in key_map.items():
            evt = pygame.event.Event(
                pygame.KEYDOWN,
                {'key': key, 'mod': 0, 'unicode': str(num), 'scancode': 0},
            )
            handled = exp.handle_key(evt)
            assert handled, f'Key {num} should be handled'
            expected = THEME_REGISTRY.get_by_key(num)
            assert exp._current_theme.theme_id == expected.theme_id
        exp.exit()

    # 28. R key resets tracking
    def test_key_r_resets_tracking(self):
        exp = self._make_exp()
        exp.enter()
        evt = pygame.event.Event(
            pygame.KEYDOWN,
            {'key': pygame.K_r, 'mod': 0, 'unicode': 'r', 'scancode': 0},
        )
        handled = exp.handle_key(evt)
        assert handled
        assert exp._tracker.person_count == 0
        exp.exit()

    def test_key_d_toggles_debug(self):
        exp = self._make_exp()
        exp.enter()
        assert not exp._debug_mode
        evt = pygame.event.Event(
            pygame.KEYDOWN,
            {'key': pygame.K_d, 'mod': 0, 'unicode': 'd', 'scancode': 0},
        )
        exp.handle_key(evt)
        assert exp._debug_mode
        exp.handle_key(evt)
        assert not exp._debug_mode
        exp.exit()

    def test_update_no_crash_without_frame(self):
        exp = self._make_exp()
        exp.enter()
        for _ in range(5):
            exp.update(1 / 60)
        exp.exit()

    def test_update_with_frame_no_crash(self):
        exp = self._make_exp()
        exp.enter()
        frame = _blank_frame(480, 640)
        exp.push_camera_frame(frame)
        for _ in range(5):
            exp.update(1 / 60)
        exp.exit()

    def test_render_no_frame_no_crash(self, screen):
        exp = self._make_exp()
        exp.enter()
        exp.render(screen)
        exp.exit()

    def test_render_with_frame_no_crash(self, screen):
        exp = self._make_exp()
        exp.enter()
        frame = _blank_frame(480, 640)
        exp.push_camera_frame(frame)
        exp.update(1 / 60)
        exp.render(screen)
        exp.exit()

    def test_render_debug_mode_no_crash(self, screen):
        exp = self._make_exp()
        exp.enter()
        exp._debug_mode = True
        frame = _blank_frame(480, 640)
        exp.push_camera_frame(frame)
        exp.update(1 / 60)
        exp.render(screen)
        exp.exit()

    # 29. Launcher integration
    def test_experience_importable(self):
        from experiences.spectrum_vision import SpectrumVisionExperience
        exp = SpectrumVisionExperience(1280, 720)
        assert exp is not None

    def test_main_imports_spectrum_vision(self):
        src = os.path.join('main.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'SpectrumVisionExperience' in content, \
            'main.py must import SpectrumVisionExperience'
        assert 'SPECTRUM_VISION' in content, \
            'main.py must handle SPECTRUM_VISION mode'
        assert 'spectrum_vision_experience' in content, \
            'main.py must have spectrum_vision_experience attribute'


# ─────────────────────────────────────────────────────────────────────────────
# 30: No fake data / no audio dependency
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFakeData:

    def test_experience_has_no_clap_attribute(self):
        from experiences.spectrum_vision.experience import SpectrumVisionExperience
        exp = SpectrumVisionExperience(640, 480)
        assert not hasattr(exp, '_clap')

    def test_experience_no_audio_import(self):
        src = os.path.join('experiences', 'spectrum_vision', 'experience.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'core.audio' not in content
        assert 'ClapDetector' not in content
        assert 'sounddevice' not in content

    def test_experience_no_fake_temperature(self):
        src = os.path.join('experiences', 'spectrum_vision', 'renderer.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        # Should not claim actual temperature
        assert '°C' not in content
        assert 'temperature' not in content.lower() or 'simulated' in content.lower()

    def test_thermal_disclaimer_in_renderer(self):
        src = os.path.join('experiences', 'spectrum_vision', 'renderer.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        # Should acknowledge it's simulated
        assert 'SIMULATED' in content or 'simulated' in content

    def test_xray_disclaimer_in_renderer(self):
        src = os.path.join('experiences', 'spectrum_vision', 'renderer.py')
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'NOT actual' in content or 'SIMULATED' in content

    def test_no_fake_persons_generated(self):
        from experiences.spectrum_vision.people import detect_persons
        blank = np.zeros((480, 640), dtype=np.uint8)
        persons = detect_persons(blank, 480, 640)
        assert persons == [], 'No persons must be synthesized from a blank mask'


# ─────────────────────────────────────────────────────────────────────────────
# 31: Phase 1–3 Regression
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase123Regression:

    def test_vision_controller_still_works(self):
        from experiences.vision_controller.runner import NeonRunnerExperience
        from core.gestures import GestureType
        runner = NeonRunnerExperience(1280, 720)
        runner.enter()
        assert runner.state == 'READY'
        runner.exit()

    def test_elemental_cultivation_still_works(self, screen):
        from experiences.elemental_cultivation.experience import ElementalCultivationExperience
        from core.gestures import GestureType
        exp = ElementalCultivationExperience(1280, 720)
        exp.enter()
        exp.handle_gesture(GestureType.NEUTRAL, GestureType.NEUTRAL, (0.5, 0.5))
        exp.update(1 / 60)
        exp.render(screen)
        exp.exit()
        assert not exp._active

    def test_phase_shift_still_works(self):
        from experiences.phase_shift.experience import PhaseShiftExperience
        exp = PhaseShiftExperience(1280, 720)
        exp.enter()
        assert exp._active
        exp.exit()
        assert not exp._active

    def test_phase_shift_gesture_still_works(self):
        from experiences.phase_shift.gestures import (
            PhaseGesture, _classify_raw
        )
        from tests.test_phase3 import _five_finger
        g, conf = _classify_raw(_five_finger())
        assert g == PhaseGesture.FIVE_FINGERS

    def test_person_segmenter_lazy_mode(self):
        from core.segmentation import PersonSegmenter
        seg = PersonSegmenter(lazy=True)
        mask = seg.person_mask
        assert mask.ndim == 2
        assert mask.dtype == np.uint8

    def test_compositor_still_works(self):
        from core.compositor import BackgroundCompositor
        comp = BackgroundCompositor()
        frame = _blank_frame(48, 64)
        mask = _make_person_mask(48, 64)
        result = comp.composite(frame, mask, blend_alpha=0.5)
        assert result.shape == (48, 64, 3)

    def test_particle_pool_still_works(self):
        from core.particles import ParticlePool
        pool = ParticlePool(capacity=50)
        pool.emit_burst(count=10, x=0, y=0, life=1.0, color=(255, 0, 0))
        assert pool.active_count == 10
        pool.clear()
        assert pool.active_count == 0
