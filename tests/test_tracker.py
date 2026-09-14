import numpy as np
import pytest
from core.tracker import HandTracker, HandLandmarkData

def test_tracker_initialization():
    tracker = HandTracker()
    assert tracker.detector is not None

def test_tracker_blank_frame():
    tracker = HandTracker()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = tracker.process_frame(blank)
    assert isinstance(hands, list)
    assert len(hands) == 0

def test_tracker_draw_landmarks():
    tracker = HandTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_lms = [(0.5, 0.5, 0.0)] * 21
    hand = HandLandmarkData(
        landmarks=mock_lms,
        wrist=(0.5, 0.5, 0.0),
        index_tip=(0.5, 0.5, 0.0),
        middle_tip=(0.5, 0.5, 0.0),
        palm_center=(0.5, 0.5, 0.0),
        confidence=0.99,
        handedness='Right'
    )
    # Ensure drawing does not raise any exception
    tracker.draw_landmarks(frame, [hand])
    assert frame.shape == (480, 640, 3)
