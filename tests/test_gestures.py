import pytest
from core.gestures import GestureProcessor, GestureType
from core.tracker import HandLandmarkData

def create_mock_hand(x: float, y: float, confidence: float = 0.95) -> HandLandmarkData:
    lms = [(x, y, 0.0)] * 21
    return HandLandmarkData(
        landmarks=lms,
        wrist=(x, y, 0.0),
        index_tip=(x, y, 0.0),
        middle_tip=(x, y, 0.0),
        palm_center=(x, y, 0.0),
        confidence=confidence,
        handedness='Right'
    )

def test_initial_state():
    gp = GestureProcessor()
    g, a, pos = gp.process(None)
    assert g == GestureType.NEUTRAL
    assert a == GestureType.NEUTRAL
    assert not gp.is_hand_present

def test_center_neutral():
    gp = GestureProcessor()
    hand = create_mock_hand(0.5, 0.5)
    # Warm up smoothing
    for _ in range(5):
        g, a, pos = gp.process(hand)
    assert g == GestureType.NEUTRAL

def test_left_gesture():
    gp = GestureProcessor()
    # Move hand clearly to the left zone (x=0.25)
    for _ in range(8):
        g, a, pos = gp.process(create_mock_hand(0.25, 0.5))
    assert g == GestureType.LEFT

def test_right_gesture():
    gp = GestureProcessor()
    # Move hand clearly to the right zone (x=0.75)
    for _ in range(8):
        g, a, pos = gp.process(create_mock_hand(0.75, 0.5))
    assert g == GestureType.RIGHT

def test_jump_gesture():
    gp = GestureProcessor()
    # Move hand high into jump zone (y=0.20)
    for _ in range(8):
        g, a, pos = gp.process(create_mock_hand(0.5, 0.20))
    assert g == GestureType.JUMP

def test_slide_gesture():
    gp = GestureProcessor()
    # Move hand low into slide zone (y=0.80)
    for _ in range(8):
        g, a, pos = gp.process(create_mock_hand(0.5, 0.80))
    assert g == GestureType.SLIDE

def test_hysteresis_left_to_neutral():
    gp = GestureProcessor()
    # Trigger LEFT first
    for _ in range(8):
        gp.process(create_mock_hand(0.25, 0.5))
    assert gp.current_gesture == GestureType.LEFT
    
    # Hand shifts slightly inward to 0.36 (above 0.33, but below 0.43 exit threshold)
    for _ in range(4):
        g, a, pos = gp.process(create_mock_hand(0.36, 0.5))
    assert gp.current_gesture == GestureType.LEFT

    # Hand moves clearly to 0.50 -> returns to NEUTRAL
    for _ in range(12):
        g, a, pos = gp.process(create_mock_hand(0.50, 0.5))
    assert gp.current_gesture == GestureType.NEUTRAL
