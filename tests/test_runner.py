import pytest
import pygame
from experiences.vision_controller.runner import NeonRunnerExperience, Obstacle, ObstacleType
from core.gestures import GestureType

@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
        except Exception:
            pass
    yield
    pygame.quit()

def test_runner_initial_state():
    runner = NeonRunnerExperience(1280, 720)
    runner.enter()
    assert runner.state == "READY"
    assert runner.target_lane == 1
    assert runner.score == 0

def test_runner_start_game():
    runner = NeonRunnerExperience(1280, 720)
    runner.enter()
    runner.last_state_change = 0 # Force cooldown expired
    runner.handle_gesture(GestureType.NEUTRAL, GestureType.JUMP, (0.5, 0.2))
    assert runner.state == "PLAYING"

def test_runner_lane_navigation():
    runner = NeonRunnerExperience(1280, 720)
    runner.enter()
    runner.state = "PLAYING"
    assert runner.target_lane == 1
    
    # Move Left
    runner.handle_gesture(GestureType.LEFT, GestureType.LEFT, (0.2, 0.5))
    assert runner.target_lane == 0
    
    # Cannot move left past lane 0
    runner.handle_gesture(GestureType.LEFT, GestureType.LEFT, (0.2, 0.5))
    assert runner.target_lane == 0

    # Move Right back to center
    runner.handle_gesture(GestureType.RIGHT, GestureType.RIGHT, (0.8, 0.5))
    assert runner.target_lane == 1

    # Move Right to lane 2
    runner.handle_gesture(GestureType.RIGHT, GestureType.RIGHT, (0.8, 0.5))
    assert runner.target_lane == 2

def test_runner_jump_and_slide():
    runner = NeonRunnerExperience(1280, 720)
    runner.enter()
    runner.state = "PLAYING"
    
    # Trigger Jump
    runner.handle_gesture(GestureType.JUMP, GestureType.JUMP, (0.5, 0.2))
    assert runner.is_jumping
    assert not runner.is_sliding

    # Advance jump cycle
    runner.update(runner.jump_duration + 0.1)
    assert not runner.is_jumping

    # Trigger Slide
    runner.handle_gesture(GestureType.SLIDE, GestureType.SLIDE, (0.5, 0.8))
    assert runner.is_sliding
    assert not runner.is_jumping

def test_runner_collision_hurdle_jump():
    runner = NeonRunnerExperience(1280, 720)
    runner.enter()
    runner.state = "PLAYING"
    runner.current_lane_pos = 1.0
    
    # Obstacle approaching player at z=80
    obs = Obstacle(lane=1, z=80.0, obs_type=ObstacleType.HURDLE)
    runner.obstacles = [obs]
    
    # Not jumping -> Game Over
    runner.update(0.01)
    assert runner.state == "GAMEOVER"
