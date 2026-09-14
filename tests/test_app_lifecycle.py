import os
import time
import pygame
import pytest
from main import AetherApp
from core.gestures import GestureType

@pytest.fixture(scope='module', autouse=True)
def init_sdl_headless():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    yield
    if pygame.get_init():
        pygame.quit()

def test_full_application_lifecycle():
    app = AetherApp()
    assert app.mode == 'MENU'
    assert app.camera.running
    
    # 1. Run 10 frames in MENU
    for _ in range(10):
        ret, frame = app.camera.read()
        assert ret
        assert frame is not None
        hands = app.tracker.process_frame(frame)
        app.clock.tick(60)

    # 2. Transition into VISION_CONTROLLER
    app._enter_vision_controller()
    assert app.mode == 'VISION_CONTROLLER'
    assert app.runner_experience.state in ('READY', 'PLAYING')

    # 3. Simulate gameplay frames
    for i in range(15):
        ret, frame = app.camera.read()
        hands = app.tracker.process_frame(frame)
        app.runner_experience.handle_gesture(GestureType.NEUTRAL, GestureType.NEUTRAL, (0.5, 0.5))
        app.runner_experience.update(0.016)
        app.runner_experience.render(app.screen)
        app.clock.tick(60)

    # 4. Simulate Jump and Slide in game
    app.runner_experience.state = 'PLAYING'
    app.runner_experience.handle_gesture(GestureType.JUMP, GestureType.JUMP, (0.5, 0.2))
    assert app.runner_experience.is_jumping
    app.runner_experience.update(0.7)
    assert not app.runner_experience.is_jumping

    app.runner_experience.handle_gesture(GestureType.SLIDE, GestureType.SLIDE, (0.5, 0.8))
    assert app.runner_experience.is_sliding
    app.runner_experience.update(0.6)
    assert not app.runner_experience.is_sliding

    # 5. Return to MENU
    app.runner_experience.exit()
    app.mode = 'MENU'
    assert app.mode == 'MENU'

    # 6. Clean release
    app._cleanup()
    assert not app.camera.running
    assert app.camera.cap is None
