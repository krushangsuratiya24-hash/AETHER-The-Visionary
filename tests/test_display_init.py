import os
import pytest
import pygame
from core.config import display_config

@pytest.fixture(autouse=True)
def headless_env():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    yield
    if pygame.get_init():
        pygame.quit()

def test_display_surface_creation():
    screen = pygame.display.set_mode((display_config.width, display_config.height))
    assert screen is not None, 'Display surface must not be None'
    assert screen.get_size() == (1280, 720), f'Expected (1280, 720), got {screen.get_size()}'
    assert screen.get_width() == 1280
    assert screen.get_height() == 720

def test_display_caption():
    pygame.display.set_caption(display_config.window_title)
    caption = pygame.display.get_caption()
    assert len(caption) >= 1
    assert "AETHER" in caption[0]
    assert "VISIONARY" in caption[0]

def test_display_event_pump():
    screen = pygame.display.set_mode((display_config.width, display_config.height))
    screen.fill((10, 14, 23))
    # Must not raise exceptions
    pygame.display.flip()
    pygame.event.pump()
    events = pygame.event.get()
    assert isinstance(events, list)

def test_app_display_initialization():
    from main import AetherApp
    app = AetherApp()
    try:
        assert app.screen is not None, 'app.screen must be a valid surface'
        assert app.screen.get_size() == (1280, 720)
        assert os.environ.get('SDL_VIDEO_CENTERED') == '1'
        assert app.running is True
    finally:
        app._cleanup()
