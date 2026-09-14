import pygame
from abc import ABC, abstractmethod
from typing import Tuple
from core.gestures import GestureType

class BaseExperience(ABC):
    """
    Abstract interface for all AETHER experiences.
    Ensures unified lifecycle, input handling, update loop, and rendering pipeline.
    """
    @abstractmethod
    def enter(self):
        """Called when transitioning into this experience."""
        pass

    @abstractmethod
    def handle_gesture(self, gesture: GestureType, action: GestureType, hand_pos: Tuple[float, float]):
        """Dispatches real-time computer vision gesture events."""
        pass

    @abstractmethod
    def handle_key(self, event: pygame.event.Event) -> bool:
        """Dispatches keyboard inputs. Returns True if handled, False otherwise."""
        pass

    @abstractmethod
    def update(self, dt: float):
        """Updates game and VFX simulation state."""
        pass

    @abstractmethod
    def render(self, surface: pygame.Surface):
        """Draws experience visuals onto the target surface."""
        pass

    @abstractmethod
    def exit(self):
        """Cleans up resources upon switching away."""
        pass
