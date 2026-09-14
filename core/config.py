import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class DisplayConfig:
    window_title: str = "AETHER — THE VISIONARY"
    width: int = 1280
    height: int = 720
    target_fps: int = 60
    fullscreen: bool = False

@dataclass
class CameraConfig:
    device_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    fps: int = 30
    mirror_feed: bool = True
    enable_mock_fallback: bool = True

@dataclass
class GestureConfig:
    # Normalized X boundaries (0.0=left, 1.0=right on mirrored feed)
    left_threshold: float = 0.38
    right_threshold: float = 0.62
    
    # Normalized Y boundaries (0.0=top, 1.0=bottom)
    jump_threshold: float = 0.32
    slide_threshold: float = 0.68
    
    # Hysteresis margins to prevent boundary flickering
    hysteresis_x: float = 0.05
    hysteresis_y: float = 0.05
    
    # Temporal smoothing alpha (0 < alpha <= 1). Lower = smoother, higher = faster
    smoothing_factor: float = 0.35
    
    # Debounce / cooldown timings in seconds
    jump_cooldown: float = 0.35
    slide_cooldown: float = 0.35
    lane_change_cooldown: float = 0.20

@dataclass
class Palette:
    # Cyberpunk / Hologram Exhibition Palette
    VOID_DARK: Tuple[int, int, int] = (10, 14, 23)
    SURFACE_DARK: Tuple[int, int, int] = (18, 25, 38)
    PANEL_BG: Tuple[int, int, int] = (24, 34, 52)
    CYAN_NEON: Tuple[int, int, int] = (0, 240, 255)
    CYAN_GLOW: Tuple[int, int, int] = (0, 180, 220)
    MAGENTA_LASER: Tuple[int, int, int] = (255, 0, 85)
    GOLD_ACCENT: Tuple[int, int, int] = (255, 215, 0)
    GREEN_MATRIX: Tuple[int, int, int] = (0, 255, 102)
    TEXT_WHITE: Tuple[int, int, int] = (245, 247, 250)
    TEXT_MUTED: Tuple[int, int, int] = (130, 145, 165)
    BORDER_DIM: Tuple[int, int, int] = (45, 60, 85)
    LOCKED_GRAY: Tuple[int, int, int] = (70, 80, 95)

@dataclass
class PathsConfig:
    root_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"))
    models_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models"))
    hand_landmarker_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models", "hand_landmarker.task"
    ))

display_config = DisplayConfig()
camera_config = CameraConfig()
gesture_config = GestureConfig()
palette = Palette()
paths_config = PathsConfig()
