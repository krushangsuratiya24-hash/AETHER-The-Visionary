"""
AETHER — Spectrum Vision: Theme Registry  (Phase 4)

Defines and registers all seven visual themes.
Each theme is a lightweight descriptor; rendering logic lives in renderer.py.

IMPORTANT — technical honesty:
    THERMAL: simulated thermal visualization from RGB data (NOT real temperature).
    X-RAY:   simulated X-ray-inspired structural visualization (NOT actual X-ray).
    RADAR:   tracking/ID visualization (NOT physical depth measurements).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Theme Enum
# ─────────────────────────────────────────────────────────────────────────────

class ThemeID(Enum):
    THERMAL   = 1
    XRAY      = 2
    NEON_EDGE = 3
    SILHOUETTE = 4
    RADAR     = 5
    ENERGY    = 6
    CYBER     = 7


# ─────────────────────────────────────────────────────────────────────────────
# Theme Descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThemeDescriptor:
    """Static metadata for a visual theme."""
    theme_id:     ThemeID
    name:         str
    short_name:   str                     # for HUD display
    key:          int                     # 1–7 keyboard number
    primary_color:   Tuple[int, int, int]
    secondary_color: Tuple[int, int, int]
    description:  str = ''
    is_showcase:  bool = False            # True for the primary showcase mode


# ─────────────────────────────────────────────────────────────────────────────
# Theme Registry
# ─────────────────────────────────────────────────────────────────────────────

class ThemeRegistry:
    """
    Immutable registry of all Spectrum Vision themes.

    Usage::

        reg = ThemeRegistry()
        theme = reg.get(ThemeID.CYBER)
        all_themes = reg.all_themes()
        current = reg.get_by_key(7)
    """

    def __init__(self):
        self._themes: Dict[ThemeID, ThemeDescriptor] = {}
        self._build()

    def _build(self) -> None:
        descriptors = [
            ThemeDescriptor(
                theme_id=ThemeID.THERMAL,
                name='THERMAL SIMULATION',
                short_name='THERMAL',
                key=1,
                primary_color=(255, 140, 0),
                secondary_color=(255, 60, 0),
                description='[SIMULATED] Thermal-style heat-map visualization from RGB data.',
            ),
            ThemeDescriptor(
                theme_id=ThemeID.XRAY,
                name='X-RAY STYLE',
                short_name='X-RAY',
                key=2,
                primary_color=(180, 220, 255),
                secondary_color=(100, 160, 255),
                description='[SIMULATED] X-ray-inspired structural overlay visualization.',
            ),
            ThemeDescriptor(
                theme_id=ThemeID.NEON_EDGE,
                name='NEON EDGE',
                short_name='NEON EDGE',
                key=3,
                primary_color=(0, 255, 180),
                secondary_color=(0, 200, 255),
                description='Glowing segmentation-edge neon contour visualization.',
            ),
            ThemeDescriptor(
                theme_id=ThemeID.SILHOUETTE,
                name='SILHOUETTE',
                short_name='SILHOUETTE',
                key=4,
                primary_color=(0, 240, 255),
                secondary_color=(80, 80, 255),
                description='Clean futuristic human silhouette scanner.',
            ),
            ThemeDescriptor(
                theme_id=ThemeID.RADAR,
                name='RADAR',
                short_name='RADAR',
                key=5,
                primary_color=(0, 255, 80),
                secondary_color=(0, 200, 60),
                description='Radar/targeting visualization with stable tracking IDs.',
            ),
            ThemeDescriptor(
                theme_id=ThemeID.ENERGY,
                name='ENERGY',
                short_name='ENERGY',
                key=6,
                primary_color=(255, 220, 0),
                secondary_color=(255, 100, 0),
                description='Animated energy-field and particle aura around each person.',
            ),
            ThemeDescriptor(
                theme_id=ThemeID.CYBER,
                name='CYBER VISION',
                short_name='CYBER',
                key=7,
                primary_color=(0, 240, 255),
                secondary_color=(140, 0, 255),
                description='Primary showcase: holographic multi-layer cyber scanning.',
                is_showcase=True,
            ),
        ]
        for d in descriptors:
            self._themes[d.theme_id] = d

    def get(self, theme_id: ThemeID) -> ThemeDescriptor:
        return self._themes[theme_id]

    def get_by_key(self, key: int) -> Optional[ThemeDescriptor]:
        for d in self._themes.values():
            if d.key == key:
                return d
        return None

    def all_themes(self) -> list:
        """Return all theme descriptors sorted by key."""
        return sorted(self._themes.values(), key=lambda d: d.key)

    def __len__(self) -> int:
        return len(self._themes)

    def __contains__(self, theme_id: ThemeID) -> bool:
        return theme_id in self._themes


# Module-level singleton
THEME_REGISTRY = ThemeRegistry()
