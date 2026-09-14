"""
AETHER — Elemental Cultivation: VFX Presets
Per-element ambient effect configuration, accessible as data for the HUD.
"""
from __future__ import annotations
from typing import Dict, Any

from experiences.elemental_cultivation.elements import ElementID, ELEMENT_INFO


def get_element_hud_config(element_id: ElementID) -> Dict[str, Any]:
    """Return display colour and name info for the HUD."""
    info = ELEMENT_INFO[element_id]
    return {
        'name': info['name'],
        'icon': info['icon'],
        'color': info['color'],
        'color_b': info['color_b'],
        'key': info['key'],
    }


# Ordered list for element selector display
ELEMENT_ORDER = [
    ElementID.PHOENIX_FLAME,
    ElementID.GOLDEN_SOLAR,
    ElementID.FROST,
    ElementID.THUNDER,
    ElementID.EARTH,
    ElementID.WIND,
    ElementID.VOID,
]
