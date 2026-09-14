"""
AETHER — Phase Shift Configuration  (Phase 3 — Hand Power rework)
All tuneable parameters for the Phase Shift experience in one place.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PhaseShiftConfig:
    # ── Visual transition timings ────────────────────────────────────────────
    # Speed of alpha interpolation toward target (higher = faster)
    # Value is the lerp coefficient per frame (applied per-frame, not per-second)
    # Effective half-life ≈ ln(2) / (lerp_speed * fps) seconds
    alpha_lerp_speed: float = 4.0       # fast but smooth

    # ── Segmentation update rate ─────────────────────────────────────────────
    # Update segmentation every N main-loop frames (1 = every frame, 2 = every other)
    seg_update_interval: int = 2

    # ── Visual effect colours ─────────────────────────────────────────────────
    # Primary phase energy colour
    energy_color:     Tuple[int,int,int] = (0,  220, 255)   # cyan
    # Secondary accent colour
    accent_color:     Tuple[int,int,int] = (180,  60, 255)  # violet
    # Ghost/silhouette edge colour
    silhouette_color: Tuple[int,int,int] = (0,  255, 180)   # teal

    # ── VFX particle counts ──────────────────────────────────────────────────
    # Number of fragment particles spawned on transparency change
    fragment_count:   int = 80
    # Number of reconstruction particles spawned on transparency decrease
    recon_count:      int = 60

    # ── Scanline overlay ─────────────────────────────────────────────────────
    scanline_spacing: int  = 4       # pixels between scanlines
    scanline_alpha:   float = 0.18   # opacity of scanline overlay

    # ── Background model ─────────────────────────────────────────────────────
    # Warmup period — seconds to collect background before going invisible
    bg_warmup_secs: float = 2.0

    # ── Debug ────────────────────────────────────────────────────────────────
    show_mask_debug: bool = False


# Module-level singleton
phase_shift_config = PhaseShiftConfig()
