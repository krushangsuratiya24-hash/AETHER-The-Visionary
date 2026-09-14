"""
AETHER — Phase Shift Configuration  (Phase 3)
All tuneable parameters for the Phase Shift experience in one place.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PhaseShiftConfig:
    # ── Visual transition timings ────────────────────────────────────────────
    # Duration (seconds) of the PHASING_OUT transition
    phase_out_duration: float = 1.8
    # Duration (seconds) of the PHASING_IN transition
    phase_in_duration:  float = 1.8

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
    # Number of fragment particles spawned during phase-out burst
    fragment_count:   int = 120
    # Number of reconstruction particles spawned during phase-in
    recon_count:      int = 90

    # ── Scanline overlay ─────────────────────────────────────────────────────
    scanline_spacing: int  = 4       # pixels between scanlines
    scanline_alpha:   float = 0.18   # opacity of scanline overlay

    # ── Background model ─────────────────────────────────────────────────────
    # Warmup period — seconds to collect background before going invisible
    bg_warmup_secs: float = 2.0

    # ── Clap sensitivity ─────────────────────────────────────────────────────
    # Passed to ClapDetector.  1.0 = default, >1 = more sensitive
    clap_sensitivity: float = 1.0

    # ── Debug ────────────────────────────────────────────────────────────────
    show_mask_debug: bool = False


# Module-level singleton
phase_shift_config = PhaseShiftConfig()
