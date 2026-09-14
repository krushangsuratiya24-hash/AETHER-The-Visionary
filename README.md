# AETHER — THE VISIONARY
### *"Your hands are the controller."*

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-00F0FF?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://microsoft.com/windows)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV%205.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/AI-MediaPipe%201.0-FF0055?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Pygame-CE](https://img.shields.io/badge/Engine-Pygame--CE%202.5-FFD700?style=for-the-badge&logo=python&logoColor=black)](https://pyga.me/)
[![License](https://img.shields.io/badge/License-MIT-00FF66?style=for-the-badge)](LICENSE)

**AETHER — The Visionary** is an exhibition-grade Computer Vision suite engineered for high-impact live interactive demonstrations. By merging sub-millisecond optical hand tracking, temporal gesture filtering, and real-time CPU rendering, AETHER eliminates traditional input hardware, turning human gesture dynamics directly into virtual interaction.

---

## 🌟 Exhibition Roadmap & Phase Status

| Phase | Experience | Status | Description |
| :---: | :--- | :---: | :--- |
| **01** | **🎮 VISION CONTROLLER** | **READY // UNLOCKED** | Full hand-gesture-driven endless 3D arcade cyber-runner with zero-latency controls. |
| **02** | **🔥 ELEMENTAL CULTIVATION** | **READY // UNLOCKED** | 7-Element real-time particle VFX engine driven by mudras and hand stances. |
| **03** | **👻 PHASE SHIFT** | *LOCKED (PHASE 3)* | Acoustic transient clap detection + person segmentation for optical cloaking. |
| **04** | **🌀 REALITY SCULPTOR** | *LOCKED (PHASE 4)* | Spatial 6-DOF 2D & 3D wireframe mesh manipulation (grab, rotate, scale, clone). |

---

## 🎮 Phase 1: Vision Controller (Operational)

The first operational experience in AETHER is **Neon Runner**, a high-octane 3D perspective cyber-runner rendered directly in real-time. Players pilot an advanced hovering craft down a neon perspective grid, dodging obstacles, jumping laser barriers, sliding beneath plasma gates, and collecting luminous Aether Cores.

### Gesture Controls (Zero Keyboard Required)

| Action | Physical Gesture | Feedback & HUD Indicator |
| :--- | :--- | :--- |
| **STEER LEFT** | Move hand into the Left Zone ($X < 0.33$) | Lane shift Left + Cyan particle spray |
| **STEER RIGHT** | Move hand into the Right Zone ($X > 0.67$) | Lane shift Right + Cyan particle spray |
| **JUMP** | Raise hand upward ($Y < 0.32$) or swift upward swipe | Parabolic vertical leap + Golden particle arc |
| **SLIDE** | Lower hand downward ($Y > 0.68$) or swift downward swipe | Aerodynamic crouch glide + Laser magenta sparks |
| **NEUTRAL** | Maintain hand in centered zone ($0.38 \le X \le 0.62$) | Balanced forward thrust trajectory |
| **ENGAGE / RESTART** | Raise hand or hold neutral in Game Over / Ready screen | Instant match countdown & restart |

> **Anti-Jitter & Hysteresis**: AETHER utilizes an exponential moving average (EMA) smoothing filter alongside dual-boundary hysteresis corridors. Small accidental hand tremors will never cause false-positive lane changes or unintended jumps.

---

## 🔥 Phase 2: Elemental Cultivation (Operational)

**Elemental Cultivation** is a real-time interactive particle VFX experience where the user's hands directly sculpt elemental energy fields rendered in the Pygame window. Seven distinct elemental disciplines are implemented, each with a unique visual identity powered by a modular VFX engine.

### Seven Elemental Disciplines

| Key | Element | Visual Identity |
| :---: | :--- | :--- |
| `1` | 🔥 **PHOENIX FLAME** | Rising flames, embers, fire trails, heat shimmer, explosive bursts |
| `2` | ☀️ **GOLDEN SOLAR** | Radial rays, orbiting particles, pulsing solar core, orbit rings |
| `3` | ❄️ **FROST** | Ice shards, crystalline fragments, frosty aura, radial freeze burst |
| `4` | ⚡ **THUNDER** | Branching lightning arcs, electrical charge buildup, lightning strike |
| `5` | 🪨 **EARTH** | Rock debris orbit, dust clouds, shockwave, ground-crack illusion |
| `6` | 🌪️ **WIND** | Spiral vortex arms, curved trail ribbons, wind pulse projectile |
| `7` | 🌑 **VOID** | Dark vortex, gravitational orbiting, implosion → explosion |

### Elemental Gesture Controls

| Gesture | Effect |
| :--- | :--- |
| **Open Palm** | Activates elemental energy field; regenerates ENERGY bar |
| **Hand Position** | Controls the position of the elemental effect in screen space |
| **Fist** | Charges POWER; element-specific charge behavior (lightning arcs, embers, etc.) |
| **Pinch** | Compresses/concentrates energy (fire sphere, ice orb, void distortion) |
| **Swipe / Fast Movement** | Launches elemental projectile + shockwave; builds flow combo |
| **Two Hands** | Activates energy field connector between hands; scales effect by distance |
| **Clap (hands together)** | Elemental burst (detected from two-hand close event) |
| **Keys 1–7** | Instantly switch active element |

### Phase 2 Technical Architecture

```
AETHER-The-Visionary/
│
├── core/
│   ├── particles.py           # Particle class, ParticlePool with object pooling (2500 cap)
│   ├── vfx.py                 # Reusable VFX primitives: Shockwave, LightningArc, EnergyOrb,
│   │                          #   OrbitRing, Vortex, EnergyTrail, Projectile
│   └── effects.py             # EffectComposer: owns all live VFX + particle emitter library
│
└── experiences/
    └── elemental_cultivation/
        ├── __init__.py
        ├── gestures.py        # ElementalGestureProcessor — rich mudra detection from
        │                      #   MediaPipe landmarks (open palm, fist, pinch, swipe,
        │                      #   two-hand, clap); EMA smoothed, cooldown gated
        ├── elements.py        # 7 element classes (PhoenixFlame, GoldenSolar, Frost,
        │                      #   Thunder, Earth, Wind, Void) with energy/power state
        │                      #   machines and per-element VFX behaviour
        ├── vfx_presets.py     # Element metadata (name, colour, key) for HUD
        └── experience.py      # ElementalCultivationExperience — implements BaseExperience;
                               #   orchestrates gesture, element, composer, and HUD
```

#### VFX Engine Design

- **ParticlePool** — pre-allocates 2500 `Particle` objects; reuses dead particles to avoid per-frame allocation.
- **Particle** — full property set: position, velocity, acceleration, lifetime, size, opacity, rotation, angular velocity, gravity, drag, trail, colour gradient, glow, orbit, attraction, turbulence. Supports shapes: `circle`, `square`, `shard`, `ring`.
- **EffectComposer** — single-call `update(dt)` + `draw(surface)` drives all live VFX. Provides named emitters: `emit_rising_flames`, `emit_crystals`, `emit_rock_chunks`, `emit_void_fragments`, `emit_orbiting`, `emit_sparks`, `emit_radial_burst`, `emit_directed_stream`.
- **Layered rendering**: particles → vortices → rings → trails → shockwaves → lightning → orbs → projectiles.

#### Energy & Power Mechanic

- **ENERGY** regenerates while holding Open Palm; consumed on launch.
- **POWER** builds during Fist; drains passively; influences effect scale and intensity.
- **FLOW / COMBO** builds with fast hand movement; decays when stationary.
- All three bars are displayed in the live HUD panel.

#### Two-Hand Field

When two hands are detected, a procedural energy arc is drawn between the palms and a midpoint power orb appears. Field scale is proportional to the normalised distance between the two palms.

---

## 🏗 System Architecture

```
AETHER-The-Visionary/
│
├── core/
│   ├── config.py              # Central display, gesture, palette, and camera settings
│   ├── camera.py              # Threaded camera frame grabber with auto-fallback
│   ├── tracker.py             # MediaPipe HandLandmarker wrapper with auto-model caching
│   ├── gestures.py            # Temporal smoothing, hysteresis, and gesture debounce engine
│   ├── ui.py                  # Futuristic exhibition HUD, telemetry cards, and debug overlay
│   ├── particles.py           # Particle system with object pooling (Phase 2)
│   ├── vfx.py                 # VFX primitive classes (Phase 2)
│   └── effects.py             # EffectComposer and emitter library (Phase 2)
│
├── experiences/
│   ├── base.py                # Abstract BaseExperience interface
│   ├── vision_controller/     # Phase 1: Built-in arcade cyber-runner
│   │   └── runner.py          # 3D perspective projection, procedural audio synth, entities
│   └── elemental_cultivation/ # Phase 2: Elemental VFX experience
│       ├── gestures.py        # Elemental gesture processor
│       ├── elements.py        # Seven element classes
│       ├── vfx_presets.py     # Element HUD metadata
│       └── experience.py      # Main Phase 2 experience
│
├── assets/
│   └── models/                # Local model weights (hand_landmarker.task)
│
├── tests/
│   ├── test_gestures.py       # Gesture thresholding & hysteresis tests
│   ├── test_tracker.py        # Landmark extraction & blank frame tests
│   ├── test_runner.py         # Game logic, state transitions & collision tests
│   ├── test_app_lifecycle.py  # Full application boot, mode switch, and release tests
│   └── test_phase2.py         # 58 Phase 2 tests: particles, VFX, gestures, elements,
│                              #   experience lifecycle, and Phase 1 regression
│
├── main.py                    # Master exhibition launcher & state machine
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Tuned repository ignore rules
├── LICENSE                    # MIT License
└── README.md                  # Project landing page & documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.12 or 3.13 (64-bit recommended).
- **Webcam**: Any standard USB or integrated webcam (automatic synthetic fallback included if no camera is connected).

### 2. Setup Environment
```powershell
# Create and activate an isolated virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch AETHER
```powershell
python main.py
```

---

## ⌨️ Presentation & Navigation Hotkeys

| Key | Action |
| :--- | :--- |
| `1` | Launch Phase 1 — Vision Controller |
| `2` | Launch Phase 2 — Elemental Cultivation |
| `1`–`7` *(in Phase 2)* | Switch active element |
| `D` | Toggle live Debug Telemetry Panel |
| `ESC` | Return to AETHER Master Hub (or exit from Hub) |

---

## 🧪 Automated Verification Suite

Run the complete automated test suite (no webcam required):
```powershell
pytest -v
```

The suite covers 78 tests across:
- Gesture classification and boundary hysteresis (Phase 1 + Phase 2)
- Landmark extraction and blank frame handling
- Particle lifecycle, pool exhaustion, and reuse
- VFX object lifecycle (shockwaves, lightning, orbs, projectiles)
- EffectComposer integration
- All 7 element enter/exit/update cycles
- Energy/power/flow state mechanics
- Swipe projectile and shockwave spawning
- Element switching via key events
- Two-hand distance detection
- Full application lifecycle with both Phase 1 and Phase 2
- Phase 1 regression (runner game logic unchanged)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
