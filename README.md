# AETHER — THE VISIONARY

> *Your hands are the controller.*
> *See people beyond the ordinary.*
> *What if your hands could control reality?*

**AETHER — The Visionary** is a real-time interactive Computer Vision exhibition platform built in Python. It combines webcam-driven human segmentation, gesture recognition, multi-person visualization, particle systems, and full-screen visual effects into four progressively complex experiences — all running live from a standard RGB webcam.

The project was developed as a four-phase engineering journey, evolving from gesture-controlled interaction into a futuristic multi-person visualization platform.

---

## 🎥 Demo

> Demo video coming soon.

<!-- TODO: Add final AETHER demo video link here once captured from the working application -->

---

## 📸 Screenshots

> Real exhibition screenshots coming soon.

<!-- TODO: Add real screenshots captured from the running application for each of the four experiences -->

---

## ✨ Four Exhibition Experiences

### 🎮 Phase 1 — Vision Controller

A hand-gesture-controlled endless arcade experience.

- Live webcam tracks your hand position and gesture state in real time.
- **Controls:** Move hand LEFT / RIGHT to change lanes, raise hand to JUMP, lower to SLIDE.
- Gesture detection uses MediaPipe HandLandmarker with temporal smoothing and hysteresis to prevent false triggers.
- Keyboard fallback available.
- Futuristic cyberpunk HUD with FPS counter, gesture telemetry, and reticle overlay.

---

### 🔥 Phase 2 — Elemental Cultivation

A gesture-driven elemental particle and VFX experience.

- Both hands tracked simultaneously via MediaPipe HandLandmarker.
- Seven elemental visual modes selectable by keyboard:

| Key | Element |
|-----|---------|
| 1 | 🔥 Phoenix Flame |
| 2 | ☀️ Golden Solar |
| 3 | ❄️ Frost |
| 4 | ⚡ Thunder |
| 5 | 🪨 Earth |
| 6 | 🌪️ Wind |
| 7 | 🌑 Void |

- Gesture interactions: open palm regens energy, fist charges power, swipe launches a projectile shockwave.
- Custom particle engine with object pooling, glow effects, trails, orbit rings, and energy strands.
- Two-hand distance field creates a visual "summoning" effect between hands.

---

### 👻 Phase 3 — Phase Shift

Gesture-controlled real-time body transparency using person segmentation and background compositing.

| Gesture | Effect |
|---------|--------|
| ☝ One finger | 25% transparent |
| ✌ Two fingers | 50% transparent |
| 🖖 Three fingers | 75% transparent |
| 🖐 Five fingers (open palm) | 100% invisible |
| Four fingers | Ignored (dead zone) |

- At 100% invisible: MediaPipe person segmentation replaces the person region with a running background estimate — creating genuine see-through compositing.
- Smooth lerp transitions between transparency levels.
- Animated edge glow, holographic distortion, and particle effects during phase transitions.
- **No microphone or clap detection required.** All interaction is through hand gestures only.
- Four-finger gesture is intentionally a dead zone to prevent accidental triggering of full invisibility.

---

### 🌀 Phase 4 — Spectrum Vision

> *"See people beyond the ordinary."*

Spectrum Vision transforms a live RGB webcam feed into a futuristic real-time people-visualization system. The camera detects and segments people in the scene, then applies one of seven visual themes independently to each detected person.

**Seven selectable themes (keys 1–7):**

| Key | Theme | Description |
|-----|-------|-------------|
| 1 | 🔥 Thermal Simulation | `[SIMULATED]` Thermal-style heat-map derived from RGB luminance with animated shimmer and scanlines |
| 2 | 🦴 X-Ray Style | `[SIMULATED]` Translucent body fill with glowing silhouette, procedural skeleton, joint nodes, and scan sweep |
| 3 | 🟢 Neon Edge | Multi-layer glowing segmentation-contour with animated neon pulse and contour particles |
| 4 | 👤 Silhouette | Clean holographic body fill with internal scanline texture and dual-layer edge glow |
| 5 | 🛰️ Radar | Green-tinted targeting overlay with radar sweep, per-person tracking brackets, trail, and IDs |
| 6 | ⚡ Energy | Pulsing energy aura, procedural arc strands, and particle emission from person contour |
| 7 | 🌌 Cyber Vision | **Primary showcase.** Moving grid, holographic body fill, animated scan sweep, multi-layer edge glow, energy pulse through body, glitch accents, target brackets, data labels, per-person particles |

**Multi-person support:**
- Up to 6 people can be detected and visualized independently.
- Each person receives their own segmentation mask, visual treatment, and tracking identity.
- Stable temporary tracking IDs (`PERSON 01`, `PERSON 02`, …) persist across frames using centroid-distance matching.
- Graceful handling of people entering, leaving, and re-entering the scene.

---

## ⚠️ Important Technical Notes

### Thermal Simulation
- The Thermal theme creates a **simulated** thermal-camera-inspired visualization using RGB luminance data, not actual temperature.
- It does **not** require a thermal camera.
- It does **not** measure body temperature.

### X-Ray Style
- The X-Ray theme creates an X-ray-**inspired** structural overlay.
- It is **not** actual X-ray imaging.
- It does **not** see through clothing or objects.
- The skeleton is procedurally estimated from the person's bounding box — not from actual bone structure.

### Radar
- The Radar theme is a **visual** tracking and scanning system.
- It does **not** provide real-world physical distance measurements (no depth hardware is used).
- Tracking IDs are session-based temporary assignments.

**All four experiences use a standard RGB webcam only.**

---

## 🔬 Computer Vision Pipeline

### Phase 3 & 4 — Person Segmentation

```
Webcam (RGB, 1280×720)
        │
        ▼
Frame capture (ThreadedCamera — non-blocking background thread)
        │
        ▼
MediaPipe Selfie Segmenter (selfie_segmenter.tflite, 256×144 processing)
        │
        ▼
Temporal EMA smoothing + morphological cleanup + hole fill
        │
        ▼
Full-resolution binary person mask (0 = background, 255 = person)
        │
        ├─── Phase 3: Background compositor → invisible compositing
        │
        └─── Phase 4: Connected-component analysis → per-person blobs
                │
                ▼
        Multi-person tracker (centroid-distance matching, stable IDs)
                │
                ▼
        Theme renderer (7 visual modes)
                │
                ▼
        Particles / VFX overlay
                │
                ▼
        HUD (mode, people count, FPS, tracking state, theme selector)
                │
                ▼
        Real-time fullscreen display (Pygame, target 60 FPS)
```

**Known multi-person limitation:** The MediaPipe selfie segmenter produces a single unified foreground mask. When two people physically overlap or stand very close together, connected-component analysis may merge them into a single blob. Separation improves as soon as physical distance between people increases.

---

## 🏗️ Project Architecture

```
AETHER-The-Visionary/
│
├── core/
│   ├── camera.py         # ThreadedCamera — non-blocking webcam capture with mock fallback
│   ├── tracker.py        # MediaPipe HandLandmarker — 21-point hand landmark detection
│   ├── segmentation.py   # PersonSegmenter — MediaPipe selfie segmentation with EMA smoothing
│   ├── compositor.py     # BackgroundCompositor — running background model + invisibility compositing
│   ├── particles.py      # ParticlePool — object-pooled particle engine (no per-frame allocation)
│   ├── vfx.py            # VFX classes — shockwaves, lightning arcs, energy orbs, projectiles
│   ├── effects.py        # EffectComposer — high-level VFX orchestration
│   ├── gestures.py       # GestureProcessor — hand position → gesture state with hysteresis
│   ├── ui.py             # HUD — futuristic telemetry, reticles, gesture cards
│   └── config.py         # DisplayConfig, CameraConfig, Palette, PathsConfig
│
├── experiences/
│   ├── base.py                       # BaseExperience — abstract lifecycle interface
│   ├── vision_controller/
│   │   └── runner.py                 # Phase 1 — arcade runner experience
│   ├── elemental_cultivation/
│   │   ├── experience.py             # Phase 2 — main orchestrator
│   │   ├── elements.py               # Seven elemental VFX definitions
│   │   ├── gestures.py               # Two-hand gesture processor
│   │   └── vfx_presets.py            # Element-specific particle/effect presets
│   ├── phase_shift/
│   │   ├── experience.py             # Phase 3 — main orchestrator
│   │   ├── gestures.py               # Finger-count gesture classifier
│   │   ├── state.py                  # AlphaController — smooth transparency lerp
│   │   ├── vfx.py                    # Phase Shift edge glow and particle effects
│   │   └── config.py                 # Phase Shift configuration constants
│   └── spectrum_vision/
│       ├── experience.py             # Phase 4 — main orchestrator
│       ├── themes.py                 # ThemeRegistry — seven theme descriptors
│       ├── people.py                 # DetectedPerson — per-person mask decomposition
│       ├── renderer.py               # SpectrumRenderer — all seven theme pipelines
│       └── tracking.py               # MultiPersonTracker — centroid-distance stable IDs
│
├── tests/
│   ├── test_phase2.py                # Phase 2 test suite
│   ├── test_phase3.py                # Phase 3 test suite
│   ├── test_phase4.py                # Phase 4 test suite (65 tests)
│   ├── test_runner.py                # Phase 1 runner tests
│   ├── test_tracker.py               # Hand tracker tests
│   ├── test_gestures.py              # Gesture processor tests
│   ├── test_app_lifecycle.py         # Application lifecycle tests
│   └── test_display_init.py          # Display initialization tests
│
├── assets/
│   └── models/
│       ├── selfie_segmenter.tflite   # MediaPipe person segmentation model
│       └── hand_landmarker.task      # MediaPipe hand landmark detection model
│
├── main.py                           # Application entry point + master launcher
└── requirements.txt
```

---

## 🌀 Spectrum Vision Module Detail

```
experiences/spectrum_vision/
│
├── experience.py    # SpectrumVisionExperience — main orchestrator
│                    # Owns camera frame injection, segmentation sub-sampling,
│                    # person detection, tracking updates, rendering dispatch
│
├── themes.py        # ThemeID enum + ThemeDescriptor dataclass + ThemeRegistry
│                    # Immutable registry of all seven themes with colors/metadata
│
├── people.py        # DetectedPerson dataclass + detect_persons()
│                    # Decomposes the global segmentation mask into per-person blobs
│                    # via OpenCV connectedComponentsWithStats; filters by area
│
├── renderer.py      # SpectrumRenderer — full-screen theme rendering pipeline
│                    # One rendering method per theme + shared HUD + particle pool
│
└── tracking.py      # MultiPersonTracker — centroid-distance frame-to-frame matching
                     # Assigns stable temporary IDs; handles entry, exit, re-entry
```

---

## ⌨️ Controls

### Main Menu

| Key | Action |
|-----|--------|
| `1` | Launch Vision Controller |
| `2` | Launch Elemental Cultivation |
| `3` | Launch Phase Shift |
| `4` | Launch Spectrum Vision |
| `D` | Toggle debug overlay |
| `ESC` | Exit |

### Spectrum Vision (in-experience)

| Key | Action |
|-----|--------|
| `1` | Thermal Simulation |
| `2` | X-Ray Style |
| `3` | Neon Edge |
| `4` | Silhouette |
| `5` | Radar |
| `6` | Energy |
| `7` | Cyber Vision |
| `R` | Reset tracking (clear all track IDs) |
| `D` | Toggle debug overlay |
| `ESC` | Return to main menu |
| `Q` | Quit |

### Phase Shift (in-experience)

| Gesture | Transparency |
|---------|-------------|
| ☝ One finger | 25% |
| ✌ Two fingers | 50% |
| 🖖 Three fingers | 75% |
| 🖐 Five fingers | 100% invisible |
| Four fingers | Ignored |

---

## 🛠️ Installation

**Requirements:** Python 3.12 or 3.13 on Windows. A webcam is recommended; the application falls back to a synthetic test feed if no camera is detected.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AETHER-The-Visionary.git
cd AETHER-The-Visionary

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

> On first launch, MediaPipe will download the hand landmark model (~8 MB) automatically if it is not already present in `assets/models/`. The selfie segmentation model (`selfie_segmenter.tflite`) must be placed in `assets/models/` before using Phase 3 or Phase 4. See the comment in `core/segmentation.py` for the download URL.

---

## 🧪 Testing

The project has a comprehensive automated test suite covering all four development phases — without requiring a webcam, microphone, or GPU.

```bash
# Run the full test suite
.venv\Scripts\activate
pytest tests/ -v
```

**Final test result: 251 passed**

- Phase 1 (runner, gestures, tracker, display): included in the general test files
- Phase 2 (`test_phase2.py`): particle system, VFX, elemental experience, multi-hand
- Phase 3 (`test_phase3.py`): gesture classification, alpha controller, segmentation, compositing
- Phase 4 (`test_phase4.py`): 65 tests — theme registry, person detection, multi-person tracking, renderer, experience lifecycle, launcher integration, no-fake-data assertions, Phase 1–3 regression

---

## ⚡ Performance

Real-time performance depends on hardware and active configuration:

- **CPU / GPU:** MediaPipe inference is CPU-bound on most laptops.
- **Camera resolution:** Default capture is 1280×720; downsampling to 256×144 for segmentation.
- **Segmentation frequency:** Segmentation runs every 2nd frame in Spectrum Vision to reduce CPU load while maintaining smooth visual continuity.
- **Number of people:** Each additional detected person adds connected-component analysis and per-person rendering.
- **Active theme:** Cyber Vision and Energy emit particles on every frame; the particle pool is bounded to 3,000 particles maximum to prevent unbounded allocation.

The application targets 60 FPS. Actual FPS will vary with hardware and active effects. The in-experience HUD displays the real-time FPS for reference.

---

## ⚙️ Technology Stack

| Technology | Version used | Role |
|------------|-------------|------|
| Python | 3.13.7 (3.12+ supported) | Runtime |
| [OpenCV](https://opencv.org/) (`opencv-python`) | 5.0.0 | Image processing, connected components, morphology |
| [MediaPipe](https://developers.google.com/mediapipe) | 1.0.1 | Hand landmark detection, person segmentation (TFLite) |
| [NumPy](https://numpy.org/) | 2.5.3 | Array operations, mask processing, colour math |
| [pygame-ce](https://pyga.me/) | 2.5.8 | Window management, rendering, event loop |
| [pytest](https://pytest.org/) | 9.1.1 | Automated test suite |

---

## ⚠️ Known Limitations

| Limitation | Detail |
|------------|--------|
| **Single-channel segmentation mask** | MediaPipe selfie segmenter produces one unified foreground mask. Multi-person separation via connected components degrades when people physically overlap or stand very close together. |
| **RGB webcam only** | No depth camera, thermal camera, or specialized hardware required or used. Thermal/X-Ray themes are RGB-based visual simulations. |
| **No actual temperature measurement** | Thermal Simulation is derived from image luminance, not heat sensors. |
| **No actual X-ray imaging** | X-Ray Style is a structural visualization, not radiographic imaging. |
| **No physical depth measurement** | Radar theme uses visual tracking only; no distance data unless depth hardware is added. |
| **Session-based tracking IDs** | Tracking IDs reset when the experience is entered or `R` is pressed. IDs are not persisted between sessions. |
| **Performance is hardware-dependent** | No specific FPS guarantee on all hardware. |

---

## 🚀 Future Work

- **Dedicated instance segmentation:** Replace the connected-component decomposition with a true multi-instance segmentation model (e.g. YOLOv8-seg, Mask R-CNN) for more robust person separation when people are close.
- **Pose estimation integration:** Integrate MediaPipe Pose or equivalent for accurate skeleton landmarks in the X-Ray theme.
- **GPU-accelerated rendering:** Offload particle and compositing operations to GPU using pygame with hardware surfaces or a compute shader pipeline.
- **Depth camera support:** Integrate Intel RealSense or similar for real-world spatial tracking in the Radar theme.
- **Actual thermal camera integration:** Optional integration path for USB thermal cameras (e.g. FLIR Lepton) to replace the simulated thermal mode with real thermographic data.
- **Additional visualization themes:** Night-vision, sonar, heat-signature trails, skeletal motion capture visualization.
- **Interactive installation mode:** Kiosk/exhibition mode with auto-reset, loop-back, and remote monitoring.
- **Cross-platform / web:** Explore deployment via WebAssembly + WebRTC for browser-based exhibition without local installation.

---

## 📈 Development Journey

AETHER evolved over four phases from a gesture-controlled arcade game into a full computer-vision visualization platform:

```
Phase 1 — Vision Controller
  Hand gesture → game controller replacement
  Hand landmark detection → LEFT / RIGHT / JUMP / SLIDE

Phase 2 — Elemental Cultivation
  Both hands tracked simultaneously
  Gesture-driven elemental particle and VFX system
  Custom particle engine with object pooling

Phase 3 — Phase Shift
  Person segmentation + background modeling
  Finger-count → real-time transparency control
  Background compositor → genuine see-through compositing

Phase 4 — Spectrum Vision
  Multi-person connected-component decomposition
  Centroid-distance temporal tracking → stable person IDs
  Seven full-screen futuristic visualization themes
  Cyber Vision primary showcase mode
```

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

*AETHER — The Visionary is an independent engineering project built for computer vision exhibition and portfolio demonstration.*
