# AETHER — THE VISIONARY
### *"Your hands are the controller."*

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-00F0FF?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://microsoft.com/windows)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV%205.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/AI-MediaPipe%201.0-FF0055?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Pygame-CE](https://img.shields.io/badge/Engine-Pygame--CE%202.5-FFD700?style=for-the-badge&logo=python&logoColor=black)](https://pyga.me/)
[![License](https://img.shields.io/badge/License-MIT-00FF66?style=for-the-badge)](LICENSE)

**AETHER — The Visionary** is an exhibition-grade Computer Vision suite engineered for high-impact live interactive demonstrations. By merging sub-millisecond optical hand tracking, temporal gesture filtering, and real-time GPU/CPU rendering, AETHER eliminates traditional input hardware, turning human gesture dynamics directly into virtual interaction.

---

## 🌟 Exhibition Roadmap & Phase Status

| Phase | Experience | Status | Description |
| :---: | :--- | :---: | :--- |
| **01** | **🎮 VISION CONTROLLER** | **READY // UNLOCKED** | Full hand-gesture-driven endless 3D arcade cyber-runner with zero-latency controls. |
| **02** | **🔥 ELEMENTAL CULTIVATION** | *LOCKED (PHASE 2)* | 7-Element real-time particle VFX engine driven by mudras and hand stances. |
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

## 🏗 System Architecture

```
AETHER-The-Visionary/
│
├── core/
│   ├── config.py              # Central display, gesture, palette, and camera settings
│   ├── camera.py              # Threaded camera frame grabber with auto-fallback
│   ├── tracker.py             # MediaPipe HandLandmarker wrapper with auto-model caching
│   ├── gestures.py            # Temporal smoothing, hysteresis, and gesture debounce engine
│   └── ui.py                  # Futuristic exhibition HUD, telemetry cards, and debug overlay
│
├── experiences/
│   ├── base.py                # Abstract BaseExperience interface
│   └── vision_controller/     # Phase 1: Built-in arcade cyber-runner
│       └── runner.py          # 3D perspective projection, procedural audio synth, entities
│
├── assets/
│   └── models/                # Local model weights (hand_landmarker.task)
│
├── tests/                     # Comprehensive automated pytest suite
│   ├── test_gestures.py       # Gesture thresholding & hysteresis tests
│   ├── test_tracker.py        # Landmark extraction & blank frame tests
│   ├── test_runner.py         # Game logic, state transitions & collision tests
│   └── test_app_lifecycle.py  # Full application boot, mode switch, and release tests
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

While AETHER is designed to be played 100% hands-free during an exhibition, presenters have access to global management hotkeys:

- **`1`** : Launch Phase 1 (Vision Controller) from Master Menu.
- **`D`** : Toggle live **Debug Telemetry Panel** (displays active FPS, frame latency, tracked hands, normalized coordinates, and gesture confidence).
- **`R` / `SPACE`** : Quick-restart current game session.
- **`ESC`** : Return to AETHER Master Hub (or exit application from Master Hub).

---

## 🧪 Automated Verification Suite

Run the complete automated test suite:
```powershell
pytest -v
```

All 16 unit and integration test suites verify gesture classification, boundary hysteresis, procedural audio synthesis, 3D perspective projection, collision matrices, and clean camera resource release.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
