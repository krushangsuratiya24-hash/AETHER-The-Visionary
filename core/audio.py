"""
AETHER — Microphone Clap Detector  (Phase 3)
Real non-blocking audio pipeline for genuine clap detection.

Detection pipeline per audio chunk:
    microphone stream (sounddevice callback, runs in audio thread)
        → RMS energy calculation
        → adaptive noise floor (EMA, updated only in quiet frames)
        → onset / transient detection (sudden energy spike relative to noise)
        → spectral band energy check (clap has energy across wide frequency range)
        → confidence score combining transient strength + spectral shape
        → debounce + cooldown (prevent sustained noise retriggering)
        → ClapEvent placed in thread-safe deque

The Pygame main loop only needs to call:
    detector.poll_clap() → True if a new confirmed clap occurred

Audio capture is released when stop() is called (safe re-entry).

Dependencies: sounddevice (already in venv)
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Tuneable parameters
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_RATE    = 44100          # Hz
_CHUNK_FRAMES   = 1024           # frames per callback (~23 ms at 44.1 kHz)
_CHANNELS       = 1

# Noise floor EMA — how fast the noise estimate adapts to quiet background
_NOISE_ALPHA    = 0.05           # slow upward drift when quiet

# RMS to noise-floor ratio required to declare a transient (onset)
_TRANSIENT_RATIO = 4.5           # clap RMS must be ≥ 4.5× noise floor

# Minimum absolute RMS to consider (avoids false positives from mic DC offset)
_MIN_ABS_RMS    = 0.005

# Spectral band thresholds (normalised energy in bands)
# A hand clap has broad energy; speech concentrates in 80-4000 Hz.
_CLAP_LOW_BAND  = (1500,  6000)  # Hz — "snap" of a clap
_CLAP_HIGH_BAND = (6000, 12000)  # Hz — upper harmonics

# Minimum fraction of total energy in mid+high bands to qualify as a clap
_SPECTRAL_CLAP_FRAC = 0.20       # at least 20% energy above 1.5 kHz

# Confidence threshold to trigger
_CONFIDENCE_THRESHOLD = 0.55

# After a clap fires, block new clap detections for this long (seconds)
_COOLDOWN = 0.8

# Minimum time between the end of one onset and a new onset (seconds)
_DEBOUNCE  = 0.12

# Maximum duration of a valid clap transient (seconds) — sustained sounds fail
_MAX_TRANSIENT_DURATION = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioTelemetry:
    """Snapshot of real-time audio diagnostics (for HUD / debug overlay)."""
    rms:               float = 0.0
    noise_floor:       float = 0.0
    ratio:             float = 0.0       # rms / noise_floor
    confidence:        float = 0.0
    cooldown_remaining: float = 0.0
    in_transient:      bool  = False
    mic_available:     bool  = False


@dataclass
class _TransientState:
    """Internal transient tracker across audio chunks."""
    active:     bool  = False
    start_time: float = 0.0
    peak_rms:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

class ClapDetector:
    """
    Non-blocking microphone clap detector.

    Lifecycle::

        det = ClapDetector()
        det.start()              # opens microphone stream (background thread)
        ...
        if det.poll_clap():      # check from main thread each frame
            trigger_phase_shift()
        ...
        det.stop()               # closes stream; safe to call multiple times

    Properties::

        det.available     → bool — True if mic stream is open
        det.telemetry     → AudioTelemetry snapshot (for debug display)
        det.sensitivity   → float — detection sensitivity (1.0 = default)
                             increase to detect quieter claps,
                             decrease to suppress false positives
    """

    def __init__(self, sensitivity: float = 1.0):
        self._sensitivity    = 1.0   # will be set via property (enforces clamp)
        self.sensitivity     = sensitivity   # use setter so clamp is applied
        self._stream         = None
        self._lock           = threading.Lock()
        self._clap_events: deque[float] = deque(maxlen=8)

        self._noise_floor    = _MIN_ABS_RMS * 4.0
        self._last_clap_time = 0.0
        self._last_onset_end = 0.0
        self._transient      = _TransientState()

        self._tel            = AudioTelemetry()
        self.available       = False

        # Keep last N chunks of audio for spectral analysis
        self._audio_buffer: deque[np.ndarray] = deque(maxlen=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the microphone stream.  Returns True on success."""
        if self._stream is not None:
            return self.available

        try:
            import sounddevice as sd

            def _callback(indata: np.ndarray, frames: int, time_info, status):
                # indata shape: (frames, channels) — use mono
                mono = indata[:, 0].copy()
                self._process_chunk(mono)

            self._stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_CHUNK_FRAMES,
                channels=_CHANNELS,
                dtype='float32',
                callback=_callback,
                latency='low',
            )
            self._stream.start()
            self.available = True
            with self._lock:
                self._tel.mic_available = True
            print('[AETHER-AUDIO] Microphone stream opened.')
            return True

        except Exception as exc:
            self.available = False
            with self._lock:
                self._tel.mic_available = False
            print(f'[AETHER-AUDIO] Microphone unavailable: {exc}')
            return False

    def stop(self):
        """Close the microphone stream safely."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self.available = False
            with self._lock:
                self._tel.mic_available = False
            print('[AETHER-AUDIO] Microphone stream closed.')

    # ─────────────────────────────────────────────────────────────────────────
    # Main thread API
    # ─────────────────────────────────────────────────────────────────────────

    def poll_clap(self) -> bool:
        """
        Called from the Pygame main thread each frame.
        Returns True (exactly once) when a new clap event has been detected
        since the last call.
        """
        with self._lock:
            if self._clap_events:
                self._clap_events.clear()
                return True
        return False

    @property
    def telemetry(self) -> AudioTelemetry:
        """Thread-safe copy of the latest audio diagnostics."""
        with self._lock:
            import copy
            return copy.copy(self._tel)

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float):
        self._sensitivity = max(0.1, min(5.0, value))

    # ─────────────────────────────────────────────────────────────────────────
    # Audio thread processing  (called from sounddevice callback)
    # ─────────────────────────────────────────────────────────────────────────

    def _process_chunk(self, mono: np.ndarray):
        """
        Core detection logic.  Runs in the sounddevice audio callback thread.
        Must not block or allocate excessively.
        """
        now = time.time()

        # 1. RMS energy of this chunk
        rms = float(np.sqrt(np.mean(mono ** 2)))

        # 2. Adaptive noise floor (update only when signal is quiet)
        if rms < self._noise_floor * 2.5:
            self._noise_floor = (
                _NOISE_ALPHA * rms
                + (1.0 - _NOISE_ALPHA) * self._noise_floor
            )
        # Never let noise floor collapse below absolute minimum
        self._noise_floor = max(self._noise_floor, _MIN_ABS_RMS)

        ratio = rms / max(self._noise_floor, 1e-9)

        # 3. Transient / onset detection
        effective_ratio = _TRANSIENT_RATIO / max(0.1, self._sensitivity)
        is_onset = (rms >= _MIN_ABS_RMS) and (ratio >= effective_ratio)

        if is_onset and not self._transient.active:
            # Rising edge of transient — start tracking
            self._transient.active     = True
            self._transient.start_time = now
            self._transient.peak_rms   = rms
        elif self._transient.active:
            self._transient.peak_rms = max(self._transient.peak_rms, rms)
            if not is_onset:
                # Falling edge — transient ended
                duration = now - self._transient.start_time
                self._transient.active = False
                self._last_onset_end   = now

                # 4. Validity checks on the transient
                if duration <= _MAX_TRANSIENT_DURATION:
                    # 5. Spectral analysis (use buffered audio)
                    conf = self._compute_confidence(
                        rms=self._transient.peak_rms,
                        ratio=ratio,
                        duration=duration,
                    )

                    # 6. Cooldown and debounce gating
                    since_last_clap  = now - self._last_clap_time
                    since_last_onset = now - self._last_onset_end + duration  # re-add duration

                    if (conf >= _CONFIDENCE_THRESHOLD
                            and since_last_clap >= _COOLDOWN
                            and since_last_onset >= _DEBOUNCE):
                        self._last_clap_time = now
                        with self._lock:
                            self._clap_events.append(now)

        # Buffer audio for spectral use
        self._audio_buffer.append(mono.copy())

        # 7. Update telemetry
        cooldown_rem = max(0.0, _COOLDOWN - (now - self._last_clap_time))
        with self._lock:
            self._tel.rms               = rms
            self._tel.noise_floor       = self._noise_floor
            self._tel.ratio             = ratio
            self._tel.cooldown_remaining = cooldown_rem
            self._tel.in_transient      = self._transient.active

    def _compute_confidence(self, rms: float, ratio: float, duration: float) -> float:
        """
        Score how clap-like this transient is.
        Returns 0.0 (definitely not clap) to 1.0 (definitely a clap).

        Factors:
        - transient ratio vs noise floor (strength)
        - duration (claps are short: 20-120 ms)
        - spectral shape (claps have energy in mid+high frequencies)
        """
        # ── Ratio score (capped at some max for normalisation) ──────────────
        # A clap at 4.5× threshold → ratio_score ≈ 0.5; at 10× → ≈ 1.0
        ratio_score = min(1.0, (ratio - _TRANSIENT_RATIO * 0.8) / (_TRANSIENT_RATIO * 3.0))
        ratio_score = max(0.0, ratio_score)

        # ── Duration score (penalty for sustained sounds) ───────────────────
        # Ideal clap: 20-100 ms.  Score drops off sharply after 120 ms.
        ideal_dur = 0.06
        dur_score = math.exp(-((duration - ideal_dur) ** 2) / (2 * 0.05 ** 2))
        dur_score = max(0.0, min(1.0, dur_score))

        # ── Spectral score using buffered audio ─────────────────────────────
        spectral_score = self._spectral_clap_score()

        # ── Combine ─────────────────────────────────────────────────────────
        # Weighted combination; spectral has the most weight to reject speech/music
        conf = 0.35 * ratio_score + 0.20 * dur_score + 0.45 * spectral_score

        with self._lock:
            self._tel.confidence = conf

        return conf

    def _spectral_clap_score(self) -> float:
        """
        Estimate how much energy is in the mid-to-high frequency bands
        characteristic of a hand clap.  Returns 0..1.

        Uses the last few audio chunks in self._audio_buffer.
        """
        if not self._audio_buffer:
            return 0.5  # no data — neutral

        try:
            audio = np.concatenate(list(self._audio_buffer))
            n = len(audio)
            if n < 64:
                return 0.5

            # FFT
            fft_mag = np.abs(np.fft.rfft(audio * np.hanning(n), n=n))
            freqs   = np.fft.rfftfreq(n, d=1.0 / _SAMPLE_RATE)

            total_energy = np.sum(fft_mag ** 2) + 1e-12

            # Mid+high band energy (1.5 kHz – 12 kHz)
            mid_high_mask = (freqs >= _CLAP_LOW_BAND[0]) & (freqs <= _CLAP_HIGH_BAND[1])
            mid_high_energy = np.sum(fft_mag[mid_high_mask] ** 2)

            frac = mid_high_energy / total_energy

            # A clap typically has 25-60% energy in mid+high
            # Speech peaks 80-4000 Hz, so high-freq fraction is lower
            score = min(1.0, frac / _SPECTRAL_CLAP_FRAC)
            return float(score)

        except Exception:
            return 0.5

    # ─────────────────────────────────────────────────────────────────────────
    # Inject synthetic clap (testing / keyboard fallback)
    # ─────────────────────────────────────────────────────────────────────────

    def inject_clap(self):
        """
        DEV / TEST ONLY — inject a synthetic clap event.
        Used by keyboard fallback in exhibition mode.
        Clearly marked so it can never be confused with real clap detection.
        """
        now = time.time()
        with self._lock:
            self._clap_events.append(now)
            self._last_clap_time = now

    def __del__(self):
        self.stop()
