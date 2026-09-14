import cv2
import time
import threading
import numpy as np
from typing import Tuple, Optional
from core.config import camera_config

class ThreadedCamera:
    """
    High-performance, threaded camera capture engine.
    Runs video acquisition in a dedicated background worker to eliminate
    I/O frame-wait latency from the primary rendering loop.
    Includes automated fallback to synthetic feed if no camera is detected.
    """
    def __init__(self, device_index: int = None, width: int = None, height: int = None, mirror: bool = None):
        self.device_index = device_index if device_index is not None else camera_config.device_index
        self.width = width if width is not None else camera_config.capture_width
        self.height = height if height is not None else camera_config.capture_height
        self.mirror = mirror if mirror is not None else camera_config.mirror_feed
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.is_mock: bool = False
        self.fps_measured: float = 0.0
        self._last_fps_time = time.time()
        self._fps_counter = 0

    def start(self) -> 'ThreadedCamera':
        if self.running:
            return self
        
        # Try initializing hardware camera
        try:
            # On Windows, cv2.CAP_DSHOW or default CAP_ANY
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                # Fallback to CAP_ANY
                self.cap = cv2.VideoCapture(self.device_index)
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
                
                # Check if we can actually read a valid frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    print(f'[AETHER-CAMERA] Hardware camera at index {self.device_index} failed frame read. Activating fallback.')
                    self._enable_mock()
                else:
                    self.is_mock = False
                    print(f'[AETHER-CAMERA] Hardware camera initialized successfully ({test_frame.shape[1]}x{test_frame.shape[0]}).')
            else:
                print(f'[AETHER-CAMERA] No camera device found at index {self.device_index}. Activating fallback.')
                self._enable_mock()
        except Exception as e:
            print(f'[AETHER-CAMERA] Camera initialization error: {e}. Activating fallback.')
            self._enable_mock()

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True, name='AetherCameraWorker')
        self.thread.start()
        return self

    def _enable_mock(self):
        self.is_mock = True
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _update_loop(self):
        while self.running:
            if not self.is_mock and self.cap:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if self.mirror:
                        frame = cv2.flip(frame, 1)
                    with self.lock:
                        self.current_frame = frame
                        self.frame_count += 1
                        self._measure_fps()
                else:
                    time.sleep(0.005)
            else:
                # Generate synthetic test frame (moving neon grid for headless or camera-less environments)
                frame = self._generate_mock_frame()
                with self.lock:
                    self.current_frame = frame
                    self.frame_count += 1
                    self._measure_fps()
                time.sleep(1.0 / camera_config.fps)

    def _generate_mock_frame(self) -> np.ndarray:
        t = time.time()
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background dark blue
        frame[:, :] = (23, 14, 10)
        
        # Draw moving horizontal and vertical grid lines
        grid_spacing = 40
        shift_y = int((t * 50) % grid_spacing)
        for y in range(shift_y, self.height, grid_spacing):
            cv2.line(frame, (0, y), (self.width, y), (35, 25, 18), 1)
        for x in range(0, self.width, grid_spacing):
            cv2.line(frame, (x, 0), (x, self.height), (35, 25, 18), 1)
            
        cv2.putText(frame, "AETHER SIMULATION FEED (NO WEBCAM DETECTED)", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Headless / Synthetic Mode Active", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 145, 165), 1, cv2.LINE_AA)
        return frame

    def _measure_fps(self):
        self._fps_counter += 1
        now = time.time()
        dt = now - self._last_fps_time
        if dt >= 1.0:
            self.fps_measured = self._fps_counter / dt
            self._fps_counter = 0
            self._last_fps_time = now

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print('[AETHER-CAMERA] Camera released cleanly.')

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
