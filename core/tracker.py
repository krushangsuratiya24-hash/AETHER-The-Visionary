import os
import urllib.request
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.config import paths_config, palette

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# MediaPipe 21 Hand Landmark Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

@dataclass
class HandLandmarkData:
    landmarks: List[Tuple[float, float, float]]  # 21 points (x, y, z)
    wrist: Tuple[float, float, float]
    index_tip: Tuple[float, float, float]
    middle_tip: Tuple[float, float, float]
    palm_center: Tuple[float, float, float]
    confidence: float
    handedness: str  # 'Left' or 'Right'

class HandTracker:
    """
    Robust hand tracking pipeline using Google MediaPipe HandLandmarker.
    Auto-caches the model asset and outputs normalized 3D hand coordinates.
    """
    def __init__(self, model_path: Optional[str] = None, num_hands: int = 1, min_confidence: float = 0.5):
        self.model_path = model_path or paths_config.hand_landmarker_path
        self.num_hands = num_hands
        self.min_confidence = min_confidence
        self.detector: Optional[vision.HandLandmarker] = None
        self._ensure_model_exists()
        self._init_detector()

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f'[AETHER-TRACKER] Model not found locally. Downloading Hand Landmarker asset to {self.model_path}...')
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print(f'[AETHER-TRACKER] Model successfully downloaded ({os.path.getsize(self.model_path)} bytes).')
            except Exception as e:
                print(f'[AETHER-TRACKER] ERROR: Failed to download model: {e}')
                raise

    def _init_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_confidence,
                min_hand_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print('[AETHER-TRACKER] HandLandmarker pipeline initialized successfully.')
        except Exception as e:
            print(f'[AETHER-TRACKER] ERROR initializing HandLandmarker: {e}')
            raise

    def process_frame(self, frame_bgr: np.ndarray) -> List[HandLandmarkData]:
        """
        Process a BGR frame and return detected hands with normalized 3D landmarks.
        """
        if self.detector is None or frame_bgr is None:
            return []

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.detector.detect(mp_image)
            
            detected_hands: List[HandLandmarkData] = []
            
            if detection_result.hand_landmarks:
                for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                    # 21 landmarks
                    lms_list = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                    
                    wrist = lms_list[0]
                    index_tip = lms_list[8]
                    middle_tip = lms_list[12]
                    
                    # Approximate palm center: average of wrist, index MCP, middle MCP, ring MCP, pinky MCP
                    palm_indices = [0, 5, 9, 13, 17]
                    palm_x = sum(lms_list[i][0] for i in palm_indices) / len(palm_indices)
                    palm_y = sum(lms_list[i][1] for i in palm_indices) / len(palm_indices)
                    palm_z = sum(lms_list[i][2] for i in palm_indices) / len(palm_indices)
                    palm_center = (palm_x, palm_y, palm_z)
                    
                    # Handedness
                    handedness = "Right"
                    confidence = 0.95
                    if detection_result.handedness and idx < len(detection_result.handedness):
                        categories = detection_result.handedness[idx]
                        if categories:
                            handedness = categories[0].category_name
                            confidence = categories[0].score

                    detected_hands.append(HandLandmarkData(
                        landmarks=lms_list,
                        wrist=wrist,
                        index_tip=index_tip,
                        middle_tip=middle_tip,
                        palm_center=palm_center,
                        confidence=confidence,
                        handedness=handedness
                    ))

            return detected_hands
        except Exception as e:
            # Prevent single-frame parsing anomalies from crashing the exhibition
            return []

    def draw_landmarks(self, frame: np.ndarray, hands: List[HandLandmarkData]):
        """
        Render futuristic neon hand skeleton and joint nodes directly on the frame.
        """
        h, w = frame.shape[:2]
        
        for hand in hands:
            # Draw connections with neon cyan lines
            for (p1_idx, p2_idx) in HAND_CONNECTIONS:
                p1 = hand.landmarks[p1_idx]
                p2 = hand.landmarks[p2_idx]
                pt1 = (int(p1[0] * w), int(p1[1] * h))
                pt2 = (int(p2[0] * w), int(p2[1] * h))
                cv2.line(frame, pt1, pt2, palette.CYAN_GLOW, 2, cv2.LINE_AA)
            
            # Draw joints
            for i, lm in enumerate(hand.landmarks):
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(frame, (cx, cy), 6, palette.MAGENTA_LASER, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 8, palette.TEXT_WHITE, 1, cv2.LINE_AA)
                elif i == 0:  # Wrist
                    cv2.circle(frame, (cx, cy), 7, palette.GOLD_ACCENT, -1, cv2.LINE_AA)
                else:  # Normal knuckles
                    cv2.circle(frame, (cx, cy), 4, palette.CYAN_NEON, -1, cv2.LINE_AA)
                    
            # Palm center reticle
            pcx, pcy = int(hand.palm_center[0] * w), int(hand.palm_center[1] * h)
            cv2.circle(frame, (pcx, pcy), 10, palette.GREEN_MATRIX, 1, cv2.LINE_AA)
            cv2.circle(frame, (pcx, pcy), 3, palette.GREEN_MATRIX, -1, cv2.LINE_AA)
