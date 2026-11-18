"""
Real-time Hand Detection module using MediaPipe and OpenCV.

Supports:
- Multi-hand detection (up to 2 hands simultaneously)
- Extraction of all 21 3D hand landmarks in both normalized and pixel space
- Handedness classification (Left vs. Right Hand identification)
- Landmark ID and coordinate visual overlays
- Graceful handling when no hand is present
- Smooth real-time tracking
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import logging
import time
import cv2
import mediapipe as mp
import numpy as np

# Landmark anatomical definitions according to MediaPipe Hands model
LANDMARK_NAMES: Dict[int, str] = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}


@dataclass
class DetectedHand:
    """Encapsulates detection metrics and landmarks for a single hand."""
    hand_index: int
    label: str                     # "Left" or "Right"
    confidence: float              # Classification confidence (0.0 to 1.0)
    landmarks: List[Tuple[int, int, int]]  # List of 21 (id, x_pixel, y_pixel)
    bbox: Tuple[int, int, int, int]        # Bounding box (x_min, y_min, width, height)


class HandDetectorError(Exception):
    """Raised when MediaPipe Hand detector encounters fatal errors."""
    pass


class HandDetector:
    """Detects and tracks hands, landmarks, and handedness in real time."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        mirror_mode: bool = True
    ) -> None:
        """
        Initialize the MediaPipe Hand Landmarker.

        Args:
            static_image_mode: Treat frames as static images (False for video streaming).
            max_num_hands: Maximum number of hands to detect concurrently (1 or 2).
            min_detection_confidence: Minimum confidence [0.0, 1.0] to detect hands.
            min_tracking_confidence: Minimum confidence [0.0, 1.0] to track hands.
            model_complexity: Landmark model complexity (0 or 1).
            mirror_mode: Set True if the incoming video is horizontally mirrored.
                         Inverts MediaPipe's raw handedness so physical Left/Right matches.
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.mirror_mode = mirror_mode

        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles
        except Exception as exc:
            raise HandDetectorError(f"Failed to initialize MediaPipe Hands: {exc}") from exc

        self.last_results: Optional[Any] = None

    def process(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True,
        draw_bbox: bool = True,
        draw_ids: bool = False
    ) -> Tuple[np.ndarray, List[DetectedHand]]:
        """
        Detect hands, calculate 21 pixel-coordinate landmarks, and draw overlays.

        Args:
            frame: Input BGR frame from OpenCV.
            draw_landmarks: Render skeleton connections and landmark points.
            draw_bbox: Render bounding box and Handedness label.
            draw_ids: Render individual numeric landmark ID tags near key points.

        Returns:
            Tuple of (annotated_frame, list_of_detected_hands).
        """
        if frame is None or frame.size == 0:
            return frame, []

        h, w, _ = frame.shape
        detected_hands: List[DetectedHand] = []

        # MediaPipe requires RGB color format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.last_results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Handle case where no hands are detected
        if not self.last_results or not self.last_results.multi_hand_landmarks:
            cv2.putText(
                frame,
                "Status: No Hand Detected",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            return frame, detected_hands

        # Process each detected hand
        num_hands = len(self.last_results.multi_hand_landmarks)
        for idx in range(num_hands):
            hand_landmarks = self.last_results.multi_hand_landmarks[idx]

            # Determine Handedness ("Left" vs "Right")
            raw_label = "Unknown"
            confidence = 1.0
            if (
                self.last_results.multi_handedness
                and len(self.last_results.multi_handedness) > idx
            ):
                classification = self.last_results.multi_handedness[idx].classification[0]
                raw_label = classification.label
                confidence = float(classification.score)

            # Invert label if video feed is horizontally flipped (mirror view)
            if self.mirror_mode:
                hand_label = "Right" if raw_label == "Left" else "Left"
            else:
                hand_label = raw_label

            # Extract 21 landmarks in pixel coordinates
            lm_list: List[Tuple[int, int, int]] = []
            x_coords: List[int] = []
            y_coords: List[int] = []

            for lm_id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((lm_id, cx, cy))
                x_coords.append(cx)
                y_coords.append(cy)

                # Optional: draw numeric ID tags next to landmarks
                if draw_ids:
                    cv2.putText(
                        frame,
                        str(lm_id),
                        (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

            # Calculate bounding box
            x_min, x_max = max(0, min(x_coords) - 15), min(w, max(x_coords) + 15)
            y_min, y_max = max(0, min(y_coords) - 15), min(h, max(y_coords) + 15)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Draw MediaPipe landmark skeleton
            if draw_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )

            # Draw bounding box and handedness badge
            if draw_bbox:
                # Color code: Right Hand = Cyan (255, 200, 0), Left Hand = Magenta (255, 0, 255)
                box_color = (255, 200, 0) if hand_label == "Right" else (255, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

                label_text = f"{hand_label} ({int(confidence * 100)}%)"
                cv2.putText(
                    frame,
                    label_text,
                    (x_min, max(25, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                    cv2.LINE_AA
                )

                # Display Wrist (0) and Index Tip (8) coordinates directly under bounding box
                wrist = lm_list[0]
                index_tip = lm_list[8]
                coord_text = f"Wrist:({wrist[1]},{wrist[2]}) IdxTip:({index_tip[1]},{index_tip[2]})"
                cv2.putText(
                    frame,
                    coord_text,
                    (x_min, min(h - 5, y_max + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

            detected_hands.append(
                DetectedHand(
                    hand_index=idx,
                    label=hand_label,
                    confidence=confidence,
                    landmarks=lm_list,
                    bbox=bbox
                )
            )

        # Global status header
        hand_count_str = f"Hands Tracked: {len(detected_hands)} / {self.max_num_hands}"
        cv2.putText(
            frame,
            hand_count_str,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 120),
            2,
            cv2.LINE_AA
        )

        return frame, detected_hands

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
