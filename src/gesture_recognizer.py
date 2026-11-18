"""
Gesture Recognition Module.

Analyzes 21 MediaPipe hand landmarks using geometric distances, joint angles,
and finger-state classification. Features:
- Recognition of 7 core gestures:
  1. Open Palm
  2. Closed Fist
  3. Thumbs Up
  4. Thumbs Down
  5. Index Finger Up (Pointing)
  6. Victory Sign (Two fingers)
  7. Pinch Gesture (Thumb & Index touching)
- Scale-invariant distance normalization using palm reference metrics
- Robust stability/anti-flicker filtering via temporal consensus and hysteresis
- Fully decoupled from OpenCV capture and landmark detection
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np


class GestureType(Enum):
    """Enumeration of recognizable hand gestures."""
    NONE = "None"
    OPEN_PALM = "Open Palm"
    CLOSED_FIST = "Closed Fist"
    THUMBS_UP = "Thumbs Up"
    THUMBS_DOWN = "Thumbs Down"
    INDEX_UP = "Index Finger Up"
    VICTORY = "Victory Sign"
    PINCH = "Pinch"


@dataclass(frozen=True)
class FingerStates:
    """Extension state for all 5 fingers (True = extended, False = curled)."""
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_list(self) -> List[bool]:
        """Return states as list [thumb, index, middle, ring, pinky]."""
        return [self.thumb, self.index, self.middle, self.ring, self.pinky]

    def count_extended(self) -> int:
        """Count total number of extended fingers."""
        return sum(self.as_list())


@dataclass
class GestureResult:
    """Output result from gesture recognition analysis."""
    gesture: GestureType
    raw_gesture: GestureType
    confidence: float              # 0.0 to 1.0 stability confidence
    finger_states: FingerStates
    pinch_distance: float          # Normalized distance between thumb & index tips
    is_stable: bool                # True when consensus threshold is satisfied


class GestureRecognizer:
    """Classifies hand gestures from 21 landmarks with temporal stabilization."""

    def __init__(
        self,
        history_length: int = 7,
        stability_threshold: float = 0.70,
        pinch_ratio_threshold: float = 0.35
    ) -> None:
        """
        Initialize the GestureRecognizer.

        Args:
            history_length: Size of rolling window buffer for temporal stabilization.
            stability_threshold: Fraction of recent frames (e.g. 70%) required to confirm a gesture.
            pinch_ratio_threshold: Distance ratio between Thumb Tip and Index Tip
                                   relative to palm size to classify a pinch.
        """
        self.history_length = max(3, history_length)
        self.stability_threshold = stability_threshold
        self.pinch_ratio_threshold = pinch_ratio_threshold

        # Sliding window buffer of recent candidate gestures
        self._history: deque = deque(maxlen=self.history_length)
        self._current_stable_gesture: GestureType = GestureType.NONE

    @staticmethod
    def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        dx = float(p1[0] - p2[0])
        dy = float(p1[1] - p2[1])
        return float(np.hypot(dx, dy))

    def get_palm_scale(self, landmarks: List[Tuple[int, int, int]]) -> float:
        """
        Compute reference palm size (distance between Wrist 0 and Middle MCP 9).
        Used to normalize distances across varying hand-to-camera distances.
        """
        if len(landmarks) < 21:
            return 1.0
        wrist = (landmarks[0][1], landmarks[0][2])
        middle_mcp = (landmarks[9][1], landmarks[9][2])
        return max(1.0, self.calculate_distance(wrist, middle_mcp))

    def detect_finger_states(
        self,
        landmarks: List[Tuple[int, int, int]],
        handedness: str = "Right"
    ) -> FingerStates:
        """
        Evaluate extension state (extended vs. curled) for all 5 fingers.

        Args:
            landmarks: 21 landmarks [(id, x, y), ...].
            handedness: "Right" or "Left".

        Returns:
            FingerStates dataclass with booleans for each finger.
        """
        if len(landmarks) < 21:
            return FingerStates(False, False, False, False, False)

        wrist = (landmarks[0][1], landmarks[0][2])

        # 1. Four fingers (Index, Middle, Ring, Pinky):
        # A finger is extended when tip is further from the wrist than its PIP and MCP joints
        def is_finger_extended(tip_id: int, pip_id: int, mcp_id: int) -> bool:
            tip = (landmarks[tip_id][1], landmarks[tip_id][2])
            pip = (landmarks[pip_id][1], landmarks[pip_id][2])
            mcp = (landmarks[mcp_id][1], landmarks[mcp_id][2])

            dist_wrist_tip = self.calculate_distance(wrist, tip)
            dist_wrist_pip = self.calculate_distance(wrist, pip)
            dist_wrist_mcp = self.calculate_distance(wrist, mcp)

            # Tip must extend well beyond PIP joint
            return (dist_wrist_tip > dist_wrist_pip * 1.1) and (dist_wrist_tip > dist_wrist_mcp * 1.25)

        index_ext = is_finger_extended(8, 6, 5)
        middle_ext = is_finger_extended(12, 10, 9)
        ring_ext = is_finger_extended(16, 14, 13)
        pinky_ext = is_finger_extended(20, 18, 17)

        # 2. Thumb extension:
        # Distance between thumb tip (4) and pinky MCP (17) compared to thumb IP (3) to pinky MCP
        thumb_tip = (landmarks[4][1], landmarks[4][2])
        thumb_ip = (landmarks[3][1], landmarks[3][2])
        thumb_mcp = (landmarks[2][1], landmarks[2][2])
        pinky_mcp = (landmarks[17][1], landmarks[17][2])

        dist_thumb_pinky = self.calculate_distance(thumb_tip, pinky_mcp)
        dist_ip_pinky = self.calculate_distance(thumb_ip, pinky_mcp)
        dist_wrist_thumb = self.calculate_distance(wrist, thumb_tip)
        dist_wrist_thumb_mcp = self.calculate_distance(wrist, thumb_mcp)

        thumb_ext = (dist_thumb_pinky > dist_ip_pinky * 1.15) and (dist_wrist_thumb > dist_wrist_thumb_mcp * 1.15)

        return FingerStates(
            thumb=thumb_ext,
            index=index_ext,
            middle=middle_ext,
            ring=ring_ext,
            pinky=pinky_ext
        )

    def classify_raw_gesture(
        self,
        landmarks: List[Tuple[int, int, int]],
        handedness: str = "Right"
    ) -> Tuple[GestureType, float, FingerStates]:
        """
        Classify the single-frame instantaneous gesture without temporal smoothing.

        Returns:
            Tuple of (GestureType, normalized_pinch_ratio, FingerStates).
        """
        if len(landmarks) < 21:
            return GestureType.NONE, 1.0, FingerStates(False, False, False, False, False)

        finger_states = self.detect_finger_states(landmarks, handedness)
        palm_scale = self.get_palm_scale(landmarks)

        # Calculate normalized distance between Thumb Tip (4) and Index Tip (8)
        thumb_tip = (landmarks[4][1], landmarks[4][2])
        index_tip = (landmarks[8][1], landmarks[8][2])
        pinch_pixel_dist = self.calculate_distance(thumb_tip, index_tip)
        norm_pinch_ratio = pinch_pixel_dist / palm_scale

        # Geometry coordinates for orientation-specific gestures
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        thumb_tip_pt = landmarks[4]

        # 1. Check if all 4 main fingers (Index, Middle, Ring, Pinky) are folded:
        all_four_folded = (
            not finger_states.index
            and not finger_states.middle
            and not finger_states.ring
            and not finger_states.pinky
        )

        if all_four_folded:
            # Check thumb direction relative to wrist and thumb MCP
            thumb_y = thumb_tip_pt[2]
            thumb_mcp_y = thumb_mcp[2]
            wrist_y = wrist[2]

            # Thumbs Up: Thumb tip is significantly higher (lower y in image space)
            if thumb_y < thumb_mcp_y - (0.15 * palm_scale) and thumb_y < wrist_y:
                return GestureType.THUMBS_UP, norm_pinch_ratio, finger_states

            # Thumbs Down: Thumb tip is significantly lower (higher y in image space)
            if thumb_y > thumb_mcp_y + (0.15 * palm_scale) and thumb_y > wrist_y:
                return GestureType.THUMBS_DOWN, norm_pinch_ratio, finger_states

            # Closed Fist: All 4 fingers folded, thumb neutral or balled in
            return GestureType.CLOSED_FIST, norm_pinch_ratio, finger_states

        # 2. PINCH GESTURE:
        # Thumb tip and Index tip touching/close (when hand is not in a closed fist)
        if norm_pinch_ratio < self.pinch_ratio_threshold:
            return GestureType.PINCH, norm_pinch_ratio, finger_states

        # 3. OPEN PALM:
        # All 5 fingers extended
        if (
            finger_states.thumb
            and finger_states.index
            and finger_states.middle
            and finger_states.ring
            and finger_states.pinky
        ):
            return GestureType.OPEN_PALM, norm_pinch_ratio, finger_states

        # 4. INDEX FINGER UP:
        # Index extended; Middle, Ring, and Pinky folded
        if finger_states.index and not finger_states.middle and not finger_states.ring and not finger_states.pinky:
            return GestureType.INDEX_UP, norm_pinch_ratio, finger_states

        # 5. VICTORY SIGN:
        # Index and Middle extended; Ring and Pinky folded
        if finger_states.index and finger_states.middle and not finger_states.ring and not finger_states.pinky:
            return GestureType.VICTORY, norm_pinch_ratio, finger_states

        return GestureType.NONE, norm_pinch_ratio, finger_states

    def recognize(
        self,
        landmarks: List[Tuple[int, int, int]],
        handedness: str = "Right"
    ) -> GestureResult:
        """
        Analyze landmarks with sliding-window temporal stability filtering.

        Prevents rapid jitter and accidental flickers by requiring consensus
        across recent consecutive frames.

        Args:
            landmarks: 21 (id, x, y) pixel coordinates.
            handedness: "Right" or "Left".

        Returns:
            GestureResult with smoothed gesture, confidence, and finger metrics.
        """
        if not landmarks or len(landmarks) < 21:
            self._history.append(GestureType.NONE)
            self._current_stable_gesture = GestureType.NONE
            return GestureResult(
                gesture=GestureType.NONE,
                raw_gesture=GestureType.NONE,
                confidence=0.0,
                finger_states=FingerStates(False, False, False, False, False),
                pinch_distance=1.0,
                is_stable=True
            )

        # 1. Raw single-frame classification
        raw_gesture, pinch_ratio, finger_states = self.classify_raw_gesture(landmarks, handedness)
        self._history.append(raw_gesture)

        # 2. Calculate consensus frequency in rolling buffer
        counts: Dict[GestureType, int] = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1

        most_frequent_gesture = max(counts, key=counts.get)
        frequency = counts[most_frequent_gesture]
        confidence = frequency / len(self._history)

        # 3. Apply hysteresis / stability threshold
        is_stable = confidence >= self.stability_threshold
        if is_stable:
            self._current_stable_gesture = most_frequent_gesture

        return GestureResult(
            gesture=self._current_stable_gesture,
            raw_gesture=raw_gesture,
            confidence=confidence,
            finger_states=finger_states,
            pinch_distance=pinch_ratio,
            is_stable=is_stable
        )

    def reset(self) -> None:
        """Reset historical buffer."""
        self._history.clear()
        self._current_stable_gesture = GestureType.NONE
