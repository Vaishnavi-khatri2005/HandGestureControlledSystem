"""
Backwards-compatibility shim re-exporting HandDetector and DetectedHand
from src.hand_detection.
"""

from .hand_detection import (
    HandDetector,
    HandDetectorError,
    DetectedHand,
    LANDMARK_NAMES,
)

__all__ = [
    "HandDetector",
    "HandDetectorError",
    "DetectedHand",
    "LANDMARK_NAMES",
]
