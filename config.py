"""
Global configuration settings for Hand Gesture Controlled System.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    """Camera hardware and capture parameters."""
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    flip_horizontal: bool = True  # Mirror view for intuitive user experience


@dataclass(frozen=True)
class HandDetectorConfig:
    """MediaPipe Hands detector configuration parameters."""
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1


@dataclass(frozen=True)
class MouseConfig:
    """Virtual mouse movement and interaction thresholds."""
    smoothening: float = 5.0
    frame_reduction: int = 100  # Active boundary box reduction in pixels
    click_distance_threshold: float = 30.0  # Pixel distance threshold for pinch-clicks
    drag_distance_threshold: float = 25.0


@dataclass(frozen=True)
class VolumeConfig:
    """System audio volume control parameters."""
    min_distance: float = 20.0   # Pinch distance mapped to 0% volume
    max_distance: float = 200.0  # Pinch distance mapped to 100% volume
    smooth_steps: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Consolidated application configuration."""
    camera: CameraConfig = CameraConfig()
    hand_detector: HandDetectorConfig = HandDetectorConfig()
    mouse: MouseConfig = MouseConfig()
    volume: VolumeConfig = VolumeConfig()
    window_name: str = "Hand Gesture Controlled System"


# Default application configuration instance
CONFIG = AppConfig()
