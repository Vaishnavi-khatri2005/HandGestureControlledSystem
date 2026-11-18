"""
FPS tracker and display overlay helper.
"""

import time
from typing import Tuple
import cv2
import numpy as np


class FPSTracker:
    """Calculates and renders real-time frames-per-second (FPS) metrics."""

    def __init__(self, smoothing: float = 0.9) -> None:
        """
        Args:
            smoothing: Weight for exponential moving average of FPS.
        """
        self.prev_time: float = time.time()
        self.fps: float = 0.0
        self.smoothing = smoothing

    def update(self) -> float:
        """
        Calculate FPS based on delta time between frames.

        Returns:
            Smoothed FPS value.
        """
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta > 0.0:
            instant_fps = 1.0 / delta
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = (self.smoothing * self.fps) + ((1.0 - self.smoothing) * instant_fps)

        return self.fps

    def draw_fps(
        self,
        frame: np.ndarray,
        origin: Tuple[int, int] = (20, 40),
        color: Tuple[int, int, int] = (0, 255, 0),
        scale: float = 0.8,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw FPS text onto the given frame.

        Args:
            frame: OpenCV image frame.
            origin: (x, y) coordinates for text placement.
            color: Text BGR color.
            scale: Font scale.
            thickness: Line thickness.

        Returns:
            Modified frame.
        """
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(
            frame,
            fps_text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        return frame
