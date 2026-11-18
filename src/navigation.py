"""
Navigation Control Module.

Provides gesture-based system and presentation navigation using PyAutoGUI:
- Swipe Right: Forward / Next Slide / Page
- Swipe Left: Backward / Previous Slide / Page
- Index Finger Up: Continuous smooth scroll up
- Thumbs Down: Continuous smooth scroll down
- Victory Sign: Toggle navigation mode (Presentation vs. Browser)

Features:
- Multi-frame trajectory tracking for accurate velocity and direction calculation
- Accidental trigger filtering (displacement threshold + horizontal dominance)
- Cooldown protection and state latching
- Coexistence with volume control by isolating gestures and dynamic movement
- Dedicated on-screen navigation HUD
"""

import time
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import cv2
import numpy as np
import pyautogui

from src.gesture_recognizer import GestureType

# PyAutoGUI safety configuration
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


class NavigationControlError(Exception):
    """Raised when navigation actions fail."""
    pass


class NavigationMode(Enum):
    """Operating modes for navigation controls."""
    PRESENTATION = "Presentation (Slides)"
    BROWSER = "Browser (Web / Tabs)"


class NavigationAction(Enum):
    """Discrete navigation actions triggered."""
    NONE = "Idle"
    SWIPE_RIGHT = "Next Page / Forward ->"
    SWIPE_LEFT = "<- Previous Page / Back"
    SCROLL_UP = "Scrolling Up ^"
    SCROLL_DOWN = "Scrolling Down v"
    MODE_SWITCHED = "Mode Switched"


class SwipeDirection(Enum):
    """Direction of horizontal motion."""
    NONE = "None"
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class NavigationState:
    """Current navigation system status."""
    mode: NavigationMode
    action: NavigationAction
    speed: float                  # Computed velocity in pixels/second
    displacement: float           # Horizontal displacement in pixels
    status_message: str
    is_swipe_active: bool         # True if dynamic high-speed motion is detected


class SwipeDetector:
    """Detects horizontal swipe motions across consecutive video frames."""

    def __init__(
        self,
        min_displacement: float = 90.0,
        min_velocity: float = 300.0,
        history_window_seconds: float = 0.35,
        swipe_cooldown: float = 0.80
    ) -> None:
        """
        Args:
            min_displacement: Minimum horizontal travel in pixels.
            min_velocity: Minimum required speed in pixels per second.
            history_window_seconds: Time window for tracking trajectory.
            swipe_cooldown: Minimum seconds between swipe detections.
        """
        self.min_displacement = min_displacement
        self.min_velocity = min_velocity
        self.history_window_seconds = history_window_seconds
        self.swipe_cooldown = swipe_cooldown

        # Stores (x, y, timestamp)
        self._trajectory: deque = deque(maxlen=20)
        self._last_swipe_time: float = 0.0

    def add_point(self, x: int, y: int) -> None:
        """Record current hand coordinate with high-precision timestamp."""
        now = time.time()
        self._trajectory.append((x, y, now))

        # Evict points older than history window
        while self._trajectory and (now - self._trajectory[0][2] > self.history_window_seconds):
            self._trajectory.popleft()

    def detect_swipe(self) -> Tuple[SwipeDirection, float, float]:
        """
        Analyze recent trajectory for horizontal swipe gestures.

        Returns:
            Tuple[SwipeDirection, velocity_px_per_sec, displacement_px]
        """
        now = time.time()
        if now - self._last_swipe_time < self.swipe_cooldown:
            return SwipeDirection.NONE, 0.0, 0.0

        if len(self._trajectory) < 4:
            return SwipeDirection.NONE, 0.0, 0.0

        oldest = self._trajectory[0]
        newest = self._trajectory[-1]

        dx = float(newest[0] - oldest[0])
        dy = float(newest[1] - oldest[1])
        dt = float(newest[2] - oldest[2])

        if dt <= 0.0:
            return SwipeDirection.NONE, 0.0, 0.0

        velocity_x = dx / dt
        speed_x = abs(velocity_x)

        # Accidental motion rejection filters:
        # 1. Minimum displacement requirement
        if abs(dx) < self.min_displacement:
            return SwipeDirection.NONE, speed_x, dx

        # 2. Minimum velocity requirement
        if speed_x < self.min_velocity:
            return SwipeDirection.NONE, speed_x, dx

        # 3. Horizontal dominance: |dx| must exceed 1.7x vertical movement |dy|
        if abs(dx) < 1.7 * abs(dy):
            return SwipeDirection.NONE, speed_x, dx

        # Valid swipe detected
        self._last_swipe_time = now
        self._trajectory.clear()

        direction = SwipeDirection.RIGHT if dx > 0 else SwipeDirection.LEFT
        return direction, speed_x, dx

    def reset(self) -> None:
        """Clear trajectory buffer."""
        self._trajectory.clear()


class NavigationController:
    """Manages presentation and desktop navigation actions via gestures."""

    def __init__(
        self,
        default_mode: NavigationMode = NavigationMode.PRESENTATION,
        scroll_speed: int = 40,
        scroll_throttle_interval: float = 0.07,
        mode_switch_cooldown: float = 1.3
    ) -> None:
        """
        Args:
            default_mode: Starting navigation mode.
            scroll_speed: PyAutoGUI scroll units per tick.
            scroll_throttle_interval: Delay between scroll ticks to prevent flooding.
            mode_switch_cooldown: Cooldown in seconds between mode toggles.
        """
        self.mode = default_mode
        self.scroll_speed = scroll_speed
        self.scroll_throttle_interval = scroll_throttle_interval
        self.mode_switch_cooldown = mode_switch_cooldown

        self.swipe_detector = SwipeDetector()
        self._last_scroll_time: float = 0.0
        self._last_mode_switch_time: float = 0.0
        self._victory_latched: bool = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def switch_mode(self) -> NavigationMode:
        """Toggle between Presentation and Browser navigation modes."""
        now = time.time()
        if now - self._last_mode_switch_time >= self.mode_switch_cooldown:
            if self.mode == NavigationMode.PRESENTATION:
                self.mode = NavigationMode.BROWSER
            else:
                self.mode = NavigationMode.PRESENTATION
            self._last_mode_switch_time = now
            self._logger.info("Navigation mode switched to: %s", self.mode.value)
        return self.mode

    def execute_next(self) -> None:
        """Perform forward navigation (Right arrow or Browser Forward)."""
        try:
            if self.mode == NavigationMode.PRESENTATION:
                pyautogui.press("right")
            else:
                pyautogui.hotkey("alt", "right")
            self._logger.info("Executed: Forward navigation (%s)", self.mode.name)
        except Exception as exc:
            self._logger.error("Error executing forward navigation: %s", exc)

    def execute_previous(self) -> None:
        """Perform backward navigation (Left arrow or Browser Back)."""
        try:
            if self.mode == NavigationMode.PRESENTATION:
                pyautogui.press("left")
            else:
                pyautogui.hotkey("alt", "left")
            self._logger.info("Executed: Backward navigation (%s)", self.mode.name)
        except Exception as exc:
            self._logger.error("Error executing backward navigation: %s", exc)

    def execute_scroll(self, direction: int) -> None:
        """
        Perform smooth scrolling.

        Args:
            direction: +1 for Up, -1 for Down.
        """
        now = time.time()
        if now - self._last_scroll_time >= self.scroll_throttle_interval:
            try:
                pyautogui.scroll(int(direction * self.scroll_speed))
                self._last_scroll_time = now
            except Exception as exc:
                self._logger.error("Error executing scroll: %s", exc)

    def process(
        self,
        gesture: GestureType,
        landmarks: List[Tuple[int, int, int]]
    ) -> NavigationState:
        """
        Analyze current frame landmarks and gesture to trigger navigation actions.

        Mapping:
        - Swipe Right -> Next page
        - Swipe Left -> Previous page
        - Index Finger Up -> Scroll Up
        - Thumbs Down -> Scroll Down
        - Victory Sign -> Switch Navigation Mode

        Args:
            gesture: The detected stable GestureType.
            landmarks: 21 (id, x, y) pixel coordinates.

        Returns:
            NavigationState snapshot.
        """
        action = NavigationAction.NONE
        status_msg = "Ready"
        speed = 0.0
        disp = 0.0
        is_swipe = False

        if landmarks and len(landmarks) >= 21:
            # Track palm center (Middle Finger MCP Landmark 9)
            palm_x = landmarks[9][1]
            palm_y = landmarks[9][2]
            self.swipe_detector.add_point(palm_x, palm_y)

            # Check for dynamic swipe motion
            direction, speed, disp = self.swipe_detector.detect_swipe()

            if direction == SwipeDirection.RIGHT:
                self.execute_next()
                action = NavigationAction.SWIPE_RIGHT
                status_msg = "Next Page / Forward (Swipe Right)"
                is_swipe = True
            elif direction == SwipeDirection.LEFT:
                self.execute_previous()
                action = NavigationAction.SWIPE_LEFT
                status_msg = "Previous Page / Back (Swipe Left)"
                is_swipe = True
        else:
            self.swipe_detector.reset()

        # Handle static navigation gestures when not in the middle of a swipe
        if not is_swipe:
            # 1. Victory Sign: Toggle Navigation Mode (with latching)
            if gesture == GestureType.VICTORY:
                if not self._victory_latched:
                    now = time.time()
                    if now - self._last_mode_switch_time >= self.mode_switch_cooldown:
                        self.switch_mode()
                        action = NavigationAction.MODE_SWITCHED
                        status_msg = f"Mode -> {self.mode.value}"
                    self._victory_latched = True
                else:
                    status_msg = f"Mode Locked ({self.mode.value})"
            else:
                self._victory_latched = False

            # 2. Index Finger Up: Scroll Up
            if gesture == GestureType.INDEX_UP:
                self.execute_scroll(+1)
                action = NavigationAction.SCROLL_UP
                status_msg = "Scrolling Up"

            # 3. Thumbs Down: Scroll Down
            elif gesture == GestureType.THUMBS_DOWN:
                self.execute_scroll(-1)
                action = NavigationAction.SCROLL_DOWN
                status_msg = "Scrolling Down"

        return NavigationState(
            mode=self.mode,
            action=action,
            speed=speed,
            displacement=disp,
            status_message=status_msg,
            is_swipe_active=is_swipe
        )

    @staticmethod
    def draw_navigation_hud(
        frame: np.ndarray,
        state: NavigationState,
        origin: Tuple[int, int] = (20, 110)
    ) -> np.ndarray:
        """
        Draw an on-screen HUD panel displaying active mode and navigation feedback.

        Args:
            frame: Video frame.
            state: NavigationState.
            origin: Top-left (x, y) coordinates of HUD panel.

        Returns:
            Annotated frame.
        """
        x, y = origin
        panel_w, panel_h = 320, 95

        # Background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (80, 80, 80), 1)

        # Mode Indicator
        mode_text = f"MODE: {state.mode.value}"
        cv2.putText(
            frame,
            mode_text,
            (x + 12, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Action Display with Color Coding
        action_color = (0, 255, 0) if state.action != NavigationAction.NONE else (180, 180, 180)
        action_text = f"Action: {state.action.value}"
        cv2.putText(
            frame,
            action_text,
            (x + 12, y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            action_color,
            2,
            cv2.LINE_AA
        )

        # Description / Velocity Info
        if state.is_swipe_active:
            sub_text = f"Swipe Speed: {int(state.speed)} px/s"
        else:
            sub_text = "[Victory] Switch Mode | [Index/Thumbs] Scroll"

        cv2.putText(
            frame,
            sub_text,
            (x + 12, y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        return frame
