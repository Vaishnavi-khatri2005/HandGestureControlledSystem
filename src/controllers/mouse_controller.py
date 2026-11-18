"""
Virtual Mouse Controller Module.

Translates hand gestures and landmark positions into desktop mouse events using PyAutoGUI:
- Index Finger Up: Move mouse cursor smoothly across screen
- Single Pinch (< 0.35s): Single Left Click
- Two Quick Pinches (< 0.40s apart): Double Click
- Hold Pinch (> 0.35s) and Move: Click, Drag, and Drop
- Open Palm: Stop mouse movement / freeze cursor

Features:
- Frame reduction boundary box mapping camera space to screen space
- Exponential smoothing filter eliminating tremor and cursor jitter
- Configurable sensitivity
- Timing-based state machine for robust click, double-click, and drag separation
- Dedicated on-screen visual overlay
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Optional
import cv2
import numpy as np
import pyautogui

from src.gesture_recognizer import GestureType

# PyAutoGUI safety configuration
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


class MouseAction(Enum):
    """Actions performed by the virtual mouse."""
    IDLE = "Idle"
    MOVING = "Moving Cursor"
    CLICK = "Left Click"
    DOUBLE_CLICK = "Double Click"
    DRAGGING = "Drag & Drop"
    STOPPED = "Movement Stopped (Open Palm)"


@dataclass
class MouseStatus:
    """Snapshot of virtual mouse state for HUD display."""
    action: MouseAction
    screen_x: int
    screen_y: int
    pinch_distance: float
    is_dragging: bool
    status_message: str


class MouseControlError(Exception):
    """Raised when mouse control operations fail."""
    pass


class MouseController:
    """Translates hand gestures into desktop mouse actions."""

    def __init__(
        self,
        frame_reduction: int = 90,
        smoothening: float = 4.5,
        sensitivity: float = 1.0,
        pinch_click_threshold: float = 0.30,
        drag_hold_delay: float = 0.35,
        double_click_interval: float = 0.40,
        click_cooldown: float = 0.15
    ) -> None:
        """
        Initialize the virtual mouse controller.

        Args:
            frame_reduction: Margin inside camera frame defining active movement region.
            smoothening: Smoothing divisor (higher = smoother, slower tracking).
            sensitivity: Speed multiplier for cursor movement.
            pinch_click_threshold: Normalized thumb-index distance to register pinch.
            drag_hold_delay: Seconds of sustained pinch before transitioning to Drag mode.
            double_click_interval: Max seconds between two pinches to trigger double-click.
            click_cooldown: Minimum seconds between consecutive click events.
        """
        self.frame_reduction = frame_reduction
        self.smoothening = max(1.0, smoothening)
        self.sensitivity = max(0.2, min(3.0, sensitivity))
        self.pinch_click_threshold = pinch_click_threshold
        self.drag_hold_delay = drag_hold_delay
        self.double_click_interval = double_click_interval
        self.click_cooldown = click_cooldown

        self._logger = logging.getLogger(self.__class__.__name__)

        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception as exc:
            self.screen_width, self.screen_height = 1920, 1080
            self._logger.warning("Could not query screen size: %s. Using 1920x1080 default.", exc)

        # Coordinate state
        self.prev_x: float = float(self.screen_width / 2)
        self.prev_y: float = float(self.screen_height / 2)
        self.current_screen_x: int = int(self.prev_x)
        self.current_screen_y: int = int(self.prev_y)
        self._last_sent_x: int = -999
        self._last_sent_y: int = -999

        # Pinch & Click State Machine
        self._is_pinching: bool = False
        self._pinch_start_time: float = 0.0
        self._last_click_time: float = 0.0
        self._is_dragging: bool = False
        self._pending_click: bool = False

    def map_coordinates(
        self,
        cam_x: int,
        cam_y: int,
        cam_w: int,
        cam_h: int
    ) -> Tuple[int, int]:
        """
        Map camera pixel coordinates to smoothed desktop screen coordinates.

        Args:
            cam_x: X coordinate on camera frame.
            cam_y: Y coordinate on camera frame.
            cam_w: Frame width.
            cam_h: Frame height.

        Returns:
            Tuple of final (screen_x, screen_y).
        """
        margin = self.frame_reduction
        effective_w = max(1, cam_w - 2 * margin)
        effective_h = max(1, cam_h - 2 * margin)

        # Normalized position inside active bounding box [0.0, 1.0]
        norm_x = (cam_x - margin) / effective_w
        norm_y = (cam_y - margin) / effective_h

        # Center and apply sensitivity scaling
        center_x = (norm_x - 0.5) * self.sensitivity + 0.5
        center_y = (norm_y - 0.5) * self.sensitivity + 0.5

        # Map to screen pixels
        target_x = center_x * self.screen_width
        target_y = center_y * self.screen_height

        # Exponential moving average smoothing: current = prev + (target - prev) / smoothening
        curr_x = self.prev_x + (target_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (target_y - self.prev_y) / self.smoothening

        # Clamp within screen bounds
        final_x = int(np.clip(curr_x, 0, self.screen_width - 1))
        final_y = int(np.clip(curr_y, 0, self.screen_height - 1))

        self.prev_x, self.prev_y = curr_x, curr_y
        self.current_screen_x, self.current_screen_y = final_x, final_y

        return final_x, final_y

    def move_to(self, screen_x: int, screen_y: int, deadband: int = 2) -> None:
        """
        Issue cursor movement to operating system if displacement exceeds deadband.
        Prevents redundant micro-spams and reduces CPU overhead.
        """
        if (
            abs(screen_x - self._last_sent_x) >= deadband
            or abs(screen_y - self._last_sent_y) >= deadband
        ):
            try:
                pyautogui.moveTo(screen_x, screen_y)
                self._last_sent_x = screen_x
                self._last_sent_y = screen_y
            except Exception as exc:
                self._logger.error("Error moving mouse cursor: %s", exc)

    def trigger_left_click(self) -> None:
        """Execute single left click."""
        try:
            pyautogui.click()
            self._logger.info("Executed: Left Click")
        except Exception as exc:
            self._logger.error("Left click failed: %s", exc)

    def trigger_double_click(self) -> None:
        """Execute double click."""
        try:
            pyautogui.doubleClick()
            self._logger.info("Executed: Double Click")
        except Exception as exc:
            self._logger.error("Double click failed: %s", exc)

    def start_drag(self) -> None:
        """Initiate mouse down for dragging."""
        if not self._is_dragging:
            try:
                pyautogui.mouseDown()
                self._is_dragging = True
                self._logger.info("Drag Started (Mouse Down)")
            except Exception as exc:
                self._logger.error("Start drag failed: %s", exc)

    def stop_drag(self) -> None:
        """Release mouse button completing drag."""
        if self._is_dragging:
            try:
                pyautogui.mouseUp()
                self._is_dragging = False
                self._logger.info("Drag Released (Mouse Up)")
            except Exception as exc:
                self._logger.error("Stop drag failed: %s", exc)

    def process(
        self,
        gesture: GestureType,
        landmarks: List[Tuple[int, int, int]],
        cam_w: int,
        cam_h: int,
        pinch_ratio: float = 1.0
    ) -> MouseStatus:
        """
        Process hand landmarks and gesture into mouse actions.

        Mapping:
        - Index Finger Up: Move mouse cursor.
        - Pinch (Thumb + Index): Left click (short pinch).
        - Two Quick Pinches: Double click.
        - Hold Pinch (> 0.35s) and Move: Drag and drop.
        - Open Palm: Stop mouse movement.

        Args:
            gesture: The detected stable GestureType.
            landmarks: 21 (id, x, y) pixel coordinates.
            cam_w: Camera frame width.
            cam_h: Camera frame height.
            pinch_ratio: Normalized pinch ratio from GestureRecognizer.

        Returns:
            MouseStatus snapshot for UI display.
        """
        action = MouseAction.IDLE
        status_msg = "Idle"
        now = time.time()

        if not landmarks or len(landmarks) < 21:
            if self._is_dragging:
                self.stop_drag()
            return MouseStatus(
                action=MouseAction.IDLE,
                screen_x=self.current_screen_x,
                screen_y=self.current_screen_y,
                pinch_distance=pinch_ratio,
                is_dragging=False,
                status_message="No Hand"
            )

        # Index tip (8) and Thumb tip (4)
        index_x, index_y = landmarks[8][1], landmarks[8][2]
        thumb_x, thumb_y = landmarks[4][1], landmarks[4][2]

        is_pinching_now = (pinch_ratio < self.pinch_click_threshold)

        # 1. OPEN PALM -> Stop Mouse Movement
        if gesture == GestureType.OPEN_PALM:
            if self._is_dragging:
                self.stop_drag()
            self._is_pinching = False
            return MouseStatus(
                action=MouseAction.STOPPED,
                screen_x=self.current_screen_x,
                screen_y=self.current_screen_y,
                pinch_distance=pinch_ratio,
                is_dragging=False,
                status_message="Stopped (Open Palm)"
            )

        # 2. PINCH HANDLING (Click, Double Click, Drag)
        if is_pinching_now:
            if not self._is_pinching:
                # Transition: Pinch just started
                self._is_pinching = True
                self._pinch_start_time = now
                action = MouseAction.IDLE
                status_msg = "Pinch Started"
            else:
                # Pinch is being held
                hold_duration = now - self._pinch_start_time
                if hold_duration >= self.drag_hold_delay:
                    # Hold exceeds threshold -> DRAG MODE
                    if not self._is_dragging:
                        self.start_drag()

                    # Move cursor along with the pinched position (midpoint between thumb & index)
                    mid_x = int((index_x + thumb_x) / 2)
                    mid_y = int((index_y + thumb_y) / 2)
                    sx, sy = self.map_coordinates(mid_x, mid_y, cam_w, cam_h)
                    self.move_to(sx, sy)

                    action = MouseAction.DRAGGING
                    status_msg = f"Dragging ({sx}, {sy})"
                else:
                    action = MouseAction.IDLE
                    status_msg = f"Pinching ({int(hold_duration * 1000)}ms)"

        else:
            # Pinch was released or is not active
            if self._is_pinching:
                # Transition: Pinch just released!
                hold_duration = now - self._pinch_start_time
                self._is_pinching = False

                if self._is_dragging:
                    # Released after drag
                    self.stop_drag()
                    action = MouseAction.DRAGGING
                    status_msg = "Drag Released"
                elif hold_duration < self.drag_hold_delay:
                    # Short pinch release -> CLICK or DOUBLE CLICK
                    time_since_last_click = now - self._last_click_time

                    if (
                        self._pending_click
                        and time_since_last_click < self.double_click_interval
                    ):
                        # Two quick pinches detected -> DOUBLE CLICK
                        self.trigger_double_click()
                        action = MouseAction.DOUBLE_CLICK
                        status_msg = "Double Click Executed"
                        self._pending_click = False
                        self._last_click_time = 0.0
                    else:
                        # Single click
                        if now - self._last_click_time >= self.click_cooldown:
                            self.trigger_left_click()
                            action = MouseAction.CLICK
                            status_msg = "Left Click Executed"
                            self._pending_click = True
                            self._last_click_time = now

            # 3. INDEX FINGER UP -> Move Cursor
            elif gesture == GestureType.INDEX_UP:
                sx, sy = self.map_coordinates(index_x, index_y, cam_w, cam_h)
                self.move_to(sx, sy)
                action = MouseAction.MOVING
                status_msg = f"Moving -> ({sx}, {sy})"

        return MouseStatus(
            action=action,
            screen_x=self.current_screen_x,
            screen_y=self.current_screen_y,
            pinch_distance=pinch_ratio,
            is_dragging=self._is_dragging,
            status_message=status_msg
        )

    def draw_mouse_hud(
        self,
        frame: np.ndarray,
        status: MouseStatus,
        landmarks: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draw active mouse bounding box, fingertip pointer, and diagnostic HUD on frame.

        Args:
            frame: OpenCV video frame.
            status: MouseStatus from process().
            landmarks: 21 landmarks list to highlight pointer fingertips.

        Returns:
            Annotated frame.
        """
        h, w, _ = frame.shape
        margin = self.frame_reduction

        # 1. Active Control Zone Boundary Box (Mapped to entire screen)
        box_color = (0, 255, 255) if not status.is_dragging else (0, 0, 255)
        cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), box_color, 2)
        cv2.putText(
            frame,
            "Active Mouse Zone",
            (margin + 5, margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA
        )

        # 2. Fingertip pointer circles
        if landmarks and len(landmarks) >= 21:
            idx_x, idx_y = landmarks[8][1], landmarks[8][2]
            thb_x, thb_y = landmarks[4][1], landmarks[4][2]

            # Index pointer dot
            cv2.circle(frame, (idx_x, idx_y), 9, (0, 255, 0), -1)

            # If pinching/dragging, draw connection line
            if status.is_dragging:
                cv2.line(frame, (thb_x, thb_y), (idx_x, idx_y), (0, 0, 255), 3)
                cv2.circle(frame, (thb_x, thb_y), 8, (0, 0, 255), -1)
            elif status.pinch_distance < self.pinch_click_threshold:
                cv2.line(frame, (thb_x, thb_y), (idx_x, idx_y), (0, 255, 255), 2)
                cv2.circle(frame, (thb_x, thb_y), 7, (0, 255, 255), -1)

        # 3. Mouse HUD Panel on bottom-right
        card_w, card_h = 310, 110
        card_x = w - card_w - 20
        card_y = h - card_h - 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

        # Header Title
        cv2.putText(
            frame,
            "VIRTUAL MOUSE",
            (card_x + 12, card_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # State / Action
        if status.action == MouseAction.CLICK:
            act_col = (0, 255, 0)
        elif status.action == MouseAction.DOUBLE_CLICK:
            act_col = (255, 0, 255)
        elif status.action == MouseAction.DRAGGING:
            act_col = (0, 0, 255)
        elif status.action == MouseAction.STOPPED:
            act_col = (160, 160, 160)
        else:
            act_col = (255, 200, 0)

        cv2.putText(
            frame,
            f"State: {status.action.value}",
            (card_x + 12, card_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            act_col,
            2,
            cv2.LINE_AA
        )

        # Coordinates & Details
        coord_str = f"Screen: ({status.screen_x}, {status.screen_y}) | Sens: {self.sensitivity:.1f}x"
        cv2.putText(
            frame,
            coord_str,
            (card_x + 12, card_y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            status.status_message,
            (card_x + 12, card_y + 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        return frame
