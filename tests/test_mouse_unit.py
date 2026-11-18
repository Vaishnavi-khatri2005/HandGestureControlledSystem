"""
Unit tests for the Virtual Mouse Controller module (src.controllers.mouse_controller).

Tests cursor movement, clicking, and dragging separately:
1. Coordinate mapping & exponential smoothing (anti-jitter).
2. Adjustable mouse sensitivity.
3. Left click detection on short pinch.
4. Double click detection on two quick pinches.
5. Drag and drop operation (hold pinch, move, release).
6. Open Palm stops mouse movement.
"""

import time
import unittest
from unittest.mock import patch, MagicMock

from src.gesture_recognizer import GestureType
from src.controllers.mouse_controller import MouseController, MouseAction, MouseStatus


class TestMouseController(unittest.TestCase):
    """Test suite verifying cursor movement, clicking, and dragging separately."""

    def setUp(self):
        self.mouse = MouseController(
            frame_reduction=100,
            smoothening=3.0,
            sensitivity=1.0,
            pinch_click_threshold=0.30,
            drag_hold_delay=0.25,
            double_click_interval=0.35,
            click_cooldown=0.10
        )
        self.cam_w = 640
        self.cam_h = 480
        # Mock screen size to 1920x1080 for deterministic test assertions
        self.mouse.screen_width = 1920
        self.mouse.screen_height = 1080
        self.mouse.prev_x = 960.0
        self.mouse.prev_y = 540.0

    @patch("pyautogui.moveTo")
    def test_cursor_movement_and_smoothing(self, mock_move):
        """1. Verify coordinate interpolation and exponential smoothing."""
        # Index tip at center of frame: (320, 240)
        # Bounding box is (100, 100) to (540, 380). Center is exactly normalized (0.5, 0.5)
        # Should map to screen center (960, 540)
        sx, sy = self.mouse.map_coordinates(320, 240, self.cam_w, self.cam_h)
        self.assertAlmostEqual(sx, 960, delta=2)
        self.assertAlmostEqual(sy, 540, delta=2)

        # Move index tip to top-left of active zone (100, 100)
        # Target is (0, 0). With smoothing factor 3.0, it should smooth towards 0
        sx_smoothed, sy_smoothed = self.mouse.map_coordinates(100, 100, self.cam_w, self.cam_h)
        self.assertLess(sx_smoothed, 960)
        self.assertGreater(sx_smoothed, 0)

    def test_adjustable_sensitivity(self):
        """2. Verify mouse sensitivity scaling."""
        self.mouse.sensitivity = 1.0
        self.mouse.prev_x = 960.0
        self.mouse.prev_y = 540.0
        sx1, _ = self.mouse.map_coordinates(400, 240, self.cam_w, self.cam_h)

        # Higher sensitivity (2.0) should produce a larger displacement from center
        self.mouse.sensitivity = 2.0
        self.mouse.prev_x = 960.0
        self.mouse.prev_y = 540.0
        sx2, _ = self.mouse.map_coordinates(400, 240, self.cam_w, self.cam_h)

        self.assertGreater(sx2, sx1)

    @patch("pyautogui.click")
    def test_single_left_click(self, mock_click):
        """3. Verify Left Click triggered by a short pinch-and-release."""
        # 21 mock landmarks
        lms = [(i, 300, 200) for i in range(21)]
        t_base = time.time()

        with patch("time.time") as mock_time:
            # A. Start pinch (ratio 0.15 < threshold 0.30)
            mock_time.return_value = t_base
            s1 = self.mouse.process(GestureType.PINCH, lms, self.cam_w, self.cam_h, pinch_ratio=0.15)
            self.assertFalse(self.mouse._is_dragging)

            # B. Release pinch quickly after 0.10s (< 0.25s drag threshold)
            mock_time.return_value = t_base + 0.10
            s2 = self.mouse.process(GestureType.INDEX_UP, lms, self.cam_w, self.cam_h, pinch_ratio=0.80)

            self.assertEqual(s2.action, MouseAction.CLICK)
            mock_click.assert_called_once()

    @patch("pyautogui.click")
    @patch("pyautogui.doubleClick")
    def test_double_click(self, mock_double_click, mock_click):
        """4. Verify Double Click triggered by two quick pinches."""
        lms = [(i, 300, 200) for i in range(21)]
        t_base = time.time()

        with patch("time.time") as mock_time:
            # 1st Pinch & Release (Click 1)
            mock_time.return_value = t_base
            self.mouse.process(GestureType.PINCH, lms, self.cam_w, self.cam_h, pinch_ratio=0.15)
            mock_time.return_value = t_base + 0.08
            self.mouse.process(GestureType.INDEX_UP, lms, self.cam_w, self.cam_h, pinch_ratio=0.80)
            mock_click.assert_called_once()

            # 2nd Pinch & Release within 0.20s (< double_click_interval 0.35s)
            mock_time.return_value = t_base + 0.20
            self.mouse.process(GestureType.PINCH, lms, self.cam_w, self.cam_h, pinch_ratio=0.15)
            mock_time.return_value = t_base + 0.28
            s_double = self.mouse.process(GestureType.INDEX_UP, lms, self.cam_w, self.cam_h, pinch_ratio=0.80)

            self.assertEqual(s_double.action, MouseAction.DOUBLE_CLICK)
            mock_double_click.assert_called_once()

    @patch("pyautogui.mouseDown")
    @patch("pyautogui.mouseUp")
    @patch("pyautogui.moveTo")
    def test_drag_and_drop(self, mock_move, mock_mouse_up, mock_mouse_down):
        """5. Verify Drag and Drop: Hold pinch > 0.25s activates drag, moves, and releases on open."""
        lms = [(i, 300, 200) for i in range(21)]
        t_base = time.time()

        with patch("time.time") as mock_time:
            # Start pinch
            mock_time.return_value = t_base
            self.mouse.process(GestureType.PINCH, lms, self.cam_w, self.cam_h, pinch_ratio=0.15)
            self.assertFalse(self.mouse._is_dragging)

            # Hold pinch past drag threshold (> 0.25s)
            mock_time.return_value = t_base + 0.30
            status_drag = self.mouse.process(GestureType.PINCH, lms, self.cam_w, self.cam_h, pinch_ratio=0.15)

            self.assertEqual(status_drag.action, MouseAction.DRAGGING)
            self.assertTrue(status_drag.is_dragging)
            mock_mouse_down.assert_called_once()

            # Release pinch -> mouseUp
            mock_time.return_value = t_base + 0.50
            status_release = self.mouse.process(GestureType.INDEX_UP, lms, self.cam_w, self.cam_h, pinch_ratio=0.80)
            self.assertFalse(self.mouse._is_dragging)
            mock_mouse_up.assert_called_once()

    def test_open_palm_stops_movement(self):
        """6. Verify Open Palm stops mouse movement."""
        lms = [(i, 300, 200) for i in range(21)]
        status = self.mouse.process(GestureType.OPEN_PALM, lms, self.cam_w, self.cam_h, pinch_ratio=0.90)
        self.assertEqual(status.action, MouseAction.STOPPED)


if __name__ == "__main__":
    unittest.main()
