"""
Unit tests for the Navigation module (src.navigation).

Tests all navigation operations independently:
1. Swipe Right detection with velocity and displacement.
2. Swipe Left detection with velocity and displacement.
3. Accidental movement rejection (slow, short, vertical).
4. Cooldown enforcement on consecutive swipes.
5. Mode switching via Victory Sign with state latching.
6. Scroll Up via Index Finger Up.
7. Scroll Down via Thumbs Down.
8. Non-interference verification between Navigation and Volume Control.
"""

import time
import unittest
from unittest.mock import patch

from src.gesture_recognizer import GestureType
from src.navigation import (
    NavigationController,
    NavigationMode,
    NavigationAction,
    SwipeDetector,
    SwipeDirection,
)
from src.controllers.volume_controller import VolumeController, VolumeAction


class TestNavigationModule(unittest.TestCase):
    """Test suite for NavigationController and SwipeDetector."""

    def setUp(self):
        self.nav = NavigationController(
            default_mode=NavigationMode.PRESENTATION,
            mode_switch_cooldown=0.5
        )

    def test_swipe_right_detection(self):
        """1. Verify Swipe Right detection with valid displacement and speed."""
        detector = SwipeDetector(min_displacement=80.0, min_velocity=250.0, swipe_cooldown=0.5)

        t_base = time.time()
        with patch("time.time") as mock_time:
            mock_time.return_value = t_base
            detector.add_point(200, 200)

            mock_time.return_value = t_base + 0.05
            detector.add_point(240, 202)

            mock_time.return_value = t_base + 0.10
            detector.add_point(280, 201)

            mock_time.return_value = t_base + 0.15
            detector.add_point(330, 203)

            direction, speed, dx = detector.detect_swipe()

        self.assertEqual(direction, SwipeDirection.RIGHT)
        self.assertGreater(speed, 250.0)
        self.assertGreater(dx, 80.0)

    def test_swipe_left_detection(self):
        """2. Verify Swipe Left detection with valid displacement and speed."""
        detector = SwipeDetector(min_displacement=80.0, min_velocity=250.0, swipe_cooldown=0.5)

        t_base = time.time()
        with patch("time.time") as mock_time:
            mock_time.return_value = t_base
            detector.add_point(400, 200)

            mock_time.return_value = t_base + 0.05
            detector.add_point(350, 202)

            mock_time.return_value = t_base + 0.10
            detector.add_point(300, 201)

            mock_time.return_value = t_base + 0.15
            detector.add_point(260, 203)

            direction, speed, dx = detector.detect_swipe()

        self.assertEqual(direction, SwipeDirection.LEFT)
        self.assertGreater(speed, 250.0)
        self.assertLess(dx, -80.0)

    def test_accidental_movement_rejected(self):
        """3. Verify slow or vertical movements are rejected as swipes."""
        detector = SwipeDetector(min_displacement=80.0, min_velocity=300.0)

        t_base = time.time()
        with patch("time.time") as mock_time:
            # A. Very slow movement (under velocity threshold)
            mock_time.return_value = t_base
            detector.add_point(200, 200)
            mock_time.return_value = t_base + 0.10
            detector.add_point(210, 200)
            mock_time.return_value = t_base + 0.20
            detector.add_point(220, 200)
            mock_time.return_value = t_base + 0.50
            detector.add_point(230, 200)
            dir_slow, _, _ = detector.detect_swipe()
            self.assertEqual(dir_slow, SwipeDirection.NONE)

            # B. Dominantly vertical movement (dy > dx)
            detector.reset()
            mock_time.return_value = t_base + 1.0
            detector.add_point(200, 200)
            mock_time.return_value = t_base + 1.05
            detector.add_point(210, 250)
            mock_time.return_value = t_base + 1.10
            detector.add_point(215, 300)
            mock_time.return_value = t_base + 1.15
            detector.add_point(220, 360)
            dir_vert, _, _ = detector.detect_swipe()
            self.assertEqual(dir_vert, SwipeDirection.NONE)

    def test_swipe_cooldown(self):
        """4. Verify cooldown prevents multiple consecutive swipe triggers."""
        detector = SwipeDetector(min_displacement=50.0, min_velocity=200.0, swipe_cooldown=0.6)

        t_base = time.time()
        with patch("time.time") as mock_time:
            # Add 4 points to trigger first swipe
            mock_time.return_value = t_base
            detector.add_point(100, 100)
            mock_time.return_value = t_base + 0.03
            detector.add_point(150, 100)
            mock_time.return_value = t_base + 0.06
            detector.add_point(200, 100)
            mock_time.return_value = t_base + 0.10
            detector.add_point(250, 100)
            dir1, _, _ = detector.detect_swipe()
            self.assertEqual(dir1, SwipeDirection.RIGHT)

            # Immediate second swipe within cooldown window
            mock_time.return_value = t_base + 0.15
            detector.add_point(250, 100)
            mock_time.return_value = t_base + 0.18
            detector.add_point(300, 100)
            mock_time.return_value = t_base + 0.22
            detector.add_point(350, 100)
            mock_time.return_value = t_base + 0.25
            detector.add_point(400, 100)
            dir2, _, _ = detector.detect_swipe()
            self.assertEqual(dir2, SwipeDirection.NONE)

    def test_mode_switching_with_victory_sign(self):
        """5. Verify Victory Sign toggles navigation mode with latching."""
        self.assertEqual(self.nav.mode, NavigationMode.PRESENTATION)

        # 1. Victory sign triggers mode switch to BROWSER
        state1 = self.nav.process(GestureType.VICTORY, [])
        self.assertEqual(self.nav.mode, NavigationMode.BROWSER)
        self.assertEqual(state1.action, NavigationAction.MODE_SWITCHED)

        # 2. Holding Victory sign does not re-trigger (latching)
        state2 = self.nav.process(GestureType.VICTORY, [])
        self.assertEqual(self.nav.mode, NavigationMode.BROWSER)
        self.assertEqual(state2.action, NavigationAction.NONE)

        # 3. Release gesture
        self.nav.process(GestureType.NONE, [])

        # 4. Wait for cooldown and trigger Victory again -> switches back to PRESENTATION
        time.sleep(0.55)
        state3 = self.nav.process(GestureType.VICTORY, [])
        self.assertEqual(self.nav.mode, NavigationMode.PRESENTATION)
        self.assertEqual(state3.action, NavigationAction.MODE_SWITCHED)

    @patch("pyautogui.scroll")
    def test_scroll_actions(self, mock_scroll):
        """6 & 7. Verify Index Up triggers scroll up and Thumbs Down triggers scroll down."""
        # Index Up -> Scroll Up
        state_up = self.nav.process(GestureType.INDEX_UP, [])
        self.assertEqual(state_up.action, NavigationAction.SCROLL_UP)
        mock_scroll.assert_called_with(self.nav.scroll_speed)

        # Allow throttle interval
        time.sleep(0.08)

        # Thumbs Down -> Scroll Down
        state_down = self.nav.process(GestureType.THUMBS_DOWN, [])
        self.assertEqual(state_down.action, NavigationAction.SCROLL_DOWN)
        mock_scroll.assert_called_with(-self.nav.scroll_speed)

    def test_non_interference_with_volume_control(self):
        """8. Verify navigation gestures do not trigger volume modifications."""
        vol = VolumeController()
        orig_vol = vol.get_volume_scalar()
        orig_mute = vol.is_muted()

        # Feed navigation gestures to volume controller:
        # None of INDEX_UP, THUMBS_DOWN, VICTORY should change volume or mute!
        for nav_gesture in (GestureType.INDEX_UP, GestureType.THUMBS_DOWN, GestureType.VICTORY):
            v_state = vol.process_gesture(nav_gesture)
            self.assertEqual(v_state.action, VolumeAction.NONE)
            self.assertEqual(vol.get_volume_scalar(), orig_vol)
            self.assertEqual(vol.is_muted(), orig_mute)


if __name__ == "__main__":
    unittest.main()
