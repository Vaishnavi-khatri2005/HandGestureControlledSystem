"""
Unit tests for the VolumeController module.

Tests all volume control operations independently:
1. Volume scalar & percentage querying.
2. Smooth target interpolation (preventing sudden jumps).
3. Open Palm: Volume increase.
4. Pinch Gesture: Finger distance to volume mapping (0-100%).
5. Closed Fist: Mute/Unmute toggling.
6. Cooldown enforcement on mute.
7. Fist latching (preventing rapid toggle chattering while fist is held).
"""

import time
import unittest
from src.gesture_recognizer import GestureType
from src.controllers.volume_controller import VolumeController, VolumeAction


class TestVolumeController(unittest.TestCase):
    """Test suite for VolumeController operations."""

    def setUp(self):
        self.controller = VolumeController(
            sensitivity=1.0,
            smoothing_factor=0.5,
            mute_cooldown=0.6,
            min_pinch_ratio=0.10,
            max_pinch_ratio=0.35
        )
        # Store original volume and mute to restore after tests
        self.original_scalar = self.controller.get_volume_scalar()
        self.original_mute = self.controller.is_muted()

    def tearDown(self):
        # Restore system state
        self.controller.set_volume_scalar(self.original_scalar, smooth=False)
        self.controller._set_mute_raw(self.original_mute)

    def test_volume_query(self):
        """1. Verify volume scalar and percentage query."""
        scalar = self.controller.get_volume_scalar()
        pct = self.controller.get_volume_percent()
        self.assertGreaterEqual(scalar, 0.0)
        self.assertLessEqual(scalar, 1.0)
        self.assertEqual(pct, int(round(scalar * 100)))

    def test_smoothing_prevents_jumps(self):
        """2. Verify that smooth=True prevents instant jumps."""
        self.controller.set_volume_scalar(0.20, smooth=False)
        # Target 0.80 with 0.5 smoothing
        intermediate = self.controller.set_volume_scalar(0.80, smooth=True)
        # Should interpolate rather than jump directly to 0.80
        self.assertGreater(intermediate, 0.20)
        self.assertLess(intermediate, 0.80)

    def test_open_palm_increases_volume(self):
        """3. Verify Open Palm triggers volume increment."""
        self.controller.set_volume_scalar(0.30, smooth=False)
        initial_vol = self.controller.get_volume_scalar()
        state = self.controller.process_gesture(GestureType.OPEN_PALM)
        self.assertEqual(state.action, VolumeAction.INCREASE)
        self.assertGreater(state.volume_scalar, initial_vol)

    def test_pinch_maps_distance_to_volume(self):
        """4. Verify Pinch gesture maps distance ratio to volume (0–100%)."""
        # Tight pinch at min_pinch_ratio (0.10) -> Should map to near 0%
        self.controller.set_volume_by_pinch_distance(0.10)
        self.controller.set_volume_by_pinch_distance(0.10)  # Apply smoothing step
        low_vol = self.controller.get_volume_scalar()

        # Wide pinch at max_pinch_ratio (0.35) -> Should map to higher volume
        for _ in range(5):
            self.controller.set_volume_by_pinch_distance(0.35)
        high_vol = self.controller.get_volume_scalar()

        self.assertGreater(high_vol, low_vol)

    def test_closed_fist_mute_toggle(self):
        """5. Verify Closed Fist toggles mute state."""
        initial_mute = self.controller.is_muted()
        state = self.controller.process_gesture(GestureType.CLOSED_FIST)
        self.assertEqual(state.action, VolumeAction.MUTE_TOGGLED)
        self.assertEqual(self.controller.is_muted(), not initial_mute)

    def test_mute_cooldown_and_fist_latching(self):
        """6. Verify that holding closed fist does not toggle mute repeatedly."""
        # 1. First fist triggers mute toggle
        state1 = self.controller.process_gesture(GestureType.CLOSED_FIST)
        self.assertEqual(state1.action, VolumeAction.MUTE_TOGGLED)
        muted_state_1 = self.controller.is_muted()

        # 2. Immediate second fist frame while held -> must be locked (fist held)
        state2 = self.controller.process_gesture(GestureType.CLOSED_FIST)
        self.assertEqual(state2.action, VolumeAction.NONE)
        self.assertEqual(self.controller.is_muted(), muted_state_1)

        # 3. Release fist
        self.controller.process_gesture(GestureType.NONE)

        # 4. Immediate re-trigger while cooldown (0.6s) is still active -> blocked by cooldown
        state3 = self.controller.process_gesture(GestureType.CLOSED_FIST)
        self.assertEqual(state3.action, VolumeAction.NONE)

        # 5. Release fist again and wait for cooldown to expire
        self.controller.process_gesture(GestureType.NONE)
        time.sleep(0.65)

        # 6. New fist after cooldown -> successfully toggles mute
        state4 = self.controller.process_gesture(GestureType.CLOSED_FIST)
        self.assertEqual(state4.action, VolumeAction.MUTE_TOGGLED)
        self.assertEqual(self.controller.is_muted(), not muted_state_1)


if __name__ == "__main__":
    unittest.main()
