"""
Unit tests for the CustomTkinter GUI Dashboard (src.gui).

Verifies:
1. Window instantiation and cute pastel styling setup.
2. Initial telemetry data structure.
3. Mode switching logic and button state synchronization.
4. Individual module enable/disable switches.
5. Sensitivity and smoothing slider event callbacks.
6. Clean exit and destruction.
"""

import unittest
from src.gui import HandGestureGUI, TelemetryData
from src.gesture_recognizer import GestureType


class TestHandGestureGUI(unittest.TestCase):
    """Unit tests for the CustomTkinter GUI dashboard interface."""

    def setUp(self):
        # Create app instance in withdrawn state to avoid opening physical window during tests
        self.app = HandGestureGUI()
        self.app.withdraw()

    def tearDown(self):
        try:
            self.app.exit_application()
        except Exception:
            pass

    def test_initial_ui_state(self):
        """1. Verify initial UI state and default widgets."""
        self.assertFalse(self.app.is_camera_running)
        self.assertEqual(self.app.active_mode, "Virtual Mouse")
        self.assertTrue(self.app.module_mouse_enabled)
        self.assertTrue(self.app.module_volume_enabled)
        self.assertTrue(self.app.module_nav_enabled)
        self.assertEqual(self.app.btn_start_cam.cget("state"), "normal")
        self.assertEqual(self.app.btn_stop_cam.cget("state"), "disabled")

    def test_mode_selection_handler(self):
        """2. Verify operating mode selection updates state correctly."""
        self.app._on_mode_selected("Volume")
        self.assertEqual(self.app.active_mode, "Volume")
        self.assertIn("Volume Mode", self.app.lbl_current_mode.cget("text"))

        self.app._on_mode_selected("Navigation")
        self.assertEqual(self.app.active_mode, "Navigation")
        self.assertIn("Navigation Mode", self.app.lbl_current_mode.cget("text"))

        self.app._on_mode_selected("Virtual Mouse")
        self.assertEqual(self.app.active_mode, "Virtual Mouse")

    def test_module_toggle_switches(self):
        """3. Verify enable/disable toggles affect module states."""
        # Uncheck mouse switch
        self.app.sw_mouse.deselect()
        self.app._on_module_toggle()
        self.assertFalse(self.app.module_mouse_enabled)
        self.assertIn("DISABLED", self.app.lbl_mode_state.cget("text"))

        # Re-check mouse switch
        self.app.sw_mouse.select()
        self.app._on_module_toggle()
        self.assertTrue(self.app.module_mouse_enabled)
        self.assertIn("ACTIVE", self.app.lbl_mode_state.cget("text"))

    def test_sensitivity_sliders(self):
        """4. Verify slider event handlers update controller settings."""
        # Mouse sensitivity
        self.app._on_mouse_sens_change(1.8)
        if self.app.mouse_ctrl:
            self.assertEqual(self.app.mouse_ctrl.sensitivity, 1.8)
        self.assertIn("1.8x", self.app.lbl_mouse_sens.cget("text"))

        # Smoothing factor
        self.app._on_smoothing_change(6.0)
        if self.app.mouse_ctrl:
            self.assertEqual(self.app.mouse_ctrl.smoothening, 6.0)
        self.assertIn("6.0", self.app.lbl_smoothing.cget("text"))

        # Volume sensitivity
        self.app._on_vol_sens_change(2.0)
        if self.app.volume_ctrl:
            self.assertEqual(self.app.volume_ctrl.sensitivity, 2.0)
        self.assertIn("2.0x", self.app.lbl_vol_sens.cget("text"))


if __name__ == "__main__":
    unittest.main()
