"""
Hand Gesture Controlled System - Centralized Main Application.

Integrates all subsystems into a unified, high-performance touchless computer interface:
- Webcam Capture (OpenCV DirectShow with fallback)
- Hand Landmark Detection (MediaPipe Hands)
- Gesture Classification (7 core gestures + temporal anti-flicker filtering)
- 3 Operating Modes:
    1. Virtual Mouse Mode (Cursor, Left Click, Double Click, Drag & Drop, Freeze)
    2. Volume Control Mode (Open Palm Vol+, Fist Mute, Pinch Distance Modulation)
    3. Navigation Mode (Swipe Forward/Backward, Scroll Up/Down, Mode Switch)

Features:
- Dual frontend support:
    - Standard OpenCV Window Interface: `python main.py`
    - Modern CustomTkinter GUI Dashboard: `python main.py --gui` or `python run_gui.py`
- Strict mode isolation: only the active mode executes commands, eliminating conflicting actions
- Interactive mode switching via keyboard (1, 2, 3, or Tab)
- Comprehensive on-screen dashboard and dynamic mode-specific HUDs
- Graceful exception handling, frame rate smoothing, and resource cleanup
"""

import sys
import logging
from enum import Enum
from typing import List, Tuple, Optional
import cv2

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError, DetectedHand
from src.gesture_recognizer import GestureRecognizer, GestureType, GestureResult
from src.controllers.volume_controller import VolumeController, VolumeControlError, VolumeState
from src.controllers.mouse_controller import MouseController, MouseControlError, MouseStatus
from src.controllers.navigation_controller import NavigationController, NavigationState, NavigationMode
from src.utils.fps_tracker import FPSTracker


class OperatingMode(Enum):
    """Available system operating modes."""
    MOUSE = "Virtual Mouse Mode"
    VOLUME = "Volume Control Mode"
    NAVIGATION = "Navigation Mode"


def setup_logger() -> logging.Logger:
    """Configure structured application logger."""
    logger = logging.getLogger("HandGestureSystem")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


class HandGestureSystemApp:
    """Central orchestrator managing pipeline lifecycle, modes, and controllers."""

    def __init__(self) -> None:
        self.logger = setup_logger()
        self.active_mode = OperatingMode.MOUSE

        # Subsystems
        self.camera: Optional[CameraManager] = None
        self.detector: Optional[HandDetector] = None
        self.recognizer: Optional[GestureRecognizer] = None
        self.mouse_ctrl: Optional[MouseController] = None
        self.volume_ctrl: Optional[VolumeController] = None
        self.nav_ctrl: Optional[NavigationController] = None
        self.fps_tracker = FPSTracker()

    def initialize(self) -> bool:
        """Initialize all controllers and models with robust exception safety."""
        self.logger.info("Initializing Hand Gesture Controlled System...")

        # 1. Camera
        self.camera = CameraManager(
            device_index=CONFIG.camera.device_index,
            width=CONFIG.camera.frame_width,
            height=CONFIG.camera.frame_height,
            fps=CONFIG.camera.fps,
            flip_horizontal=CONFIG.camera.flip_horizontal
        )
        try:
            self.camera.start()
            self.logger.info("Webcam started on device index %d.", CONFIG.camera.device_index)
        except CameraError as e:
            self.logger.critical("Failed to start webcam: %s", e)
            return False

        # 2. Hand Detector
        try:
            self.detector = HandDetector(
                max_num_hands=1,
                min_detection_confidence=CONFIG.hand_detector.min_detection_confidence,
                min_tracking_confidence=CONFIG.hand_detector.min_tracking_confidence,
                mirror_mode=CONFIG.camera.flip_horizontal
            )
            self.logger.info("MediaPipe Hand Detector initialized.")
        except HandDetectorError as e:
            self.logger.critical("Failed to initialize Hand Detector: %s", e)
            self.shutdown()
            return False

        # 3. Gesture Recognizer
        self.recognizer = GestureRecognizer(
            history_length=6,
            stability_threshold=0.65,
            pinch_ratio_threshold=0.32
        )

        # 4. Virtual Mouse Controller
        try:
            self.mouse_ctrl = MouseController(
                frame_reduction=CONFIG.mouse.frame_reduction,
                smoothening=CONFIG.mouse.smoothening,
                sensitivity=1.1,
                pinch_click_threshold=0.30,
                drag_hold_delay=0.35,
                double_click_interval=0.40,
                click_cooldown=0.15
            )
            self.logger.info(
                "Virtual Mouse Controller ready (Screen: %dx%d).",
                self.mouse_ctrl.screen_width,
                self.mouse_ctrl.screen_height
            )
        except MouseControlError as e:
            self.logger.warning("Virtual Mouse Controller unavailable: %s", e)

        # 5. Volume Controller
        try:
            self.volume_ctrl = VolumeController(
                sensitivity=1.2,
                smoothing_factor=0.25,
                mute_cooldown=1.5
            )
            self.logger.info(
                "Volume Controller ready (Initial Volume: %d%%).",
                self.volume_ctrl.get_volume_percent()
            )
        except VolumeControlError as e:
            self.logger.warning("Volume Controller unavailable: %s", e)

        # 6. Navigation Controller
        self.nav_ctrl = NavigationController(
            default_mode=NavigationMode.PRESENTATION,
            scroll_speed=45,
            scroll_throttle_interval=0.07,
            mode_switch_cooldown=1.3
        )
        self.logger.info("Navigation Controller ready.")

        return True

    def set_mode(self, new_mode: OperatingMode) -> None:
        """Switch operating mode and release any active holds or latches."""
        if self.active_mode == new_mode:
            return

        # Clean up previous mode state
        if self.active_mode == OperatingMode.MOUSE and self.mouse_ctrl:
            self.mouse_ctrl.stop_drag()

        self.active_mode = new_mode
        self.logger.info("Operating Mode switched to: >>> %s <<<", new_mode.value)

    def cycle_mode(self) -> None:
        """Cycle sequentially through available operating modes."""
        modes = list(OperatingMode)
        current_idx = modes.index(self.active_mode)
        next_mode = modes[(current_idx + 1) % len(modes)]
        self.set_mode(next_mode)

    def draw_top_dashboard(
        self,
        frame,
        gesture_res: GestureResult,
        fps: float
    ) -> None:
        """Render universal header dashboard displaying Mode tabs, active gesture, and FPS."""
        h, w, _ = frame.shape

        bar_h = 75
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), (60, 60, 60), 1)

        # 1. Mode Tabs
        tab_defs = [
            ("[1] MOUSE", OperatingMode.MOUSE),
            ("[2] VOLUME", OperatingMode.VOLUME),
            ("[3] NAVIGATION", OperatingMode.NAVIGATION),
        ]
        tab_x = 20
        for label, mode in tab_defs:
            is_active = (self.active_mode == mode)
            tab_bg = (0, 180, 255) if is_active else (45, 45, 45)
            text_color = (0, 0, 0) if is_active else (180, 180, 180)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            padding_x = 12
            btn_w = tw + (padding_x * 2)
            btn_h = 24
            btn_y = 12

            cv2.rectangle(frame, (tab_x, btn_y), (tab_x + btn_w, btn_y + btn_h), tab_bg, -1)
            if is_active:
                cv2.rectangle(frame, (tab_x, btn_y), (tab_x + btn_w, btn_y + btn_h), (255, 255, 255), 1)

            cv2.putText(
                frame,
                label,
                (tab_x + padding_x, btn_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1 if not is_active else 2,
                cv2.LINE_AA
            )
            tab_x += btn_w + 10

        # FPS counter on top-right
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(
            frame,
            fps_text,
            (w - 110, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
            cv2.LINE_AA
        )

        # 2. Gesture feedback on sub-row
        g_name = gesture_res.gesture.value
        g_conf = int(gesture_res.confidence * 100)
        g_col = (0, 255, 0) if gesture_res.gesture != GestureType.NONE else (160, 160, 160)

        cv2.putText(
            frame,
            f"Detected Gesture: {g_name} ({g_conf}%)",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            g_col,
            2,
            cv2.LINE_AA
        )

        # Instructions footer at bottom of window
        footer_bg = (20, 20, 20)
        cv2.rectangle(frame, (0, h - 28), (w, h), footer_bg, -1)
        cv2.putText(
            frame,
            "Press [1/2/3] Mode | [Tab] Cycle Mode | [ESC] Quit",
            (20, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

    def run(self) -> int:
        """Main real-time application processing loop."""
        if not self.initialize():
            return 1

        window_name = CONFIG.window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        self.logger.info("Main loop running. Press [1/2/3] to switch modes, [ESC] to quit.")

        try:
            while True:
                success, frame = self.camera.read_frame()
                if not success or frame is None:
                    continue

                cam_w, cam_h = frame.shape[1], frame.shape[0]

                # 1. Detect Hand Landmarks
                show_mouse_zone = (self.active_mode == OperatingMode.MOUSE)
                frame, detected_hands = self.detector.process(
                    frame,
                    draw_landmarks=True,
                    draw_bbox=not show_mouse_zone,
                    draw_ids=False
                )

                # 2. Extract Landmarks & Recognize Gesture
                landmarks: List[Tuple[int, int, int]] = []
                pinch_ratio = 1.0
                handedness = "Right"

                if detected_hands:
                    primary_hand = detected_hands[0]
                    landmarks = primary_hand.landmarks
                    handedness = primary_hand.label
                    gesture_res = self.recognizer.recognize(landmarks, handedness=handedness)
                    pinch_ratio = gesture_res.pinch_distance
                else:
                    gesture_res = self.recognizer.recognize([])

                # 3. Execute Mode-Specific Commands (Strict Isolation: Only active mode executes)
                if self.active_mode == OperatingMode.MOUSE and self.mouse_ctrl is not None:
                    mouse_status = self.mouse_ctrl.process(
                        gesture=gesture_res.gesture,
                        landmarks=landmarks,
                        cam_w=cam_w,
                        cam_h=cam_h,
                        pinch_ratio=pinch_ratio
                    )
                    self.mouse_ctrl.draw_mouse_hud(frame, mouse_status, landmarks)

                elif self.active_mode == OperatingMode.VOLUME and self.volume_ctrl is not None:
                    vol_state = self.volume_ctrl.process_gesture(
                        gesture=gesture_res.gesture,
                        pinch_ratio=pinch_ratio
                    )
                    hud_pos = (cam_w - 175, 95)
                    VolumeController.draw_volume_hud(frame, vol_state, origin=hud_pos)

                elif self.active_mode == OperatingMode.NAVIGATION and self.nav_ctrl is not None:
                    nav_state = self.nav_ctrl.process(
                        gesture=gesture_res.gesture,
                        landmarks=landmarks
                    )
                    nav_hud_pos = (20, cam_h - 135)
                    NavigationController.draw_navigation_hud(frame, nav_state, origin=nav_hud_pos)

                # 4. Render Universal Header & FPS
                fps = self.fps_tracker.update()
                self.draw_top_dashboard(frame, gesture_res, fps)

                # 5. Display Frame
                cv2.imshow(window_name, frame)

                # 6. Keyboard listener for mode switching and application exit
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    self.logger.info("Exit command received.")
                    break
                elif key == ord('1'):
                    self.set_mode(OperatingMode.MOUSE)
                elif key == ord('2'):
                    self.set_mode(OperatingMode.VOLUME)
                elif key == ord('3'):
                    self.set_mode(OperatingMode.NAVIGATION)
                elif key == 9:  # Tab key cycles modes
                    self.cycle_mode()

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        except Exception as exc:
            self.logger.exception("Unexpected error in main application loop: %s", exc)
            return 1
        finally:
            self.shutdown()

        return 0

    def shutdown(self) -> None:
        """Release all allocated system and hardware resources cleanly."""
        self.logger.info("Shutting down Hand Gesture Controlled System...")
        if self.mouse_ctrl:
            self.mouse_ctrl.stop_drag()
        if self.camera:
            self.camera.release()
        if self.detector:
            self.detector.close()
        cv2.destroyAllWindows()
        self.logger.info("Shutdown completed successfully.")


def main() -> int:
    """Application entry point. Supports --gui flag to launch CustomTkinter dashboard."""
    if len(sys.argv) > 1 and sys.argv[1] in ("--gui", "-g", "gui"):
        from src.gui import run_gui
        return run_gui()

    app = HandGestureSystemApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
