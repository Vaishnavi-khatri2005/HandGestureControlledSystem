"""
Real-time Virtual Mouse Control Test Application.

Implements and demonstrates hand gesture-controlled virtual mouse via PyAutoGUI:
- Index Finger Up: Move mouse cursor smoothly across screen
- Pinch (Thumb + Index): Left click (short pinch)
- Two Quick Pinches: Double click
- Hold Pinch (> 0.35s) and Move: Click, Drag, and Drop
- Open Palm: Stop mouse movement / freeze cursor

Features:
- Active control zone boundary box with coordinate interpolation
- Anti-jitter exponential moving average smoothing filter
- Fingertip pointer and pinch drag line overlay
- Real-time HUD displaying cursor position, sensitivity, and active action
- On-the-fly sensitivity and smoothing adjustments
- ESC key exit and safe resource cleanup
"""

import sys
import logging
import cv2

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError
from src.gesture_recognizer import GestureRecognizer, GestureType
from src.controllers.mouse_controller import MouseController, MouseStatus, MouseAction
from src.utils.fps_tracker import FPSTracker


def run_virtual_mouse_test() -> int:
    """Run real-time virtual mouse test with webcam feed."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("VirtualMouseTest")
    logger.info("Initializing Virtual Mouse Control System...")

    # 1. Initialize Mouse Controller
    mouse_ctrl = MouseController(
        frame_reduction=CONFIG.mouse.frame_reduction,
        smoothening=CONFIG.mouse.smoothening,
        sensitivity=1.1,
        pinch_click_threshold=0.30,
        drag_hold_delay=0.35,
        double_click_interval=0.40,
        click_cooldown=0.15
    )
    logger.info(
        "Virtual Mouse ready (Screen: %dx%d | Sensitivity: %.1fx | Smoothing: %.1f).",
        mouse_ctrl.screen_width,
        mouse_ctrl.screen_height,
        mouse_ctrl.sensitivity,
        mouse_ctrl.smoothening
    )

    # 2. Initialize Camera
    cam = CameraManager(
        device_index=CONFIG.camera.device_index,
        width=CONFIG.camera.frame_width,
        height=CONFIG.camera.frame_height,
        fps=CONFIG.camera.fps,
        flip_horizontal=CONFIG.camera.flip_horizontal
    )

    # 3. Initialize Hand Detector & Gesture Recognizer
    try:
        detector = HandDetector(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            mirror_mode=CONFIG.camera.flip_horizontal
        )
    except HandDetectorError as exc:
        logger.critical("Failed to initialize hand detector: %s", exc)
        return 1

    recognizer = GestureRecognizer(
        history_length=5,
        stability_threshold=0.60,
        pinch_ratio_threshold=0.32
    )

    fps_tracker = FPSTracker()

    try:
        cam.start()
    except CameraError as exc:
        logger.critical("Failed to start camera: %s", exc)
        detector.close()
        return 1

    window_name = "Virtual Mouse Test - Press ESC to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 65)
    print("  Virtual Mouse Active (PyAutoGUI Control)!")
    print("  Mouse Controls:")
    print("    - [Index Finger Up]  : Move cursor inside yellow bounding box")
    print("    - [Quick Pinch]      : Single Left Click")
    print("    - [2 Quick Pinches]  : Double Click")
    print("    - [Hold Pinch & Move]: Drag and Drop")
    print("    - [Open Palm]        : Stop / Freeze cursor")
    print("  Keyboard Adjustments:")
    print("    - [+] / [-]          : Increase / decrease sensitivity")
    print("    - '[' / ']'          : Decrease / increase smoothing factor")
    print("    - [ESC] or [q]       : Exit application")
    print("=" * 65 + "\n")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success or frame is None:
                continue

            # Process frame for hand landmarks
            frame, hands = detector.process(
                frame,
                draw_landmarks=True,
                draw_bbox=False,
                draw_ids=False
            )

            # Recognize hand gesture and coordinates
            landmarks = []
            pinch_ratio = 1.0
            if hands:
                primary = hands[0]
                landmarks = primary.landmarks
                gesture_res = recognizer.recognize(landmarks, handedness=primary.label)
                stable_gesture = gesture_res.gesture
                pinch_ratio = gesture_res.pinch_distance
            else:
                gesture_res = recognizer.recognize([])
                stable_gesture = GestureType.NONE

            # Process mouse interactions
            mouse_status = mouse_ctrl.process(
                stable_gesture,
                landmarks,
                cam_w=frame.shape[1],
                cam_h=frame.shape[0],
                pinch_ratio=pinch_ratio
            )

            # Render Virtual Mouse HUD and pointer indicators
            mouse_ctrl.draw_mouse_hud(frame, mouse_status, landmarks)

            # Draw current gesture and FPS on top-left
            fps = fps_tracker.update()
            fps_tracker.draw_fps(frame, origin=(20, 35))

            g_color = (0, 255, 0) if stable_gesture != GestureType.NONE else (160, 160, 160)
            cv2.putText(
                frame,
                f"Gesture: {stable_gesture.value}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                g_color,
                2,
                cv2.LINE_AA
            )

            # Instructions footer
            cv2.putText(
                frame,
                "[+] / [-] Sens | '[' / ']' Smooth | [ESC] Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            # Key listener
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                logger.info("Exit requested by user.")
                break
            elif key in (ord('+'), ord('=')):
                mouse_ctrl.sensitivity = min(3.0, mouse_ctrl.sensitivity + 0.2)
                logger.info("Sensitivity increased to %.1fx", mouse_ctrl.sensitivity)
            elif key in (ord('-'), ord('_')):
                mouse_ctrl.sensitivity = max(0.2, mouse_ctrl.sensitivity - 0.2)
                logger.info("Sensitivity decreased to %.1fx", mouse_ctrl.sensitivity)
            elif key == ord(']'):
                mouse_ctrl.smoothening = min(10.0, mouse_ctrl.smoothening + 1.0)
                logger.info("Smoothing increased to %.1f (smoother)", mouse_ctrl.smoothening)
            elif key == ord('['):
                mouse_ctrl.smoothening = max(1.0, mouse_ctrl.smoothening - 1.0)
                logger.info("Smoothing decreased to %.1f (faster)", mouse_ctrl.smoothening)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard.")
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("Virtual mouse shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(run_virtual_mouse_test())
