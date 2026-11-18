"""
Real-time Hand Gesture Navigation Control Test Application.

Implements and demonstrates:
- Swipe Right: Next Page / Slide / Forward
- Swipe Left: Previous Page / Slide / Backward
- Index Finger Up: Continuous smooth scroll up
- Thumbs Down: Continuous smooth scroll down
- Victory Sign: Toggle Navigation Mode (Presentation vs. Browser)

Features:
- Trajectory tracking with velocity and displacement filters
- Real-time HUD displaying active mode and executed action
- Accidental trigger rejection and cooldown protection
- ESC key exit and resource cleanup
"""

import sys
import logging
import cv2

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError
from src.gesture_recognizer import GestureRecognizer, GestureType
from src.navigation import NavigationController, NavigationState, NavigationMode, NavigationAction
from src.utils.fps_tracker import FPSTracker


def run_navigation_test() -> int:
    """Run real-time gesture navigation test with live webcam feed."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("NavigationTest")
    logger.info("Initializing Gesture Navigation System...")

    # 1. Initialize Navigation Controller
    nav_ctrl = NavigationController(
        default_mode=NavigationMode.PRESENTATION,
        scroll_speed=50,
        scroll_throttle_interval=0.07,
        mode_switch_cooldown=1.3
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
        stability_threshold=0.60
    )

    fps_tracker = FPSTracker()

    try:
        cam.start()
    except CameraError as exc:
        logger.critical("Failed to start camera: %s", exc)
        detector.close()
        return 1

    window_name = "Hand Gesture Navigation Test - Press ESC to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 65)
    print("  Gesture Navigation Active (PyAutoGUI Control)!")
    print("  Navigation Controls:")
    print("    - [Swipe Right]     : Next Page / Slide (Forward)")
    print("    - [Swipe Left]      : Previous Page / Slide (Backward)")
    print("    - [Index Finger Up] : Scroll Up")
    print("    - [Thumbs Down]     : Scroll Down")
    print("    - [Victory Sign]    : Switch Mode (Presentation <-> Browser)")
    print("  Keyboard Shortcuts:")
    print("    - [Tab]             : Manual toggle navigation mode")
    print("    - [ESC] or [q]      : Exit application")
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
                draw_bbox=True,
                draw_ids=False
            )

            # Recognize hand gesture and coordinates
            if hands:
                primary = hands[0]
                landmarks = primary.landmarks
                gesture_res = recognizer.recognize(landmarks, handedness=primary.label)
                stable_gesture = gesture_res.gesture
            else:
                landmarks = []
                gesture_res = recognizer.recognize([])
                stable_gesture = GestureType.NONE

            # Process navigation operations
            nav_state = nav_ctrl.process(stable_gesture, landmarks)

            # Render Navigation HUD on bottom-left
            hud_origin = (20, frame.shape[0] - 120)
            NavigationController.draw_navigation_hud(frame, nav_state, origin=hud_origin)

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

            cv2.imshow(window_name, frame)

            # Key listener
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                logger.info("Exit requested by user.")
                break
            elif key == 9:  # Tab key
                new_mode = nav_ctrl.switch_mode()
                logger.info("Mode manually switched to %s", new_mode.value)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard.")
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("Navigation test shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(run_navigation_test())
