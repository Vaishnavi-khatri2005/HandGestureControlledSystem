"""
Real-time System Volume Control Test Application.

Implements and demonstrates gesture-based Windows volume control via Pycaw:
- Open Palm: Smoothly increase volume
- Closed Fist: Toggle Mute / Unmute (protected with cooldown & state latching)
- Pinch Gesture: Modulate / decrease volume mapped directly to thumb-index finger distance

Features:
- Real-time vertical volume bar with live percentage and color gradient
- Mute status indicator
- Anti-jump smoothing filter preventing audio spikes
- Sensitivity scaling and cooldown protection
- ESC key exit and safe resource cleanup
"""

import sys
import logging
import cv2

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError
from src.gesture_recognizer import GestureRecognizer, GestureType
from src.controllers.volume_controller import VolumeController, VolumeControlError, VolumeState, VolumeAction
from src.utils.fps_tracker import FPSTracker


def run_volume_control_test() -> int:
    """Run real-time gesture-based volume control test with live webcam feed."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("VolumeControlTest")
    logger.info("Initializing Gesture Volume Control System...")

    # 1. Initialize Pycaw Volume Controller
    try:
        volume_ctrl = VolumeController(
            sensitivity=1.2,
            smoothing_factor=0.25,
            mute_cooldown=1.5,
            min_pinch_ratio=0.08,
            max_pinch_ratio=0.38
        )
        logger.info(
            "Volume Controller ready. Initial Master Volume: %d%% (Muted: %s)",
            volume_ctrl.get_volume_percent(),
            volume_ctrl.is_muted()
        )
    except VolumeControlError as exc:
        logger.critical("Failed to access Windows Core Audio: %s", exc)
        return 1

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
        pinch_ratio_threshold=0.38
    )

    fps_tracker = FPSTracker()

    try:
        cam.start()
    except CameraError as exc:
        logger.critical("Failed to start camera: %s", exc)
        detector.close()
        return 1

    window_name = "System Volume Control Test - Press ESC to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 65)
    print("  System Volume Control Active (Pycaw Windows Audio)!")
    print("  Gestures:")
    print("    - [Open Palm]     : Smoothly increase volume")
    print("    - [Closed Fist]   : Toggle Mute / Unmute (1.5s cooldown)")
    print("    - [Pinch Gesture] : Decrease / modulate volume by finger distance")
    print("  Keyboard Shortcuts:")
    print("    - [+] / [-]       : Increase / decrease sensitivity")
    print("    - [m]             : Manual mute toggle")
    print("    - [ESC] or [q]    : Exit application")
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

            # Recognize hand gesture and process volume command
            if hands:
                primary = hands[0]
                gesture_res = recognizer.recognize(primary.landmarks, handedness=primary.label)
                stable_gesture = gesture_res.gesture
                pinch_ratio = gesture_res.pinch_distance
            else:
                gesture_res = recognizer.recognize([])
                stable_gesture = GestureType.NONE
                pinch_ratio = 1.0

            # Execute volume operation
            vol_state = volume_ctrl.process_gesture(stable_gesture, pinch_ratio)

            # Draw volume gauge HUD on the right side of the screen
            hud_origin = (frame.shape[1] - 170, 70)
            VolumeController.draw_volume_hud(frame, vol_state, origin=hud_origin)

            # Draw current gesture information card on top-left
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

            # Sensitivity indicator
            cv2.putText(
                frame,
                f"Sensitivity: {volume_ctrl.sensitivity:.1f}x ([+] / [-])",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
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
                volume_ctrl.sensitivity = min(3.0, volume_ctrl.sensitivity + 0.2)
                logger.info("Sensitivity increased to %.1fx", volume_ctrl.sensitivity)
            elif key in (ord('-'), ord('_')):
                volume_ctrl.sensitivity = max(0.2, volume_ctrl.sensitivity - 0.2)
                logger.info("Sensitivity decreased to %.1fx", volume_ctrl.sensitivity)
            elif key == ord('m'):
                volume_ctrl.toggle_mute()

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard.")
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("System volume control shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(run_volume_control_test())
