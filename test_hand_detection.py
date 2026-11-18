"""
Real-time Hand Detection Test Runner.

Demonstrates and verifies:
1. Multi-hand detection (up to 2 hands concurrently).
2. Continuous tracking of all 21 landmarks per hand.
3. Left vs. Right hand classification.
4. On-screen visualization of landmarks, connections, bounding boxes, IDs, and coordinates.
5. Graceful handling of empty frames (no hands detected).
6. ESC key exit and resource cleanup.
"""

import sys
import logging
import time
import cv2

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError, LANDMARK_NAMES
from src.utils.fps_tracker import FPSTracker


def run_hand_detection_test() -> int:
    """Run real-time hand detection test with webcam feed."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("HandDetectionTest")
    logger.info("Initializing Real-Time Hand Detection Test...")

    # Initialize Camera
    cam = CameraManager(
        device_index=CONFIG.camera.device_index,
        width=CONFIG.camera.frame_width,
        height=CONFIG.camera.frame_height,
        fps=CONFIG.camera.fps,
        flip_horizontal=CONFIG.camera.flip_horizontal
    )

    # Initialize Hand Detector (detects up to 2 hands)
    try:
        detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            mirror_mode=CONFIG.camera.flip_horizontal
        )
        logger.info("MediaPipe HandDetector initialized (Max Hands: 2).")
    except HandDetectorError as exc:
        logger.critical("Failed to initialize HandDetector: %s", exc)
        return 1

    try:
        cam.start()
    except CameraError as exc:
        logger.critical("Failed to start camera: %s", exc)
        detector.close()
        return 1

    window_name = "Hand Detection Test (Press ESC to Quit | 'i' to Toggle IDs)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    fps_tracker = FPSTracker()
    show_landmark_ids = True

    print("\n" + "=" * 65)
    print("  Real-time Hand Detection Active!")
    print("  - Place one or both hands in front of the camera.")
    print("  - Press [i] to toggle individual Landmark ID numbers on/off.")
    print("  - Press [ESC] or [q] to quit.")
    print("=" * 65 + "\n")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success or frame is None:
                continue

            # Process frame for hand landmarks & handedness
            frame, detected_hands = detector.process(
                frame,
                draw_landmarks=True,
                draw_bbox=True,
                draw_ids=show_landmark_ids
            )

            # Draw FPS metric
            fps = fps_tracker.update()
            fps_tracker.draw_fps(frame, origin=(20, 35))

            # Render key controls reminder
            cv2.putText(
                frame,
                "[i] Toggle IDs | [ESC] Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # Display frame
            cv2.imshow(window_name, frame)

            # Handle user keypresses
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                logger.info("Exit requested by user.")
                break
            elif key == ord('i'):
                show_landmark_ids = not show_landmark_ids
                logger.info("Landmark IDs overlay toggled: %s", show_landmark_ids)

            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard.")
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("Resources released and shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(run_hand_detection_test())
