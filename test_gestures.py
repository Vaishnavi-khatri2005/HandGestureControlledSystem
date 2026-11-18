"""
Real-time Gesture Recognition Test Application.

Verifies and visualizes all 7 hand gestures:
1. Open Palm
2. Closed Fist
3. Thumbs Up
4. Thumbs Down
5. Index Finger Up (Pointing)
6. Victory Sign
7. Pinch Gesture

Includes:
- Anti-flicker stability filtering with confidence indicator
- Real-time finger-state dashboard ([T, I, M, R, P] indicators)
- Pinch distance ratio meter
- Handedness (Left/Right) integration
- ESC key exit and resource management
"""

import sys
import logging
import cv2

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError
from src.gesture_recognizer import GestureRecognizer, GestureType, GestureResult
from src.utils.fps_tracker import FPSTracker


def draw_hud(
    frame,
    gesture_result: GestureResult,
    handedness: str,
    fps: float
) -> None:
    """Render a clean, informative HUD displaying gesture metrics."""
    h, w, _ = frame.shape

    # 1. Semi-transparent background card on top-left
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (340, 185), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 2. Border
    cv2.rectangle(frame, (15, 15), (340, 185), (80, 80, 80), 1)

    # 3. FPS & Handedness Header
    header_text = f"FPS: {int(fps)} | Hand: {handedness}"
    cv2.putText(
        frame,
        header_text,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 180),
        1,
        cv2.LINE_AA
    )

    # 4. Active Stable Gesture with Color Coding
    gesture = gesture_result.gesture
    if gesture == GestureType.NONE:
        color = (160, 160, 160)
    elif gesture in (GestureType.OPEN_PALM, GestureType.VICTORY):
        color = (0, 255, 0)       # Green
    elif gesture in (GestureType.THUMBS_UP, GestureType.INDEX_UP):
        color = (255, 200, 0)     # Cyan
    elif gesture == GestureType.THUMBS_DOWN:
        color = (0, 0, 255)       # Red
    elif gesture == GestureType.CLOSED_FIST:
        color = (0, 165, 255)     # Orange
    elif gesture == GestureType.PINCH:
        color = (255, 0, 255)     # Magenta
    else:
        color = (255, 255, 255)

    gesture_text = f"{gesture.value}"
    cv2.putText(
        frame,
        gesture_text,
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )

    # 5. Stability & Confidence Bar
    conf_pct = int(gesture_result.confidence * 100)
    conf_label = f"Stability: {conf_pct}%"
    cv2.putText(
        frame,
        conf_label,
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # Draw small progress bar for confidence
    bar_x, bar_y, bar_w, bar_h = 135, 95, 180, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
    fill_w = int(bar_w * (gesture_result.confidence))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 0), -1)

    # 6. Finger States Dashboard: [T, I, M, R, P]
    fs = gesture_result.finger_states
    fingers = [("T", fs.thumb), ("I", fs.index), ("M", fs.middle), ("R", fs.ring), ("P", fs.pinky)]
    cv2.putText(
        frame,
        "Fingers:",
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    for i, (name, is_up) in enumerate(fingers):
        fx = 95 + (i * 45)
        fy = 140
        f_color = (0, 255, 0) if is_up else (80, 80, 80)
        cv2.circle(frame, (fx, fy - 5), 11, f_color, -1)
        cv2.putText(
            frame,
            name,
            (fx - 5, fy - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0) if is_up else (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    # 7. Pinch Metric
    pinch_str = f"Pinch Ratio: {gesture_result.pinch_distance:.2f}"
    cv2.putText(
        frame,
        pinch_str,
        (25, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )


def run_gesture_test() -> int:
    """Run real-time gesture recognition preview loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("GestureTest")
    logger.info("Starting Gesture Recognition Test Runner...")

    cam = CameraManager(
        device_index=CONFIG.camera.device_index,
        width=CONFIG.camera.frame_width,
        height=CONFIG.camera.frame_height,
        fps=CONFIG.camera.fps,
        flip_horizontal=CONFIG.camera.flip_horizontal
    )

    try:
        detector = HandDetector(
            max_num_hands=1,  # Single primary hand for gesture testing
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            mirror_mode=CONFIG.camera.flip_horizontal
        )
    except HandDetectorError as exc:
        logger.critical("Failed to initialize hand detector: %s", exc)
        return 1

    recognizer = GestureRecognizer(
        history_length=6,
        stability_threshold=0.65,
        pinch_ratio_threshold=0.35
    )

    fps_tracker = FPSTracker()

    try:
        cam.start()
    except CameraError as exc:
        logger.critical("Failed to start webcam: %s", exc)
        detector.close()
        return 1

    window_name = "Gesture Recognition Test - Press ESC to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 65)
    print("  Gesture Recognition Active!")
    print("  Try the 7 gestures in front of the camera:")
    print("    1. Open Palm       - Spread all 5 fingers")
    print("    2. Closed Fist     - Curl all fingers into a ball")
    print("    3. Thumbs Up       - Thumb pointing up, fingers folded")
    print("    4. Thumbs Down     - Thumb pointing down, fingers folded")
    print("    5. Index Up        - Point index finger up")
    print("    6. Victory Sign    - Index and middle fingers up")
    print("    7. Pinch           - Touch thumb and index tips together")
    print("  Press [ESC] or [q] to quit.")
    print("=" * 65 + "\n")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success or frame is None:
                continue

            frame, hands = detector.process(
                frame,
                draw_landmarks=True,
                draw_bbox=True,
                draw_ids=False
            )

            # Analyze hand gestures
            if hands:
                primary_hand = hands[0]
                gesture_res = recognizer.recognize(
                    primary_hand.landmarks,
                    handedness=primary_hand.label
                )
                hand_label = primary_hand.label
            else:
                gesture_res = recognizer.recognize([])
                hand_label = "None"

            # Render HUD
            fps = fps_tracker.update()
            draw_hud(frame, gesture_res, hand_label, fps)

            # Footer instructions
            cv2.putText(
                frame,
                "Press [ESC] to Exit",
                (frame.shape[1] - 170, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                logger.info("Exit requested.")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed.")

    return 0


if __name__ == "__main__":
    sys.exit(run_gesture_test())
