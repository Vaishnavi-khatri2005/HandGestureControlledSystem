"""
Independent test runner for the webcam integration module.

Verifies:
1. Webcam access via OpenCV.
2. Real-time video streaming with live FPS counter.
3. Proper exit handling using the ESC key.
4. Clean hardware and window resource release upon termination.
"""

import sys
import logging
from config import CONFIG
from src.camera import CameraManager, CameraError


def test_webcam() -> int:
    """Run independent webcam preview test."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("WebcamTest")
    logger.info("Starting independent webcam test...")

    cam = CameraManager(
        device_index=CONFIG.camera.device_index,
        width=CONFIG.camera.frame_width,
        height=CONFIG.camera.frame_height,
        fps=CONFIG.camera.fps,
        flip_horizontal=CONFIG.camera.flip_horizontal
    )

    try:
        cam.run_preview(window_name="Webcam Test - Press ESC to Quit")
        return 0
    except CameraError as exc:
        logger.error("Webcam test failed with CameraError: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during webcam test: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(test_webcam())
