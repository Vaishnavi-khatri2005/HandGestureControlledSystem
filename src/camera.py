"""
Webcam handling module.

Provides a robust, object-oriented interface for real-time video capture using OpenCV.
Features:
- Windows DirectShow backend optimization with automatic fallback
- Graceful error handling for missing, disconnected, or busy webcams
- Frame dimension and FPS configuration
- Optional horizontal mirroring for natural interaction UX
- Clean context manager support (__enter__ / __exit__)
- Dedicated standalone preview mode with ESC key exit and resource cleanup
"""

import sys
import time
import logging
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraError(Exception):
    """Custom exception raised when webcam initialization or streaming fails."""
    pass


class CameraManager:
    """Manages OpenCV VideoCapture lifecycle, streaming, and preview display."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = True
    ) -> None:
        """
        Initialize the camera manager.

        Args:
            device_index: Hardware camera index (0 is typically the default webcam).
            width: Requested capture frame width in pixels.
            height: Requested capture frame height in pixels.
            fps: Requested frames per second.
            flip_horizontal: Whether to mirror the image horizontally.
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.cap: Optional[cv2.VideoCapture] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def start(self) -> None:
        """
        Open the camera device, configure capture properties, and verify frame access.

        Raises:
            CameraError: If the device cannot be opened or fails the initial frame read.
        """
        try:
            # On Windows, cv2.CAP_DSHOW provides fast, stable webcam startup
            self._logger.info("Attempting to open camera %d via DirectShow...", self.device_index)
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                self._logger.warning("DirectShow failed. Falling back to default backend...")
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                raise CameraError(
                    f"Unable to access camera (Index {self.device_index}). "
                    "Please ensure the webcam is connected, drivers are installed, "
                    "and no other application (Zoom, Teams, etc.) is currently using it."
                )

            # Configure video capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify by capturing a test frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise CameraError(
                    f"Camera at index {self.device_index} opened but failed to capture a frame. "
                    "Check camera privacy permissions in Windows Settings."
                )

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._logger.info(
                "Camera %d successfully initialized (Resolution: %dx%d).",
                self.device_index,
                actual_w,
                actual_h
            )

        except Exception as exc:
            self.release()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Unexpected error while initializing webcam: {exc}") from exc

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the active video capture stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - True and the BGR frame if capture succeeded.
                - False and None if reading failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Release the camera hardware and free underlying system resources."""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    self._logger.info("Camera %d released successfully.", self.device_index)
            except Exception as e:
                self._logger.error("Error releasing camera: %s", e)
            finally:
                self.cap = None

    def run_preview(self, window_name: str = "Webcam Live Stream") -> None:
        """
        Display a live, smooth video stream window.

        Exit options:
            - Press ESC key (ASCII 27)
            - Press 'q' key
            - Close the window

        Args:
            window_name: Title of the OpenCV display window.
        """
        try:
            self.start()
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            prev_time = time.time()
            fps_display = 0.0

            print("\n" + "=" * 50)
            print(f"  Live Webcam Preview Active: '{window_name}'")
            print("  - Press [ESC] or [q] to stop and exit.")
            print("=" * 50 + "\n")

            while True:
                success, frame = self.read_frame()
                if not success or frame is None:
                    self._logger.warning("Missed video frame. Attempting recovery...")
                    time.sleep(0.01)
                    continue

                # Calculate smooth FPS display
                curr_time = time.time()
                delta = curr_time - prev_time
                prev_time = curr_time
                if delta > 0.0:
                    fps_inst = 1.0 / delta
                    fps_display = (0.9 * fps_display) + (0.1 * fps_inst) if fps_display > 0 else fps_inst

                # Draw diagnostic on-screen HUD
                cv2.putText(
                    frame,
                    f"FPS: {int(fps_display)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    "Press ESC to Quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )

                # Render frame
                cv2.imshow(window_name, frame)

                # Wait 1ms for key press; check for ESC (27) or 'q'
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    self._logger.info("Exit requested via keypress (Keycode: %d).", key)
                    break

                # Also detect if user clicked the window's close (X) button
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._logger.info("Window close button clicked.")
                    break

        except CameraError as err:
            print(f"\n[ERROR] Camera failure: {err}\n", file=sys.stderr)
        except KeyboardInterrupt:
            self._logger.info("Preview interrupted by user.")
        finally:
            self.release()
            cv2.destroyAllWindows()
            self._logger.info("Window destroyed and cleanup completed.")

    def __enter__(self) -> "CameraManager":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit guaranteed cleanup."""
        self.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    cam = CameraManager()
    cam.run_preview()
