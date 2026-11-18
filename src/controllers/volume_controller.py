"""
System Volume Control Module.

Integrates with Windows Core Audio APIs via Pycaw to provide touchless audio
control using hand gestures:
- Open Palm: Smoothly increase volume
- Closed Fist: Toggle Mute / Unmute (protected with cooldown & state latching)
- Pinch Gesture: Decrease / modulate volume directly mapped to thumb-index finger distance

Features:
- Real-time smooth interpolation filter to prevent sudden volume spikes
- Configurable sensitivity and responsiveness thresholds
- On-screen visual HUD rendering of volume percentage, level bar, and mute badge
- Full exception safety with automatic recovery
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import cv2
import numpy as np

from src.gesture_recognizer import GestureType


class VolumeAction(Enum):
    """Categorization of executed volume operations."""
    NONE = "Idle"
    INCREASE = "Increasing Volume (+)"
    DECREASE = "Decreasing Volume (-)"
    MUTE_TOGGLED = "Mute Toggled"
    PINCH_ADJUST = "Pinch Volume Modulated"


@dataclass
class VolumeState:
    """Current state of system volume and recent gesture actions."""
    volume_scalar: float           # 0.0 to 1.0
    volume_percent: int            # 0 to 100
    is_muted: bool                 # True if system audio is muted
    action: VolumeAction           # Current action triggered by gesture
    status_message: str            # Descriptive status for UI display


class VolumeControlError(Exception):
    """Raised when Windows Audio endpoint fails or is inaccessible."""
    pass


class VolumeController:
    """Manages system volume using Pycaw with smoothing and gesture mapping."""

    def __init__(
        self,
        sensitivity: float = 1.0,
        smoothing_factor: float = 0.30,
        mute_cooldown: float = 1.5,
        min_pinch_ratio: float = 0.08,
        max_pinch_ratio: float = 0.38
    ) -> None:
        """
        Initialize the Volume Controller.

        Args:
            sensitivity: Scaling factor for volume increments/decrements (0.2 to 3.0).
            smoothing_factor: Alpha for exponential smoothing (0.05 = very smooth, 1.0 = instant).
            mute_cooldown: Minimum elapsed seconds required between consecutive mute toggles.
            min_pinch_ratio: Finger pinch ratio mapped to 0% volume.
            max_pinch_ratio: Finger pinch ratio mapped to 100% volume.
        """
        self.sensitivity = max(0.1, min(3.0, sensitivity))
        self.smoothing_factor = max(0.05, min(1.0, smoothing_factor))
        self.mute_cooldown = max(0.5, mute_cooldown)
        self.min_pinch_ratio = min_pinch_ratio
        self.max_pinch_ratio = max_pinch_ratio

        self._logger = logging.getLogger(self.__class__.__name__)
        self.volume_endpoint = None
        self._target_scalar: float = 0.5
        self._current_scalar: float = 0.5
        self._last_applied_scalar: float = -1.0
        self._last_mute_time: float = 0.0
        self._fist_held: bool = False
        self._last_step_time: float = time.time()

        self._init_endpoint()

    def _init_endpoint(self) -> None:
        """Initialize Windows Core Audio volume interface."""
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            from ctypes import cast, POINTER

            devices = AudioUtilities.GetSpeakers()
            if devices is None:
                raise VolumeControlError("No active audio output device detected.")

            if hasattr(devices, "EndpointVolume"):
                self.volume_endpoint = devices.EndpointVolume
            else:
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume_endpoint = cast(interface, POINTER(IAudioEndpointVolume))

            # Query initial values
            self._current_scalar = float(self.volume_endpoint.GetMasterVolumeLevelScalar())
            self._target_scalar = self._current_scalar
            self._logger.info(
                "Windows Audio Endpoint ready. Initial Volume: %d%%",
                int(self._current_scalar * 100)
            )
        except Exception as exc:
            self.volume_endpoint = None
            raise VolumeControlError(f"Failed to access Windows Core Audio: {exc}") from exc

    def get_volume_scalar(self) -> float:
        """Return current volume scalar [0.0, 1.0]."""
        if self.volume_endpoint is None:
            return self._current_scalar
        try:
            return float(self.volume_endpoint.GetMasterVolumeLevelScalar())
        except Exception:
            return self._current_scalar

    def get_volume_percent(self) -> int:
        """Return current volume percentage [0, 100]."""
        return int(round(self.get_volume_scalar() * 100.0))

    def is_muted(self) -> bool:
        """Return whether system audio is currently muted."""
        if self.volume_endpoint is None:
            return False
        try:
            return bool(self.volume_endpoint.GetMute())
        except Exception:
            return False

    def set_volume_scalar(self, target: float, smooth: bool = True) -> float:
        """
        Set master volume scalar with anti-jump smoothing.

        Args:
            target: Desired target volume scalar [0.0, 1.0].
            smooth: If True, exponentially interpolate towards target to prevent spikes.

        Returns:
            Updated volume scalar.
        """
        target = float(np.clip(target, 0.0, 1.0))
        self._target_scalar = target

        if smooth:
            # Exponential smoothing: current = current + alpha * (target - current)
            self._current_scalar += self.smoothing_factor * (self._target_scalar - self._current_scalar)
        else:
            self._current_scalar = self._target_scalar

        self._current_scalar = float(np.clip(self._current_scalar, 0.0, 1.0))

        # Avoid redundant COM dispatch if difference is negligible (< 0.004)
        if abs(self._current_scalar - self._last_applied_scalar) >= 0.004:
            if self.volume_endpoint is not None:
                try:
                    self.volume_endpoint.SetMasterVolumeLevelScalar(self._current_scalar, None)
                    self._last_applied_scalar = self._current_scalar
                except Exception as exc:
                    self._logger.error("Error setting volume scalar: %s", exc)

        return self._current_scalar

    def set_volume_percent(self, target_percent: float, smooth: bool = True) -> int:
        """Set volume by percentage [0, 100]."""
        scalar = target_percent / 100.0
        new_scalar = self.set_volume_scalar(scalar, smooth=smooth)
        return int(round(new_scalar * 100.0))

    def increase_volume(self, base_step: float = 0.015) -> int:
        """
        Smoothly increase volume by a calibrated step.

        Args:
            base_step: Base fractional increment before sensitivity scaling.

        Returns:
            Updated volume percentage.
        """
        # If currently muted, un-mute on volume increase
        if self.is_muted():
            self._set_mute_raw(False)

        delta = base_step * self.sensitivity
        return self.set_volume_percent((self._target_scalar + delta) * 100.0, smooth=True)

    def decrease_volume(self, base_step: float = 0.015) -> int:
        """
        Smoothly decrease volume by a calibrated step.

        Args:
            base_step: Base fractional decrement before sensitivity scaling.

        Returns:
            Updated volume percentage.
        """
        delta = base_step * self.sensitivity
        return self.set_volume_percent((self._target_scalar - delta) * 100.0, smooth=True)

    def _set_mute_raw(self, mute_state: bool) -> None:
        """Internal helper to set mute without cooldown checks."""
        if self.volume_endpoint is not None:
            try:
                self.volume_endpoint.SetMute(bool(mute_state), None)
            except Exception as exc:
                self._logger.error("Error setting mute state: %s", exc)

    def toggle_mute(self) -> Tuple[bool, bool]:
        """
        Toggle mute state with cooldown protection and state latching.

        Returns:
            Tuple[bool, bool]: (toggled_successfully, is_now_muted)
        """
        now = time.time()
        if now - self._last_mute_time < self.mute_cooldown:
            return False, self.is_muted()

        current_mute = self.is_muted()
        new_mute = not current_mute
        self._set_mute_raw(new_mute)
        self._last_mute_time = now
        self._logger.info("System mute toggled -> %s", "MUTED" if new_mute else "UNMUTED")
        return True, new_mute

    def set_volume_by_pinch_distance(self, pinch_ratio: float) -> int:
        """
        Directly map thumb-index finger pinch distance ratio to volume percentage (0–100%).
        As finger distance decreases, volume smoothly decreases down to 0%.

        Args:
            pinch_ratio: Normalized pinch distance ratio (from GestureRecognizer).

        Returns:
            Computed volume percentage.
        """
        # Map pinch_ratio between min_pinch_ratio (0%) and max_pinch_ratio (100%)
        target_pct = float(
            np.interp(
                pinch_ratio,
                [self.min_pinch_ratio, self.max_pinch_ratio],
                [0.0, 100.0]
            )
        )
        # Apply smoothing to eliminate jitter from minor hand tremors
        return self.set_volume_percent(target_pct, smooth=True)

    def process_gesture(
        self,
        gesture: GestureType,
        pinch_ratio: float = 0.0
    ) -> VolumeState:
        """
        High-level dispatcher mapping gestures to volume actions.

        Mapping:
        - Open Palm: Increase volume
        - Closed Fist: Mute / Unmute (cooldown protected)
        - Pinch Gesture: Decrease volume based on pinch distance

        Args:
            gesture: The detected stable GestureType.
            pinch_ratio: Normalized thumb-to-index distance ratio.

        Returns:
            VolumeState snapshot.
        """
        action = VolumeAction.NONE
        status_msg = "Ready"

        now = time.time()
        elapsed = now - self._last_step_time
        self._last_step_time = now

        # 1. Closed Fist -> Mute / Unmute
        if gesture == GestureType.CLOSED_FIST:
            if not self._fist_held:
                toggled, is_muted = self.toggle_mute()
                if toggled:
                    action = VolumeAction.MUTE_TOGGLED
                    status_msg = "Muted" if is_muted else "Unmuted"
                else:
                    cooldown_remaining = max(0.0, self.mute_cooldown - (now - self._last_mute_time))
                    status_msg = f"Mute Cooldown ({cooldown_remaining:.1f}s)"
                self._fist_held = True
            else:
                status_msg = "Fist Held (Mute Locked)"

        # Reset fist latch when hand is not in a closed fist
        if gesture != GestureType.CLOSED_FIST:
            self._fist_held = False

        # 2. Open Palm -> Increase Volume
        if gesture == GestureType.OPEN_PALM:
            new_vol = self.increase_volume(base_step=0.015)
            action = VolumeAction.INCREASE
            status_msg = f"Increasing Volume -> {new_vol}%"

        # 3. Pinch Gesture -> Modulate / Decrease volume based on distance
        elif gesture == GestureType.PINCH:
            new_vol = self.set_volume_by_pinch_distance(pinch_ratio)
            action = VolumeAction.PINCH_ADJUST
            status_msg = f"Pinch Distance Volume -> {new_vol}%"

        current_scalar = self.get_volume_scalar()
        current_pct = int(round(current_scalar * 100.0))
        muted = self.is_muted()

        return VolumeState(
            volume_scalar=current_scalar,
            volume_percent=current_pct,
            is_muted=muted,
            action=action,
            status_message=status_msg
        )

    @staticmethod
    def draw_volume_hud(
        frame: np.ndarray,
        state: VolumeState,
        origin: Tuple[int, int] = (50, 180)
    ) -> np.ndarray:
        """
        Draw a vertical volume gauge and status badge on the OpenCV frame.

        Args:
            frame: Video frame.
            state: VolumeState from process_gesture().
            origin: (x, y) top-left coordinates of the vertical volume bar.

        Returns:
            Annotated frame.
        """
        x, y = origin
        bar_w, bar_h = 28, 160

        # Semi-transparent background card for volume panel
        card_x1, card_y1 = x - 20, y - 45
        card_x2, card_y2 = x + bar_w + 130, y + bar_h + 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (80, 80, 80), 1)

        # Title Header
        cv2.putText(
            frame,
            "SYSTEM VOLUME",
            (x - 10, y - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Draw volume vertical outline
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (120, 120, 120), 2)

        # Calculate fill height based on percentage
        pct = state.volume_percent
        fill_h = int((pct / 100.0) * bar_h)
        fill_y = y + bar_h - fill_h

        # Fill color: Red if muted, gradient green/yellow/red otherwise
        if state.is_muted:
            bar_color = (0, 0, 255)  # Red when muted
        elif pct > 60:
            bar_color = (0, 255, 0)  # Green
        elif pct > 25:
            bar_color = (0, 255, 255)  # Yellow
        else:
            bar_color = (0, 165, 255)  # Orange

        # Fill bar
        if fill_h > 0 and not state.is_muted:
            cv2.rectangle(frame, (x + 2, fill_y), (x + bar_w - 2, y + bar_h - 2), bar_color, -1)

        # Percentage label next to bar
        pct_text = f"{pct}%"
        cv2.putText(
            frame,
            pct_text,
            (x + bar_w + 12, y + int(bar_h / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            bar_color,
            2,
            cv2.LINE_AA
        )

        # Mute Badge
        if state.is_muted:
            cv2.putText(
                frame,
                "MUTED",
                (x + bar_w + 12, y + int(bar_h / 2) + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame,
                "ACTIVE",
                (x + bar_w + 12, y + int(bar_h / 2) + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 100),
                1,
                cv2.LINE_AA
            )

        # Bottom action description
        cv2.putText(
            frame,
            state.status_message,
            (x - 10, y + bar_h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        return frame
