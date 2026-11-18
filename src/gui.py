"""
Graphical User Interface for Hand Gesture Controlled System.

Built using CustomTkinter with a modern, cute pastel dashboard theme.
Features:
- Separate multi-threaded video processing ensuring silky-smooth UI responsiveness
- Start & Stop Webcam buttons with hardware lifecycle management
- Live webcam video canvas with hand landmark tracking overlay
- Real-time telemetry:
    * Detected Gesture & confidence bar
    * Active Operating Mode indicator
    * System Volume percentage & Mute badge
    * Virtual Mouse action & screen coordinates
    * Navigation command & sub-mode status
- Sensitivity adjustment sliders for mouse, smoothing, and volume
- Individual module toggle switches (Mouse, Volume, Navigation)
- Graceful shutdown and resource cleanup
"""

import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk

from config import CONFIG
from src.camera import CameraManager, CameraError
from src.hand_detection import HandDetector, HandDetectorError
from src.gesture_recognizer import GestureRecognizer, GestureType, GestureResult
from src.controllers.volume_controller import VolumeController, VolumeState, VolumeAction
from src.controllers.mouse_controller import MouseController, MouseStatus, MouseAction
from src.controllers.navigation_controller import NavigationController, NavigationState, NavigationMode, NavigationAction


# Pastel & Cute Color Palette
PALETTE = {
    "bg_dark": "#161324",            # Midnight lavender dark background
    "card_bg": "#211D36",            # Soft purple card background
    "card_border": "#3B335E",        # Subtle lavender card border
    "accent_pink": "#FF79C6",        # Sakura rose pink accent
    "accent_purple": "#BD93F9",      # Soft lavender accent
    "accent_mint": "#50FA7B",        # Cute pastel mint green
    "accent_cyan": "#8BE9FD",        # Baby sky cyan
    "accent_coral": "#FF5555",       # Soft strawberry coral for stops/mutes
    "accent_peach": "#FFB86C",       # Soft warm peach
    "text_main": "#F8F8F2",          # Bright cream white
    "text_muted": "#B3A9D6",         # Muted pastel lavender text
}

# Cute gesture emoji mappings
GESTURE_EMOJIS = {
    GestureType.NONE: ("•", "None"),
    GestureType.OPEN_PALM: ("🖐️", "Open Palm"),
    GestureType.CLOSED_FIST: ("✊", "Closed Fist"),
    GestureType.THUMBS_UP: ("👍", "Thumbs Up"),
    GestureType.THUMBS_DOWN: ("👎", "Thumbs Down"),
    GestureType.INDEX_UP: ("👆", "Pointing Up"),
    GestureType.VICTORY: ("✌️", "Victory Sign"),
    GestureType.PINCH: ("🤏", "Pinch Gesture"),
}


@dataclass
class TelemetryData:
    """Thread-safe snapshot of video processing pipeline state."""
    fps: float = 0.0
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    volume_percent: int = 0
    is_muted: bool = False
    volume_action: str = "Idle"
    mouse_action: str = "Idle"
    mouse_x: int = 0
    mouse_y: int = 0
    is_dragging: bool = False
    nav_mode: str = "Presentation"
    nav_action: str = "Idle"


class HandGestureGUI(ctk.CTk):
    """Main application window using CustomTkinter."""

    def __init__(self) -> None:
        super().__init__()

        # Window Configuration
        self.title("🌸 Hand Gesture Studio - Touchless Control Hub")
        self.geometry("1280x820")
        self.minsize(1100, 720)
        self.configure(fg_color=PALETTE["bg_dark"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._logger = logging.getLogger("HandGestureGUI")

        # System State
        self.is_camera_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._latest_frame_bgr = None
        self.telemetry = TelemetryData()

        # Module Toggles
        self.module_mouse_enabled = True
        self.module_volume_enabled = True
        self.module_nav_enabled = True

        # Active Mode Selection
        self.active_mode = "Virtual Mouse"

        # Controllers & Models
        self.camera: Optional[CameraManager] = None
        self.detector: Optional[HandDetector] = None
        self.recognizer = GestureRecognizer(history_length=6, stability_threshold=0.65)
        self.mouse_ctrl: Optional[MouseController] = None
        self.volume_ctrl: Optional[VolumeController] = None
        self.nav_ctrl: Optional[NavigationController] = None

        self._init_controllers()
        self._build_ui()

        self._update_job = None
        # Start periodic GUI telemetry refresh loop (every 25ms ~ 40fps)
        self._update_job = self.after(25, self._gui_update_loop)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.exit_application)

    def _init_controllers(self) -> None:
        """Instantiate controllers safely."""
        try:
            self.mouse_ctrl = MouseController(
                frame_reduction=CONFIG.mouse.frame_reduction,
                smoothening=CONFIG.mouse.smoothening,
                sensitivity=1.1
            )
        except Exception as e:
            self._logger.warning("MouseController error: %s", e)

        try:
            self.volume_ctrl = VolumeController(
                sensitivity=1.2,
                smoothing_factor=0.25,
                mute_cooldown=1.5
            )
        except Exception as e:
            self._logger.warning("VolumeController error: %s", e)

        self.nav_ctrl = NavigationController(
            default_mode=NavigationMode.PRESENTATION,
            scroll_speed=45
        )

    def _build_ui(self) -> None:
        """Construct the polished dashboard layout."""
        # Main Grid Layout (3 columns: Left Sidebar, Live Preview, Right Telemetry)
        self.grid_columnconfigure(0, weight=0, minsize=300)  # Left Dock
        self.grid_columnconfigure(1, weight=1)               # Center Video
        self.grid_columnconfigure(2, weight=0, minsize=320)  # Right Telemetry
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # 1. LEFT SIDEBAR: Controls & Settings
        # ==========================================
        left_frame = ctk.CTkFrame(
            self,
            fg_color=PALETTE["card_bg"],
            corner_radius=18,
            border_width=1,
            border_color=PALETTE["card_border"]
        )
        left_frame.grid(row=0, column=0, padx=(18, 10), pady=18, sticky="nsew")

        # Cute Title Banner
        app_title = ctk.CTkLabel(
            left_frame,
            text="🌸 Gesture Studio",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=PALETTE["accent_pink"]
        )
        app_title.pack(padx=16, pady=(18, 2), anchor="w")

        subtitle = ctk.CTkLabel(
            left_frame,
            text="Touchless Computer Control Hub",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["text_muted"]
        )
        subtitle.pack(padx=16, pady=(0, 16), anchor="w")

        # --- Webcam Hardware Controls ---
        webcam_section = ctk.CTkFrame(left_frame, fg_color="transparent")
        webcam_section.pack(fill="x", padx=16, pady=(0, 12))

        self.btn_start_cam = ctk.CTkButton(
            webcam_section,
            text="▶ Start Webcam",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=PALETTE["accent_mint"],
            text_color="#0D2818",
            hover_color="#70FFA0",
            corner_radius=12,
            height=38,
            command=self.start_webcam
        )
        self.btn_start_cam.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_stop_cam = ctk.CTkButton(
            webcam_section,
            text="⏹ Stop",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=PALETTE["accent_coral"],
            text_color="#FFFFFF",
            hover_color="#FF7777",
            corner_radius=12,
            height=38,
            state="disabled",
            command=self.stop_webcam
        )
        self.btn_stop_cam.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # --- Active Operating Mode Selector ---
        mode_label = ctk.CTkLabel(
            left_frame,
            text="🎯 Active Operating Mode",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        mode_label.pack(padx=16, pady=(10, 4), anchor="w")

        self.mode_selector = ctk.CTkSegmentedButton(
            left_frame,
            values=["Virtual Mouse", "Volume", "Navigation"],
            font=ctk.CTkFont(size=12, weight="bold"),
            selected_color=PALETTE["accent_purple"],
            selected_hover_color=PALETTE["accent_pink"],
            unselected_color="#2E284A",
            corner_radius=10,
            command=self._on_mode_selected
        )
        self.mode_selector.set("Virtual Mouse")
        self.mode_selector.pack(padx=16, pady=(0, 14), fill="x")

        # --- Module Enable/Disable Switches ---
        mod_label = ctk.CTkLabel(
            left_frame,
            text="🔌 Enable / Disable Modules",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        mod_label.pack(padx=16, pady=(6, 4), anchor="w")

        self.sw_mouse = ctk.CTkSwitch(
            left_frame,
            text="Virtual Mouse Module",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["text_main"],
            progress_color=PALETTE["accent_pink"],
            command=self._on_module_toggle
        )
        self.sw_mouse.select()
        self.sw_mouse.pack(padx=20, pady=4, anchor="w")

        self.sw_volume = ctk.CTkSwitch(
            left_frame,
            text="Volume Control Module",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["text_main"],
            progress_color=PALETTE["accent_purple"],
            command=self._on_module_toggle
        )
        self.sw_volume.select()
        self.sw_volume.pack(padx=20, pady=4, anchor="w")

        self.sw_nav = ctk.CTkSwitch(
            left_frame,
            text="Navigation Module",
            font=ctk.CTkFont(size=12),
            text_color=PALETTE["text_main"],
            progress_color=PALETTE["accent_cyan"],
            command=self._on_module_toggle
        )
        self.sw_nav.select()
        self.sw_nav.pack(padx=20, pady=4, anchor="w")

        # --- Sensitivity Sliders ---
        sens_heading = ctk.CTkLabel(
            left_frame,
            text="🎛️ Sensitivity Adjustments",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        sens_heading.pack(padx=16, pady=(14, 2), anchor="w")

        # Mouse Sensitivity Slider
        self.lbl_mouse_sens = ctk.CTkLabel(
            left_frame,
            text="Mouse Sensitivity: 1.1x",
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["text_muted"]
        )
        self.lbl_mouse_sens.pack(padx=18, pady=(4, 0), anchor="w")
        self.slider_mouse_sens = ctk.CTkSlider(
            left_frame,
            from_=0.4,
            to=2.5,
            number_of_steps=21,
            progress_color=PALETTE["accent_pink"],
            button_color=PALETTE["accent_pink"],
            command=self._on_mouse_sens_change
        )
        self.slider_mouse_sens.set(1.1)
        self.slider_mouse_sens.pack(padx=16, pady=(0, 6), fill="x")

        # Mouse Smoothing Slider
        self.lbl_smoothing = ctk.CTkLabel(
            left_frame,
            text="Cursor Smoothing: 4.5",
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["text_muted"]
        )
        self.lbl_smoothing.pack(padx=18, pady=(4, 0), anchor="w")
        self.slider_smoothing = ctk.CTkSlider(
            left_frame,
            from_=1.0,
            to=10.0,
            number_of_steps=18,
            progress_color=PALETTE["accent_purple"],
            button_color=PALETTE["accent_purple"],
            command=self._on_smoothing_change
        )
        self.slider_smoothing.set(4.5)
        self.slider_smoothing.pack(padx=16, pady=(0, 6), fill="x")

        # Volume Sensitivity Slider
        self.lbl_vol_sens = ctk.CTkLabel(
            left_frame,
            text="Volume Sensitivity: 1.2x",
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["text_muted"]
        )
        self.lbl_vol_sens.pack(padx=18, pady=(4, 0), anchor="w")
        self.slider_vol_sens = ctk.CTkSlider(
            left_frame,
            from_=0.4,
            to=2.5,
            number_of_steps=21,
            progress_color=PALETTE["accent_cyan"],
            button_color=PALETTE["accent_cyan"],
            command=self._on_vol_sens_change
        )
        self.slider_vol_sens.set(1.2)
        self.slider_vol_sens.pack(padx=16, pady=(0, 14), fill="x")

        # Spacer to push Exit button to bottom
        spacer = ctk.CTkLabel(left_frame, text="")
        spacer.pack(expand=True, fill="both")

        # Exit Button
        btn_exit = ctk.CTkButton(
            left_frame,
            text="🚪 Exit Application",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#362E4A",
            text_color="#FF8A8A",
            hover_color="#522432",
            corner_radius=12,
            height=38,
            command=self.exit_application
        )
        btn_exit.pack(padx=16, pady=(0, 16), fill="x")

        # ==========================================
        # 2. CENTER PANEL: Live Webcam Preview
        # ==========================================
        center_frame = ctk.CTkFrame(
            self,
            fg_color=PALETTE["card_bg"],
            corner_radius=18,
            border_width=1,
            border_color=PALETTE["card_border"]
        )
        center_frame.grid(row=0, column=1, padx=8, pady=18, sticky="nsew")
        center_frame.grid_rowconfigure(1, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)

        # Video Header with Live Pill
        header_bar = ctk.CTkFrame(center_frame, fg_color="transparent")
        header_bar.grid(row=0, column=0, padx=16, pady=(14, 8), sticky="ew")

        v_title = ctk.CTkLabel(
            header_bar,
            text="📷 Live Video Feed & Landmarks",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        v_title.pack(side="left")

        self.pill_status = ctk.CTkLabel(
            header_bar,
            text="● STANDBY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=PALETTE["text_muted"],
            fg_color="#2E284A",
            corner_radius=8,
            padx=10,
            pady=3
        )
        self.pill_status.pack(side="right")

        # Webcam Display Label
        self.video_canvas = ctk.CTkLabel(
            center_frame,
            text="Click 'Start Webcam' to begin touchless gesture control 🖐️",
            font=ctk.CTkFont(size=14),
            text_color=PALETTE["text_muted"],
            fg_color="#181528",
            corner_radius=14
        )
        self.video_canvas.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")

        # ==========================================
        # 3. RIGHT SIDEBAR: Telemetry & Status Cards
        # ==========================================
        right_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=PALETTE["card_bg"],
            corner_radius=18,
            border_width=1,
            border_color=PALETTE["card_border"]
        )
        right_frame.grid(row=0, column=2, padx=(10, 18), pady=18, sticky="nsew")

        t_title = ctk.CTkLabel(
            right_frame,
            text="📊 Real-Time Telemetry",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=PALETTE["accent_purple"]
        )
        t_title.pack(padx=12, pady=(12, 10), anchor="w")

        # --- Card A: Detected Gesture ---
        card_gesture = ctk.CTkFrame(right_frame, fg_color="#2A2445", corner_radius=14)
        card_gesture.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(
            card_gesture,
            text="DETECTED GESTURE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_pink"]
        ).pack(padx=14, pady=(10, 2), anchor="w")

        self.lbl_gesture_name = ctk.CTkLabel(
            card_gesture,
            text="✨ None",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        self.lbl_gesture_name.pack(padx=14, pady=(2, 4), anchor="w")

        self.bar_confidence = ctk.CTkProgressBar(
            card_gesture,
            progress_color=PALETTE["accent_mint"],
            fg_color="#1C1830",
            height=8,
            corner_radius=4
        )
        self.bar_confidence.set(0.0)
        self.bar_confidence.pack(fill="x", padx=14, pady=(2, 10))

        # --- Card B: Active Operating Mode ---
        card_mode = ctk.CTkFrame(right_frame, fg_color="#2A2445", corner_radius=14)
        card_mode.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(
            card_mode,
            text="CURRENT OPERATING MODE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_cyan"]
        ).pack(padx=14, pady=(10, 2), anchor="w")

        self.lbl_current_mode = ctk.CTkLabel(
            card_mode,
            text="🖱️ Virtual Mouse Mode",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        self.lbl_current_mode.pack(padx=14, pady=(2, 4), anchor="w")

        self.lbl_mode_state = ctk.CTkLabel(
            card_mode,
            text="● Module Status: ACTIVE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_mint"]
        )
        self.lbl_mode_state.pack(padx=14, pady=(0, 10), anchor="w")

        # --- Card C: System Volume Status ---
        card_vol = ctk.CTkFrame(right_frame, fg_color="#2A2445", corner_radius=14)
        card_vol.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(
            card_vol,
            text="SYSTEM VOLUME",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_peach"]
        ).pack(padx=14, pady=(10, 2), anchor="w")

        self.lbl_volume_pct = ctk.CTkLabel(
            card_vol,
            text="Volume: 50%",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        self.lbl_volume_pct.pack(padx=14, pady=(2, 4), anchor="w")

        self.bar_volume = ctk.CTkProgressBar(
            card_vol,
            progress_color=PALETTE["accent_peach"],
            fg_color="#1C1830",
            height=10,
            corner_radius=5
        )
        self.bar_volume.set(0.50)
        self.bar_volume.pack(fill="x", padx=14, pady=(2, 4))

        self.lbl_mute_badge = ctk.CTkLabel(
            card_vol,
            text="🔊 UNMUTED",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_mint"]
        )
        self.lbl_mute_badge.pack(padx=14, pady=(0, 10), anchor="w")

        # --- Card D: Virtual Mouse Telemetry ---
        card_mouse = ctk.CTkFrame(right_frame, fg_color="#2A2445", corner_radius=14)
        card_mouse.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(
            card_mouse,
            text="VIRTUAL MOUSE TELEMETRY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_pink"]
        ).pack(padx=14, pady=(10, 2), anchor="w")

        self.lbl_mouse_action = ctk.CTkLabel(
            card_mouse,
            text="State: Idle",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        self.lbl_mouse_action.pack(padx=14, pady=(2, 2), anchor="w")

        self.lbl_mouse_coords = ctk.CTkLabel(
            card_mouse,
            text="Screen: (0, 0)",
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["text_muted"]
        )
        self.lbl_mouse_coords.pack(padx=14, pady=(0, 10), anchor="w")

        # --- Card E: Navigation Command Status ---
        card_nav = ctk.CTkFrame(right_frame, fg_color="#2A2445", corner_radius=14)
        card_nav.pack(fill="x", padx=8, pady=(6, 12))

        ctk.CTkLabel(
            card_nav,
            text="NAVIGATION STATUS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=PALETTE["accent_cyan"]
        ).pack(padx=14, pady=(10, 2), anchor="w")

        self.lbl_nav_mode = ctk.CTkLabel(
            card_nav,
            text="Mode: Presentation (Slides)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=PALETTE["accent_cyan"]
        )
        self.lbl_nav_mode.pack(padx=14, pady=(2, 2), anchor="w")

        self.lbl_nav_action = ctk.CTkLabel(
            card_nav,
            text="Last Action: Idle",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=PALETTE["text_main"]
        )
        self.lbl_nav_action.pack(padx=14, pady=(2, 10), anchor="w")

    # ==========================================
    # Video Worker Thread & Frame Pipeline
    # ==========================================
    def start_webcam(self) -> None:
        """Launch background worker thread to process video and gestures."""
        if self.is_camera_running:
            return

        self._logger.info("Starting webcam feed...")
        self.is_camera_running = True
        self.btn_start_cam.configure(state="disabled")
        self.btn_stop_cam.configure(state="normal")
        self.pill_status.configure(
            text="● STREAMING LIVE",
            text_color="#0D2818",
            fg_color=PALETTE["accent_mint"]
        )

        self._worker_thread = threading.Thread(target=self._webcam_worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_webcam(self) -> None:
        """Halt background video thread and release camera."""
        if not self.is_camera_running:
            return

        self._logger.info("Stopping webcam feed...")
        self.is_camera_running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

        if self.camera:
            self.camera.release()

        self.btn_start_cam.configure(state="normal")
        self.btn_stop_cam.configure(state="disabled")
        self.pill_status.configure(
            text="● STOPPED",
            text_color=PALETTE["text_muted"],
            fg_color="#2E284A"
        )
        self.video_canvas.configure(
            image=None,
            text="Webcam stopped. Click 'Start Webcam' to resume 🖐️"
        )

    def _webcam_worker_loop(self) -> None:
        """Background thread executing capture, detection, recognition, and controller actions."""
        try:
            self.camera = CameraManager(
                device_index=CONFIG.camera.device_index,
                width=CONFIG.camera.frame_width,
                height=CONFIG.camera.frame_height,
                fps=CONFIG.camera.fps,
                flip_horizontal=CONFIG.camera.flip_horizontal
            )
            self.camera.start()

            self.detector = HandDetector(
                max_num_hands=1,
                min_detection_confidence=CONFIG.hand_detector.min_detection_confidence,
                min_tracking_confidence=CONFIG.hand_detector.min_tracking_confidence,
                mirror_mode=CONFIG.camera.flip_horizontal
            )

            prev_time = time.time()
            fps_smoothed = 0.0

            while self.is_camera_running:
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0.0:
                    fps_inst = 1.0 / dt
                    fps_smoothed = (0.85 * fps_smoothed) + (0.15 * fps_inst) if fps_smoothed > 0 else fps_inst

                cam_w, cam_h = frame.shape[1], frame.shape[0]

                # 1. Detect Hand Landmarks
                show_mouse_zone = (self.active_mode == "Virtual Mouse" and self.module_mouse_enabled)
                frame, detected_hands = self.detector.process(
                    frame,
                    draw_landmarks=True,
                    draw_bbox=not show_mouse_zone,
                    draw_ids=False
                )

                landmarks = []
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

                # Telemetry buffer
                t_gesture = gesture_res.gesture
                t_conf = gesture_res.confidence
                t_vol_pct = self.volume_ctrl.get_volume_percent() if self.volume_ctrl else 50
                t_is_muted = self.volume_ctrl.is_muted() if self.volume_ctrl else False
                t_vol_act = "Idle"
                t_mouse_act = "Idle"
                t_mouse_x = self.mouse_ctrl.current_screen_x if self.mouse_ctrl else 0
                t_mouse_y = self.mouse_ctrl.current_screen_y if self.mouse_ctrl else 0
                t_dragging = False
                t_nav_mode = self.nav_ctrl.mode.value if self.nav_ctrl else "Presentation"
                t_nav_act = "Idle"

                # 2. Execute ONLY the active mode's controller (Strict Isolation)
                if self.active_mode == "Virtual Mouse" and self.module_mouse_enabled and self.mouse_ctrl:
                    m_status = self.mouse_ctrl.process(
                        gesture=gesture_res.gesture,
                        landmarks=landmarks,
                        cam_w=cam_w,
                        cam_h=cam_h,
                        pinch_ratio=pinch_ratio
                    )
                    self.mouse_ctrl.draw_mouse_hud(frame, m_status, landmarks)
                    t_mouse_act = m_status.action.value
                    t_mouse_x = m_status.screen_x
                    t_mouse_y = m_status.screen_y
                    t_dragging = m_status.is_dragging

                elif self.active_mode == "Volume" and self.module_volume_enabled and self.volume_ctrl:
                    v_state = self.volume_ctrl.process_gesture(
                        gesture=gesture_res.gesture,
                        pinch_ratio=pinch_ratio
                    )
                    hud_origin = (cam_w - 170, 70)
                    VolumeController.draw_volume_hud(frame, v_state, origin=hud_origin)
                    t_vol_pct = v_state.volume_percent
                    t_is_muted = v_state.is_muted
                    t_vol_act = v_state.status_message

                elif self.active_mode == "Navigation" and self.module_nav_enabled and self.nav_ctrl:
                    n_state = self.nav_ctrl.process(
                        gesture=gesture_res.gesture,
                        landmarks=landmarks
                    )
                    nav_hud_origin = (20, cam_h - 120)
                    NavigationController.draw_navigation_hud(frame, n_state, origin=nav_hud_origin)
                    t_nav_mode = n_state.mode.value
                    t_nav_act = n_state.action.value

                # Update shared thread-safe telemetry & frame
                with self._thread_lock:
                    self._latest_frame_bgr = frame
                    self.telemetry = TelemetryData(
                        fps=fps_smoothed,
                        gesture=t_gesture,
                        confidence=t_conf,
                        volume_percent=t_vol_pct,
                        is_muted=t_is_muted,
                        volume_action=t_vol_act,
                        mouse_action=t_mouse_act,
                        mouse_x=t_mouse_x,
                        mouse_y=t_mouse_y,
                        is_dragging=t_dragging,
                        nav_mode=t_nav_mode,
                        nav_action=t_nav_act
                    )

        except Exception as exc:
            self._logger.exception("Exception in webcam worker thread: %s", exc)
        finally:
            if self.camera:
                self.camera.release()
            if self.detector:
                self.detector.close()

    # ==========================================
    # GUI Telemetry & Frame Update (Main Thread)
    # ==========================================
    def _gui_update_loop(self) -> None:
        """Periodic main-thread updater for frame rendering and telemetry widgets."""
        if self.is_camera_running:
            # 1. Render Video Frame on Canvas
            with self._thread_lock:
                frame_to_render = None
                if self._latest_frame_bgr is not None:
                    frame_to_render = self._latest_frame_bgr.copy()
                data = self.telemetry

            if frame_to_render is not None:
                # Resize to fit display label nicely
                lbl_w = max(400, self.video_canvas.winfo_width())
                lbl_h = max(300, self.video_canvas.winfo_height())

                # Maintain 4:3 camera aspect ratio
                target_ratio = 4.0 / 3.0
                if lbl_w / max(1, lbl_h) > target_ratio:
                    disp_h = lbl_h
                    disp_w = int(disp_h * target_ratio)
                else:
                    disp_w = lbl_w
                    disp_h = int(disp_w / target_ratio)

                resized = cv2.resize(frame_to_render, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

                self.video_canvas.configure(image=ctk_img, text="")

            # 2. Update Telemetry Widgets
            # Gesture Badge
            emoji, g_text = GESTURE_EMOJIS.get(data.gesture, ("✨", "None"))
            self.lbl_gesture_name.configure(text=f"{emoji} {g_text}")
            self.bar_confidence.set(data.confidence)

            # Volume Card
            self.lbl_volume_pct.configure(text=f"Volume: {data.volume_percent}%")
            self.bar_volume.set(data.volume_percent / 100.0)
            if data.is_muted:
                self.lbl_mute_badge.configure(text="🔇 MUTED", text_color=PALETTE["accent_coral"])
                self.bar_volume.configure(progress_color=PALETTE["accent_coral"])
            else:
                self.lbl_mute_badge.configure(text="🔊 UNMUTED", text_color=PALETTE["accent_mint"])
                self.bar_volume.configure(progress_color=PALETTE["accent_peach"])

            # Mouse Card
            self.lbl_mouse_action.configure(text=f"State: {data.mouse_action}")
            self.lbl_mouse_coords.configure(text=f"Screen: ({data.mouse_x}, {data.mouse_y})")

            # Navigation Card
            self.lbl_nav_mode.configure(text=f"Mode: {data.nav_mode}")
            self.lbl_nav_action.configure(text=f"Action: {data.nav_action}")

            # Header Status Pill
            self.pill_status.configure(text=f"● LIVE ({int(data.fps)} FPS)")

        self._update_job = self.after(25, self._gui_update_loop)

    # ==========================================
    # Interactive Event Handlers
    # ==========================================
    def _on_mode_selected(self, value: str) -> None:
        """Handle operating mode segment selection."""
        # Clean up mouse drag if switching away
        if self.active_mode == "Virtual Mouse" and self.mouse_ctrl:
            self.mouse_ctrl.stop_drag()

        self.active_mode = value
        self.lbl_current_mode.configure(text=f"🎯 {value} Mode")
        self._on_module_toggle()

    def _on_module_toggle(self) -> None:
        """Update module enable/disable states."""
        self.module_mouse_enabled = bool(self.sw_mouse.get())
        self.module_volume_enabled = bool(self.sw_volume.get())
        self.module_nav_enabled = bool(self.sw_nav.get())

        # Check if currently active mode is enabled
        active_enabled = True
        if self.active_mode == "Virtual Mouse":
            active_enabled = self.module_mouse_enabled
        elif self.active_mode == "Volume":
            active_enabled = self.module_volume_enabled
        elif self.active_mode == "Navigation":
            active_enabled = self.module_nav_enabled

        if active_enabled:
            self.lbl_mode_state.configure(text="● Module Status: ACTIVE", text_color=PALETTE["accent_mint"])
        else:
            self.lbl_mode_state.configure(text="● Module Status: DISABLED", text_color=PALETTE["accent_coral"])

    def _on_mouse_sens_change(self, value: float) -> None:
        """Update virtual mouse cursor sensitivity."""
        if self.mouse_ctrl:
            self.mouse_ctrl.sensitivity = value
        self.lbl_mouse_sens.configure(text=f"Mouse Sensitivity: {value:.1f}x")

    def _on_smoothing_change(self, value: float) -> None:
        """Update cursor smoothing factor."""
        if self.mouse_ctrl:
            self.mouse_ctrl.smoothening = value
        self.lbl_smoothing.configure(text=f"Cursor Smoothing: {value:.1f}")

    def _on_vol_sens_change(self, value: float) -> None:
        """Update volume modulation sensitivity."""
        if self.volume_ctrl:
            self.volume_ctrl.sensitivity = value
        self.lbl_vol_sens.configure(text=f"Volume Sensitivity: {value:.1f}x")

    def exit_application(self) -> None:
        """Cleanly terminate camera, background threads, and destroy window."""
        self._logger.info("Exiting Hand Gesture Studio...")
        self.stop_webcam()
        if hasattr(self, "_update_job") and self._update_job:
            try:
                self.after_cancel(self._update_job)
            except Exception:
                pass
            self._update_job = None
        try:
            self.destroy()
        except Exception:
            pass


def run_gui() -> int:
    """Launch the CustomTkinter GUI dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    app = HandGestureGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run_gui())
