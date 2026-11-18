# Hand Gesture Controlled System

A modular, real-time Computer Vision system allowing touchless computer interaction using hand gestures. Powered by OpenCV, MediaPipe, NumPy, PyAutoGUI, Pycaw, and CustomTkinter.

---

## 📁 Directory Structure

```text
HandGestureControlledSystem/
│
├── .venv/                          # Python isolated virtual environment
├── requirements.txt                # Third-party dependencies
├── config.py                       # Centralized application parameters and thresholds
├── run_gui.py                      # Dedicated launcher for CustomTkinter GUI Dashboard
├── main.py                         # Centralized application (OpenCV or CustomTkinter GUI via --gui)
├── test_webcam.py                  # Standalone webcam integration test script
├── test_hand_detection.py          # Standalone multi-hand detection & landmark test script
├── test_gestures.py                # Standalone real-time gesture recognition HUD runner
├── test_volume_control.py          # Standalone gesture volume control test runner
├── test_navigation.py              # Standalone gesture navigation test runner
├── test_virtual_mouse.py           # Standalone virtual mouse test runner
├── README.md                       # Documentation & setup guide
│
├── src/
│   ├── __init__.py                 # Source package initialization
│   ├── camera.py                   # Camera feed capture and OpenCV lifecycle management
│   ├── hand_detection.py           # Real-time multi-hand landmark detection & handedness
│   ├── hand_detector.py            # Backwards-compatible import wrapper
│   ├── gesture_recognizer.py       # Geometric calculation and gesture classification
│   ├── navigation.py               # Gesture navigation (Swipe, Scroll, Mode Switch)
│   ├── gui.py                      # CustomTkinter professional pastel dashboard
│   │
│   ├── controllers/
│   │   ├── __init__.py             # Controllers package export
│   │   ├── volume_controller.py    # Windows system audio control (Pycaw)
│   │   ├── mouse_controller.py     # Virtual mouse cursor, click, drag, scroll (PyAutoGUI)
│   │   └── navigation_controller.py# Re-export shim for navigation module
│   │
│   └── utils/
│       ├── __init__.py             # Utilities package export
│       └── fps_tracker.py          # Real-time FPS calculation and visual overlay
│
└── tests/
    ├── test_gestures_unit.py       # Unit tests verifying all 7 gestures individually
    ├── test_volume_unit.py         # Unit tests verifying all volume operations independently
    ├── test_navigation_unit.py     # Unit tests verifying all navigation operations independently
    ├── test_mouse_unit.py          # Unit tests verifying mouse movement, clicks, and dragging
    └── test_gui_unit.py            # Unit tests verifying GUI state, sliders, and switches
```

---

## 🛠️ Prerequisites

- **Operating System**: Windows 10/11
- **Python**: Version 3.10 to 3.12
- **Hardware**: Working webcam / integrated camera

---

## 🚀 Setup & Execution

### 1. Open Terminal or PowerShell
Navigate to the project root:
```powershell
cd d:\HandGestureControlledSystem
```

### 2. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
# (Or in cmd.exe: .\.venv\Scripts\activate.bat)
```

### 3. Launch the CustomTkinter GUI Dashboard
```powershell
python run_gui.py
```
*(Alternatively: `python main.py --gui`)*

---

## 🌸 CustomTkinter Dashboard Features

1. **Start / Stop Webcam**: Clean hardware lifecycle buttons with status pills (`● STREAMING LIVE` / `● STANDBY`).
2. **Live Webcam Canvas**: Real-time video preview with landmark connections and bounding boxes.
3. **Detected Gesture Card**: Large emoji + name badge (`🖐️ Open Palm`, `✊ Closed Fist`, `✌️ Victory Sign`, `👆 Pointing Up`, etc.) and stability progress bar.
4. **Operating Mode Switcher**:
   - `Virtual Mouse Mode`
   - `Volume Mode`
   - `Navigation Mode`
5. **System Volume Card**: Live volume percentage, animated volume progress bar, and dynamic mute status pill (`🔊 UNMUTED` / `🔇 MUTED`).
6. **Virtual Mouse Telemetry**: Real-time cursor coordinates `(x, y)` and active state (`Moving Cursor`, `Left Click`, `Double Click`, `Drag & Drop`, `Stopped`).
7. **Navigation Status Card**: Sub-mode display (`Presentation` vs. `Browser`) and last detected navigation action.
8. **Sensitivity Adjustment Sliders**:
   - Mouse Sensitivity ($0.4\times$ to $2.5\times$)
   - Cursor Smoothing factor ($1.0$ to $10.0$)
   - Volume Modulation Sensitivity ($0.4\times$ to $2.5\times$)
9. **Module Enable/Disable Switches**: Individual toggle switches to isolate or disable Mouse, Volume, or Navigation without restarting.
10. **Thread-Safe & Ultra-Responsive**: Video capture and MediaPipe processing run in a dedicated background worker thread, ensuring the GUI remains silky-smooth and never hangs.

---

## 🧪 Verification & Testing

Run the full automated test suite across all 30 tests:
```powershell
python -m unittest discover tests -v
```

Run independent module tests:
```powershell
python test_virtual_mouse.py    # Test mouse separately
python test_volume_control.py   # Test volume separately
python test_navigation.py       # Test navigation separately
python test_gestures.py         # Test 7-gesture classification HUD
python test_hand_detection.py   # Test multi-hand landmark detection
python test_webcam.py           # Test webcam integration
```
