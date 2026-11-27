# Greeto - Environment Setup

Two environments must stay compatible: the Windows development machine and the
Jetson Xavier NX deployment target.

> Nothing in the Jetson section has been executed on hardware. Commands are
> labelled as verified (run on this machine) or **unverified**.

---

## 1. Windows (development)

### 1.1 Verified environment

| Item | Value |
|---|---|
| Python | 3.11.1 (`MSC v.1934 64 bit`) |
| Interpreter | `<project>/.venv/Scripts/python.exe` |
| pip | 26.1.2 |
| PyQt5 | 5.15.11 (Qt 5.15.2, sip 12.19.0) |
| OpenCV | `cv2.__version__ == 4.11.0` |
| NumPy | 2.4.6 |
| OS | Windows 10/11 |

The project's `.venv` is created at the repository root and is git-ignored.

### 1.2 Create / activate

```bash
# from the repository root
python -m venv .venv

# Git Bash
source .venv/Scripts/activate
# cmd.exe
.venv\Scripts\activate.bat
```

### 1.3 Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_windows.txt
```

`requirements_windows.txt` is UTF-8 and pins the versions that were actually
validated. The older `requirements.txt` in the repository root is UTF-16 and
stale; it is superseded and should not be used.

Two known environment conflicts are **not** resolved by this file (see
`docs/PROJECT_AUDIT.md`, V8):

- `opencv-python==5.0.0.93` and `opencv-contrib-python==4.11.0.86` are both
  installed. Only one OpenCV distribution may be present:
  `python -m pip uninstall opencv-python` (or the contrib variant).
- `onnxruntime==1.28.0` and `onnxruntime-gpu==1.28.0` share one package
  directory. Keep exactly one:
  `python -m pip uninstall onnxruntime-gpu` for CPU-only, or keep the GPU build
  and remove the plain one.

Do not run those uninstalls until the GPU requirement is decided.

### 1.4 Run

```bash
python -m app.main
```

Must be run from the repository root. Resource paths are now resolved against
the project root rather than the process working directory, but the
`-m` invocation itself needs the root on `sys.path`.

Expected console output (verified):

```
Opening camera...
Camera ready (1280x720 @ 30 FPS)
Backend services started.
GUI started.
```

Logs: `logs/system.log` (rotating, 1 MB x 5). The window opens maximised at
1700x900 with a 1400x800 minimum.

### 1.5 Common errors

| Symptom | Cause | Action |
|---|---|---|
| `ModuleNotFoundError: No module named 'PySide6'` | Stale code from before the PyQt5 migration | Update the tree; the migration is complete |
| `ModuleNotFoundError: No module named 'PyQt5'` | Wrong interpreter / venv not active | `python -c "import sys; print(sys.executable)"` must show `.venv` |
| `ImportError: Unable to import dependency onnxruntime.` | Real cause is `DLL load failed while importing onnxruntime_pybind11_state` - PyQt5 was imported before onnxruntime | Keep the import order in `app/main.py` (vision before Qt). Root cause still open (V4) |
| `FileNotFoundError: Configuration directory ... not found` | `config/` is missing | Restore `config/*.yaml` |
| `Failed to start Vision Service.` | No camera, or a wrong `index` in `config/camera.yaml` | Connect a camera or edit `index`. Exit code is 1 |
| `FileNotFoundError: LLM model not found` | `models/llm/*.gguf` missing | Restore the model file (git-ignored) |
| `FileNotFoundError: TTS model missing` | `models/tts/*.onnx` missing | Restore the Piper voice |
| Silent TTS / `FileNotFoundError: 'piper'` | The `piper` executable is not on `PATH` | Install `piper-tts` in the active venv |
| `QFontDatabase: Cannot find font directory` | Benign PyQt5 warning about bundled fonts | Ignore |

---

## 2. Jetson Xavier NX (deployment) - UNVERIFIED

Target: JetPack 5.x / L4T R35.6.0, Python 3.8.10, aarch64, CUDA 11.4.

### 2.1 System packages (apt)

```bash
sudo apt-get update
sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtsvg
sudo apt-get install -y python3-opencv
sudo apt-get install -y portaudio19-dev
sudo apt-get install -y build-essential cmake python3-dev
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y alsa-utils
```

Rationale: PyQt5 has **no** cp38/aarch64 wheel on PyPI, and apt OpenCV is built
for the L4T GStreamer stack. `alsa-utils` provides `aplay`, which
`app/speech/tts_service.py` uses to play the Piper output.

### 2.2 Virtual environment and PyQt5 visibility

The critical detail: a plain `python3 -m venv` does **not** see apt-installed
PyQt5 or OpenCV. Create the environment with system site packages:

```bash
cd /path/to/Greeto
python3 -m venv --system-site-packages .venv-jetson
source .venv-jetson/bin/activate
python -c "import PyQt5, cv2; print('PyQt5 + cv2 visible')"
```

Verification step (unverified on device) - both imports must succeed **before**
any pip install of a GUI or OpenCV package is attempted. Do not try to
`pip install PyQt5` as a fix: no wheel exists for this platform, and mixing a
pip Qt with the apt Qt produces the same class of DLL/symbol conflict seen on
Windows.

If an environment already exists without `--system-site-packages`, recreate it
rather than modifying or deleting the existing one in place.

### 2.3 Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_jetson.txt
```

Items in that file marked `[HARDWARE TEST]` (InsightFace, pywhispercpp,
webrtcvad, llama-cpp-python, onnxruntime) are source-only builds for this
platform.

### 2.4 LLM with CUDA (unverified)

`app/brain/llm.py` requests `n_gpu_layers=-1`. A CPU-only build ignores it and
still works, just slowly. For GPU inference:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  python -m pip install --no-cache-dir --no-binary :all: llama-cpp-python==0.3.34
```

### 2.5 Models

`models/` is git-ignored, so the device needs the files copied over:

- `models/llm/gemma-3-1b-it-Q4_K_M.gguf`
- `models/whisper/ggml-tiny.en.bin`
- `models/tts/en_US-lessac-medium.onnx` (+ `.onnx.json`)
- `models/faces/<name>/*.jpg`
- `models/cache/face_embeddings.pkl` (optional - regenerated if deleted)

InsightFace `buffalo_l` is **not** in the repository; it is fetched to
`~/.insightface` on first use. Copy it manually if the kiosk must be offline on
first boot.

If the pickled embeddings were generated on Windows with `insightface 1.0.1`,
regenerate the cache on the Jetson before trusting recognition results.

### 2.6 Run

```bash
cd /path/to/Greeto
source .venv-jetson/bin/activate
python -m app.main
```

### 2.7 Jetson checks that still require hardware

1. `import PyQt5` and `import cv2` inside the venv.
2. Camera open through V4L2 (`CameraManager` now uses `cv2.CAP_V4L2` on Linux).
3. Whether the PyQt5-before-onnxruntime DLL conflict reproduces.
4. Piper output through `aplay`.
5. Microphone capture through PortAudio.
6. llama.cpp CUDA build and token throughput on Xavier NX.
7. Full-screen kiosk behaviour under the display manager.
