# Greeto - Dependency & Hardware Compatibility Matrix

- Windows column: checked on the development machine (Python 3.11.1, `.venv`).
- Jetson column: **never executed on hardware.** It records Python 3.8.10 /
  aarch64 wheel availability, probed with
  `pip download --no-deps --python-version 3.8 --implementation cp --abi cp38 --only-binary=:all: --platform manylinux2014_aarch64`.
- "Wheel exists" is a packaging fact. It is **not** proof that the package
  works on JetPack 5 with CUDA 11.4.

Status values: `VERIFIED`, `INSTALLED`, `NOT INSTALLED`, `INCOMPATIBLE`,
`REQUIRES HARDWARE TEST`, `NOT REQUIRED`, `UNKNOWN`.

---

## 1. Matrix

| Package | Purpose | Windows | Jetson | Status |
|---|---|---|---|---|
| PyQt5 | GUI | 5.15.11 installed, imported, window rendered | No cp38/aarch64 wheel (verified: none). Must come from `apt install python3-pyqt5`. Venv needs `--system-site-packages` | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| PyQt5-Qt5 / PyQt5_sip | Qt runtime + bindings | 5.15.2 / 12.19.0 (transitive) | Supplied by apt | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| OpenCV (`cv2`) | Camera processing | Effective `cv2.__version__ == 4.11.0`; **two** distributions installed (`opencv-contrib-python==4.11.0.86`, `opencv-python==5.0.0.93`) | `python3-opencv` from apt (L4T/GStreamer build). pip `opencv-python==4.10.0.84` also has a cp38/aarch64 wheel | Windows INSTALLED (conflict) / Jetson REQUIRES HARDWARE TEST |
| NumPy | Numerical operations | 2.4.6 | Max **1.24.4** for Python 3.8 (1.26.4 has no cp38 build - verified). 2.x requires >= 3.9 | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| InsightFace | Face detection + embeddings | 1.0.1, `buffalo_l` loaded in 2.5 s | sdist only, compiles Cython/C++ on device | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| ONNX Runtime | InsightFace inference backend | 1.28.0 **and** onnxruntime-gpu 1.28.0 both installed in one directory; providers `Tensorrt/CUDA/CPU` | `onnxruntime==1.16.3` is the newest cp38/aarch64 wheel (1.19.2 has no cp38 build). GPU providers not in the PyPI aarch64 wheel | Windows INSTALLED (conflict) / Jetson REQUIRES HARDWARE TEST |
| ONNX | Model format support | 1.22.0 | `onnx==1.16.2` available for cp38/aarch64 | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| Whisper implementation (`pywhispercpp`) | Speech recognition | 1.5.0 with `ggml-tiny.en.bin` | sdist only (no cp38/aarch64 wheel); needs a C++ toolchain | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| Piper (`piper-tts`) | Text-to-speech | 1.6.0; app shells out to the `piper` executable | 1.6.0 not verified for 3.8; `piper-tts==1.2.0` has a cp38/aarch64 wheel. Only the `piper` binary matters to the app | Windows INSTALLED (`piper` on PATH unverified) / Jetson REQUIRES HARDWARE TEST |
| Local LLM runtime (`llama-cpp-python`) | LLM inference | 0.3.34, `n_gpu_layers=-1` | sdist only; must be built with `-DGGML_CUDA=on` against CUDA 11.4 | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| `webrtcvad` | Voice activity detection | 2.0.10 | sdist only; C extension, needs `build-essential` | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| `sounddevice` | Microphone capture | 0.5.5 | cp38/aarch64 wheels exist for 0.4.6 and 0.5.5; needs `portaudio19-dev` | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| PyYAML | Configuration | 6.0.3 | 6.0.1 cp38/aarch64 wheel verified | Windows VERIFIED / Jetson REQUIRES HARDWARE TEST |
| SciPy | Transitive (InsightFace) | 1.17.1 | Available for cp38/aarch64 | Windows INSTALLED / Jetson UNKNOWN |
| Pillow | Transitive (InsightFace) | 12.3.0 | Available for cp38/aarch64 | Windows INSTALLED / Jetson UNKNOWN |
| requests | Transitive (InsightFace model download) | 2.34.2 | Pure Python | Windows INSTALLED / Jetson UNKNOWN |
| `winsound` | WAV playback | stdlib, used | **Not available on Linux** - replaced by `aplay`/`paplay`/`ffplay` in `tts_service.py` | Windows VERIFIED / Jetson INCOMPATIBLE (handled) |
| `PySide6` | Former GUI toolkit | **Not installed** | Incompatible: no Python 3.8 support | NOT REQUIRED / INCOMPATIBLE |
| `torch` | - | Not installed | - | NOT REQUIRED (no source imports it) |
| `cv2.imshow` / highgui windows | - | Removed from the shutdown path | Not usable on a headless kiosk | NOT REQUIRED |

---

## 2. Python version requirements

| Constraint | Detail |
|---|---|
| Windows dev | 3.11.1 (`.venv`) |
| Jetson | 3.8.10, fixed by JetPack 5 / L4T R35.6.0 |
| Source compatibility | Verified statically for 3.8: every `app/**/*.py` parses with `ast.parse(feature_version=(3, 8))`, and both `@dataclass(slots=True)` uses were removed. PEP 604 `X \| None` annotations only appear in modules that have `from __future__ import annotations` |
| NumPy ceiling | 3.8 -> 1.24.x maximum |
| InsightFace / ORT | must pair a numpy 1.24-compatible build on Jetson |

---

## 3. CUDA / ARM64 / native build notes

| Item | Detail |
|---|---|
| CUDA | Windows: CUDA-capable GPU present (onnxruntime-gpu exposes CUDA/TensorRT). Jetson: CUDA 11.4 with JetPack 5 |
| ARM64 wheels | PyQt5 (none), InsightFace (none), pywhispercpp (none), webrtcvad (none), llama-cpp-python (none), onnxruntime-gpu (none) - all require apt or on-device compilation |
| Native compilation needed | InsightFace, pywhispercpp, webrtcvad, llama-cpp-python (+ optional CUDA), onnxruntime GPU provider |
| System packages | `python3-pyqt5`, `python3-pyqt5.qtsvg`, `python3-opencv`, `portaudio19-dev`, `build-essential`, `cmake`, `python3-dev`, `libopenblas-dev`, `liblapack-dev`, `alsa-utils` |
| Venv requirement | `python3 -m venv --system-site-packages ...` so the apt PyQt5 and OpenCV are importable. Without it, `import PyQt5` fails inside the venv |

---

## 4. Model-specific requirements

| Model | File | Runtime | Notes |
|---|---|---|---|
| LLM | `models/llm/gemma-3-1b-it-Q4_K_M.gguf` | llama.cpp | `n_ctx=1024`, `n_batch=128`, `n_gpu_layers=-1` (CPU fallback if the build has no CUDA) |
| STT | `models/whisper/ggml-tiny.en.bin` | pywhispercpp | 16 kHz mono float32 input; `n_threads=8` |
| TTS | `models/tts/en_US-lessac-medium.onnx` + `.onnx.json` | Piper executable | Requires the `piper` binary on `PATH` |
| Faces | `models/faces/<name>/*.jpg` -> `models/cache/face_embeddings.pkl` | InsightFace `buffalo_l` | Model pack is downloaded to `~/.insightface`, not shipped in the repository |
| Knowledge | `app/knowledge/*.md` | pure Python | 12 files / 355 sections |

---

## 5. Open compatibility questions

1. Does the Windows PyQt5-before-onnxruntime DLL conflict also occur on Jetson
   with apt PyQt5 + NVIDIA ONNX Runtime? Untested.
2. Which OpenCV distribution should be authoritative on Windows
   (`opencv-python` 5.0.0.93 vs `opencv-contrib-python` 4.11.0.86)?
3. Which ONNX Runtime should remain: the CPU package or the GPU package
   (both 1.28.0 currently share one directory)?
4. Does `piper-tts==1.6.0` work on Python 3.8, or must Jetson use 1.2.0?
5. Are the InsightFace `buffalo_l` embeddings byte-compatible between Windows
   (`insightface 1.0.1`) and Jetson (`0.7.3`)? The pickle cache in
   `models/cache/` may need regenerating.
