# Greeto - Project Audit (Phase 0)

AI-powered offline Admission Receptionist for I.T.S Engineering College, Greater Noida.

- Audit date: 2026-11-17
- Branch audited: `test`
- Codebase state: single commit, working tree containing the developer's uncommitted edits
- Scope: audit, environment stabilisation, PyQt5 migration. No new features.

Everything in this document was checked against the source in this repository.
Claims are tagged **VERIFIED** (reproduced with a command) or **SUSPECTED**
(reasoned from code, not reproduced). Verified test output is in
`docs/PHASE_0_REPORT.md`.

---

## 1. Repository overview

| Area | Contents |
|---|---|
| Entry point | `app/main.py` (`python -m app.main`) |
| Core | `app/core/` - `container.py`, `event_bus.py`, `config_manager.py`, `state_manager.py`, `logging_setup.py`, `constants.py`, `exceptions.py` |
| Vision | `app/vision/` - 13 modules (camera, frame grabbing, InsightFace embeddings, recognition, tracking, visitor database, cache) |
| Speech | `app/speech/` - `audio_capture.py`, `vad.py`, `stt.py`, `tts_service.py`, `speech_service.py`, `models.py` |
| Brain | `app/brain/` - `conversation_manager.py`, `llm.py`, `knowledge_base.py`, `prompt_manager.py`, `parser.py`, `emotion.py` |
| GUI | `app/gui/` - `main_window.py`, `camera_widget.py`, `robot_widget.py`, `theme.py`, `chat_widget.py`, `status_widget.py`, `resources/` |
| Knowledge | `app/knowledge/` - 12 Markdown documents (355 parsed sections) |
| Config | `config/` - 7 YAML files |
| Models | `models/` - `llm/gemma-3-1b-it-Q4_K_M.gguf`, `whisper/ggml-tiny.en.bin`, `tts/en_US-lessac-medium.onnx`, `faces/Saksham/1.jpg`, `cache/face_embeddings.pkl` |
| Data | `data/visitors.json` |
| Empty scaffolding | `docs/` (used by this phase), `knowledge_base/`, `scripts/`, `tests/` - all empty |
| Dependency files | `requirements.txt` (stale, UTF-16), `requirements_windows.txt` (was UTF-16 freeze), `requirements_jetson.txt` (new) |

There is no `setup.py`/`pyproject.toml`; the project runs as a plain package from
the repository root.

---

## 2. Existing architecture

```
                      app/main.py
                           |
          +----------------+----------------+
          |                |                |
     ServiceContainer   QApplication    MainWindow
     (config, logger,       (PyQt5)          |
      event bus,                             +-- CameraWidget --+
      state)                                 +-- RobotWidget    |
          |                                  +-- GlassCard x2   |
          |                                                     |
   +------+--------+-------------+                              |
   |               |             |                              |
VisionService  SpeechService  ConversationManager               |
   |               |             |                              |
CameraManager  AudioCapture   KnowledgeBase                     |
FrameGrabber   VAD            PromptManager                     |
EmbeddingModel STT (whisper)  LocalLLM (llama.cpp)              |
FaceDatabase   TTSService     ResponseParser                    |
FaceRecognizer                                 |
VisitorTracker                                 |
VisitorDatabase ---------------------------> EventBus <----------+
```

### Service dependencies

| Service | Constructs | Reads config | Publishes | Subscribes |
|---|---|---|---|---|
| `ServiceContainer` | `ConfigManager`, `LoggerManager`, `EventBus`, `StateManager` | `logging` | - | - |
| `VisionService` | `CameraManager`, `FrameGrabber`, `EmbeddingModel`, `FaceDatabase`, `FaceRecognizer`, `VisitorTracker`, `VisitorDatabase` | `camera` (via `CameraManager`) | `visitor.detected`, `visitor.unknown` | - |
| `SpeechService` | `AudioCapture`, `VoiceActivityDetector`, `SpeechToText` | - | `speech_recognized` | - |
| `ConversationManager` | `LocalLLM`, `PromptManager`, `ResponseParser`, `KnowledgeBase`, `TTSService` | - | `llm_response_ready` | `speech_recognized` |
| `MainWindow` | `CameraWidget`, `RobotWidget`, `GlassCard` | - | - | polls `VisionService.process()` on a `QTimer` |

Flow: camera frame -> grabber thread -> `VisionService.process()` (called from the
Qt timer) -> detection -> recognition -> tracking -> events; microphone ->
VAD -> Whisper -> `speech_recognized` -> retrieval + LLM -> parser -> TTS.

Nothing subscribes to `visitor.detected` / `visitor.unknown` today, so greeting
by name is **not implemented yet** (the roadmap item exists only as events).

---

## 3. Main application entry point

`python -m app.main` -> `app/main.py:main()`.

Sequence after this phase:

1. `QApplication(sys.argv)`
2. `ServiceContainer()` -> loads `config/*.yaml`, creates the `System` logger
3. `VisionService`, `SpeechService`, `ConversationManager` are constructed
   (this is where InsightFace, Whisper and the Gemma GGUF are loaded)
4. `vision.start()` - opens the camera; returns exit code 1 if it fails
5. `speech.start()`
6. `MainWindow(vision=...).showMaximized()`
7. `app.exec_()` - single Qt event loop
8. `finally:` `speech.stop()`, `vision.stop()`

The command is unchanged and remains the only supported entry point.
`app/gui/main_window.py` also has a standalone `__main__` block.

---

## 4. Module descriptions

### Core
- `container.py` - service locator. Creates shared config/logger/event bus/state.
  No shutdown method, no service registry, no duplicate-instance protection.
- `event_bus.py` - thread-safe pub/sub with a `Lock` around the subscriber map.
  Callbacks run outside the lock. Per-callback exception isolation was added in
  this phase.
- `config_manager.py` - loads every `config/*.yaml` into a dict and reads nested
  keys. Now resolves the directory against the project root and returns a default
  for missing keys.
- `state_manager.py` - `RobotState` enum (14 states) plus a locked get/set. No
  transition validation.
- `logging_setup.py` - `RotatingFileHandler` + console handler for a named logger.
- `constants.py`, `exceptions.py` - empty files (0 bytes).

### Vision
- `camera_manager.py` - OpenCV capture; backend now chosen per platform.
- `frame_grabber.py` - background thread keeping only the newest frame.
- `embedding_model.py` - InsightFace `buffalo_l`, `CPUExecutionProvider`.
- `face_database.py` - loads `models/faces/<name>/*.jpg`, or the pickle cache.
- `face_cache.py` - pickle of the embedding dict.
- `face_recognizer.py` - cosine similarity, threshold 0.55.
- `similarity.py` - cosine similarity helper.
- `visitor_tracker.py` - centroid tracking, assigns stable `visitor_id`.
- `visitor_database.py` - JSON visitor statistics in `data/visitors.json`.
- `models.py` - `Face` dataclass.
- `vision_service.py` - orchestrates the pipeline and draws overlays.
- `motion_detector.py`, `enrollment.py` - empty files (0 bytes).

### Speech
- `audio_capture.py` - `sounddevice` input stream into a queue.
- `vad.py` - `webrtcvad` energy/speech decision.
- `stt.py` - `pywhispercpp` with `ggml-tiny.en.bin`.
- `tts_service.py` - shells out to the `piper` executable, then plays the WAV.
- `speech_service.py` - listen loop: VAD -> buffer -> silence -> transcribe -> event.
- `models.py` - empty file (0 bytes).

### Brain
- `knowledge_base.py` - splits Markdown into sections, naive keyword scoring, top 3.
- `prompt_manager.py` - forces a strict JSON reply contract.
- `llm.py` - `llama_cpp.Llama`, `n_gpu_layers=-1`, `n_ctx=1024`, `n_threads=8`.
- `parser.py` - extracts JSON, falls back to a handover response.
- `conversation_manager.py` - event subscriber, history window of 6, TTS call.
- `emotion.py` - empty file (0 bytes).

### GUI
- `main_window.py` - `GlassCard` + two-column layout (camera 3 : robot 2).
- `camera_widget.py` - `QLabel` refreshed by `QTimer(33ms)` from `vision.process()`.
- `robot_widget.py` - `QSvgWidget` showing `resources/robot.svg`; state setters are stubs.
- `theme.py` - one Qt stylesheet string.
- `chat_widget.py`, `status_widget.py` - empty files (0 bytes).
- `resources/` - `robot.svg` (used), `background.png` and `logo.png` (both 0 bytes).

---

## 5. Identified problems

### 5.1 VERIFIED problems (reproduced)

| # | Problem | Impact | Status |
|---|---|---|---|
| V1 | `PySide6` imports in `app/main.py`, `gui/main_window.py`, `gui/camera_widget.py`, `gui/robot_widget.py` | `import app.main` raised `ModuleNotFoundError: No module named 'PySide6'`; the application could not start at all on the PyQt5/Jetson stack | **Fixed** |
| V2 | `@dataclass(slots=True)` in `vision/models.py` and `vision/visitor_tracker.py` | `slots=True` is Python 3.10+. On Python 3.8 import raises `TypeError: dataclass() got an unexpected keyword argument 'slots'` | **Fixed** |
| V3 | `requirements.txt` and `requirements_windows.txt` were UTF-16 (`pip freeze >` in PowerShell) | `pip install -r` cannot parse them | `requirements_windows.txt` **rewritten as UTF-8**; `requirements.txt` left untouched and documented |
| V4 | PyQt5 loaded before `onnxruntime` breaks ORT on Windows: `ImportError: DLL load failed while importing onnxruntime_pybind11_state: A dynamic link library (DLL) initialization routine failed` | Every import of `app.main` failed. `insightface` swallows the real error and raises `Unable to import dependency onnxruntime.` Any PyQt5 submodule (even `QtCore`) triggers it; `os.add_dll_directory` does not help | **Mitigated** by importing the vision stack before Qt in both entry points; root cause open |
| V5 | Paths resolved against the process working directory: `config`, `models/faces`, `models/cache/...`, `data/visitors.json`, `models/whisper/...`, `models/tts/...`, `logs` | The app only worked when launched from the repository root - not acceptable for a systemd/kiosk launch on Jetson | **Fixed** (project-root-relative) |
| V6 | `cv2.VideoCapture(index, cv2.CAP_DSHOW)` hardcoded | DirectShow does not exist on Linux; the camera would never open on Xavier NX | **Fixed** (DShow on Windows, V4L2 on Linux, plain fallback) |
| V7 | TTS playback used `winsound` unconditionally | `ImportError: No module named 'winsound'` on Linux - no speech output on the device | **Fixed** (`winsound` on Windows, `aplay`/`paplay`/`ffplay` on Linux) |
| V8 | Conflicting duplicate packages in the Windows venv: `opencv-python==5.0.0.93` **and** `opencv-contrib-python==4.11.0.86`; `onnxruntime==1.28.0` **and** `onnxruntime-gpu==1.28.0` sharing one `onnxruntime/` directory | Effective `cv2` is 4.11.0 (whichever installed last wins); ORT's single directory holds a mix of two distributions. A plausible contributor to V4 | **Not fixed** - requires an environment change the developer must approve |
| V9 | `VisionService._publish_events` called `VisitorDatabase.register()` on every detection cycle (~10x/sec while a visitor is present), and `register()` rewrites the whole JSON file each time | Constant disk writes; `data/visitors.json` records `"visits": 1143` for one person, which is an artefact, not real traffic | **Fixed** (announced once per tracked visitor) |
| V10 | `AudioCapture._capture_loop` used `while self.running: pass` | Spins one CPU core at 100% for the whole session (serious on a 6-core Xavier NX) | **Fixed** (`time.sleep(0.05)`), microphone open failure now logged instead of silently killing the thread |
| V11 | `EventBus.publish` invoked callbacks with no error handling | One raising subscriber aborted the remaining subscribers and propagated into the publishing background thread, killing the listen loop | **Fixed** (per-callback try/except with logging) |
| V12 | `LoggerManager.create_logger` used `mkdir(exist_ok=True)` without `parents=True`; no `propagate = False` | Startup crash if `logs/` had no parent; possible duplicate records through the root logger | **Fixed** |
| V13 | `app/main.py` created backend services before `QApplication`, returned `None` on camera failure with no exit code, and could stop services twice | Non-standard Qt start-up order, unhelpful failure signal | **Fixed** (single QApplication, `int` exit code, guarded single shutdown) |
| V14 | `config/vision.yaml`, `config/speech.yaml`, `config/ui.yaml`, `config/avatar.yaml` are 0 bytes; `config/llm.yaml` contains the stray text `ui` | `yaml.safe_load` returns `None`/`"ui"`. No module reads these keys today, so there is no runtime effect yet, but the files are not valid configuration | **Not fixed** - content is unknown, invented defaults would be speculative |
| V15 | Eight 0-byte modules (`core/constants.py`, `core/exceptions.py`, `gui/chat_widget.py`, `gui/status_widget.py`, `speech/models.py`, `vision/motion_detector.py`, `vision/enrollment.py`, `brain/emotion.py`) and two 0-byte images (`resources/background.png`, `resources/logo.png`) | Dead placeholders; nothing imports the modules | **Left untouched** (deletion must not happen in Phase 0) |
| V16 | `models/tts/provisional_patent_specification.pdf` (239 KB) sits inside the model directory | Misplaced document, not a model | **Not moved** - protected data |
| V17 | `assets/robot.png` (231 KB) is unused; `RobotWidget` renders the SVG | Dead asset | **Left untouched** |
| V18 | Hardcoded `n_threads=8` in `brain/llm.py` and `speech/stt.py` | Xavier NX has 6 cores; oversubscription | **Not changed** (tuning belongs to a later phase) |
| V19 | `requirements.txt` is stale (UTF-16, lists `opencv 5.0.0.93`/`numpy 2.4.6`, no PyQt5 or PySide6) | Misleading dependency manifest | **Not fixed** - superseded by `requirements_windows.txt` / `requirements_jetson.txt` |

### 5.2 SUSPECTED problems (reasoned, not reproduced)

| # | Problem | Why it matters |
|---|---|---|
| S1 | The Qt main thread runs the AI pipeline: `CameraWidget.update_frame` (33 ms timer) calls `VisionService.process()`, which runs InsightFace detect + recognise on every 3rd frame | Frame drops and input lag. The documented Phase 1 work is a worker thread with Qt signals. Not implemented here because startup is not blocked by it |
| S2 | `QObject::~QObject: Timers cannot be stopped from another thread` appeared once, while the process was being killed | Daemon threads outlive the Qt widgets. May also occur on a normal shutdown path; needs confirmation with a worker-thread design |
| S3 | `SpeechService.start()` reports success even when the microphone cannot be opened (`AudioCapture` now logs and stops, but the service keeps polling an empty queue) | The kiosk can appear healthy while deaf |
| S4 | `MainWindow.closeEvent` calls `vision.stop()` and `main()`'s `finally` calls it again | Double stop. `VisionService.stop()` is effectively idempotent (grabber join is None-guarded, `camera.release()` is None-guarded), so it is benign today |
| S5 | Event naming is inconsistent: `visitor.detected` / `visitor.unknown` (dotted) vs `speech_recognized` / `llm_response_ready` (underscored) | Easy to subscribe to the wrong name; no constants module exists (`core/constants.py` is empty) |
| S6 | The face embedding cache (`models/cache/face_embeddings.pkl`) is used whenever it exists, with no staleness check against `models/faces/` | Newly enrolled faces are ignored until the cache is deleted by hand |
| S7 | `VisionService.detection_interval = 3` counts frames, not time | Recognition rate varies with camera FPS |
| S8 | `main()` constructs all models before checking whether the camera exists | ~4 s of model loading before a predictable camera failure |
| S9 | `knowledge_base.search` scores a fixed +10 per matching word with the top-3 cut-off and no stop-word list | Common words dominate retrieval; answer quality depends on the prompt contract |

---

## 6. Recommended fixes (not performed in Phase 0)

1. Resolve V8 (duplicate OpenCV / ONNX Runtime distributions) and re-test V4.
   This is the single highest-value environment action.
2. Give `config/*.yaml` real contents (V14) or delete the unused files.
3. Add a `core/constants.py` event-name registry and adopt it (S5).
4. Move the vision pipeline off the Qt thread (S1) - the core Phase 1 task.
5. Add a microphone-open health check to `SpeechService` (S3).
6. Add cache invalidation for `FaceCache` (S6).
7. Delete the empty placeholder modules and 0-byte images once confirmed unused (V15).
8. Delete the stale `requirements.txt` after `requirements_windows.txt` /
   `requirements_jetson.txt` are adopted (V19).

---

## 7. Existing functionality that must be preserved

- Knowledge-base loading and keyword retrieval (12 files, 355 sections, working).
- The strict JSON prompt/parse contract in `prompt_manager.py` + `parser.py`.
- Streaming LLM output and the 6-message history window.
- Face detection, cosine-similarity recognition (threshold 0.55), centroid tracking.
- The `data/visitors.json` schema (`first_seen`, `visits`, `last_seen`).
- VAD-gated continuous listening and the silence-chunk segmentation.
- Piper-based TTS and the installed model files.
- The current GUI layout, theme and `robot.svg` mascot artwork.
- All model files, face images, knowledge files and `config/*.yaml`.
- The `python -m app.main` entry point.

---

## 8. Known limitations

- **Jetson compatibility is unverified.** No JetPack 5 / Python 3.8 device was
  available. Python 3.8 compatibility was checked statically (`ast.parse` with
  `feature_version=(3, 8)` plus a manual 3.10-only feature scan), not by running
  on 3.8.
- Only the Windows paths in `camera_manager.py` and `tts_service.py` were
  exercised. Their Linux branches are code-reviewed but untested.
- Visitor greeting by name is not wired up: `visitor.detected` has no subscribers.
- `chat_widget.py` / `status_widget.py` are empty, so the UI has no transcript or
  status display; the window is camera + mascot only.
- The offline requirement holds: no cloud AI calls exist in the source. Only
  InsightFace model download and Piper binary installation need network access,
  both one-time.
