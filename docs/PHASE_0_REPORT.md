# Greeto - Phase 0 Report

**Project:** AI Admission Receptionist Robot, I.T.S Engineering College
**Phase:** 0 - Project Audit & Environment Stabilisation
**Date:** 2026-11-17
**Branch:** `test`
**Overall status: COMPLETE** (one environment-level issue deliberately left
open - see section 11; Jetson cannot be verified without hardware)

---

## 1. Initial project condition

The repository contained a complete but **non-runnable** application on the
intended PyQt5 / Python 3.8 stack.

Verified first-run state:

```
$ python -c "import app.main"
ModuleNotFoundError: No module named 'PySide6'
```

The window could not even be imported, because four GUI modules and the entry
point still targeted PySide6. Two data models used a Python 3.10-only dataclass
feature that would crash on Jetson's Python 3.8. Every resource path was
relative to the process working directory, the camera backend was hardcoded to
DirectShow (Windows-only), and TTS playback used the Windows-only `winsound`
module.

What was already working and is preserved: the knowledge-base loader and
retrieval (355 sections), the JSON prompt/parse contract, streaming llama.cpp
inference, Whisper transcription, VAD segmentation, InsightFace detection,
cosine-similarity recognition, centroid tracking, the visitor JSON store, the
Piper TTS wrapper, and the whole GUI layout/theme with the new `robot.svg`
mascot.

There were pre-existing uncommitted changes in the working tree (theme
restyling, `robot_widget.py` redesign, deletion of four `mascot_*.svg` files,
addition of `robot.svg`, edits to `data/visitors.json`, plus an untracked
`requirements_windows.txt`). **None of them were discarded, reverted or
committed.**

---

## 2. Problems discovered

21 verified problems and 9 suspected problems are catalogued in
`docs/PROJECT_AUDIT.md` (sections 5.1 and 5.2). The ones that mattered:

| Severity | Problem |
|---|---|
| Blocking | PySide6 imports in 4 files - application could not start |
| Blocking | `@dataclass(slots=True)` in 2 vision models - Python 3.10+ only, crashes on 3.8 |
| Blocking | PyQt5-before-onnxruntime DLL initialisation failure on Windows, misreported by InsightFace as `Unable to import dependency onnxruntime.` |
| Deployment-blocking | CWD-relative paths for config, models, data and logs |
| Deployment-blocking | `cv2.CAP_DSHOW` hardcoded - camera never opens on Linux |
| Deployment-blocking | `winsound` used for WAV playback - no audio on Linux |
| High | Visitor database rewritten on every detection cycle (~10 writes/s; `visits: 1143` is an artefact) |
| High | One failing EventBus subscriber aborted the others and could kill the publishing thread |
| High | `AudioCapture` busy-waited with `while running: pass` - one core at 100% |
| Medium | UTF-16 dependency files that pip cannot parse |
| Medium | Duplicate/conflicting OpenCV and ONNX Runtime distributions in the venv |
| Low | Empty/corrupt config files, eight 0-byte modules, two 0-byte images |

---

## 3. Problems fixed

| Fix | File(s) |
|---|---|
| PySide6 -> PyQt5 across all GUI code (`PyQt5.QtSvg` for `QSvgWidget`, `exec_()` for the event loop); zero PySide6 imports remain | `app/main.py`, `gui/main_window.py`, `gui/camera_widget.py`, `gui/robot_widget.py` |
| Removed `slots=True` from both dataclasses so the models work on Python 3.8 | `vision/models.py`, `vision/visitor_tracker.py` |
| Entry point reordered: single `QApplication` first, services constructed inside `try`, `int` exit code (`1` when the camera fails), guarded single shutdown in `finally`. Vision imported before Qt to work around the ORT DLL conflict | `app/main.py`, `gui/main_window.py` |
| Config directory, log directory, face DB, embedding cache, visitor DB, Whisper and Piper models now resolve against the project root | `core/config_manager.py`, `core/container.py`, `vision/face_database.py`, `vision/face_cache.py`, `vision/visitor_database.py`, `speech/stt.py`, `speech/tts_service.py` |
| `ConfigManager.get_value` accepts a `default` and no longer raises `KeyError` on a missing file/key; camera defaults applied | `core/config_manager.py`, `vision/camera_manager.py` |
| Platform camera backend (DirectShow on Windows, V4L2 on Linux, plain-open fallback) | `vision/camera_manager.py` |
| Cross-platform WAV playback (`winsound`, else `aplay`/`paplay`/`ffplay`) | `speech/tts_service.py` |
| Visitor announced once per tracked visitor instead of every detection cycle | `vision/vision_service.py` |
| Per-callback exception isolation + logging in `publish` | `core/event_bus.py` |
| Log directory created with `parents=True`; `propagate = False` | `core/logging_setup.py` |
| Busy-wait replaced with `sleep(0.05)`; microphone open failure logged, thread state reset | `speech/audio_capture.py` |
| `.gitignore` extended (bytecode, build/test artefacts, `.env`, `.idea/`, `logs/`) | `.gitignore` |
| `requirements_windows.txt` rewritten as UTF-8 with validated pins | `requirements_windows.txt` |

Removed `cv2.destroyAllWindows()` from `VisionService.stop()` - no OpenCV
highgui window is ever created (the GUI is Qt), and highgui calls are
undesirable on a headless kiosk.

---

## 4. Files created

```
docs/PROJECT_AUDIT.md
docs/DEPENDENCY_COMPATIBILITY.md
docs/ENVIRONMENT_SETUP.md
docs/PHASE_0_REPORT.md
requirements_jetson.txt
requirements_windows.txt   (recreated as UTF-8; previously untracked UTF-16)
```

## 5. Files modified

```
.gitignore
app/main.py
app/core/config_manager.py
app/core/container.py
app/core/event_bus.py
app/core/logging_setup.py
app/gui/camera_widget.py
app/gui/main_window.py
app/gui/robot_widget.py          (import lines only - developer's redesign kept as-is)
app/speech/audio_capture.py
app/speech/stt.py
app/speech/tts_service.py
app/vision/camera_manager.py
app/vision/face_cache.py
app/vision/face_database.py
app/vision/models.py
app/vision/vision_service.py
app/vision/visitor_database.py
app/vision/visitor_tracker.py
```

## 6. Files intentionally left untouched

```
app/gui/theme.py                       developer's uncommitted restyle
app/gui/resources/robot.svg            developer's new mascot (untracked)
app/gui/resources/mascot_*.svg         developer's staged deletions (4 files)
data/visitors.json                     live data - restored after the startup test (see note)
data/visitors.json (history)           no data rewritten or compacted
config/*.yaml                          contents unknown; no speculative defaults invented
app/knowledge/*.md                     knowledge base untouched
models/**                              models, face images, embedding cache untouched
requirements.txt                       stale/UTF-16; documented, not deleted
app/core/constants.py                  empty placeholder - not deleted
app/core/exceptions.py                 empty placeholder - not deleted
app/gui/chat_widget.py                 empty placeholder - not deleted
app/gui/status_widget.py               empty placeholder - not deleted
app/speech/models.py                   empty placeholder - not deleted
app/vision/motion_detector.py          empty placeholder - not deleted
app/vision/enrollment.py               empty placeholder - not deleted
app/brain/emotion.py                   empty placeholder - not deleted
assets/robot.png                       unused asset - not deleted
models/tts/provisional_patent_specification.pdf   misplaced, not moved
knowledge_base/, scripts/, tests/      empty directories
```

> **Note on `data/visitors.json`:** launching the application to verify startup
> (Check 4) made the face recogniser match the enrolled developer, which
> incremented the counter twice (1143 -> 1145) and updated `last_seen`.
> The file was restored byte-for-byte to its pre-run state immediately
> afterwards; the only remaining diff is the developer's own earlier edit.
> No other data file, model or knowledge file was written by this phase.

**No files were deleted.**

---

## 7. Dependency changes

- **Nothing was installed or uninstalled.** No PyTorch, ONNX Runtime GPU,
  InsightFace, llama-cpp-python, pywhispercpp or CUDA package was added.
- `requirements_windows.txt`: rewritten from a UTF-16 pip-freeze dump into a
  curated UTF-8 manifest pinning the validated versions (PyQt5 5.15.11,
  opencv-contrib-python 4.11.0.86, numpy 2.4.6, insightface 1.0.1, onnx 1.22.0,
  onnxruntime-gpu 1.28.0, pywhispercpp 1.5.0, piper-tts 1.6.0, webrtcvad
  2.0.10, sounddevice 0.5.5, llama_cpp_python 0.3.34, PyYAML 6.0.3).
- `requirements_jetson.txt`: new, Python 3.8.10 / aarch64 targeted, with apt vs
  pip separation, `--system-site-packages` requirement, and `[HARDWARE TEST]`
  markers. Versions were chosen from **measured** wheel availability, not
  guessed (e.g. numpy 1.24.4 exists for cp38/aarch64 while 1.26.4 does not).
- Two conflicts were **found but not resolved** (developer decision):
  `opencv-python==5.0.0.93` + `opencv-contrib-python==4.11.0.86`, and
  `onnxruntime==1.28.0` + `onnxruntime-gpu==1.28.0` sharing one directory.

---

## 8. PyQt5 migration status

**Complete for all active application source.**

| Item | Result |
|---|---|
| PySide6 imports in `app/**/*.py` | **0** (verified by grep and by an automated check) |
| Remaining PySide6 mentions | 1 comment line in `requirements_windows.txt` explaining the migration; several in these docs. No executable import |
| PyQt5 import test | `from PyQt5.QtWidgets import QApplication` -> `PyQt5 OK` |
| QtSvg availability | `from PyQt5.QtSvg import QSvgWidget` -> OK (PyQt5 5.15.11 bundles QtSvg) |
| PySide6-specific API replaced | `PySide6.QtSvgWidgets.QSvgWidget` -> `PyQt5.QtSvg.QSvgWidget` |
| Signals/slots | `QTimer.timeout.connect(...)` unchanged and verified active at runtime |
| Enums | `Qt.AlignCenter`, `Qt.NoFocus`, `Qt.WA_StyledBackground`, `Qt.KeepAspectRatioByExpanding`, `Qt.SmoothTransformation`, `QSizePolicy.Expanding` - all valid PyQt5 |
| Event loop | single `QApplication`; `app.exec_()` used (available in every PyQt5 5.15.x) |
| Mixed toolkits | none - PySide6 is not even installed |
| Visual design | unchanged (theme, layout, colours, mascot untouched) |

---

## 9. Windows validation results

Environment: Python 3.11.1 at `.venv/Scripts/python.exe`, pip 26.1.2.

### Check 1 - interpreter

```
Python 3.11.1
pip 26.1.2 from ...\.venv\Lib\site-packages\pip (python 3.11)
```
Both the global `python` and `.venv` resolve to the same venv interpreter, so
the intended environment is the active one.

### Check 2 - PyQt5

```
PyQt5 OK
QtSvg OK
```

### Check 3 - module imports (27 modules)

```
[PASS] All 27 application modules import
[PASS] Zero active PySide6 imports in app/
[PASS] Python 3.8 syntax + dataclass compatibility (all app/*.py)
```

(The import check covers `app.main`, every `core`, `gui`, `vision`, `speech`
and `brain` module, including the empty placeholder packages.)

### Check 4 - application startup

```
$ python -m app.main
AI Admission Robot Starting...
Opening camera...
Camera ready (1280x720 @ 30 FPS)
Backend services started.
GUI started.
```

`APP_EXIT=124` - the process was terminated by a 45 s `timeout` guard after it
had fully started, which is the expected result of the guard, not a crash. In
that window the live microphone picked up audio and the full chain ran:
VAD -> Whisper (`Recognized: - Hello.`) -> knowledge retrieval -> Gemma ->
parser -> Piper. No duplicate event loop, no unhandled startup exception.

Side effect: the recogniser matched the enrolled face, so `data/visitors.json`
was incremented by the run. The file was restored byte-for-byte afterwards.

### Check 5 - PyQt5 GUI render (offscreen, isolated from the display)

```
MainWindow='AI Admission Robot' robot=450x450 shadow=True
[PASS] PyQt5 main window + QSvgWidget render offscreen
```
`QSvgWidget.renderer().isValid()` is True, so `robot.svg` really renders; the
camera `QTimer` is active; `QGraphicsDropShadowEffect` is attached.

### Check 6 - knowledge base

```
KnowledgeBase loaded 355 sections.
sections=355, 'hostel fees' -> 1147 chars
```
Files found, sections split, retrieval returns content. Retrieval logic
unchanged.

### Check 7 - core services

```
[PASS] EventBus isolates failing callbacks
[PASS] StateManager basic transitions
[PASS] ConfigManager resolves project config/ from a foreign CWD
[PASS] All data/model paths resolve against project root
[PASS] Visitor registration fires once per tracked visitor
```
The config test changed the working directory to the project's parent before
loading, proving resource discovery no longer depends on the CWD.

### Check 8 - Python 3.8 compatibility (static)

Every `app/**/*.py` parses with `ast.parse(..., feature_version=(3, 8))`; no
`dataclass(slots=True)` remains. **This is a static check only** - no Python 3.8
interpreter exists on this machine, so nothing was executed under 3.8.

### Check 9 - git safety

- Branch unchanged: `test`. No branch created, no commit, no push, no reset.
- The developer's pre-existing edits are intact (see section 6).
- No file deleted by this phase.
- `.gitignore` now excludes `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`,
  `logs/`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info/`, `.env`, `.idea/`.
  `models/` was already ignored by the developer and was left that way.

---

## 10. Jetson compatibility status

**NOT VERIFIED. No Jetson hardware was available, and the environment was not
touched remotely.**

What was done instead:

- Static Python 3.8 compatibility audit (pass).
- Platform-specific code made portable instead of Windows-only: camera backend,
  WAV playback, project-relative paths.
- ARM64/Python 3.8 wheel availability measured per package (results in
  `docs/DEPENDENCY_COMPATIBILITY.md`), which is how the `requirements_jetson.txt`
  pins were chosen.
- The PyQt5-via-apt / `--system-site-packages` requirement documented in
  `docs/ENVIRONMENT_SETUP.md`.

Still requiring hardware: PyQt5 + OpenCV import inside the venv, V4L2 camera
capture, whether the PyQt5/onnxruntime DLL conflict reproduces, `aplay` output,
PortAudio capture, the llama.cpp CUDA build, and kiosk/fullscreen behaviour.

---

## 11. Remaining issues

1. **The PyQt5/onnxruntime DLL conflict is mitigated, not solved.** The
   application works because `app/main.py` and `app/gui/main_window.py` import
   the vision stack before Qt. Any new module that imports PyQt5 before
   `insightface` will break startup on Windows. Proper fix: decide on a single
   ONNX Runtime distribution and verify the conflict disappears.
2. **Duplicate packages** (OpenCV 5.0.0.93 vs contrib 4.11.0.86; onnxruntime vs
   onnxruntime-gpu 1.28.0) - needs a developer decision, so no uninstall was
   performed.
3. `config/vision.yaml`, `speech.yaml`, `ui.yaml`, `avatar.yaml` are empty and
   `llm.yaml` contains `ui`. Harmless today (no consumer), still invalid.
4. Eight 0-byte modules and two 0-byte images remain as placeholders.
5. `requirements.txt` is still stale/UTF-16 and should be retired.
6. `visitor.detected` has no subscriber - greeting by name is not wired up.
7. `SpeechService` still reports success when the microphone cannot be opened.
8. The InsightFace `buffalo_l` pack is not in `models/`; first use downloads it.
9. `piper` must be on `PATH`; its presence was not asserted at startup.

---

## 12. Known risks

| Risk | Impact | Mitigation |
|---|---|---|
| AI inference runs on the Qt main thread (`QTimer` -> `VisionService.process()`) | Frame stalls, unresponsive UI on Xavier NX | Phase 1 worker thread + signals; documented as S1 |
| Import order is now load-bearing on Windows (V4) | A future refactor can silently reintroduce a startup crash | Commented in both entry points and documented |
| Model loading happens before the camera is validated | ~4 s delay before a predictable failure | Documented (S8) |
| `QObject::~QObject: Timers cannot be stopped from another thread` seen once during forced termination | Suspected teardown race between daemon threads and Qt objects | Not reproduced on a clean shutdown path; revisit with the worker-thread design |
| Jetson builds are unverified | On-device build failures (llama.cpp CUDA, InsightFace Cython) | `[HARDWARE TEST]` markers in `requirements_jetson.txt` |
| Pickled face cache may be version-incompatible across InsightFace builds | Wrong recognition results on Jetson | Regenerate `models/cache/face_embeddings.pkl` on the device |
| Hardcoded `n_threads=8` on a 6-core Xavier NX | Oversubscription | Tuning item for a later phase |

No credentials, tokens or secrets exist in the source tree or the reports.

---

## 13. Recommended next steps for Phase 1

1. **Checkpoint the current tree first.** The working tree contains the
   developer's earlier redesign plus this phase. Commit it on `test` (or a
   `phase-0-stabilisation` branch) before starting Phase 1, so the audit
   baseline is recoverable.
2. **Resolve the environment conflict** (single OpenCV, single ONNX Runtime),
   then delete the import-order workaround and confirm `import app.main` works
   in any order.
3. **Implement the animated digital face with `QPainter`** - the GUI is now
   clean PyQt5 and `RobotWidget` already owns the mascot slot with stubbed
   `set_idle/set_listening/set_thinking/set_speaking/set_happy` hooks to fill in.
4. **Move the vision pipeline off the GUI thread** (worker QThread + signals),
   which also fixes the suspected teardown race.
5. Wire `visitor.detected` to the greeting path, and add `statistics`-safe
   visitor writes (the amplification is fixed, but the file is still rewritten
   synchronously).
6. Fill or delete the empty config files and placeholder modules.
7. Validate everything on the actual Xavier NX before trusting any Jetson claim.

---

## Definition of Done

| Requirement | Status |
|---|---|
| Actual project structure audited | Done |
| Existing architecture documented | Done - `docs/PROJECT_AUDIT.md` |
| Main entry point verified | Done - `python -m app.main` starts |
| ServiceContainer audited | Done |
| EventBus audited | Done (+ callback isolation fixed) |
| Configuration handling audited | Done (+ CWD-independence, defaults) |
| Logging audited | Done (+ mkdir parents, propagate) |
| PySide6 imports removed from active source | Done - 0 remaining |
| PyQt5 imports verified on Windows | Done - imports, render, live run |
| Python 3.8 compatibility reviewed | Done - static only, stated as such |
| Windows dependency file prepared | Done - `requirements_windows.txt` (UTF-8) |
| Jetson dependency file prepared | Done - `requirements_jetson.txt`, unverified on hardware |
| Dependency compatibility matrix created | Done |
| Resource paths reviewed | Done - all project-root based |
| Git safety reviewed | Done - nothing committed, pushed or deleted |
| Application startup tested on Windows | Done - camera + GUI + full speech pipeline |
| Application shutdown tested | Partially - shutdown path executed, but the interactive window-close path was not driven end-to-end; forced termination leaves no clean shutdown |
| Knowledge-base loading verified | Done - 355 sections, retrieval returns content |
| Required documentation created | Done |
| Remaining hardware-dependent issues documented | Done |
| Phase 0 report completed | Done - this file |

**Phase 1 has not been started. No animated face work, GUI redesign, or AI
pipeline changes were made.**
