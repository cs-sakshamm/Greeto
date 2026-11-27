# Phase 3 — Milestone 1 Implementation Report

**Scope:** configuration and camera management hardening.
**Status:** complete; existing suites still pass.
**Date:** 23 November 2026.

## 1. What changed

### Files modified

| File | Change |
| --- | --- |
| `config/camera.yaml` | Added backend selection, discovery range, warm-up, reconnect/retry timing, release timeout and stale-frame threshold, with comments |
| `app/vision/camera_manager.py` | Rewritten around a single verified capture lifecycle: discovery, actual-resolution verification, health tracking, reconnect, in-process device registry, read/release serialisation, status events |
| `app/vision/frame_grabber.py` | Failed-read detection, reconnect through the manager, safe `stop()` that reports whether the thread really stopped, frame-age/FPS diagnostics |
| `app/vision/vision_service.py` | Optional injection for tests; degraded startup with a background camera supervisor; ordered shutdown; `is_running()` and `health()` |
| `app/main.py` | Camera failure no longer returns exit code 1; starts in voice-only degraded mode |
| `app/gui/kiosk_presenter.py` | Subscribes to `camera.status` and `vision.degraded` for diagnostics only; never changes the face or caption |
| `app/gui/main_window.py` | Diagnostics line now shows vision and camera health; 1 s refresh timer stopped on close |
| `app/gui/main_window.py` (standalone launch) | No longer raises when the camera is unavailable |

### Files created

| File | Purpose |
| --- | --- |
| `tests/test_camera.py` | 9 hardware-free checks + optional `--live` smoke test |
| `docs/CAMERA.md` | Camera architecture, configuration and failure handling |
| `docs/PHASE_3_MILESTONE_1_REPORT.md` | This report |

No files were deleted or moved. No data, script or model was changed.

## 2. Architectural decisions

1. **One owner, reused.** `CameraManager` remains the only object that
   opens the device; `FrameGrabber` and `VisionService` were extended
   rather than duplicated.
2. **In-process duplicate guard.** A module-level registry rejects a
   second open of the same index before touching the device. This is
   deliberately process-local: the OS does not reliably expose
   "camera busy" on Windows.
3. **Discovery instead of a fixed index.** The configured index is
   tried first. With `auto_detect` enabled, other indices up to
   `probe_max_index` are tried, so a camera that enumerates on a
   different index no longer prevents startup.
4. **Verify, don't assume.** A real frame is read after configuring,
   and its size/FPS are recorded in `camera.status`. A mismatch is
   logged with both values instead of claiming the requested format.
5. **Health is consecutive-read based.** Fifteen failed reads
   (~0.5 s at 30 FPS) mark the camera degraded; the capture thread then
   reconnects. No extra watchdog thread is needed for runtime loss.
6. **Initial failure is handled separately.** If no camera can be
   opened at startup, a supervisor thread retries every
   `open_retry_interval_s`; the rest of the application is already
   running. This is the voice-only degraded mode.
7. **Release cannot race a read.** Reads and release share an I/O lock.
   `release()` waits up to `release_timeout_s`; if the read is still
   blocked, the device is left to the OS and the failure is logged
   rather than pulling it out from under the read.
8. **Status uses the existing EventBus.** No new service or GUI
   coupling: `camera.status` goes through the existing thread-safe bus
   and the presenter's queued-signal bridge.
9. **Test seams, not test modes.** `CameraManager` accepts a capture
   factory and `VisionService` accepts camera/model/database
   overrides. Production defaults are unchanged.
10. **No new dependency.** Only stdlib and already-installed packages
    are used.

## 3. Tests executed and actual results

Environment: Windows, `.venv` Python 3.11.1, real DirectShow camera at
index 0.

| Command | Result |
| --- | --- |
| `python tests/test_camera.py` | **ALL CHECKS PASSED** (9/9; live test skipped by design) |
| `python tests/test_camera.py --live` | **ALL CHECKS PASSED** (9/9 + live open/read/release) |
| `python tests/test_face.py --skip-slow` | **ALL CHECKS PASSED** |
| `python tests/test_kiosk.py` | **ALL CHECKS PASSED** |

Camera checks and what each proves:

| Check | Proves |
| --- | --- |
| A1 open verifies actual resolution | 1280x720 requested vs 640x480 delivered is detected and reported |
| A2 discovery falls back | Dead configured index 3, working index 0 is used |
| A3 duplicate ownership refused | Second manager fails before creating a capture |
| A4 read failures mark unhealthy, then recover | Degraded state and recovery to ready |
| A5 capture thread reconnects | Old device released once, new device used, grabber stays alive |
| A6 release waits for an in-flight read | Release blocks for the read instead of racing it |
| A7 stuck read is not released underneath | Release times out, abandons safely, reports `release-failed` |
| A8 capture thread stops before release | Ordered shutdown, single release |
| B1 voice-only startup recovers | `start()` returns False without raising, degraded event published, pipeline starts when the camera returns |
| C1 live camera (opt-in) | Real open, frame read and release |

Measured live behaviour (same machine):

- open: **1.95 s** (includes five warm-up frames);
- requested and delivered **1280x720 @ 30 FPS**;
- direct read latency: mean **36.8 ms**, p95 **48.3 ms**, max **65.2 ms**;
- capture thread: **89 frames / 3.0 s = 29.7 FPS**;
- stop: **0.02 s**; release: **0.28 s**.

Full pipeline check (real camera, real InsightFace init, real face
cache load, no frames processed through detection):

- `VisionService` construction: **1.17 s**;
- `start()`: **2.10 s** (camera open plus database load);
- 3 s run: camera `ready`, **87 captured frames, 29.0 FPS**;
- `stop()`: **0.32 s**, camera closed, capture thread stopped.

Long stability suites (`--stability-minutes`) were **not** run; they are
opt-in and unchanged.

## 4. Existing functionality preserved

- Kiosk face/caption behaviour, greeting text, duplicate-greeting
  protection, stale-TTS handling, watchdog, and clean-shutdown checks
  all pass (`tests/test_kiosk.py`).
- Face rendering, animation, layout and MainWindow integration pass
  (`tests/test_face.py`).
- InsightFace import order (before PyQt5) is unchanged in `app/main.py`
  and both test entry points.
- Vision still runs off the GUI thread through `VisionPump`; no camera
  frame is displayed.
- Speech, Whisper, Piper and the local LLM were not modified.
- The stale `app/test_*.py` scripts and `app/gui/camera_widget.py` were
  left untouched and unreferenced.

## 5. Known limitations and unresolved risks

1. **Voice-only startup is verified at service level, not end-to-end.**
   `VisionService` degraded start/recovery is covered, and `main.py` no
   longer aborts, but the full Qt application was not launched with the
   camera physically disconnected in this milestone.
2. **Physical unplug/replug was not tested.** Reconnection is covered
   by a fake device; a real USB disconnect can block `read()` for
   longer than the test model, which is exactly why release abandons
   the device when the read cannot be drained.
3. **OS-level "in use" is only best-effort.** The duplicate guard is
   in-process. Another application holding the camera appears as
   "could not be opened" with one combined error message.
4. **Resolution fallback is accepted.** If the camera delivers
   640x480 instead of 1280x720, capture continues and the mismatch is
   logged; no lower-resolution negotiation is attempted.
5. **Discovery can be slow.** Opening an index can take up to ~2 s on
   Windows, so scanning five indices worst-case delays startup. It
   happens on the supervisor thread when the preferred device fails,
   so the kiosk remains responsive.
6. **Abandoned device after a stuck read.** In the `release-failed`
   state the OS owns the device until process exit. This is a shutdown
   path; the alternative (releasing during a read) risks a crash.
7. **`vision.degraded` is not shown to visitors.** It is diagnostics
   only; by design the kiosk does not announce hardware failures.
8. **Jetson/V4L2 was not exercised here.** Backend selection and
   defaults are platform-aware but unverified on hardware.

## 6. Recommendation and approval request

Milestone 1 meets its completion criteria: camera management is
reliable within the tested scope, voice-only startup is implemented and
tested at service level, existing functionality is intact, and all
targeted and regression suites pass.

I recommend proceeding to **Milestone 2: Recognition Engine and
Embedding Reliability** (embedding validation/normalisation, quality
gates, configurable and calibrated threshold, structured recognition
results, and unique visitor IDs).

**Approval requested:** please confirm Milestone 2. I will stop here
until you approve.
