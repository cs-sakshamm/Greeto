# Camera Management

Phase 3, Milestone 1. This document describes the camera layer only;
recognition, visitor tracking, enrolment and the visitor database are
unchanged and are covered in later milestones.

## Ownership model

`CameraManager` is the only object in the process that opens the
device. `VisionService` owns exactly one instance and `FrameGrabber`
reads from it in a background thread.

- A module-level registry (`_OPEN_INDICES`) refuses a second open of
  the same device index by any `CameraManager` in the process. Two
  managers on one index is the usual cause of a black preview and a
  crash on shutdown.
- Reads and `release()` are serialised by an I/O lock, so the device is
  never released underneath a read that is still using it.
- `FrameGrabber` asks the manager to reconnect after a run of failed
  reads. It never creates a second capture object or camera manager.

## Startup and recovery

```
CameraManager.open()
    configured index ──fail──► discovery scan (0..probe_max_index)
            │                          │
          success ─────────────────────┘
            │
VisionService._start_pipeline()
    load face database → FrameGrabber.start() → running

open() fails
    VisionService.start() returns False
    camera.status(unavailable) + vision.degraded published
    CameraSupervisor thread retries every open_retry_interval_s
    kiosk continues in voice-only mode (speech + LLM unaffected)
    when the camera returns → pipeline starts, camera.status(ready)
```

The kiosk never waits for a camera. `app/main.py` keeps running when
`vision.start()` returns `False`; only a truly fatal exception exits.

## Camera status

`CameraManager` publishes `camera.status` through the existing
`EventBus`. Subscribers run on the publishing thread, so the kiosk
presenter forwards it to the GUI with its usual queued signal.

| Field | Meaning |
| --- | --- |
| `state` | `closed`, `opening`, `ready`, `degraded`, `reconnecting`, `unavailable`, `released`, `release-failed` |
| `index` | Device index actually in use |
| `message` | Human-readable warning (for example a resolution mismatch) |
| `requested_width/height/fps` | Values from `config/camera.yaml` |
| `actual_width/height/fps` | Values verified from a real frame / the device |
| `frames` | Successfully read frames since opening |
| `read_failures` | Total failed reads since opening |
| `consecutive_failures` | Failed reads in the current run |
| `reconnects` | Reconnect attempts made |
| `open`, `healthy` | Convenience flags |
| `frame_age_s` | Seconds since the last successful frame |

No frame content, image, embedding or identity is ever included.

## Configuration

`config/camera.yaml`:

| Key | Default | Purpose |
| --- | --- | --- |
| `index` | `0` | Preferred device index |
| `width`, `height`, `fps` | 1280, 720, 30 | Requested format (verified, not assumed) |
| `backend` | `auto` | `auto`, `dshow`, `v4l2` or `any` |
| `auto_detect` | `true` | Scan other indices when the preferred one fails |
| `probe_max_index` | `5` | Highest index scanned |
| `warmup_frames` | `5` | Frames discarded after opening |
| `read_failures_before_reconnect` | `15` | Failed reads in a row before reconnecting (~0.5 s at 30 FPS) |
| `reconnect_interval_s` | `5.0` | Pause between reconnect attempts in the capture thread |
| `open_retry_interval_s` | `5.0` | Retry interval while the camera is unavailable at startup |
| `release_timeout_s` | `0.75` | How long a release waits for an in-flight read |
| `frame_stale_after_s` | `2.0` | A frame older than this marks the camera unhealthy |

Invalid or missing values fall back to the defaults above, so a partial
configuration file cannot stop the camera.

## Resolution and FPS verification

The requested values are applied to the device, then a real frame is
read and its shape recorded. If the device delivers something else, the
status message says so and a warning is logged with both values. This
replaces the old behaviour of logging the requested size as if it had
been achieved.

Measured on the development machine (DirectShow, index 0):

- requested and delivered 1280x720 @ 30 FPS;
- open took 1.95 s including five warm-up frames;
- direct frame read: mean 36.8 ms, p95 48.3 ms, max 65.2 ms;
- capture thread: 89 frames in 3.0 s (29.7 FPS);
- stop: 0.02 s; release: 0.28 s.

## Diagnostics

`F10` in the kiosk shows a development-only line with face state,
microphone state, active visitor tracks, vision state and camera state
(index, actual resolution/FPS, last-frame age, failed reads,
reconnects). It is hidden by default and never shown to visitors.

## Failure handling summary

| Failure | Behaviour |
| --- | --- |
| No camera at startup | Voice-only mode; retry supervisor keeps running |
| Configured index busy/missing | Discovery scan, then the first working index |
| Camera unplugged while running | Reads fail, status becomes `degraded`, capture thread reconnects |
| Camera returns | Reconnect succeeds, status returns to `ready` |
| Frame read blocks during shutdown | Release waits up to `release_timeout_s`; if the read is still stuck the device is left to the OS and the failure is logged instead of releasing underneath the read |
| Two managers on one index | Second open is refused before the device is touched |
| Requested format unsupported | Warning with both requested and actual values; capture continues |

## Tests

`tests/test_camera.py` covers all of the above with a fake capture
device (no hardware, no model, no GUI). `--live` adds one real
open/read/release smoke test.
