# Greeto - Phase 2 Report

**Fullscreen Kiosk Interface, Face Integration and Application State Wiring**

| Field | Value |
|---|---|
| Phase | 2 - Fullscreen Kiosk Interface |
| Branch | `test` |
| Status | Complete (scope boundaries in section 18) |
| Verification | Re-verified end to end after the continuation audit in section 20 |
| Host | Windows 10/11, Python 3.11.1, PyQt5 5.15.11, offscreen test platform |
| Target hardware | NVIDIA Jetson Xavier NX - **not tested in this phase** |
| Report date | 2026-11-18 |

Nothing was committed or pushed.

---

## 1. Initial project state

Phase 1 was complete and verified. The audit at the start of this phase
(`git status --short`, `git diff --stat`, plus reading every GUI and service
file) found:

- `app/gui/robot_widget.py`, `face_renderer.py`, `face_animation.py` - the
  completed programmatic face, untouched by this phase.
- `app/gui/main_window.py` - a desktop-style two-card layout: a `CameraWidget`
  preview on the left, the face in a `GlassCard` on the right, with
  `showMaximized()` and a 1400x800 minimum size.
- `app/gui/camera_widget.py` - a `QLabel` that scaled and displayed camera
  frames, driven by a 33 ms `QTimer` **on the GUI thread**.
- `app/core/event_bus.py` - a thread-safe publish/subscribe bus that runs each
  callback synchronously on the publishing thread and logs exceptions without
  aborting other subscribers.
- `app/core/state_manager.py` - a `RobotState` enum and a thread-safe
  `StateManager`. **Nothing calls `set_state` and nothing subscribes to it**;
  it is a passive holder.
- Existing events: `visitor.detected`, `visitor.unknown` (VisionService),
  `speech_recognized` (SpeechService), `llm_response_ready`
  (ConversationManager). Only `speech_recognized` had a subscriber.
- `vi​sitor.detected` had no subscriber, `visitor.unknown` was published on
  every detection cycle, and there was no event for microphone failure, STT
  failure, processing start, TTS playback start, TTS playback end or visitor
  departure.
- `TTSService.speak()` generated a WAV, played it synchronously and returned
  no result. It also left every generated WAV file on disk.

The Phase 1 report's own limitation list (face not wired to state, camera on
the GUI thread, no Jetson validation) matched the code exactly.

---

## 2. Phase 1 integration details

The Phase 1 face was reused as-is:

- `RobotWidget` is still the only face widget, painted by `FaceRenderer`, with
  its `FaceAnimator` `QTimer` independent of everything added in this phase.
- All eight setters (`set_idle`, `set_listening`, `set_thinking`,
  `set_speaking`, `set_happy`, `set_greeting`, `set_confused`, `set_error`) and
  `set_state()` keep their signatures.
- `set_state()` is a no-op when the requested state is already active, so the
  presenter can emit states freely without restarting animations.
- The Phase 1 test suite still passes unchanged (check A below).

Nothing in the renderer, animator or widget was modified.

---

## 3. New fullscreen architecture

```
MainWindow (frameless, fullscreen)
│
└── KioskBackground        painted dark radial light + vignette
    │
    ├── face row          RobotWidget, expanding, centred
    ├── caption row       CaptionWidget, bottom, width-limited
    └── diagnostics       QLabel, hidden until F10
```

- **Window**: `setWindowFlags(Qt.Window | Qt.FramelessWindowHint)` and
  `showFullScreen()`. No title bar, no border, no menu, no toolbar, no status
  bar, no buttons. The cursor is hidden while the kiosk is active.
- **Background**: `KioskBackground` paints base `#05080D`, a soft radial light
  centred at 38 % height (`#112433` to base), and a vignette to black at the
  edges. Gradients are rebuilt on resize only; a frame allocates nothing.
- **Face placement**: the face row takes all height left over from the caption.
  At 1920x1080 the face is drawn on an 840 px square; at 1280x720 on 534 px.
  `RobotWidget` keeps its own proportions, so nothing was enlarged inside the
  renderer.
- **Margins**: 48 px sides, 28 px top, 34 px bottom, 16 px between face and
  caption. No desktop-style chrome.
- **Development controls** (never drawn):
  - `Esc` - leave fullscreen if in kiosk mode; a second `Esc` closes the app.
  - `F11` - toggle fullscreen.
  - `F10` - toggle a one-line diagnostics readout (state / microphone /
    visitor count), hidden by default and absent from production.

The old `GlassCard` helper and the camera preview were removed from
`MainWindow`. `app/gui/camera_widget.py` is kept on disk (unused) so no
existing import path disappears; it is no longer part of the kiosk.

---

## 4. MainWindow modifications

`MainWindow.__init__` now accepts:

```python
MainWindow(
    vision,                 # unchanged, still required
    container=None,          # provides the EventBus (optional)
    speech=None,             # read once for microphone availability
    fps=30,
    pump_vision=False,       # start the vision worker thread
    parent=None,
)
```

- The `vision` argument and the "vision imported before PyQt5" module order are
  preserved, so the Phase 0 ONNX Runtime workaround is intact.
- `build_ui()` creates the background, face row, caption row and hidden
  diagnostics.
- `enter_kiosk()` / `leave_kiosk()` / `is_kiosk()` control kiosk mode.
- `keyPressEvent` implements Esc / F11 / F10.
- `closeEvent` stops, in order: the vision pump, the presenter (which
  unsubscribes every EventBus handler), the caption timers, the face timer, and
  finally calls `vision.stop()` through the existing service lifecycle.
- `resizeEvent` sizes the caption band (15 % of height, minimum 92 px) and caps
  its width at 72 % of the window, so text can never overlap the face.

`app/main.py` now reaches the kiosk through:

```python
window = MainWindow(
    vision=vision,
    container=container,
    speech=speech,
    pump_vision=True,
)
window.enter_kiosk()
```

---

## 5. Caption architecture

`CaptionWidget` is a single `QLabel` (`Qt.PlainText`, word-wrapped,
centre-aligned, transparent background, no card, no bubble).

API:

```python
show_caption(text, duration_ms=0)   # 0 = keep until replaced
clear_caption()
current_text() / opacity()          # inspection helpers
shutdown()
```

- **Fade**: `QGraphicsOpacityEffect` animated with `QPropertyAnimation`
  (320 ms in, 520 ms out, `OutCubic`). No layout jump when text changes.
- **Safe timers**: every update and every clear increments `_token`. A
  finish-callback only blanks the label if its captured token is still current,
  and a new caption stops the previous single-shot timer. Therefore an old
  timer can never erase newer text. This is verified by check C2.
- **Long answers**: whitespace-normalised, truncated to 260 characters with an
  ellipsis, wrapped, and confined to a 15 %-height band; a long reply cannot
  take over the screen.
- **Typography**: Segoe UI Light, pixel size `clamp(16, height * 0.22, 34)` -
  23 px at 1280x720, 34 px at 1920x1080. Contrast `#C7DAE6` on near-black.
- **What is shown**: greeting, "Let me check that for you.", the AI reply
  itself, and friendly recovery messages. Raw JSON, prompts, logs, confidence
  values, model names and event names are never displayed.

---

## 6. Event-to-face-state mapping

`KioskPresenter` is the only translation layer. It owns no widgets and emits
three signals: `face_state(str)`, `caption(str, int)` and `caption_clear()`.

| Application event / condition | Face state | Caption |
|---|---|---|
| Application startup, no visitor | IDLE | none |
| `visitor.detected` (recognised) | GREETING | "Hello {name}! Welcome back to I.T.S Engineering College." |
| `visitor.unknown` | GREETING | "Hello! Welcome to I.T.S Engineering College." |
| Greeting hold finished | LISTENING (mic available and visitor present), else IDLE | greeting fades after 7 s |
| `conversation.processing` | THINKING | "Let me check that for you." |
| `llm_response_ready` | THINKING (still) | the AI reply text |
| `speech.speaking_started` | SPEAKING | the reply |
| `speech.speaking_finished` | HAPPY for 1.4 s, then LISTENING/IDLE | reply fades after 7 s |
| `speech.error` / `conversation.error` | ERROR for 4.2 s, then LISTENING/IDLE | friendly message per failure kind |
| `visitor.left` | IDLE | cleared |
| Thinking longer than 45 s | ERROR (watchdog) | "Sorry, that took too long..." |

Design decisions behind the table:

- **SPEAKING only when audio plays.** `TTSService` publishes
  `speech.speaking_started` after Piper has produced the WAV and immediately
  before playback; `llm_response_ready` deliberately does not change the face,
  so Greeto never mouths a reply that is still being generated.
- **LISTENING only with a microphone.** The presenter is initialised from
  `SpeechService.microphone_ok` and permanently drops to IDLE if a microphone
  error arrives. It never claims to listen without a mic.
- **No PERMANENT THINKING.** A 45 s watchdog converts a stuck request into
  ERROR and then recovers.
- **One greeting per visitor.** VisionService announces each recognised and
  unknown visitor once per tracker ID, and the presenter also holds its own
  greeted-ID set. A person standing in front of the camera does not cause
  repeated greetings.
- **No interruption.** A second visitor arriving while Greeto is thinking or
  speaking only updates `visitor_present`; it does not cut off the answer.
- **Stale TTS completion ignored.** Each utterance carries an ID; a
  `speech.speaking_finished` for an older ID is discarded (check D2).

`StateManager` remains unused, as it was before this phase; no parallel state
machine was introduced, and `EventBus` remains the single event system.

### New event names

All additions are backward compatible; no existing event was renamed or
removed.

| Event | Payload | Publisher | Purpose |
|---|---|---|---|
| `visitor.left` | - | VisionService | All tracked visitors expired |
| `conversation.processing` | `text` | ConversationManager | Request accepted, generation starting |
| `conversation.error` | `message` | ConversationManager | Generation/retrieval failed |
| `speech.speaking_started` | `text`, `utterance` | TTSService | Playback is starting now |
| `speech.speaking_finished` | `utterance` | TTSService | Playback ended (also on failure) |
| `speech.error` | `kind`, `message` | SpeechService, TTSService | `microphone`, `stt` or `tts` failure |

Existing events are unchanged: `visitor.detected`, `visitor.unknown`,
`speech_recognized`, `llm_response_ready`.

---

## 7. Visitor detection integration

- **Source of truth**: the existing `VisitorTracker`, `FaceRecognizer` and
  `VisitorDatabase`. No second database, tracker or registration path was
  created.
- **Once per visit**: `VisionService` now records announced unknown visitor IDs
  in `_announced_unknown`, exactly as it already did for recognised visitors in
  `_announced_visitors`. `visitor.unknown` used to fire on every detection
  cycle (≈10 times a second); it now fires once per tracked visitor.
- **Departure**: `visitor.left` is published on the transition from "tracks
  present" to "no tracks". Track expiry is 30 missing frames at the existing
  detection interval (every third frame at 30 FPS), so departure is noticed
  roughly 3 seconds after the visitor leaves. That delay is inherited from the
  tracker's existing `max_missing_frames` and was not changed.
- **Known visitors**: the greeting uses `face.name` only when it is a real
  recognised name (not `Unknown`, `None` or `Visitor`). Low-confidence or
  unknown faces get the generic greeting. Recognition confidence is never
  shown, and visitor records never reach the screen.
- **No duplicate greeting**: both VisionService (per tracker ID) and the
  presenter (greeted-ID set, capped at 512) suppress repeats. Different
  visitors still get their own greeting.

---

## 8. Speech pipeline integration

The pipeline itself was not rewritten. Six small, backward-compatible changes
expose its real lifecycle:

| Component | Change |
|---|---|
| `AudioCapture` | Optional `on_error` callback, invoked from the capture thread when the microphone stream cannot be opened. |
| `SpeechService` | Wires that callback, sets `microphone_ok = False`, publishes `speech.error(kind="microphone")`, and wraps Whisper transcription in `try/except` publishing `speech.error(kind="stt")` instead of killing the listening thread. |
| `ConversationManager` | Publishes `conversation.processing` once a request is accepted, and `conversation.error` in its existing exception handler. |
| `TTSService` | Publishes `speech.speaking_started` immediately before playback, `speech.speaking_finished` in `finally` (success or failure), `speech.error(kind="tts")` on failure; returns `bool`; numbers each utterance; deletes the temporary WAV. |

Behaviour now matches the pipeline:

- **Listening**: continuous capture means LISTENING is the resting state while
  a visitor is present. A recognised utterance moves the face to THINKING when
  the conversation manager accepts it, not when the microphone hears noise.
- **Thinking**: covers retrieval + LLM generation, i.e. from
  `conversation.processing` until playback starts.
- **Speaking**: begins exactly at `speech.speaking_started` and ends at
  `speech.speaking_finished`.
- **Failures**: STT, LLM and TTS failures each produce ERROR plus a friendly
  caption, then recovery. The raw error goes to the existing logger only.
- **Greeting audio**: the project has no greeting workflow (Phase 0 recorded
  that `visitor.detected` had no subscriber and no greeting path). This phase
  therefore greets visually with GREETING plus a caption and does not synthesise
  speech on the visitor's arrival. Speaking a greeting is a conversation-design
  feature for a later phase, not a UI integration.

---

## 9. Thread-safety implementation

- **EventBus callbacks never touch widgets.** `KioskPresenter` registers one
  handler per event; each handler's only job is
  `self._dispatch.emit(name, payload)`. `_dispatch` is a `pyqtSignal(str,
  object)` connected to the mapping slot with `Qt.QueuedConnection`, so the
  mapping always runs on the GUI thread regardless of the publisher
  (vision pump, speech thread, conversation thread, TTS thread).
- **Camera processing moved off the GUI thread.** `VisionPump` is a `QThread`
  that calls `vision.process(annotate=False)` at the configured FPS and
  discards the result. It stops through a `threading.Event` and is joined
  (3 s timeout) in `closeEvent` before the camera is released. This removes the
  Phase 0 "inference on the Qt main thread" risk for the kiosk path.
- **No frames are produced for display.** `VisionService.process(annotate=False)`
  returns before copying the frame, drawing boxes and drawing the FPS counter.
- **Event loop work is non-blocking.** No `sleep`, no inference and no audio
  playback happens on the GUI thread. Piper generation and playback remain on
  the conversation thread (as before).
- **No blocking waits on close** beyond the pump join (bounded) and the
  existing service stops.
- **Leak-free subscriptions.** `KioskPresenter.shutdown()` unsubscribes every
  handler it registered. Check G asserts the subscriber count returns to zero.
- **No duplicate subscriptions**: the presenter is created exactly once per
  window; the window is created once per process.

---

## 10. Files created

| File | Purpose |
|---|---|
| `app/gui/kiosk_background.py` | Dark radial/vignette backdrop, gradients cached per resize |
| `app/gui/caption_widget.py` | Fading caption with token-safe timers and wrapping |
| `app/gui/kiosk_presenter.py` | Event-to-face/caption mapping, watchdog, dedupe |
| `app/gui/vision_pump.py` | `QThread` that runs vision processing invisibly |
| `tests/test_kiosk.py` | Phase 2 checks A-I |
| `docs/PHASE_2_REPORT.md` | This report |

## 11. Files modified

| File | Change |
|---|---|
| `app/gui/main_window.py` | Card layout replaced by the fullscreen kiosk; presenter, caption, vision pump, kiosk controls and shutdown |
| `app/gui/theme.py` | Added `KIOSK_STYLE`; `STYLE` kept for compatibility |
| `app/main.py` | Passes `container` and `speech`, launches the kiosk fullscreen |
| `app/speech/tts_service.py` | Playback lifecycle events, utterance IDs, `bool` result, temp-WAV cleanup |
| `app/speech/speech_service.py` | Microphone-error reporting, STT failure handling, `_publish` helper |
| `app/speech/audio_capture.py` | Optional `on_error` callback |
| `app/brain/conversation_manager.py` | `conversation.processing` and `conversation.error` |
| `app/vision/vision_service.py` | `annotate` switch, unknown-visitor dedupe, `visitor.left` |
| `tests/test_kiosk.py` | Added after the first run (test expectations corrected) |

## 12. Files intentionally left untouched

- `app/gui/robot_widget.py`, `app/gui/face_renderer.py`,
  `app/gui/face_animation.py` - the Phase 1 face.
- `app/core/event_bus.py` and `app/core/state_manager.py` - already
  thread-safe; no parallel event system or state machine added.
- `app/vision/*` apart from `vision_service.py` (camera, tracker, recogniser,
  database, embeddings).
- `app/brain/llm.py`, `knowledge_base.py`, `prompt_manager.py`, `parser.py`.
- `app/speech/stt.py`, `vad.py`.
- `app/gui/camera_widget.py`, `chat_widget.py`, `status_widget.py` - unused by
  the kiosk, kept on disk.
- `config/*`, `data/*`, `models/*`, `requirements*`, `docs/PHASE_0_REPORT.md`,
  `docs/PHASE_1_REPORT.md`.
- No `flow.md` or `design.md` was created.

No file was deleted in this phase. `GlassCard` (an internal helper of the old
layout) was removed as part of replacing that layout.

---

## 13. Tests performed

All automated runs were on the offscreen Qt platform with stubbed services; no
camera, microphone, models or network were used. Real-hardware checks are
listed separately and honestly.

### A. Phase 1 regression

`tests/test_face.py` (all 10 checks) is executed as a subprocess by check A. It
also covers all eight states, transitions, blinking, gaze, mouth shapes,
resizing and the Phase 1 MainWindow integration.

### B. Fullscreen behaviour

- Frameless window flag present.
- `enter_kiosk()` produces a fullscreen window; F11 toggles; Esc leaves kiosk
  mode while the app stays visible; a second Esc closes it.
- No `GlassCard`, `CameraWidget`, `QPushButton`, menu, toolbar, tab widget or
  status bar exists anywhere in the window tree.
- Diagnostics hidden by default; F10 shows and hides them.
- Caption empty and face IDLE at startup.

### C. Caption behaviour

- Show, update, fade in, fade out, clear.
- Auto-clear after its duration.
- An older timer cannot clear newer text.
- Word wrap on, truncation to 260 characters with ellipsis, font size clamped
  16-34 px.

### D. State mapping

Injected through the real EventBus and observed on the real `RobotWidget`:
`idle → greeting → listening → thinking → speaking → happy → listening →
error → listening → idle`, plus the thinking watchdog, stale-TTS protection
and microphone-unavailable behaviour.

### E. Duplicate protection

Three `visitor.detected` events for the same visitor produce one greeting, and
the number of `QTimer` children stays constant. A stale TTS completion does not
end a newer utterance. Duplicate subscriptions are impossible by construction
and are asserted after shutdown.

### F. Thread safety

Four worker threads publish `visitor.detected` simultaneously. Before the GUI
event loop is pumped the face is still IDLE (proof that no callback touched a
widget), and after pumping all four events are applied, all `face_state`
emissions are observed on the main thread, and all four visitors are recorded.

### G. Application launch

```text
AI Admission Robot Starting...
Starting Vision Service...
Camera ready (1280x720 @ 30 FPS)
Starting Speech Service...
Backend services started.
GUI started.
```

`python -m app.main` ran for 25 s in real fullscreen and was ended by an
external timeout. The gap between "camera ready" and "GUI started" is the model
load the application already did; the kiosk appeared and stayed responsive. No
traceback, no Qt rendering warning and no timer warning were logged. The face
animated for the whole run and the vision pump processed frames without
displaying them. `python -c "import app.main"` also imports cleanly, so the
vision-before-Qt order is intact.

That 25 s run proves startup, **not** shutdown: an external timeout is not a
clean close. The continuation session therefore repeated the launch and closed
the window through the normal close path instead; see section 20.4.

### H. End-to-end interaction

**Not performed with a real visitor.** It requires a person in front of the
camera speaking into the microphone. The event path that such a session
produces is covered by test D, and the real launch proves the services and GUI
run together, but no claim is made that a live camera/microphone session was
completed.

What the continuation launch did prove about the real pipeline: the real
microphone captured an utterance, Whisper transcribed it ("What are you
doing?"), retrieval returned 2,964 characters of context and llama.cpp began
generating. No face was in front of the camera, so no `visitor.detected` or
`visitor.unknown` event was produced, and the window was closed before playback
finished. A full visitor conversation therefore remains **NOT TESTED**.

### I. Continuous operation

The 15-minute run was **completed during the continuation session**: 900.0 s,
450 event cycles, no failures, no timer growth, flat RSS. Full output and the
measurements are in section 14 and section 20.3. The earlier draft of this
report carried a placeholder for this section while describing a completed
run; that inconsistency is what the continuation session resolved.

### J. Jetson readiness

Review only, no hardware:

- Python 3.8 compatibility: the new GUI modules use `from __future__ import
  annotations`, f-strings, and no 3.9+ syntax. The union syntax
  (`Callable[...] | None`) in `audio_capture.py` is guarded by the existing
  `from __future__ import annotations`, so it is annotation-only. `dict[str,
  ...]` in `vision_service.py` predates this phase.
- PyQt5 5.15.11 is the same version already validated for the project.
- `showFullScreen()` with `FramelessWindowHint` works under X11/Wayland and
  under the Qt eglfs/linuxfb plugins used on kiosk devices; the exact
  Jetson display stack was not available to test.
- CPU rendering: the face is still software-rasterised at 30 FPS; adding the
  vision pump moved inference off the GUI thread, which should *improve* the
  kiosk on a 6-core Xavier NX, but this is not measured.
- Memory: no new large buffers; the kiosk adds three small widgets and one
  thread.
- Camera invisibility: guaranteed by construction (no pixmap path, no camera
  widget) rather than by display configuration.
- Threading: the pump and the existing daemon threads are the only workers; no
  Qt object is created or destroyed from them.

Still to verify on hardware: sustained FPS during speaking + detection,
fullscreen behaviour under the actual display server, audio device
availability, and whether 30 FPS should become 24 FPS to save power.

---

## 14. Test results

### Automated

```text
Greeto kiosk checks (platform: offscreen)
[PASS] A. Phase 1 face checks
       frameless, fullscreen toggle, Esc and F10 verified
[PASS] B. fullscreen borderless kiosk
[PASS] C1. caption show, update, fade and clear
[PASS] C2. older caption timers cannot clear newer text
       max_chars=260  font=33px
[PASS] C3. caption wrapping, truncation and font size
       1024x768  face=928x575  caption=737x115  font=25px
       1280x720  face=1184x534  caption=921x108  font=23px
       1366x768  face=1270x575  caption=983x115  font=25px
       1920x1080 face=1824x840  caption=1382x162  font=34px
[PASS] B2. responsive layout at four resolutions
       no camera widget, no pixmap, no frames displayed
[PASS] B3. camera feed never on screen
       idle -> greeting -> listening -> thinking -> speaking
       -> happy -> listening -> error -> listening -> idle
[PASS] D. application events drive face states
[PASS] D2. stale TTS completion ignored
[PASS] D3. microphone unavailable is handled
       3 detection events -> 1 greeting, timer count stable
[PASS] E. one visitor causes one greeting
       4 worker threads -> 4 queued events, all applied on GUI thread
[PASS] F. cross-thread events reach the GUI safely
       subscriptions released, all timers stopped
[PASS] G. clean shutdown
RESULT: ALL CHECKS PASSED
```

Face geometry is the allocated layout region; `RobotWidget` paints a square of
`min(width, height)` centred inside it, so with `face=1184x534` the visible face
is a 534 px square centred horizontally, with the caption 16 px below.

### Simulated vs real

| Test | Nature |
|---|---|
| A, B, C, D, D2, D3, E, F, G | Automated, offscreen, stubbed services, real EventBus + real widgets |
| H | Not performed (requires a real visitor) |
| I | Automated continuous run with simulated events, real widgets and timers, 15 min |
| Application launch | Real: actual camera, real services, real fullscreen window, 25 s (startup only) |
| Continuation launch | Real: camera, microphone, Whisper, llama.cpp, fullscreen 1920x1080, closed through the normal window close path, exit code 0 |

The offscreen platform prints `This plugin does not support
propagateSizeHints()`; that is Qt-internal noise from resizing windows without a
window manager and is not produced by application code.

### Continuous run

Actually executed in the continuation session (`python tests/test_kiosk.py
--stability-minutes 15`, offscreen platform, real widgets and real timers,
stubbed backends, events injected through the real EventBus):

```text
[PASS] G. clean shutdown                                              (0.41s)
       ran 900.0s  cycles=450  face_fps=30.3  cpu=14.1%  rss=94.4MB (start 94.3MB)
       rss samples: 0s:94MB, 140s:94MB, 280s:94MB, 420s:94MB, 560s:94MB, 700s:94MB, 840s:94MB
[PASS] I. continuous kiosk run (15 min)                               (900.51s)
--------------------------------------------------------------------------
RESULT: ALL CHECKS PASSED
```

- Duration: 900.0 s (the whole command took ≈16.4 min including checks A-G).
- 450 full cycles: visitor detected → processing → reply → speaking started →
  speaking finished → visitor left, plus a recoverable `speech.error` every
  third cycle (150 injected errors).
- Face animation never stopped; the `QTimer` child count never changed from its
  startup value (asserted every loop iteration).
- Memory: 94.3 MB → 94.4 MB (+0.1 MB) across 15 minutes, flat at every 140 s
  sample.
- CPU: 14.1 % of one core; face FPS 30.3.
- No traceback, no exception, no Qt warning other than the two cosmetic ones
  already listed in section 16 (missing font directory, offscreen
  `propagateSizeHints()`).

This is a simulated, offscreen, hardware-free run. It is **not** evidence of a
real camera/microphone session on a kiosk device, and it is not Jetson
measurement.

---

## 15. Performance observations

Measured on this Windows host, offscreen platform where stated; **not measured
on Jetson Xavier NX**.

- The face animation still runs at ≈30 FPS (Phase 1 measurement: 30.3 FPS,
  7.9-8.3 % of one core, RSS flat at ~90 MB over 10 minutes).
- In the real launch (section 20.4) the face emitted 330 frames in the 13 s
  sampling window (≈25 FPS) with the camera, Whisper and the vision pump all
  active, i.e. real-device rendering costs more than the offscreen harness. The
  GUI never stalled: the vision pump delivered 94 processed frames with 0
  errors during the same window.
- The kiosk continuous run above reports its own face FPS, CPU share and RSS;
  those numbers include the presenter, caption animations and event churn.
- Moving vision processing to `VisionPump` removes the camera pipeline from the
  GUI thread. Before this phase, the 33 ms camera timer ran detection,
  recognition and annotation on the GUI thread (Phase 0 risk S1). No before/after
  on-device measurement exists, so no speed-up is claimed - only that the GUI
  thread no longer runs that work.
- `process(annotate=False)` avoids one full-frame copy and two OpenCV draw
  passes per processed frame, plus the FPS overlay text.
- Caption fades animate a `QGraphicsOpacityEffect` only for 320/520 ms per
  change; idle captions cost nothing.

Caveats: offscreen rendering does not flush to a compositor, and CPU share is
per-core process time, not system-wide load.

---

## 16. Known limitations

1. **Greeting is visual, not spoken.** No greeting workflow exists in the
   pipeline, and this phase did not invent one. GREETING + caption is shown on
   arrival; Piper is only used for answers.
2. **End-to-end interaction untested with a real visitor** (test H). The
   event path is covered by injected events, not by a live session.
3. **Departure latency.** `visitor.left` relies on the tracker's existing
   30-missing-frame expiry, so the face stays in LISTENING for roughly 3 s
   after the visitor leaves.
4. **Jetson unverified.** Fullscreen, sustained FPS, audio devices and the
   display stack must be checked on the Xavier NX.
5. **HAPPY is a short fixed beat (1.4 s)** after every successful answer. It is
   a deliberate success cue, not a reaction to the LLM's `emotion` field, which
   is still unused.
6. **CONFUSED is not driven by events.** It remains available through the face
   API and `scripts/face_preview.py`, but no application event maps to it; the
   LLM's `emotion`/`action` fields would be the natural source later.
7. **Diagnostics are minimal** (state / microphone / visitor count) and hidden;
   they are development-only and never part of the visitor experience.
8. **Pre-existing teardown warning.** The `QObject::~QObject: Timers cannot be
   stopped from another thread` message documented in Phase 0 can still appear
   when the process is force-killed with live daemon threads. It was not
   observed in this phase's clean close or in the 25 s launch (which was killed
   externally without the warning), and it is not caused by the face or the
   presenter timers.
9. **Offscreen font warning.** `QFontDatabase: Cannot find font directory ...`
   appears on this machine because Qt no longer ships fonts; it is cosmetic and
   pre-existing.
10. **Microphone failure detected asynchronously.** A microphone that fails
    after startup is reported when `AudioCapture` attempts to open the stream;
    the face then stays IDLE instead of LISTENING.

---

## 17. Remaining Jetson work

1. Run `python -m app.main` on the Xavier NX and confirm the kiosk fills the
   target panel with no window decorations under its display server.
2. Measure sustained FPS during speaking + detection at 1280x720 and
   1920x1080; decide whether 24/30 FPS is the right power trade-off.
3. Confirm `aplay`/`paplay` availability for Piper playback and that
   `speech.speaking_started`/`finished` bracket the audible output on that
   device.
4. Verify microphone open failure and recovery messages on the device.
5. Confirm camera capture still works purely as an input with the vision pump.
6. Watch RSS over a multi-hour run; the Windows 15-minute run shows the shape,
   the device run is the real test.
7. Re-check `requirements_jetson.txt` installs (llama.cpp CUDA, InsightFace).

---

## 18. Phase 3 readiness

The visitor-facing interface is ready for the next phase:

- Fullscreen, borderless kiosk with a large animated face as the only focal
  point.
- Real application events drive the face automatically, with guarded recovery.
- One caption line that updates safely and never fights with older timers.
- Camera and all processing invisible; no preview, boxes, confidence, logs or
  visitor data on screen.
- Clean shutdown with no leaked subscriptions or running timers.
- Phase 1 face suite still passes unchanged.

Phase 3 can build on this without reworking it. The main open item for later
phases is the conversation experience itself: spoken greeting on arrival,
using the LLM's `emotion` field for CONFUSED/HAPPY nuance, and possibly captions
for the visitor's own utterance.

---

## 19. Acceptance checklist

| Requirement | Status |
|---|---|
| Existing Phase 1 implementation preserved | Yes |
| MainWindow opens in fullscreen kiosk mode | Yes |
| Window is borderless | Yes |
| Dark minimalist background | Yes |
| Animated face centred and appropriately scaled | Yes - 840 px square at 1080p |
| Camera feed never displayed | Yes - no pixmap path, no camera widget |
| No unnecessary dashboard elements | Yes |
| Caption component implemented | Yes |
| Caption text updates correctly | Yes |
| Caption timers do not interfere with newer messages | Yes - token + timer stop |
| Actual application events control facial states | Yes - EventBus, real widgets |
| Visitor detection connected | Yes |
| Duplicate greetings prevented | Yes - source dedupe + presenter set |
| Listening state connected | Yes |
| Thinking state connected | Yes |
| Speaking connected to actual playback lifecycle | Yes - TTS start event |
| Speech completion returns to a valid state | Yes - HAPPY then LISTENING/IDLE |
| Recoverable errors produce feedback | Yes - microphone/STT/LLM/TTS/timeout |
| Cross-thread UI updates are safe | Yes - queued signal + worker thread |
| Existing Phase 1 tests pass | Yes |
| Fullscreen behaviour tested | Yes - offscreen + real launch |
| Application startup tested | Yes - 25 s real run |
| Clean shutdown tested | Yes - normal window close on real hardware, exit code 0, no forced kill (section 20.4) |
| No new Qt timer warnings introduced | Yes |
| `docs/PHASE_2_REPORT.md` created | Yes - this file |
| No Phase 3 work started | Yes |
| Nothing committed or pushed | Yes |

---

## 20. Continuation audit (resumed session)

Phase 2 was implemented in an earlier session that hit a usage limit. This
section records what that session left behind, what was actually missing, and
the evidence gathered when execution resumed. **No application code was changed
in the continuation session** - every Phase 2 module was inspected and found
complete, and every defect claim below was checked against a real run rather
than assumed from the earlier report.

### 20.1 What was already complete before continuation

Verified by inspection (`git status --short`, `git diff --stat`, `git diff` and
reading each file):

- `app/gui/kiosk_background.py`, `caption_widget.py`, `kiosk_presenter.py`,
  `vision_pump.py` - present and correct.
- `app/gui/main_window.py` - frameless fullscreen kiosk, presenter wiring,
  Esc/F11/F10, ordered shutdown.
- `app/main.py`, `app/speech/*`, `app/brain/conversation_manager.py`,
  `app/vision/vision_service.py` - the lifecycle events listed in section 6.
- `tests/test_face.py` and `tests/test_kiosk.py` - both suites implemented:
  checks A, B, B2, B3, C1-C3, D, D2, D3, E, F, G plus the opt-in
  `--stability-minutes` run (check I).
- `docs/PHASE_2_REPORT.md` - written, apart from unsupported test evidence.

### 20.2 What was genuinely incomplete when execution stopped

The code was done; the **evidence** was not.

1. `python tests/test_kiosk.py --stability-minutes 15` was implemented but had
   never been executed to completion. No prior 15-minute kiosk log existed
   (`.stability.log` in the repository root is the Phase 1 face-only 10-minute
   run, not the kiosk test).
2. Consequently section 14 of this report carried
   `KIOSK_STABILITY_PLACEHOLDER` while section 13.I simultaneously described a
   completed continuous run. That claim was unsupported and is now replaced by
   the real output (section 14).
3. Clean shutdown had not been demonstrated. The only real launch on record
   (`logs/system.log`, 14:40:42) stops at `GUI started.` with no
   `Shutdown complete.` - it was ended by an external timeout, which is not a
   clean close.
4. The continuation session also re-ran the full suites to confirm nothing had
   regressed since they were last executed.

No missing subscription, stuck face state, caption race, duplicate greeting,
mis-timed TTS event, worker-thread widget access, leaking timer or failing
vision worker was found. Nothing needed to be fixed.

### 20.3 Exact commands and actual results

All from the project root, branch `test`, Windows host, same interpreter.

| # | Command | Result |
|---|---|---|
| 1 | `git status --short` / `git diff --stat` / `git branch --show-current` | branch `test`; 27 modified, 4 deleted, 7 new paths; unchanged by this session |
| 2 | `python tests/test_face.py --skip-slow` | **PASS** - all 10 Phase 1 checks; test 8 skipped by flag |
| 3 | `python tests/test_kiosk.py` | **PASS** - checks A-G (13 checks), ≈82 s total; I skipped as opt-in |
| 4 | `python tests/test_kiosk.py --stability-minutes 0.5` | **PASS** - shortened validation run before committing to 15 min: 30.0 s, 15 cycles, 30.3 FPS, 11.2 % CPU, RSS 94.0 → 94.0 MB |
| 5 | `python tests/test_kiosk.py --stability-minutes 15` | **PASS** - all checks including I; 900.51 s, 450 cycles, 30.3 FPS, 14.1 % CPU, RSS 94.3 → 94.4 MB |
| 6 | `python -c "import app.main"` | **PASS** - `import OK`; vision-before-Qt order intact |
| 7 | Real launch, then normal window close (see 20.4) | **PASS** - exit code 0, `Shutdown complete.` logged |

Run 7 executed the real `app.main` startup path unchanged; the only addition was
a `QTimer.singleShot(25000, window.close)` so the window closes itself through
`closeEvent` instead of being killed. That harness was fed to Python on stdin
and is not stored in the repository, so it cannot drift from `app/main.py`.

### 20.4 Real launch, runtime measurement and clean shutdown

Executed at 15:32:19 on the Windows host with the actual camera, microphone,
Whisper `ggml-tiny.en`, llama.cpp, InsightFace and the real fullscreen window.

```text
AI Admission Robot Starting...
Conversation Manager started.
Starting Vision Service...
Camera ready (1280x720 @ 30 FPS)
Vision Service started.
Starting Speech Service...
Speech Service started.
Backend services started.
GUI started.

RUNTIME: fullscreen=True visible=True kiosk=True size=1920x1080
RUNTIME: face_state=listening animator_running=True face_frames=330
RUNTIME: vision_pump_running=True frames_processed=94 errors=0
RUNTIME: camera_widget_present=False caption=''
RUNTIME: top_level_widgets=1

Releasing camera...
Camera released.
Vision stopped.
Stopping services...
Speech Service stopped.
Vision stopped.
Shutdown complete.
EXIT=0
```

Interpretation:

- Borderless fullscreen at 1920x1080, exactly one top-level widget: no camera
  preview, no dashboard, no buttons.
- The face was already in LISTENING at the sample point, driven by a real
  microphone event path, with the animator running.
- The vision pump processed 94 frames with 0 errors on its own thread while the
  GUI stayed responsive, and nothing was drawn from those frames.
- **Clean shutdown through the normal close path:** the window's close event
  stopped the pump, the presenter (unsubscribing every handler), the caption and
  face timers, then released the camera and stopped the services. The process
  exited with code 0 and never needed a forced kill. This replaces the earlier
  externally-timed-out launch as shutdown evidence.
- No `QObject::~QObject: Timers cannot be stopped from another thread` warning
  appeared, in this run or in the 15-minute run.
- The real microphone captured an utterance and the conversation pipeline ran,
  but no face was detected and playback had not finished when the window closed.

### 20.5 Hardware and interaction status

| Item | Status | Meaning |
|---|---|---|
| Windows host, real camera 1280x720@30 | **PASS** | Opened, captured and was released cleanly |
| Windows host, real microphone + Whisper | **PARTIAL** | Real utterance transcribed and the LLM started generating; no complete spoken answer was observed in this run |
| Real fullscreen kiosk on Windows | **PASS** | 1920x1080, frameless, single window, face animated |
| Real visitor / face detection | **NOT TESTED** | No person in front of the camera; event path covered by tests D, D2, D3, E |
| Simulated continuous operation, 15 min | **PASS** | Offscreen, stubbed backends, real widgets, timers and EventBus |
| Sustained hardware kiosk run (≥15 min, camera + speech) | **NOT TESTED** | Would tie up the camera and microphone for the full duration |
| Jetson Xavier NX | **NOT TESTED** | No device available in this session |

No claim is made that a live visitor interaction, a real microphone
conversation end to end, or any Jetson behaviour passed.

### 20.6 Final Phase 2 acceptance checklist

| Criterion | Status | Evidence |
|---|---|---|
| Fullscreen borderless kiosk works | **PASS** | Test B; real 1920x1080 run |
| Face centred and animated | **PASS** | Tests 1-7; 330 frames in 13 s on real launch |
| Camera feed never displayed | **PASS** | Test B3; `camera_widget_present=False` at runtime |
| Captions work, timer races prevented | **PASS** | Tests C1-C3 |
| Actual application events drive face states | **PASS** | Test D via real EventBus |
| Visitor greeting deduplication | **PASS (simulated)** | Test E; real visitor not observed |
| SPEAKING reflects actual TTS playback | **PASS (simulated)** | Test D, D2; TTS start event published before playback |
| Cross-thread UI updates are safe | **PASS** | Test F; vision pump off the GUI thread |
| Vision does not block the GUI | **PASS** | Test F; 94 pump frames / 0 errors during real run |
| Phase 1 regression tests pass | **PASS** | Run 2 and check A |
| Phase 2 tests pass | **PASS** | Run 3 |
| Shutdown is clean | **PASS** | Run 7, exit code 0, full stop sequence |
| Stability testing accurately reported | **PASS** | Run 5; section 14 holds measured values, no placeholder |
| Documentation matches implementation | **PASS** | This report, corrected here |
| Real visitor interaction | **NOT TESTED** | Requires a person at the kiosk |
| Jetson Xavier NX validation | **NOT TESTED** | No hardware |

### 20.7 What the continuation session changed

| File | Change |
|---|---|
| `docs/PHASE_2_REPORT.md` | Replaced the stability placeholder with real output and measurements; corrected the launch/shutdown claims; added the simulated-vs-real and hardware status tables; added this section; updated the checklist |

No source file, test, configuration or data file was modified. No file was
deleted. Nothing was committed, pushed, reset or cleaned.

### 20.8 Remaining limitations

1. The 15-minute run is an offscreen simulation with stubbed backends. It proves
   timer, subscription and memory behaviour, not sustained camera/microphone
   operation on the target device.
2. Real visitor interaction is untested (no person at the camera).
3. Jetson Xavier NX is untested; all section 17 items remain open.
4. Face rendering measured ≈25 FPS under real load versus 30.3 FPS offscreen;
   whether that holds on the Xavier NX is unknown.
5. The greeting is visual, not spoken (unchanged from section 16.1).
6. Departure latency of roughly 3 s is inherited from the tracker's existing
   30-missing-frame expiry.

---

## How to run

```bash
# Full kiosk (camera + speech + face)
python -m app.main

# Phase 2 checks (offscreen, no hardware)
python tests/test_kiosk.py
python tests/test_kiosk.py --stability-minutes 15

# Phase 1 face checks
python tests/test_face.py

# Face-only preview (development)
python scripts/face_preview.py
```

Development keys: `Esc` leaves kiosk mode (second `Esc` closes), `F11` toggles
fullscreen, `F10` toggles diagnostics.
