# Greeto - Phase 1 Report

**Programmatic Animated Digital Face**

| Field | Value |
|---|---|
| Phase | 1 - Animated Digital Face Development |
| Branch | `test` |
| Status | Complete (see section 17 for scope boundaries) |
| Host | Windows 10/11, Python 3.11.1, PyQt5 5.15.11, software rasterizer (`offscreen`) |
| Target hardware | NVIDIA Jetson Xavier NX - **not tested in this phase** |
| Report date | 2026-11-18 |

This report is a continuation record. Phase 1 had been started by a previous
execution and stopped part way through; the work below documents what was
already on disk when this execution resumed, what was added, and what was
verified. Nothing was committed to Git.

---

## 1. Initial Phase 1 progress discovered

The working tree already contained a complete-looking Phase 1 slice. The audit
(`git status --short`, `git diff --stat`, plus reading every GUI file) found:

| Artefact | Lines | State found on disk |
|---|---|---|
| `app/gui/face_renderer.py` | 526 | Complete stateless `QPainter` renderer: aura, face panel, two eyes, pupils, eyebrows, mouth. Asset-free. |
| `app/gui/face_animation.py` | 501 | Complete controller: `FaceParameters`, `FaceAnimator`, eight state presets, gaze scheduler, blink scheduler, speaking waveform, exponential smoothing, one `QTimer`. |
| `app/gui/robot_widget.py` | 191 | Rewritten from the old `QLabel`/`QPixmap` mascot into a `QWidget` that paints the face. All eight state setters present. |
| `tests/test_face.py` | 949 | Nine offscreen checks: rendering, no-asset guard, blinking, gaze, mouth/speaking, all states, transitions, resizing, MainWindow integration with a stubbed vision service, optional long stability run. |
| `scripts/face_preview.py` | 240 | Standalone interactive launcher. Keys `1`-`8`, `D` for auto-demo, `Q`/`Esc` to quit, `--size`, `--fps`, `--demo`. |
| `app/gui/main_window.py` | 260 | Embeds `RobotWidget` in the right-hand card and calls `shutdown()` in `closeEvent`. The load-bearing "vision before Qt" import order from Phase 0 is intact. |
| `.face_tmp/` | - | Fourteen PNG snapshots produced during the previous visual iteration (not tracked, no runtime role). |
| `.stability.log` | - | Output of a previously completed 10-minute run: **ALL CHECKS PASSED**, 30.3 fps, 8.3 % CPU, RSS flat at ~90 MB. |

What was **missing** or unverified at resume time:

1. `docs/PHASE_1_REPORT.md` did not exist.
2. The machine-readable test suite proved four mouth looks (idle, happy,
   speaking, error) but never verified the full required mouth catalogue
   (closed, small open, large open, surprised, concerned).
3. `.gitignore` did not cover the `Phase 1` artefacts (`.face_tmp/`,
   `.stability.log`) that the test and preview scripts produce.
4. The previous test log could not be assumed valid for the current code, and
   no report recorded it.
5. One docstring in `robot_widget.py` claimed "no per-frame allocation beyond
   the painter", which is not true (gradients and paths are constructed per
   frame). Corrected rather than left as a false claim.

No file was reset, cleaned, deleted or overwritten. Existing implementation was
inspected first and preserved; only the gaps above were closed.

---

## 2. Work already completed before continuation

This is the previously implemented work, confirmed by reading the source (not
inferred from filenames or method presence):

- Asset-free `QPainter` rendering of the whole face.
- Normalised geometry: every dimension is a fraction of `min(width, height)`.
- A single-neon-palette colour identity (`FacePalette`) with warm blending.
- Independent left/right eye openness and smile geometry.
- Luminous pupils with per-state gaze drift.
- Blink system with per-eye offset, double blinks and state-scaled intervals.
- Independent left/right eyebrow raise and tilt.
- A mouth built from two symmetric cubic segments, with filled interior for
  open shapes and a stroked lip line for near-closed shapes.
- Eight expression presets with per-parameter easing time constants.
- A single `QTimer` at 30 FPS, idempotent start/stop, pause on hide.
- The standalone preview launcher and the offscreen test suite.
- `MainWindow` embedding plus `shutdown()` on window close.

---

## 3. Work completed during this continuation

1. **Re-verified the implementation end to end** instead of assuming the old
   passing log still applied: re-ran all fast checks (all passed), rendered
   all eight states to PNG and inspected them visually, and rendered the ten
   required mouth shapes for inspection.
2. **Added `tests/test_face.py` check 4b**: every required mouth shape is drawn
   from frozen parameters and all ten frames are asserted byte-distinct. This
   is what makes "surprised", "large open", "closed" and "concerned" verified
   rather than merely drawable in principle.
3. **Corrected the inaccurate `paintEvent` docstring** in `robot_widget.py`.
4. **Ignored Phase 1 test artefacts** in `.gitignore` (`.face_tmp/`,
   `.stability.log`).
5. **Ran a real application launch** (`python -m app.main`) and inspected the
   log for startup ordering, GUI creation and Qt timer warnings.
6. **Re-ran the full 10-minute continuous animation test on the final code**
   (all fast checks plus the stability run in one process).
7. **Wrote this report.**

No runtime rendering or animation behaviour was changed, because none of it
was found broken. Fixing working animation would have been out of scope.

---

## 4. Rendering architecture

Three layers with one-directional dependencies:

```
RobotWidget (QWidget)        owns the timer via FaceAnimator, receives paintEvent
      |
FaceAnimator (QObject)       produces FaceParameters numbers only
      |
FaceRenderer                 stateless; paints one frame from one parameter set
```

`FaceRenderer.paint(painter, rect, params)` is the only entry point. It derives
`side = min(rect.width(), rect.height())`, centres the drawing box, and paints
in this order: aura, panel, both eyebrows, both eyes, mouth. Everything is
`QPainterPath`, `QPen`, `QBrush`, `QLinearGradient` and `QRadialGradient`.

Layout constants (fractions of `side`):

| Element | Constant | Value |
|---|---|---|
| Panel | `PANEL_W` / `PANEL_H` / `PANEL_RADIUS` | 0.80 / 0.70 / 0.155 |
| Eye centre | `EYE_DX` / `EYE_DY` | ±0.175 / -0.045 |
| Eye size | `EYE_HALF_W` / `EYE_HALF_H` | 0.100 / 0.054 |
| Pupil | `PUPIL_R` | 0.029 |
| Brow | `BROW_DY` / `BROW_HALF_W` | -0.150 / 0.086 |
| Mouth | `MOUTH_DY` / `MOUTH_HALF_W` | 0.150 / 0.112 |

Because every value is a fraction of the smaller widget dimension, proportions
are identical at every window size. There are no fixed pixel coordinates, no
image loading, no SVG, no OpenGL and no external asset in the render path. The
required `QPainter`-only rule is enforced by test 1b, which greps the three
face modules for `QPixmap`, `QSvgWidget`, `QMovie`, `QtSvg` and for
`.png/.jpg/.jpeg/.gif/.svg` strings.

Colour identity: cyan accent `(56, 224, 255)`, warm accent `(255, 198, 108)`,
dark eye lens top `(7, 24, 36)` to lit bottom, near-white pupil core, dark mouth
interior, translucent slate panel. States change geometry and intensity; the
character stays recognisable.

---

## 5. Facial components

### Face structure

A rounded-rectangle face plate (`QPainterPath.addRoundedRect`) filled with a
three-stop vertical gradient and stroked with an accent border whose alpha
tracks the glow parameter (`18 + 26 * glow`). A horizontal "glass" highlight
line runs near the top edge with a transparent-to-accent-to-transparent linear
gradient.

### Glow / aura

A radial gradient halo behind the panel, radius `0.48 * side`, fully inside the
widget (a halo clipped by the widget edge can produce a visible seam). Alpha
responds to `glow`: `14 + 26 * glow` at the centre, `5 + 10 * glow` at 62 %,
zero at the rim. No blur effects are used, which keeps the cost predictable.

### Eyes (independent)

Each eye is a closed almond made of two cubic segments. Height is
`EYE_HALF_H * eye_open_l/r`, so openness *is* the eyelid motion — there is one
geometry, not a separate lid overlay that could desynchronise. The upper lid
bulge is `1.30 * half_h`; the lower lid is
`(1.46 - 1.95 * eye_smile) * half_h`, clamped to at least `0.50 * half_h`.
Positive `eye_smile` lifts the lower lid into the narrow curved eye that reads
as a genuine smile; negative values widen it.

Each eye gets a wide soft glow stroke (alpha `24 + 18 * glow`), a tighter inner
stroke, a dark lens gradient lit from below, and a clipped radial light pooling
in the lower half. A closed eye (`openness < 0.14`) draws a lid line whose
alpha rises as the eye closes, so a blink never looks like an empty socket.

Left and right eyes are drawn from the same routine with `direction = -1/+1`
and separate params, which is why a blink can be asymmetric.

### Pupils

A luminous focus point rather than an eyeball: a radial halo (2.35x radius,
alpha `105 + 45 * glow`) plus a core gradient from near-white through accent to
translucent. Position:
`px = ex + pupil_x * half_w * 0.52`, `py = ey + pupil_y * half_h * 0.38`.
The offsets arrive from the animator and are clamped by the smoothing, so the
pupil can never escape the eye.

### Eyebrows (independent)

A cubic arch from the outer end to the inner end. `brow_raise_*` translates it
by up to `0.030 * side`; `brow_tilt_*` moves the inner end by `0.020 * side`
while the outer end moves `0.006 * side`, which is what produces the asymmetric
questioning look. Drawn as a soft under-stroke (2.6x width, accent, low alpha)
plus a bright core stroke. Left and right are fully independent.

### Mouth

Two symmetric cubic segments, upper and lower, closed into one path. The
controlling parameters are:

- `mouth_open` -> vertical half-open `0.052 * side` (0 = closed lip line).
- `mouth_width` -> horizontal half-width `0.112 * side`, clamped to 0.5-1.6.
- `mouth_curve` -> vertical offset `0.052 * side` (positive smiles, negative
  frowns).

Open shapes get a filled dark interior gradient with an accent rim. Near-closed
shapes skip the fill and draw the lip line with a soft outer stroke and a
brighter inner stroke. This is why the same geometry can read as a closed
neutral line, a smile, or a wide open mouth without switching code paths.

---

## 6. Animation architecture

`FaceAnimator` owns exactly one `QTimer` (`Qt.PreciseTimer`,
`interval = max(8, 1000 / fps)`, default 33 ms ≈ 30 FPS). Every frame it:

1. measures real elapsed time and clamps the delta to 0.12 s (a stalled event
   loop cannot teleport the animation);
2. advances gaze scheduling and blink scheduling;
3. eases every parameter towards its state target with
   `easing = 1 - exp(-delta / tau)`, so the motion is frame-rate independent;
4. computes `breath = sin(t * 1.15)`;
5. emits `frame`, which is connected to `QWidget.update`.

Smoothing time constants (`tau`, seconds) are intentionally uneven:

| Parameter | tau |
|---|---|
| `eye_open` | 0.085 |
| `eye_smile` | 0.140 |
| `pupil_x` / `pupil_y` | 0.260 |
| eyebrow raise/tilt (each) | 0.170 |
| `mouth_open` | 0.055 |
| `mouth_width` | 0.160 |
| `mouth_curve` | 0.150 |
| `glow` | 0.320 |
| `warmth` | 0.300 |

Fast mouth response plus slow gaze/glow is what makes the face feel alive
rather than uniformly damped.

### Blinking

- Blink duration: uniform 0.160-0.240 s (the 150-250 ms target).
- Normal interval: uniform 3.0-6.0 s, multiplied by state (`speaking` 1.45,
  `error` 1.30, `listening` 1.15) so a talking face blinks less.
- 18 % of blinks are double blinks, 0.12-0.20 s apart.
- Per-eye start offset of 0-22 ms, so the two eyes are never perfectly
  synchronised.
- Closure curve: 42 % of the duration closing, 58 % opening, smoothstep-shaped.
- The blink is a multiplier applied to the state's eye openness, so a blink can
  never overwrite the expression.
- `_blink_start` is cleared when the blink ends, which is why the eyes cannot
  stay shut permanently after a transition. Re-selecting the current state is a
  no-op, so repeated `set_state` calls cannot start extra blinks or timers.

### Gaze

Per-state target lists with per-state hold times:

| State | Targets (x, y) | Hold (s) |
|---|---|---|
| idle | centre, slight left/right/up drift | 1.8-4.4 |
| listening | centre, small ±x with slight up | 2.6-5.0 |
| thinking | up-left, up-right, mild up-left | 0.9-1.9 |
| speaking | near centre with small wander | 0.8-1.7 |
| happy | centre, slight up | 2.4-4.0 |
| greeting | centre-down, slight right | 2.2-3.6 |
| confused | strongly right, then left | 1.4-2.6 |
| error | down | 3.0-5.0 |

A state change requests a new target immediately (`_next_gaze = 0`), which is
why the eyes visibly react to a transition instead of drifting later.

### Speaking mouth

Three incommensurate sine waves (8.7, 4.1, 15.3 rad/s) plus a slow phrase
envelope at 0.62 rad/s that gates the amplitude down to 0.22 between phrases.
Mouth width also breathes by ±4 % at 3.3 rad/s. The result is continuous,
non-repeating variation rather than one static open mouth. Measured opening
range in the test run: **0.08-0.52** of full open.

### Thinking

Glow pulses ±0.12 around the state target at 2.1 rad/s while the pupils step
between a few up-and-to-the-side targets. Nothing else animates differently, so
the state reads as "processing" without becoming busy.

### Timer safety

- `start()` and `stop()` are idempotent and check `isActive()`.
- `hideEvent` stops the timer, `showEvent` restarts it.
- `closeEvent` calls `shutdown()`; `MainWindow.closeEvent` also calls it.
- The stability test asserts that the number of `QTimer` children never changes
  during a 10-minute run, and that the animator is still running at the end.
- No `sleep`, no blocking work and no drawing inside the timer callback.

---

## 7. Supported states

All eight mandatory states exist as presets in `FaceAnimator.PRESETS`. Preset
values (eye_open, eye_smile, brow raise L/R, brow tilt L/R, mouth_open,
mouth_width, mouth_curve, glow, warmth):

| State | Appearance | Key parameters |
|---|---|---|
| **IDLE** | Relaxed eyes, slight smile, subtle gaze drift, natural blinking | open 0.93, smile 0.10, brows 0.05/0.05, curve 0.30, glow 0.45 |
| **LISTENING** | Wide attentive eyes, both brows raised, small closed mouth | open 1.00, brows 0.42/0.42, width 0.93, curve 0.12, glow 0.62 |
| **THINKING** | Narrowed eyes, asymmetric brows, gaze up and aside, pulsing glow | open 0.84, tilt 0.38/-0.12, width 0.82, curve -0.05 |
| **SPEAKING** | Procedural mouth motion, friendly eyes, reduced blink rate | open 0.95, curve 0.42, animated mouth, glow 0.58 |
| **HAPPY** | Squinted smiling eyes, wide open smile, warm accent, bright glow | open 0.66, smile 0.60, width 1.12, curve 0.90, glow 0.80, warmth 0.55 |
| **GREETING** | High brows, welcoming smile, attentive eyes, warm tint | open 0.97, brows 0.52/0.52, width 1.05, curve 0.68, glow 0.75, warmth 0.30 |
| **CONFUSED** | Clearly asymmetric brows, narrow questioning mouth, side gaze | brows 0.74/0.00, tilt 0.46/-0.26, width 0.74, curve 0.00 |
| **ERROR** | Inner brows dropped/tilted, downturned mouth, dimmed restrained animation | brows -0.26/-0.26, tilt 0.58/0.58, curve -0.55, glow 0.32 |

Each state differs in eye geometry, brow placement, mouth geometry and glow,
not merely in colour. The test suite grabs a frame per state and asserts all
eight are byte-distinct.

### Transitions

All of `IDLE->LISTENING`, `LISTENING->THINKING`, `THINKING->SPEAKING`,
`SPEAKING->LISTENING`, `SPEAKING->IDLE`, `IDLE->GREETING`, `GREETING->IDLE`,
`THINKING->CONFUSED`, `ANY->ERROR` and `ERROR->IDLE` were exercised. There is
no special-case transition code: switching state swaps the target dictionary
and the existing easing carries every parameter from its current position,
which is what preserves animation continuity. The measured worst single-frame
change over all tested transitions was **0.283** of the mouth-curve range - no
abrupt jumps. Since `set_state` returns early on an identical state, repeated
callers (for example a polling loop) cannot restart animations.

---

## 8. Public methods and API

`RobotWidget`:

| Method | Behaviour |
|---|---|
| `set_state(state)` | Accepts `idle`, `listening`, `thinking`, `speaking`, `happy`, `greeting`, `confused`, `error`. Case/whitespace insensitive. Unknown values and `None` fall back to `idle` instead of raising. Returns the resolved state name. |
| `set_idle()` | `set_state("idle")` |
| `set_listening()` | `set_state("listening")` |
| `set_thinking()` | `set_state("thinking")` |
| `set_speaking()` | `set_state("speaking")` |
| `set_happy()` | `set_state("happy")` |
| `set_greeting()` | `set_state("greeting")` |
| `set_confused()` | `set_state("confused")` |
| `set_error()` | `set_state("error")` |
| `current_state()` | Active state name |
| `parameters()` | Live `FaceParameters` object (read-only by convention) |
| `shutdown()` | Stops the timer; safe to call repeatedly |
| `load_idle()` | Legacy alias for `set_idle()`, kept for older call sites |

Constructor: `RobotWidget(parent=None, fps=30)`.

`set_idle/set_listening/set_thinking/set_speaking/set_happy` - the five methods
that existed before Phase 1 - are unchanged in name and call signature, so no
existing caller breaks. `RobotWidget` also still declares `sizeHint`,
`minimumSize` (220 px) and an expanding size policy, so `MainWindow`'s layout
needed no changes.

`FaceAnimator` additionally exposes `start()`, `stop()`, `is_running()`,
`STATES`, `set_state()`, `params` and the `frame` signal. `FaceParameters`
exposes `snapshot()` for tests.

---

## 9. Files created

### During this continuation

| File | Purpose |
|---|---|
| `docs/PHASE_1_REPORT.md` | This report |

### By the earlier Phase 1 execution (preserved as found)

| File | Purpose |
|---|---|
| `app/gui/face_renderer.py` | Stateless `QPainter` face renderer |
| `app/gui/face_animation.py` | Parameter model and central animation controller |
| `tests/test_face.py` | Offscreen automated checks, including the optional stability run |
| `scripts/face_preview.py` | Standalone interactive preview launcher |
| `app/gui/resources/robot.svg` | Unreferenced leftover asset. Kept because it was pre-existing; no code imports it. |

---

## 10. Files modified

### During this continuation

| File | Change |
|---|---|
| `tests/test_face.py` | Added check 4b (ten distinct mouth shapes) and its imports/registration |
| `.gitignore` | Added `.face_tmp/` and `.stability.log` |
| `app/gui/robot_widget.py` | Corrected one inaccurate `paintEvent` docstring line (no behaviour change) |
| `docs/PHASE_1_REPORT.md` | Created (see section 9) |

### By the earlier Phase 1 execution (preserved as found)

| File | Change |
|---|---|
| `app/gui/robot_widget.py` | Replaced the `QLabel`/`QPixmap`/SVG mascot with a `QWidget` that paints the programmatic face; all state setters implemented |
| `app/gui/main_window.py` | Calls `robot_widget.shutdown()` in `closeEvent`; `RobotWidget` embedded in the right-hand card |

The four `mascot_*.svg` deletions visible in `git status` predate this
continuation and were not performed here. They are not referenced anywhere in
the source.

---

## 11. Files intentionally left untouched

- `app/main.py` - startup, service wiring and the deliberate "vision before Qt"
  import order. Unchanged.
- `app/vision/*`, `app/speech/*`, `app/brain/*`, `app/core/*` - no backend work
  belongs to this phase.
- `app/gui/theme.py`, `camera_widget.py`, `chat_widget.py`, `status_widget.py` -
  unrelated GUI.
- `config/*`, `data/*`, `models/*`.
- `docs/PHASE_0_REPORT.md`, `docs/PROJECT_AUDIT.md`,
  `docs/DEPENDENCY_COMPATIBILITY.md`, `docs/ENVIRONMENT_SETUP.md`.
- `requirements_windows.txt`, `requirements_jetson.txt`, `requirements.txt` -
  no new dependency was added; the face uses only PyQt5.
- MainWindow layout and the kiosk/fullscreen design - Phase 2.
- `flow.md` and `design.md` - deliberately not created in this phase.

---

## 12. Tests performed

### Automated face checks (`python tests/test_face.py`)

Run offscreen with no camera, microphone, Whisper, LLM, Piper, visitor database
or network access. The `QT_QPA_PLATFORM=offscreen` default is set by the test
file itself.

| # | Check | What it proves |
|---|---|---|
| 1 | Rendering | Grabs a 600x600 frame: coverage in range, ink bbox not clipped on any edge, accent-cyan glow pixels present |
| 1b | No image/SVG asset | Source of the three face modules contains no pixmap/SVG/movie use and no image extension |
| 2 | Blinking | Forces a blink, observes eyes reach < 0.30 openness and reopen > 0.75; measures left/right asymmetry |
| 3 | Pupil movement | Steers gaze right, left, up, down and back to centre; asserts signs and magnitudes |
| 4 | Mouth + speaking | Idle mouth closed; happy smile wide; speaking opening has real range; error mouth downturned; four frames byte-distinct |
| 4b | Ten mouth shapes | Renders neutral, slight smile, wide smile, closed, small open, large open, speaking, surprised, confused, concerned; all ten frames byte-distinct |
| 5 | All eight states | Each preset converges within tolerance; all eight frames byte-distinct; invalid/`None`/mixed-case state names handled |
| 6 | Transitions | Ten state pairs; largest single-frame change below 0.55; each transition settles on the target |
| 7 | Resizing | Seven sizes including 800x600, 1024x768, 1280x720, 1366x768, 1920x1080, plus 300x300 and a tall 420x900; asserts centring, no clipping, sane coverage |
| 8 | Stability | Optional 10-minute continuous run with a state switch every 6 s; asserts timer count constant, animator alive throughout, memory growth bounded |
| 9 | MainWindow integration | Real `MainWindow` with a stubbed `VisionService`; face is the mascot slot, starts idle, renders, timer runs, all setters callable, timer stopped after close |

### Standalone visual review

All eight states were rendered from the live widget to `.face_tmp/v3_*.png` and
inspected, and the ten mouth shapes were rendered to
`.face_tmp/mouth_*.png`. The state images show the intended differences:
narrow smiling eyes in HAPPY, raised brows in LISTENING/GREETING, asymmetric
brows and side gaze in CONFUSED, inner-brow drop plus frown in ERROR, and
animated mouth geometry in SPEAKING.

### Application integration

- `python -c "import app.main"` -> imports cleanly (no import-order regression).
- `python -m app.main` -> ran for 25 s under an external timeout. Log order:
  `AI Admission Robot Starting...` -> `Starting Vision Service...` ->
  `Camera ready (1280x720 @ 30 FPS)` -> `Starting Speech Service...` ->
  `Backend services started.` -> `GUI started.`, followed by live LLM replies
  and speech output. No traceback, no Qt rendering warnings and no face-related
  timer warning during the run.
- On forced termination by the timeout, Qt printed the pre-existing
  `QObject::~QObject: Timers cannot be stopped from another thread` message.
  This is the teardown race already recorded in `docs/PHASE_0_REPORT.md`
  (risks table) and is caused by killing the process with live daemon threads;
  it is not produced by the face animation timer, and the clean close path
  (test 9) stops the face timer correctly.

### Test results (this continuation)

```
Greeto face checks (platform: offscreen)
       coverage=0.643  glow_pixels=11172
[PASS] 1. rendering produces a full face
[PASS] 1b. no image or SVG asset is used
       min_openness=0.02  eye_asymmetry=0.210
[PASS] 2. blinking closes and reopens the eyes
       right=0.58 left=-0.56 up=-0.58 down=0.57 centre=-0.00
[PASS] 3. pupils move in all four directions
       speaking_open=0.08..0.52  distinct_frames=4
[PASS] 4. mouth shapes and speaking motion
       10 distinct mouth shapes rendered
[PASS] 4b. ten distinct mouth shapes render
       8 distinct states rendered
[PASS] 5. all eight facial states
       largest single-frame change: 0.283
[PASS] 6. state transitions are smooth
        800x600  coverage=0.481  bbox=(118,34,680,540)
       1024x768  coverage=0.481  bbox=(150,46,870,692)
       1280x720  coverage=0.361  bbox=(302,44,976,650)
       1366x768  coverage=0.361  bbox=(322,46,1042,694)
       1920x1080 coverage=0.361  bbox=(452,66,1466,976)
        300x300  coverage=0.644  bbox=(8,18,288,270)
        420x900  coverage=0.301  bbox=(12,264,404,618)
[PASS] 7. responsive resizing
       MainWindow integration OK, timer stopped on close
[PASS] 9. MainWindow integration (stubbed vision)
RESULT: ALL CHECKS PASSED
```

Resize observations: coverage falls from ~0.48 at 4:3 sizes to ~0.36 at 16:9
sizes because the face is square and bounded by the shorter dimension - the
proportions are preserved, the face stays centred, and nothing is clipped.

---

## 13. Performance observations

**Measured on this Windows host, offscreen platform, software rendering. Not
measured on Jetson Xavier NX.**

10-minute continuous run on the final code, with a state change every 6 s:

```
       ran 600.0s  frames=18182  fps=30.3  cpu=7.9%  rss=90.0MB (start 89.8MB)
       rss samples: 0s:90MB, 90s:90MB, 180s:90MB, 270s:90MB, 360s:90MB, 450s:90MB, 540s:90MB
[PASS] 8. stability run (10 min)
RESULT: ALL CHECKS PASSED
```

The same process also asserted, every iteration for the full 10 minutes, that
the animator stayed running and that the widget's `QTimer` child count never
changed - so no blink, gaze or state transition created a duplicate timer.
Memory was flat, not merely under the threshold: 90 MB at every 90-second
sample from 90 MB at the start.

The earlier Phase 1 execution's own 10-minute log (`.stability.log`) recorded
30.3 fps, 8.3 % CPU of one core equivalent, and RSS steady at ~90.0 MB from a
89.7 MB baseline - no drift across samples. Both runs are consistent with a
cheap software-rendered face: one timer, a handful of gradients per frame, no
blur, no framebuffer copies and no per-frame `QImage` allocation in the real
paint path.

Frame pacing: the animator requests 30 FPS by default and uses
`Qt.PreciseTimer`. Elapsed time is measured with `perf_counter` and clamped, so
a delayed frame does not speed up or skip animation.

Caveats:

- The offscreen platform still rasterises through QPainter software paths, but
  it does not flush to a real compositor; on-screen cost may differ slightly.
- No profile exists for the Xavier NX. Jetson has 6 CPU cores and a weaker GPU;
  the face is CPU-rasterised, so real FPS/CPU there is unknown until measured.
- CPU percentage above is process CPU time divided by wall time, i.e. share of
  a single core, not a system-wide percentage.

---

## 14. Known limitations

1. **No audio-driven lip synchronisation.** Speaking animation is procedural and
   deliberately not phoneme-locked; Phase 1 excludes real lip sync.
2. **Visual accent drift in HAPPY/GREETING.** Warmth blends the cyan accent
   toward warm amber, but because the source is cyan the mid-blend can read as
   pale mint rather than gold. Intentional and state-scoped, not a defect.
3. **Eyelids are implicit.** There is no separate lid polygon; the eye outline
   height is the lid motion and a lid line is drawn for closed eyes. Independent
   lids that open at different rates beyond blinking are not modelled.
4. **No dedicated SURPRISED state.** Surprised is a mouth shape (narrow, widely
   open) reachable through parameters and verified by check 4b, not one of the
   eight mandatory states.
5. **Gaze targets are a fixed per-state list.** There is no target tracking or
   visitor-aware gaze; that belongs to a later integration phase.
6. **30 FPS default.** Higher rates are supported (`RobotWidget(fps=...)`,
   `scripts/face_preview.py --fps`), but nothing above 30 was measured.
7. **Full-app run was time-boxed.** The `python -m app.main` validation ran 25 s
   and was ended by an external timeout; the interactive window was not driven
   through a manual session in this continuation.
8. **Unused asset remains.** `app/gui/resources/robot.svg` is still in the tree
   but is not imported anywhere.

---

## 15. Remaining issues

Ordered by how much they matter for the next phase:

1. **No face-state wiring to the conversation pipeline.** `MainWindow` embeds
   and shuts down the face but nothing subscribes `RobotState`/`EventBus`
   changes to `set_listening/set_thinking/set_speaking/...`. The face works when
   called; the application does not call it yet. Phase 0 already recorded that
   `visitor.detected` has no subscriber. This was intentionally out of Phase 1
   scope ("do not modify conversation workflow") and is the first Phase 2
   integration task.
2. **Jetson performance unverified.** No hardware was available. The design is
   deliberately cheap (no OpenGL, no blur, one timer) but that is a design
   argument, not a measurement.
3. **Pre-existing teardown race.** The `QObject::~QObject: Timers cannot be
   stopped from another thread` message on forced termination remains, as
   documented in Phase 0. It is not caused by the face and is not fixed here.
4. **Per-frame gradient construction.** Each frame builds several
   `QLinearGradient`/`QRadialGradient`/`QPainterPath` objects. This is fine at
   30 FPS on the test host; if Xavier NX profiling shows pressure, caching the
   static gradients (they depend only on `side` and `glow`) is the first
   optimisation to try.
5. **Fullscreen kiosk presentation.** The face currently sits in the existing
   card layout; the Phase 2 fullscreen design is not started.

---

## 16. Phase 2 readiness

The face is ready to be used as the visual front end of the kiosk:

- Asset-free `QPainter` rendering, no external files or downloads.
- Eight distinct expressions with smooth, non-restarting transitions.
- Natural blinking, gaze and continuous speaking motion.
- Responsive from 300x300 to 1920x1080 without clipping or distortion.
- Backward-compatible API, so existing call sites keep working.
- Clean lifecycle: pauses when hidden, stops when closed, no timer growth over
  10 minutes.
- Standalone preview and automated checks, both independent of camera, mic,
  models and network.

What Phase 2 still needs before a kiosk deployment:

1. A fullscreen layout that hosts the face at display scale (and keeps it
   centred on 16:9 panels).
2. Subscription to the app's state/event flow so the face reacts to visitor
   detection, listening, processing, speaking and errors automatically.
3. An on-device performance check on the Xavier NX, including a sustained
   speaking/thinking run at the target resolution.
4. A decision on whether the face should also carry captions or status text
   (explicitly a later phase).

---

## 17. Acceptance checklist

| Requirement | Status |
|---|---|
| Existing Phase 1 work inspected and preserved | Yes |
| Face rendered entirely with PyQt5 drawing operations | Yes - verified by test 1b |
| No image-based face rendering | Yes - no pixmap/SVG/GIF/video in the render path |
| Independent left and right eyes | Yes - separate `eye_open_l/r`, `eye_smile` |
| Natural pupil movement | Yes - test 3, four directions + return to centre |
| Natural blinking | Yes - test 2; 0.16-0.24 s, 3-6 s intervals, asymmetric |
| Animated eyebrows | Yes - independent raise/tilt per side |
| Multiple mouth shapes | Yes - ten shapes, test 4b |
| Continuous speaking animation | Yes - test 4, opening 0.08-0.52 |
| IDLE / LISTENING / THINKING / SPEAKING | Yes |
| HAPPY / GREETING / CONFUSED / ERROR | Yes |
| Smooth state transitions | Yes - test 6, worst step 0.283 |
| Existing `RobotWidget` methods preserved | Yes - verified in test 9 |
| Responsive rendering | Yes - test 7 across seven sizes |
| Standalone testing available | Yes - `scripts/face_preview.py` and `tests/test_face.py` |
| `MainWindow` integration verified | Yes - test 9 |
| Application launch verified | Yes - 25 s `python -m app.main` run |
| No new timer or rendering regressions | Yes - constant timer count over 10 minutes |
| Stability test performed | Yes - final-code 10-minute run recorded above |
| `docs/PHASE_1_REPORT.md` updated | Yes - this file |
| No Phase 2 work started | Yes |
| Nothing committed or pushed | Yes |

---

## How to run

```bash
# Interactive face preview
python scripts/face_preview.py
python scripts/face_preview.py --demo --size 1280x720 --fps 60

# Automated checks (offscreen, no hardware required)
python tests/test_face.py
python tests/test_face.py --stability-minutes 10
```
