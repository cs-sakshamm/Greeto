"""
Animation controller for the Greeto digital face.

One QTimer drives everything: breathing, gaze drift, blinking, speaking
mouth movement, glow modulation and expression transitions. Nothing in
here touches QPainter - the controller only produces numbers, which keeps
rendering and animation independently testable.
"""

from __future__ import annotations

import math
import random
import time

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal


class FaceParameters:
    """
    One frame of animation state, read by the renderer.

    Plain attributes on purpose: this object is updated 30 times a second
    and must stay cheap to write.
    """

    __slots__ = (
        "eye_open_l",
        "eye_open_r",
        "eye_smile",
        "pupil_x",
        "pupil_y",
        "brow_raise_l",
        "brow_tilt_l",
        "brow_raise_r",
        "brow_tilt_r",
        "mouth_open",
        "mouth_width",
        "mouth_curve",
        "glow",
        "warmth",
        "breath",
    )

    def __init__(self):
        self.eye_open_l = 0.93
        self.eye_open_r = 0.93
        self.eye_smile = 0.10
        self.pupil_x = 0.0
        self.pupil_y = 0.0
        self.brow_raise_l = 0.05
        self.brow_tilt_l = 0.0
        self.brow_raise_r = 0.05
        self.brow_tilt_r = 0.0
        self.mouth_open = 0.0
        self.mouth_width = 1.0
        self.mouth_curve = 0.30
        self.glow = 0.45
        self.warmth = 0.0
        self.breath = 0.0

    def snapshot(self):
        """Return the current values as a dict (used by tests)."""

        return {name: getattr(self, name) for name in self.__slots__}


class FaceAnimator(QObject):
    """
    Central animation controller for the face.

    Responsibilities:
      * smooth transitions between expression presets
      * natural blinking (including double blinks and per-eye offset)
      * controlled gaze drift with per-state targets
      * speaking mouth movement driven by layered sine waves
      * subtle glow and breathing modulation
    """

    frame = pyqtSignal()

    STATES = (
        "idle",
        "listening",
        "thinking",
        "speaking",
        "happy",
        "greeting",
        "confused",
        "error",
    )

    # Time constant, in seconds, for each parameter's easing towards its
    # target. Small values react quickly, large values drift slowly.
    SMOOTHING = {
        "eye_open": 0.085,
        "eye_smile": 0.140,
        "pupil_x": 0.260,
        "pupil_y": 0.260,
        "brow_raise_l": 0.170,
        "brow_raise_r": 0.170,
        "brow_tilt_l": 0.170,
        "brow_tilt_r": 0.170,
        "mouth_open": 0.055,
        "mouth_width": 0.160,
        "mouth_curve": 0.150,
        "glow": 0.320,
        "warmth": 0.300,
    }

    _BASE = {
        "eye_open": 0.93,
        "eye_smile": 0.10,
        "pupil_x": 0.0,
        "pupil_y": 0.0,
        "brow_raise_l": 0.05,
        "brow_tilt_l": 0.0,
        "brow_raise_r": 0.05,
        "brow_tilt_r": 0.0,
        "mouth_open": 0.0,
        "mouth_width": 1.00,
        "mouth_curve": 0.30,
        "glow": 0.45,
        "warmth": 0.0,
    }

    PRESETS = {
        "idle": dict(_BASE),
        "listening": dict(
            _BASE,
            eye_open=1.00,
            eye_smile=0.00,
            brow_raise_l=0.42,
            brow_raise_r=0.42,
            mouth_open=0.03,
            mouth_width=0.93,
            mouth_curve=0.12,
            glow=0.62,
        ),
        "thinking": dict(
            _BASE,
            eye_open=0.84,
            eye_smile=0.05,
            brow_raise_l=0.30,
            brow_tilt_l=0.38,
            brow_raise_r=0.02,
            brow_tilt_r=-0.12,
            mouth_open=0.04,
            mouth_width=0.82,
            mouth_curve=-0.05,
            glow=0.50,
        ),
        "speaking": dict(
            _BASE,
            eye_open=0.95,
            brow_raise_l=0.20,
            brow_raise_r=0.20,
            mouth_open=0.18,
            mouth_width=1.02,
            mouth_curve=0.42,
            glow=0.58,
        ),
        "happy": dict(
            _BASE,
            eye_open=0.66,
            eye_smile=0.60,
            brow_raise_l=0.34,
            brow_raise_r=0.34,
            mouth_open=0.10,
            mouth_width=1.12,
            mouth_curve=0.90,
            glow=0.80,
            warmth=0.55,
        ),
        "greeting": dict(
            _BASE,
            eye_open=0.97,
            eye_smile=0.22,
            brow_raise_l=0.52,
            brow_raise_r=0.52,
            mouth_open=0.06,
            mouth_width=1.05,
            mouth_curve=0.68,
            glow=0.75,
            warmth=0.30,
        ),
        "confused": dict(
            _BASE,
            eye_open=0.90,
            eye_smile=0.00,
            brow_raise_l=0.74,
            brow_tilt_l=0.46,
            brow_raise_r=0.00,
            brow_tilt_r=-0.26,
            mouth_open=0.05,
            mouth_width=0.74,
            mouth_curve=0.00,
            glow=0.50,
        ),
        "error": dict(
            _BASE,
            eye_open=0.80,
            eye_smile=-0.10,
            brow_raise_l=-0.26,
            brow_tilt_l=0.58,
            brow_raise_r=-0.26,
            brow_tilt_r=0.58,
            mouth_open=0.02,
            mouth_width=0.88,
            mouth_curve=-0.55,
            glow=0.32,
        ),
    }

    # Where the pupils drift, per state. Deliberately small offsets: a
    # digital assistant should look attentive, not dart around.
    _GAZE = {
        "idle": [(0.0, 0.0), (0.24, 0.06), (-0.24, 0.06), (0.12, -0.10),
                 (-0.12, -0.10), (0.30, 0.02)],
        "listening": [(0.0, 0.0), (0.08, -0.05), (-0.08, -0.05)],
        "thinking": [(0.34, -0.46), (-0.34, -0.40), (0.18, -0.24)],
        "speaking": [(0.0, 0.05), (0.12, -0.04), (-0.12, 0.02)],
        "happy": [(0.0, 0.0), (0.10, -0.05)],
        "greeting": [(0.0, -0.02), (0.10, 0.04)],
        "confused": [(0.40, -0.10), (-0.34, -0.02)],
        "error": [(0.0, 0.26)],
    }

    # How long to hold a gaze target, per state.
    _GAZE_INTERVAL = {
        "idle": (1.8, 4.4),
        "listening": (2.6, 5.0),
        "thinking": (0.9, 1.9),
        "speaking": (0.8, 1.7),
        "happy": (2.4, 4.0),
        "greeting": (2.2, 3.6),
        "confused": (1.4, 2.6),
        "error": (3.0, 5.0),
    }

    # Blink interval multipliers: speaking gets fewer blinks, calm states
    # blink normally.
    _BLINK_SCALE = {
        "speaking": 1.45,
        "listening": 1.15,
        "error": 1.30,
    }

    def __init__(self, fps=30, parent=None):
        super().__init__(parent)

        self.params = FaceParameters()
        self.state = "idle"

        self.blink_min = 3.0
        self.blink_max = 6.0

        self._current = dict(self._BASE)
        self._targets = dict(self.PRESETS["idle"])

        now = time.perf_counter()

        self._time = 0.0
        self._last_tick = now

        self._blink_start = None
        self._blink_duration = 0.2
        self._blink_offset = 0.0
        self._double_blink = False
        self._next_blink = now + random.uniform(0.8, 2.4)
        self._blink_l = 1.0
        self._blink_r = 1.0

        self._gaze_target = (0.0, 0.0)
        self._next_gaze = now + random.uniform(1.0, 2.5)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(max(8, int(1000.0 / max(1, fps))))
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the animation timer (idempotent)."""

        if not self._timer.isActive():
            self._last_tick = time.perf_counter()
            self._timer.start()

    def stop(self):
        """Stop the animation timer (idempotent, safe to call twice)."""

        if self._timer.isActive():
            self._timer.stop()

    def is_running(self):
        return self._timer.isActive()

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def set_state(self, state):
        """
        Switch expression.

        Invalid values fall back to ``idle`` instead of raising, and
        re-selecting the active state is a no-op so animations are not
        restarted needlessly.
        """

        key = "idle" if state is None else str(state).strip().lower()

        if key not in self.PRESETS:
            key = "idle"

        if key == self.state:
            return self.state

        self.state = key
        self._targets = dict(self.PRESETS[key])

        # Pick a fresh gaze target straight away; the eyes should react to
        # a state change immediately.
        self._next_gaze = 0.0

        return self.state

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def _tick(self):

        now = time.perf_counter()
        delta = now - self._last_tick

        if delta <= 0.0:
            return

        # Clamp so a stalled event loop cannot teleport the animation.
        delta = min(delta, 0.12)
        self._last_tick = now
        self._time += delta

        self._advance_gaze(now)
        self._advance_blink(now)
        self._advance_values(delta)

        self.params.breath = math.sin(self._time * 1.15)

        self.frame.emit()

    # ------------------------------------------------------------------
    # Gaze
    # ------------------------------------------------------------------

    def _advance_gaze(self, now):

        if now < self._next_gaze:
            return

        options = self._GAZE.get(self.state, self._GAZE["idle"])
        self._gaze_target = random.choice(options)

        low, high = self._GAZE_INTERVAL.get(self.state, (2.0, 4.0))
        self._next_gaze = now + random.uniform(low, high)

    # ------------------------------------------------------------------
    # Blinking
    # ------------------------------------------------------------------

    @staticmethod
    def _blink_curve(progress):
        """1.0 = fully open, 0.0 = fully closed. Fast close, soft open."""

        if progress < 0.42:
            step = progress / 0.42
            closed = step * step * (3.0 - 2.0 * step)
        else:
            step = (progress - 0.42) / 0.58
            closed = 1.0 - step * step * (3.0 - 2.0 * step)

        return 1.0 - closed

    def _advance_blink(self, now):

        if self._blink_start is None:

            if now < self._next_blink:
                return

            self._blink_start = now
            self._blink_duration = random.uniform(0.16, 0.24)
            # A few milliseconds of difference between the eyes is what
            # makes a blink read as organic rather than mechanical.
            self._blink_offset = random.uniform(0.0, 0.022)
            self._double_blink = random.random() < 0.18
            return

        elapsed = now - self._blink_start

        if elapsed >= self._blink_duration:

            self._blink_l = 1.0
            self._blink_r = 1.0
            self._blink_start = None

            scale = self._BLINK_SCALE.get(self.state, 1.0)

            interval = random.uniform(self.blink_min, self.blink_max) * scale

            if self._double_blink:
                interval = random.uniform(0.12, 0.20)

            self._next_blink = now + interval
            return

        progress = elapsed / self._blink_duration
        self._blink_l = self._blink_curve(progress)

        offset_progress = (elapsed - self._blink_offset) / self._blink_duration
        offset_progress = min(max(offset_progress, 0.0), 1.0)
        self._blink_r = self._blink_curve(offset_progress)

    # ------------------------------------------------------------------
    # Parameter easing
    # ------------------------------------------------------------------

    def _speaking_mouth(self):
        """
        Procedural mouth movement for the speaking state.

        Three incommensurate sine waves plus a slow phrase envelope keep
        the movement from looking like a repeating loop.
        """

        t = self._time

        fast = math.sin(t * 8.7)
        slow = math.sin(t * 4.1 + 1.7)
        flutter = math.sin(t * 15.3 + 0.6)

        phrase = 0.5 + 0.5 * math.sin(t * 0.62)
        envelope = 0.22 + 0.78 * min(1.0, max(0.0, phrase * 1.7))

        opening = (
            0.14
            + 0.34 * (0.5 + 0.5 * fast) * (0.72 + 0.28 * (0.5 + 0.5 * slow))
            + 0.07 * (0.5 + 0.5 * flutter)
        ) * envelope

        return min(1.0, max(0.0, opening))

    def _advance_values(self, delta):

        desired = dict(self._targets)

        if self.state == "speaking":
            desired["mouth_open"] = self._speaking_mouth()
            desired["mouth_width"] = self._targets["mouth_width"] * (
                1.0 + 0.04 * math.sin(self._time * 3.3)
            )

        if self.state == "thinking":
            # Gentle glow pulse while the assistant is processing.
            desired["glow"] = self._targets["glow"] + 0.12 * math.sin(
                self._time * 2.1
            )

        desired["pupil_x"] = self._gaze_target[0]
        desired["pupil_y"] = self._gaze_target[1]

        for key, target in desired.items():

            current = self._current[key]
            tau = self.SMOOTHING.get(key, 0.15)

            easing = 1.0 - math.exp(-delta / tau)

            self._current[key] = current + (target - current) * easing

        params = self.params

        params.eye_open_l = self._current["eye_open"] * self._blink_l
        params.eye_open_r = self._current["eye_open"] * self._blink_r
        params.eye_smile = self._current["eye_smile"]
        params.pupil_x = self._current["pupil_x"]
        params.pupil_y = self._current["pupil_y"]
        params.brow_raise_l = self._current["brow_raise_l"]
        params.brow_tilt_l = self._current["brow_tilt_l"]
        params.brow_raise_r = self._current["brow_raise_r"]
        params.brow_tilt_r = self._current["brow_tilt_r"]
        params.mouth_open = self._current["mouth_open"]
        params.mouth_width = self._current["mouth_width"]
        params.mouth_curve = self._current["mouth_curve"]
        params.glow = self._current["glow"]
        params.warmth = self._current["warmth"]
