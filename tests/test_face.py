"""
Automated checks for the animated digital face.

Runs the widget offscreen, with no camera, microphone, LLM or database.

    python tests/test_face.py
    python tests/test_face.py --stability-minutes 10

The long stability run is opt-in because it takes real time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# onnxruntime must be imported before PyQt5 on Windows, otherwise its
# native extension fails to initialise (docs/PROJECT_AUDIT.md, V4). The
# MainWindow integration check below pulls in the vision stack, so the
# order is established here rather than inside that test.
try:

    import app.vision.vision_service  # noqa: F401

except Exception:

    pass

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtGui import QColor, QImage, QPainter  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from app.gui.face_animation import FaceAnimator, FaceParameters  # noqa: E402
from app.gui.face_renderer import FaceRenderer  # noqa: E402
from app.gui.robot_widget import RobotWidget  # noqa: E402


FAILURES = []


def check(name, fn):
    """Run one check and record the outcome."""

    started = time.perf_counter()

    try:

        fn()

        elapsed = time.perf_counter() - started

        print("[PASS] %-58s (%.2fs)" % (name, elapsed))

    except Exception as error:  # noqa: BLE001 - test harness reports everything

        print("[FAIL] %-58s -> %s: %s" % (name, type(error).__name__, error))

        FAILURES.append(name)


def pump(app, seconds):
    """Let the Qt event loop and the animation timer run for real time."""

    deadline = time.perf_counter() + seconds

    while time.perf_counter() < deadline:

        app.processEvents()

        time.sleep(0.004)


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------

def grab(widget):
    """Grab the widget and return it as a QImage."""

    image = widget.grab().toImage()

    assert not image.isNull(), "grab() returned a null image"

    return image


def ink_ratio(image, step=3):
    """Fraction of sampled pixels that differ from the top-left pixel."""

    background = image.pixelColor(1, 1)

    total = 0
    drawn = 0

    for y in range(0, image.height(), step):

        for x in range(0, image.width(), step):

            total += 1

            colour = image.pixelColor(x, y)

            if (
                abs(colour.red() - background.red()) > 8
                or abs(colour.green() - background.green()) > 8
                or abs(colour.blue() - background.blue()) > 8
                or abs(colour.alpha() - background.alpha()) > 8
            ):

                drawn += 1

    return drawn / float(total) if total else 0.0


def ink_bounds(image, step=2):
    """Bounding box of everything that is not background."""

    background = image.pixelColor(1, 1)

    left, top = image.width(), image.height()
    right = bottom = -1

    for y in range(0, image.height(), step):

        for x in range(0, image.width(), step):

            colour = image.pixelColor(x, y)

            if (
                abs(colour.red() - background.red()) > 8
                or abs(colour.green() - background.green()) > 8
                or abs(colour.blue() - background.blue()) > 8
            ):

                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    return left, top, right, bottom


def image_bytes(image):
    """Stable byte signature of a frame, for diffing two expressions."""

    return bytes(image.bits().asstring(image.byteCount()))


# ----------------------------------------------------------------------
# Test 1 - rendering
# ----------------------------------------------------------------------

def test_rendering(app):

    widget = RobotWidget()

    widget.resize(600, 600)

    widget.show()

    pump(app, 0.4)

    image = grab(widget)

    assert image.width() == 600 and image.height() == 600, "unexpected size"

    ratio = ink_ratio(image)

    assert 0.12 < ratio < 0.95, "face coverage looks wrong: %.3f" % ratio

    left, top, right, bottom = ink_bounds(image)

    assert right > left and bottom > top, "nothing was drawn"

    assert left > 0 and top > 0, "drawing is clipped at the top-left edge"

    assert right < image.width() - 1, "drawing is clipped at the right edge"

    assert bottom < image.height() - 1, "drawing is clipped at the bottom edge"

    # The eyes must glow cyan-ish somewhere: a blue channel clearly above
    # red is the signature of the accent colour.
    glowing = 0

    for y in range(0, image.height(), 2):

        for x in range(0, image.width(), 2):

            colour = image.pixelColor(x, y)

            if (
                colour.blue() > 120
                and colour.green() > 90
                and colour.blue() > colour.red() + 40
            ):

                glowing += 1

    assert glowing > 40, "no accent glow found (%d px)" % glowing

    print("       coverage=%.3f  glow_pixels=%d" % (ratio, glowing))

    widget.close()


def test_no_assets_used():

    for name in ("robot_widget.py", "face_renderer.py", "face_animation.py"):

        source = (ROOT / "app" / "gui" / name).read_text(encoding="utf-8")

        for forbidden in ("QPixmap", "QSvgWidget", "QMovie", "QtSvg"):

            assert forbidden not in source, "%s references %s" % (name, forbidden)

        for extension in (".png", ".jpg", ".jpeg", ".gif", ".svg"):

            assert extension not in source.replace(
                '""""', ""
            ), "%s references a %s asset" % (name, extension)


# ----------------------------------------------------------------------
# Test 2 - blinking
# ----------------------------------------------------------------------

def test_blinking(app):

    widget = RobotWidget()

    widget.resize(500, 500)

    widget.show()

    pump(app, 0.3)

    animator = widget._animator

    animator._next_blink = time.perf_counter()

    lowest = 1.0

    deadline = time.perf_counter() + 0.7

    while time.perf_counter() < deadline:

        app.processEvents()

        time.sleep(0.003)

        lowest = min(lowest, animator.params.eye_open_l, animator.params.eye_open_r)

    assert lowest < 0.30, "eyes never closed (minimum openness %.2f)" % lowest

    pump(app, 0.5)

    reopened = animator.params.eye_open_l

    assert reopened > 0.75, "eyes did not reopen (%.2f)" % reopened

    # Asymmetric timing: the two eyes should not be perfectly identical
    # during a blink, otherwise it looks mechanical.
    animator._next_blink = time.perf_counter()

    observed_gap = 0.0

    deadline = time.perf_counter() + 0.35

    while time.perf_counter() < deadline:

        app.processEvents()

        time.sleep(0.002)

        observed_gap = max(
            observed_gap,
            abs(animator.params.eye_open_l - animator.params.eye_open_r),
        )

    print("       min_openness=%.2f  eye_asymmetry=%.3f" % (lowest, observed_gap))

    widget.close()


# ----------------------------------------------------------------------
# Test 3 - pupil movement
# ----------------------------------------------------------------------

def test_pupil_movement(app):

    widget = RobotWidget()

    widget.resize(500, 500)

    widget.show()

    pump(app, 0.3)

    animator = widget._animator

    animator._next_gaze = time.perf_counter() + 1e6

    def settle(target, seconds=0.9):

        animator._gaze_target = target

        pump(app, seconds)

        return animator.params.pupil_x, animator.params.pupil_y

    right_x, _ = settle((0.6, 0.0))

    assert right_x > 0.25, "eyes did not look right (%.2f)" % right_x

    left_x, _ = settle((-0.6, 0.0))

    assert left_x < -0.25, "eyes did not look left (%.2f)" % left_x

    _, up_y = settle((0.0, -0.6))

    assert up_y < -0.20, "eyes did not look up (%.2f)" % up_y

    _, down_y = settle((0.0, 0.6))

    assert down_y > 0.20, "eyes did not look down (%.2f)" % down_y

    centre_x, centre_y = settle((0.0, 0.0), 1.1)

    assert abs(centre_x) < 0.10 and abs(centre_y) < 0.10, "eyes did not return"

    print("       right=%.2f left=%.2f up=%.2f down=%.2f centre=%.2f"
          % (right_x, left_x, up_y, down_y, centre_x))

    widget.close()


# ----------------------------------------------------------------------
# Test 4 - mouth
# ----------------------------------------------------------------------

def test_mouth(app):

    widget = RobotWidget()

    widget.resize(520, 520)

    widget.show()

    pump(app, 0.2)

    animator = widget._animator

    signatures = {}

    # Neutral / closed.
    widget.set_state("idle")

    pump(app, 1.3)

    neutral_open = animator.params.mouth_open

    assert neutral_open < 0.05, "idle mouth should be closed (%.2f)" % neutral_open

    signatures["idle"] = image_bytes(grab(widget))

    # Smile.
    widget.set_state("happy")

    pump(app, 1.4)

    assert animator.params.mouth_curve > 0.70, "happy mouth is not smiling"

    assert animator.params.mouth_width > 1.03, "happy mouth is not wider"

    signatures["happy"] = image_bytes(grab(widget))

    # Surprised / wide open via the speaking waveform extremes.
    widget.set_state("speaking")

    lowest = 1.0
    highest = 0.0

    deadline = time.perf_counter() + 4.0

    while time.perf_counter() < deadline:

        app.processEvents()

        time.sleep(0.003)

        lowest = min(lowest, animator.params.mouth_open)

        highest = max(highest, animator.params.mouth_open)

    # The speaking waveform is intentionally uneven and has quiet phrases,
    # so only the overall range is asserted here.
    assert highest > 0.20, "speaking mouth never opened (max %.2f)" % highest

    assert highest - lowest > 0.12, "speaking mouth does not move"

    signatures["speaking"] = image_bytes(grab(widget))

    # Concerned / frowning.
    widget.set_state("error")

    pump(app, 1.4)

    assert animator.params.mouth_curve < -0.30, "error mouth is not downturned"

    signatures["error"] = image_bytes(grab(widget))

    assert len(set(signatures.values())) == 4, "expressions render identically"

    print("       speaking_open=%.2f..%.2f  distinct_frames=%d"
          % (lowest, highest, len(set(signatures.values()))))

    widget.close()


# ----------------------------------------------------------------------
# Test 4b - the required mouth shapes
# ----------------------------------------------------------------------

def test_mouth_shapes():
    """
    Every mouth shape named in the Phase 1 requirements must be drawable
    and must look different from the other nine.

    The shapes are rendered directly from frozen parameters so this check
    does not depend on any state preset or on animation timing.
    """

    # name, mouth_open, mouth_width, mouth_curve
    shapes = [
        ("neutral", 0.00, 1.00, 0.30),
        ("slight_smile", 0.00, 1.05, 0.65),
        ("wide_smile", 0.06, 1.15, 1.00),
        ("closed", 0.00, 0.95, 0.00),
        ("small_open", 0.30, 0.95, 0.20),
        ("large_open", 0.95, 1.00, 0.10),
        ("speaking", 0.45, 1.02, 0.40),
        ("surprised", 0.80, 0.62, 0.00),
        ("confused", 0.05, 0.74, 0.00),
        ("concerned", 0.02, 0.88, -0.55),
    ]

    renderer = FaceRenderer()

    signatures = {}

    for name, opening, width, curve in shapes:

        params = FaceParameters()

        params.mouth_open = opening
        params.mouth_width = width
        params.mouth_curve = curve

        image = QImage(420, 420, QImage.Format_ARGB32)

        image.fill(QColor(13, 17, 23))

        painter = QPainter(image)

        try:

            renderer.paint(painter, image.rect(), params)

        finally:

            painter.end()

        signatures[name] = image_bytes(image)

    assert len(set(signatures.values())) == len(shapes), (
        "some mouth shapes render identically: %s" % list(signatures)
    )

    print("       %d distinct mouth shapes rendered" % len(shapes))


# ----------------------------------------------------------------------
# Test 5 - every facial state
# ----------------------------------------------------------------------

def test_all_states(app):

    widget = RobotWidget()

    widget.resize(520, 520)

    widget.show()

    pump(app, 0.2)

    animator = widget._animator

    reference = {}

    for state in FaceAnimator.STATES:

        returned = widget.set_state(state)

        assert returned == state, "set_state(%r) returned %r" % (state, returned)

        assert widget.current_state() == state

        pump(app, 1.6)

        preset = FaceAnimator.PRESETS[state]

        # Convergence is asserted on the eased expression value, not on
        # params.eye_open_l, which is multiplied by the blink factor and
        # would fail whenever a blink happens to land on the sample.
        assert abs(animator._current["eye_open"] - preset["eye_open"]) < 0.20, (
            "%s: eye openness did not converge" % state
        )

        assert abs(animator._current["mouth_curve"] - preset["mouth_curve"]) < 0.15, (
            "%s: mouth curve did not converge" % state
        )

        assert abs(
            animator._current["brow_raise_l"] - preset["brow_raise_l"]
        ) < 0.15, "%s: eyebrow did not converge" % state

        reference[state] = image_bytes(grab(widget))

    # Every state must look different from the others.
    assert len(set(reference.values())) == len(FaceAnimator.STATES), (
        "some facial states render identically"
    )

    # Invalid states must not crash and must fall back to idle.
    assert widget.set_state("not-a-state") == "idle"

    assert widget.set_state(None) == "idle"

    assert widget.set_state("  HAPPY  ") == "happy"

    print("       %d distinct states rendered" % len(set(reference.values())))

    widget.close()


# ----------------------------------------------------------------------
# Test 6 - transitions
# ----------------------------------------------------------------------

def test_transitions(app):

    widget = RobotWidget()

    widget.resize(420, 420)

    widget.show()

    pump(app, 0.2)

    animator = widget._animator

    pairs = [
        ("idle", "listening"),
        ("listening", "thinking"),
        ("thinking", "speaking"),
        ("speaking", "listening"),
        ("speaking", "idle"),
        ("idle", "greeting"),
        ("greeting", "idle"),
        ("error", "happy"),
        ("confused", "idle"),
    ]

    worst = 0.0

    for source, target in pairs:

        widget.set_state(source)

        pump(app, 1.2)

        previous = animator.params.mouth_curve

        widget.set_state(target)

        # Measure the largest change between two consecutive frames across
        # the whole transition. A teleporting expression would show up here
        # as a step close to the full distance.
        largest_step = 0.0

        deadline = time.perf_counter() + 0.7

        while time.perf_counter() < deadline:

            app.processEvents()

            time.sleep(0.004)

            current = animator.params.mouth_curve

            largest_step = max(largest_step, abs(current - previous))

            previous = current

        worst = max(worst, largest_step)

        assert largest_step < 0.55, (
            "%s -> %s stepped %.2f in one frame" % (source, target, largest_step)
        )

        pump(app, 1.5)

        assert abs(
            animator.params.mouth_curve
            - FaceAnimator.PRESETS[target]["mouth_curve"]
        ) < 0.17, "%s -> %s did not settle" % (source, target)

    print("       largest single-frame change: %.3f" % worst)

    widget.close()


# ----------------------------------------------------------------------
# Test 7 - resizing
# ----------------------------------------------------------------------

def test_resizing(app):

    widget = RobotWidget()

    widget.show()

    sizes = [
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1366, 768),
        (1920, 1080),
        (300, 300),
        (420, 900),
    ]

    for width, height in sizes:

        widget.resize(width, height)

        pump(app, 0.22)

        image = grab(widget)

        assert image.width() == width and image.height() == height

        ratio = ink_ratio(image)

        assert 0.08 < ratio < 0.95, "%dx%d: coverage %.3f" % (width, height, ratio)

        left, top, right, bottom = ink_bounds(image)

        assert right > left and bottom > top, "%dx%d: nothing drawn" % (width, height)

        assert left > 0 and top > 0 and right < width - 1 and bottom < height - 1, (
            "%dx%d: face is clipped" % (width, height)
        )

        centre_x = (left + right) * 0.5

        assert abs(centre_x - width * 0.5) < width * 0.06, (
            "%dx%d: face is not centred" % (width, height)
        )

        print("       %4dx%-4d coverage=%.3f  bbox=(%d,%d,%d,%d)"
              % (width, height, ratio, left, top, right, bottom))

    widget.close()


# ----------------------------------------------------------------------
# Test 9 - integration with the real MainWindow (stubbed vision)
# ----------------------------------------------------------------------

def test_main_window_integration(app):

    from app.gui.main_window import MainWindow

    class StubVision:

        def process(self):

            return None

        def stop(self):

            return None

    window = MainWindow(vision=StubVision())

    window.resize(1400, 800)

    window.show()

    pump(app, 0.5)

    face = window.robot_widget

    assert isinstance(face, RobotWidget), "mascot slot is not the digital face"

    assert face.current_state() == "idle", "face did not start idle"

    image = grab(face)

    assert ink_ratio(image) > 0.10, "face slot rendered empty"

    assert face._animator.is_running(), "animation timer is not running"

    # State methods must still work through the widget that MainWindow owns.
    for method in (
        face.set_listening,
        face.set_thinking,
        face.set_speaking,
        face.set_happy,
        face.set_greeting,
        face.set_confused,
        face.set_error,
        face.set_idle,
    ):

        method()

        pump(app, 0.12)

    window.close()

    pump(app, 0.2)

    assert not face._animator.is_running(), "timer not stopped after close"

    print("       MainWindow integration OK, timer stopped on close")


# ----------------------------------------------------------------------
# Test 8 - stability
# ----------------------------------------------------------------------

class _MemoryProbe:
    """Best-effort process RSS reading, with a clear fallback."""

    def __init__(self):

        self._read = None

        try:

            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):

                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi

            # Explicit signatures. Without them the pseudo-handle is
            # truncated to 32 bits, the call silently returns 0 and every
            # sample reads 0.0 MB.
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p

            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(Counters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int

            def read():

                ok = psapi.GetProcessMemoryInfo(
                    kernel32.GetCurrentProcess(),
                    ctypes.byref(counters),
                    counters.cb,
                )

                if not ok:

                    return None

                return counters.WorkingSetSize / (1024.0 * 1024.0)

            if read() is None:

                raise RuntimeError("GetProcessMemoryInfo unavailable")

            self._read = read

        except Exception:

            self._read = None

    def megabytes(self):

        if self._read is None:

            return None

        return self._read()


def test_stability(app, minutes):

    widget = RobotWidget()

    widget.resize(560, 560)

    widget.show()

    pump(app, 1.0)

    animator = widget._animator

    probe = _MemoryProbe()

    frames = [0]

    animator.frame.connect(lambda: frames.__setitem__(0, frames[0] + 1))

    timer_count_at_start = len(widget.findChildren(QTimer))

    baseline = probe.megabytes()

    samples = []

    duration = minutes * 60.0

    started = time.perf_counter()
    cpu_started = time.process_time()

    states = list(FaceAnimator.STATES)

    next_switch = started
    switch_index = 0

    warnings = []

    while True:

        now = time.perf_counter()

        if now - started >= duration:

            break

        app.processEvents()

        time.sleep(0.006)

        if now >= next_switch:

            widget.set_state(states[switch_index % len(states)])

            switch_index += 1

            next_switch = now + 6.0

        if len(samples) == 0 or now - samples[-1][0] >= 15.0:

            samples.append((now, probe.megabytes()))

        assert animator.is_running(), "animation timer stopped during the run"

        assert len(widget.findChildren(QTimer)) == timer_count_at_start, (
            "timer count changed during the run"
        )

    elapsed = time.perf_counter() - started
    cpu = time.process_time() - cpu_started

    fps = frames[0] / elapsed if elapsed else 0.0

    final = probe.megabytes()

    if baseline is not None and final is not None:

        growth = final - baseline

        assert growth < 60.0, "memory grew by %.1f MB" % growth

    else:

        print("       WARNING: OS memory could not be measured on this platform")

    assert animator.is_running(), "animation stopped at the end of the run"

    print(
        "       ran %.1fs  frames=%d  fps=%.1f  cpu=%.1f%%  rss=%s"
        % (
            elapsed,
            frames[0],
            fps,
            (cpu / elapsed * 100.0) if elapsed else 0.0,
            "n/a" if final is None else "%.1fMB (start %.1fMB)" % (final, baseline),
        )
    )

    if samples:

        print(
            "       rss samples: %s"
            % ", ".join(
                "%.0fs:%s" % (when - started, "n/a" if value is None else "%.0fMB" % value)
                for when, value in samples[::max(1, len(samples) // 6)]
            )
        )

    widget.close()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stability-minutes",
        type=float,
        default=0.0,
        help="run the long stability test for this many minutes",
    )

    parser.add_argument("--skip-slow", action="store_true")

    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])

    print("Greeto face checks (platform: %s)" % app.platformName())

    check("1. rendering produces a full face", lambda: test_rendering(app))

    check("1b. no image or SVG asset is used", test_no_assets_used)

    check("2. blinking closes and reopens the eyes", lambda: test_blinking(app))

    check("3. pupils move in all four directions", lambda: test_pupil_movement(app))

    check("4. mouth shapes and speaking motion", lambda: test_mouth(app))

    check("4b. ten distinct mouth shapes render", test_mouth_shapes)

    check("5. all eight facial states", lambda: test_all_states(app))

    check("6. state transitions are smooth", lambda: test_transitions(app))

    check("7. responsive resizing", lambda: test_resizing(app))

    check(
        "9. MainWindow integration (stubbed vision)",
        lambda: test_main_window_integration(app),
    )

    if args.stability_minutes > 0:

        check(
            "8. stability run (%.0f min)" % args.stability_minutes,
            lambda: test_stability(app, args.stability_minutes),
        )

    else:

        print("[SKIP] 8. long stability run (use --stability-minutes N)")

    print("-" * 72)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))

        return 1

    print("RESULT: ALL CHECKS PASSED")

    return 0


# pytest must not collect the script-only stability run: it needs a
# minutes argument from the CLI and a real QApplication.
test_stability.__test__ = False


if __name__ == "__main__":

    sys.exit(main())
