"""
Phase 2 kiosk checks.

Everything runs offscreen: no camera, microphone, Whisper, LLM, Piper or
network access. Events are injected through the real EventBus, so the
face states below are produced by the same path the application uses.

    python tests/test_kiosk.py
    python tests/test_kiosk.py --stability-minutes 15
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# onnxruntime must be imported before PyQt5 on Windows (docs/PROJECT_AUDIT.md).
try:

    import app.vision.vision_service  # noqa: F401

except Exception:

    pass

from PyQt5.QtCore import QEvent, Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QWidget,
)

from app.brain.parser import LLMResponse  # noqa: E402
from app.core.event_bus import EventBus  # noqa: E402
from app.gui.caption_widget import CaptionWidget  # noqa: E402
from app.gui.kiosk_background import KioskBackground  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402


FAILURES = []


def check(name, fn):
    """Run one check and record the outcome."""

    started = time.perf_counter()

    try:

        fn()

        elapsed = time.perf_counter() - started

        print("[PASS] %-62s (%.2fs)" % (name, elapsed))

    except Exception as error:  # noqa: BLE001 - harness reports everything

        print("[FAIL] %-62s -> %s: %s" % (name, type(error).__name__, error))

        FAILURES.append(name)


def pump(app, seconds):
    """Run the Qt event loop for real time."""

    deadline = time.perf_counter() + seconds

    while time.perf_counter() < deadline:

        app.processEvents()

        time.sleep(0.004)


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------

class StubFace:
    """Minimal stand-in for vision.models.Face."""

    def __init__(self, visitor_id, name="Unknown"):

        self.visitor_id = visitor_id
        self.name = name


class StubVision:
    """VisionService stand-in: never touches a camera."""

    def __init__(self):

        self.container = None

        self.processed = 0

        class _Tracker:
            def active_track_count(self):
                return 0

        self.tracker = _Tracker()

    def process(self, annotate=True):

        self.processed += 1

        return None

    def stop(self):

        return None


class StubSpeech:
    """SpeechService stand-in."""

    def __init__(self, microphone_ok=True):

        self.microphone_ok = microphone_ok


class FakeContainer:
    """Container subset used by MainWindow: just the event bus."""

    def __init__(self):

        self.event_bus = EventBus()

        self.logger = logging.getLogger("kiosk-tests")


def make_window(app, resize=(1280, 720), microphone_ok=True):
    """Create the kiosk with a fake backend and shortened presenter timings."""

    container = FakeContainer()

    window = MainWindow(
        vision=StubVision(),
        container=container,
        speech=StubSpeech(microphone_ok),
    )

    presenter = window.presenter

    presenter.GREETING_HOLD_MS = 220
    presenter.SUCCESS_HOLD_MS = 150
    presenter.ERROR_RECOVERY_MS = 220
    presenter.THINKING_WATCHDOG_MS = 5000
    presenter.CAPTION_CLEAR_MS = 5000

    window.resize(*resize)
    window.show()

    pump(app, 0.2)

    return window, container


# ----------------------------------------------------------------------
# Test C - caption behaviour
# ----------------------------------------------------------------------

def test_caption_lifecycle(app):

    caption = CaptionWidget()

    caption.resize(900, 140)
    caption.show()

    pump(app, 0.1)

    assert caption.show_caption("Hello there", 0) is True

    pump(app, 0.5)

    assert caption.current_text() == "Hello there"
    assert caption.opacity() > 0.9, "caption did not fade in"

    caption.clear_caption()

    pump(app, 0.9)

    assert caption.current_text() == "", "caption did not clear"
    assert caption.opacity() < 0.05, "caption did not fade out"

    # Auto-clear after its own duration.
    caption.show_caption("Short lived", 120)

    pump(app, 0.2)

    assert caption.current_text() == "Short lived"

    pump(app, 1.2)

    assert caption.current_text() == "", "caption timer did not clear it"

    caption.shutdown()
    caption.close()


def test_caption_newer_wins(app):

    caption = CaptionWidget()

    caption.resize(900, 140)
    caption.show()

    pump(app, 0.1)

    # The first caption's timer must not blank the second one.
    caption.show_caption("old message", 100)

    pump(app, 0.05)

    caption.show_caption("new message", 0)

    pump(app, 1.4)

    assert caption.current_text() == "new message", (
        "an older caption timer cleared a newer caption"
    )

    assert caption.opacity() > 0.9

    caption.shutdown()
    caption.close()


def test_caption_wrapping_and_length(app):

    caption = CaptionWidget()

    caption.resize(760, 150)
    caption.show()

    pump(app, 0.1)

    assert caption.wordWrap() is True

    long_text = "admission " * 80

    caption.show_caption(long_text, 0)

    text = caption.current_text()

    assert len(text) <= CaptionWidget.MAX_CHARS, "caption was not truncated"
    assert text.endswith("\u2026"), "truncated caption has no ellipsis"

    assert caption.font().pixelSize() >= 16, "caption font too small"
    assert caption.font().pixelSize() <= 34, "caption font too large"

    print("       max_chars=%d  font=%dpx"
          % (len(text), caption.font().pixelSize()))

    caption.shutdown()
    caption.close()


# ----------------------------------------------------------------------
# Test B - fullscreen kiosk window
# ----------------------------------------------------------------------

def test_kiosk_window(app):

    window, _ = make_window(app)

    assert window.windowFlags() & Qt.FramelessWindowHint, "window is not frameless"
    assert isinstance(window.centralWidget(), KioskBackground)
    assert window.caption_widget is not None
    assert window.isVisible()

    # No conventional interface survives.
    names = [type(child).__name__ for child in window.findChildren(QWidget)]

    for forbidden in ("GlassCard", "CameraWidget", "QTabWidget", "QToolBar",
                      "QMenuBar", "QStatusBar", "QPushButton"):

        assert forbidden not in names, "%s is still part of the kiosk" % forbidden

    assert window.diagnostics.isVisible() is False, "diagnostics visible by default"
    assert not window.caption_widget.current_text(), "caption not empty at startup"
    assert window.robot_widget.current_state() == "idle"

    window.enter_kiosk()

    pump(app, 0.3)

    assert window.is_kiosk()
    assert window.isFullScreen() or (
        window.windowState() & Qt.WindowFullScreen
    ), "window did not enter fullscreen"

    # Esc leaves kiosk mode and keeps the application running.
    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    )

    pump(app, 0.3)

    assert not window.isFullScreen(), "Esc did not leave kiosk mode"
    assert window.isVisible(), "Esc closed the application from kiosk mode"

    # F11 toggles fullscreen.
    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_F11, Qt.NoModifier)
    )

    pump(app, 0.3)

    assert window.isFullScreen(), "F11 did not enter fullscreen"

    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_F11, Qt.NoModifier)
    )

    pump(app, 0.2)

    # F10 diagnostics are optional and off by default.
    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_F10, Qt.NoModifier)
    )

    pump(app, 0.1)

    assert window.diagnostics.isVisible(), "F10 did not show diagnostics"

    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_F10, Qt.NoModifier)
    )

    pump(app, 0.1)

    assert not window.diagnostics.isVisible(), "F10 did not hide diagnostics"

    # Esc outside kiosk mode closes the application.
    window.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    )

    pump(app, 0.3)

    assert not window.isVisible(), "Esc did not close the window"

    assert not window.robot_widget._animator.is_running()

    print("       frameless, fullscreen toggle, Esc and F10 verified")


def test_responsive_layout(app):

    window, _ = make_window(app)

    window.showNormal()

    sizes = [(1024, 768), (1280, 720), (1366, 768), (1920, 1080)]

    for width, height in sizes:

        window.resize(width, height)

        pump(app, 0.3)

        background = window.background
        face = window.robot_widget.geometry()
        caption = window.caption_widget.geometry()

        assert background.width() == width and background.height() == height, (
            "%dx%d: unexpected client size %dx%d"
            % (width, height, background.width(), background.height())
        )

        assert face.bottom() < caption.top() + 1, (
            "%dx%d: caption overlaps the face" % (width, height)
        )

        assert abs(face.center().x() - width * 0.5) <= 3, (
            "%dx%d: face is not centred" % (width, height)
        )

        assert face.height() > int(height * 0.5), (
            "%dx%d: face is too small" % (width, height)
        )

        assert caption.width() <= int(width * 0.75), (
            "%dx%d: caption is too wide" % (width, height)
        )

        assert window.caption_widget.font().pixelSize() >= 16

        print("       %4dx%-4d face=%dx%d  caption=%dx%d  font=%dpx"
              % (width, height, face.width(), face.height(),
                 caption.width(), caption.height(),
                 window.caption_widget.font().pixelSize()))

    window.close()

    pump(app, 0.2)


def test_camera_never_visible(app):

    window, _ = make_window(app)

    for child in window.findChildren(QLabel):

        pixmap = child.pixmap()

        assert pixmap is None or pixmap.isNull(), (
            "a label is displaying a camera image"
        )

    assert window.vision_pump is None, "vision pump started for a stub backend"

    source = (ROOT / "app" / "gui" / "main_window.py").read_text(
        encoding="utf-8"
    )

    assert "CameraWidget" not in source, "kiosk still imports the camera preview"
    assert "QPixmap" not in source, "kiosk still handles pixmaps"

    window.close()

    pump(app, 0.2)

    print("       no camera widget, no pixmap, no frames displayed")


# ----------------------------------------------------------------------
# Test D - event to face-state mapping
# ----------------------------------------------------------------------

def test_state_mapping(app):

    window, container = make_window(app)

    bus = container.event_bus
    presenter = window.presenter
    face = window.robot_widget

    bus.publish("visitor.detected", name="Asha", face=StubFace(1, "Asha"))

    pump(app, 0.2)

    assert face.current_state() == "greeting", (
        "visitor.detected did not produce GREETING (got %s)"
        % face.current_state()
    )

    assert "Asha" in window.caption_widget.current_text()

    pump(app, 0.4)

    assert face.current_state() == "listening", (
        "greeting did not hand over to LISTENING (got %s)"
        % face.current_state()
    )

    bus.publish("conversation.processing", text="Which courses do you offer?")

    pump(app, 0.2)

    assert face.current_state() == "thinking"
    assert "check" in window.caption_widget.current_text().lower()

    bus.publish(
        "llm_response_ready",
        response=LLMResponse(
            reply="We offer B.Tech in four branches.",
            emotion="happy",
            action="answer",
        ),
    )

    pump(app, 0.2)

    assert face.current_state() == "thinking", (
        "face entered SPEAKING before playback started"
    )

    # Caption must not display LLM response before playback starts
    assert "check" in window.caption_widget.current_text().lower()

    bus.publish(
        "speech.speaking_started",
        text="We offer B.Tech in four branches.",
        utterance=1,
    )

    pump(app, 0.2)

    assert face.current_state() == "speaking"
    assert "B.Tech" in window.caption_widget.current_text()

    bus.publish("speech.speaking_finished", utterance=1)

    pump(app, 0.05)

    assert face.current_state() == "happy", (
        "successful answer did not show a success expression"
    )

    pump(app, 0.4)

    assert face.current_state() == "listening"

    # A recoverable failure must not leave the face thinking or speaking.
    bus.publish("speech.error", kind="stt", message="empty transcription")

    pump(app, 0.2)

    assert face.current_state() == "error"
    assert "try again" in window.caption_widget.current_text().lower()

    pump(app, 0.4)

    assert face.current_state() == "listening"

    # Watchdog: a stuck request must time out.
    presenter.THINKING_WATCHDOG_MS = 150

    bus.publish("conversation.processing", text="stuck request")

    pump(app, 0.2)

    assert face.current_state() in ("thinking", "error")

    pump(app, 0.4)

    assert face.current_state() in ("listening", "error"), (
        "watchdog did not recover the face"
    )

    # No visitor left -> idle.
    bus.publish("visitor.left")

    pump(app, 0.3)

    assert face.current_state() == "idle"

    window.close()

    pump(app, 0.2)

    print("       idle -> greeting -> listening -> thinking -> speaking")
    print("       -> happy -> listening -> error -> listening -> idle")


def test_stale_speaking_finished(app):

    window, container = make_window(app)

    bus = container.event_bus
    face = window.robot_widget

    bus.publish("speech.speaking_started", text="one", utterance=1)

    pump(app, 0.15)

    bus.publish("speech.speaking_started", text="two", utterance=2)

    pump(app, 0.15)

    # A late finish for utterance 1 must not end utterance 2.
    bus.publish("speech.speaking_finished", utterance=1)

    pump(app, 0.3)

    assert face.current_state() == "speaking", (
        "a stale TTS finish ended a newer utterance"
    )

    # Hold the success beat long enough to observe it.
    window.presenter.SUCCESS_HOLD_MS = 600

    bus.publish("speech.speaking_finished", utterance=2)

    pump(app, 0.15)

    assert face.current_state() == "happy", (
        "a stale TTS finish left the face in %s" % face.current_state()
    )

    pump(app, 0.8)

    # No visitor is present in this scenario, so the face returns to idle.
    assert face.current_state() == "idle", (
        "post-speech recovery left the face in %s" % face.current_state()
    )

    window.close()

    pump(app, 0.2)


def test_microphone_unavailable(app):

    window, container = make_window(app, microphone_ok=False)

    bus = container.event_bus
    face = window.robot_widget
    presenter = window.presenter

    assert presenter.microphone_available is False

    bus.publish("visitor.detected", name="Asha", face=StubFace(5, "Asha"))

    pump(app, 0.2)

    assert face.current_state() == "greeting"

    pump(app, 0.4)

    assert face.current_state() == "idle", (
        "kiosk claimed to be LISTENING without a microphone"
    )

    bus.publish("speech.error", kind="microphone", message="no input device")

    pump(app, 0.2)

    assert face.current_state() == "error"
    assert "microphone" in window.caption_widget.current_text().lower()

    pump(app, 0.4)

    assert face.current_state() == "idle"

    window.close()

    pump(app, 0.2)


# ----------------------------------------------------------------------
# Test E - duplicate protection
# ----------------------------------------------------------------------

def test_duplicate_greetings(app):

    window, container = make_window(app)

    captions = []

    window.presenter.caption.connect(
        lambda text, duration: captions.append(text)
    )

    timers_before = len(window.findChildren(QTimer))

    face = StubFace(11, "Asha")

    for _ in range(3):

        container.event_bus.publish("visitor.detected", name="Asha", face=face)

        pump(app, 0.1)

    greetings = [text for text in captions if "Asha" in text]

    assert len(greetings) == 1, (
        "the same visitor caused %d greetings" % len(greetings)
    )

    assert len(window.findChildren(QTimer)) == timers_before, (
        "duplicate events created extra timers"
    )

    window.close()

    pump(app, 0.2)

    print("       3 detection events -> 1 greeting, timer count stable")


# ----------------------------------------------------------------------
# Test F - thread safety
# ----------------------------------------------------------------------

def test_thread_safety(app):

    window, container = make_window(app)

    bus = container.event_bus
    presenter = window.presenter

    # Keep the greeting state while the assertions run.
    presenter.GREETING_HOLD_MS = 5000

    emitted_in_main = []

    presenter.face_state.connect(
        lambda state: emitted_in_main.append(
            threading.current_thread() is threading.main_thread()
        )
    )

    def publish(index):

        bus.publish(
            "visitor.detected",
            name="Visitor %d" % index,
            face=StubFace(100 + index, "Visitor %d" % index),
        )

    threads = [
        threading.Thread(target=publish, args=(index,))
        for index in range(4)
    ]

    for thread in threads:

        thread.start()

    for thread in threads:

        thread.join()

    # The GUI thread never ran the event loop, so nothing may have changed
    # yet. If a callback touched widgets directly this would already differ.
    assert window.robot_widget.current_state() == "idle", (
        "a worker thread updated the face directly"
    )

    pump(app, 0.6)

    assert emitted_in_main and all(emitted_in_main), (
        "face state was not applied on the GUI thread"
    )

    assert window.robot_widget.current_state() == "greeting", (
        "queued events did not reach the face (got %s)"
        % window.robot_widget.current_state()
    )

    assert len(presenter._greeted) == 4, "events from worker threads were lost"

    window.close()

    pump(app, 0.2)

    print("       4 worker threads -> 4 queued events, all applied on GUI thread")


# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------

def test_shutdown(app):

    window, container = make_window(app)

    bus = container.event_bus

    assert bus.subscriber_count("visitor.detected") == 1
    assert bus.subscriber_count("speech.speaking_finished") == 1

    window.close()

    pump(app, 0.2)

    assert bus.subscriber_count("visitor.detected") == 0, (
        "event subscription leaked after close"
    )

    assert bus.subscriber_count("speech.error") == 0

    assert not window.robot_widget._animator.is_running(), (
        "face timer still running after close"
    )

    assert not window.presenter._timer.isActive(), "presenter timer still running"

    assert not window.caption_widget._timer.isActive(), "caption timer still running"

    print("       subscriptions released, all timers stopped")


# ----------------------------------------------------------------------
# Test I - continuous operation
# ----------------------------------------------------------------------

class _MemoryProbe:
    """Best-effort process RSS on Windows, None elsewhere."""

    def __init__(self):

        self._read = None

        try:

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

    window, container = make_window(app)

    presenter = window.presenter

    presenter.GREETING_HOLD_MS = 700
    presenter.SUCCESS_HOLD_MS = 350
    presenter.ERROR_RECOVERY_MS = 700
    presenter.CAPTION_CLEAR_MS = 2500
    presenter.THINKING_WATCHDOG_MS = 10000

    bus = container.event_bus

    probe = _MemoryProbe()

    frames = [0]

    window.robot_widget._animator.frame.connect(
        lambda: frames.__setitem__(0, frames[0] + 1)
    )

    timers_at_start = len(window.findChildren(QTimer))

    baseline = probe.megabytes()

    samples = []

    duration = minutes * 60.0

    started = time.perf_counter()
    cpu_started = time.process_time()

    cycle = 0
    next_cycle = started

    while True:

        now = time.perf_counter()

        if now - started >= duration:

            break

        app.processEvents()

        time.sleep(0.006)

        if now >= next_cycle:

            cycle += 1

            face = StubFace(1000 + cycle, "Visitor")

            bus.publish("visitor.detected", name="Visitor", face=face)
            bus.publish("conversation.processing", text="a question")
            bus.publish(
                "llm_response_ready",
                response=LLMResponse(
                    reply="Answer number %d." % cycle,
                    emotion="neutral",
                    action="answer",
                ),
            )
            bus.publish(
                "speech.speaking_started",
                text="Answer number %d." % cycle,
                utterance=cycle,
            )
            bus.publish("speech.speaking_finished", utterance=cycle)

            if cycle % 3 == 0:

                bus.publish("speech.error", kind="stt", message="reset")

            bus.publish("visitor.left")

            next_cycle = now + 2.0

        if len(samples) == 0 or now - samples[-1][0] >= 20.0:

            samples.append((now, probe.megabytes()))

        assert window.robot_widget._animator.is_running(), (
            "face animation stopped during the run"
        )

        assert len(window.findChildren(QTimer)) == timers_at_start, (
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

        print("       WARNING: OS memory could not be measured here")

    print(
        "       ran %.1fs  cycles=%d  face_fps=%.1f  cpu=%.1f%%  rss=%s"
        % (
            elapsed,
            cycle,
            fps,
            (cpu / elapsed * 100.0) if elapsed else 0.0,
            "n/a" if final is None else "%.1fMB (start %.1fMB)" % (final, baseline),
        )
    )

    if samples:

        print(
            "       rss samples: %s"
            % ", ".join(
                "%.0fs:%s"
                % (
                    when - started,
                    "n/a" if value is None else "%.0fMB" % value,
                )
                for when, value in samples[:: max(1, len(samples) // 6)]
            )
        )

    window.close()

    pump(app, 0.3)

    assert not window.robot_widget._animator.is_running()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stability-minutes",
        type=float,
        default=0.0,
        help="run the continuous kiosk test for this many minutes",
    )

    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])

    print("Greeto kiosk checks (platform: %s)" % app.platformName())

    check("A. Phase 1 face checks", lambda: _run_face_suite())

    check("B. fullscreen borderless kiosk", lambda: test_kiosk_window(app))

    check("C1. caption show, update, fade and clear",
          lambda: test_caption_lifecycle(app))

    check("C2. older caption timers cannot clear newer text",
          lambda: test_caption_newer_wins(app))

    check("C3. caption wrapping, truncation and font size",
          lambda: test_caption_wrapping_and_length(app))

    check("B2. responsive layout at four resolutions",
          lambda: test_responsive_layout(app))

    check("B3. camera feed never on screen", lambda: test_camera_never_visible(app))

    check("D. application events drive face states",
          lambda: test_state_mapping(app))

    check("D2. stale TTS completion ignored", lambda: test_stale_speaking_finished(app))

    check("D3. microphone unavailable is handled",
          lambda: test_microphone_unavailable(app))

    check("E. one visitor causes one greeting", lambda: test_duplicate_greetings(app))

    check("F. cross-thread events reach the GUI safely",
          lambda: test_thread_safety(app))

    check("G. clean shutdown", lambda: test_shutdown(app))

    if args.stability_minutes > 0:

        check(
            "I. continuous kiosk run (%.0f min)" % args.stability_minutes,
            lambda: test_stability(app, args.stability_minutes),
        )

    else:

        print("[SKIP] I. continuous kiosk run (use --stability-minutes N)")

    print("-" * 74)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))

        return 1

    print("RESULT: ALL CHECKS PASSED")

    return 0


def _run_face_suite():
    """Phase 1 regression, executed in-process."""

    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_face.py"), "--skip-slow"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise AssertionError(
            "tests/test_face.py failed:\n%s" % result.stdout[-3000:]
        )


# pytest must not collect the script-only stability run: it needs a
# minutes argument from the CLI and a real QApplication.
test_stability.__test__ = False


if __name__ == "__main__":

    sys.exit(main())
