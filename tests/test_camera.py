"""
Phase 3 Milestone 1 camera checks.

Camera behaviour is exercised with a fake capture device, so these
checks need no webcam, no InsightFace model and no GUI. They cover the
real CameraManager, FrameGrabber and VisionService code paths.

    python tests/test_camera.py

The optional live smoke test opens the real camera once:

    python tests/test_camera.py --live
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.event_bus import EventBus  # noqa: E402
from app.vision.camera_manager import CameraManager  # noqa: E402
from app.vision.frame_grabber import FrameGrabber  # noqa: E402


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


def wait_for(predicate, timeout=3.0, interval=0.01):
    """Wait until predicate is true. Raises on timeout."""

    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:

        if predicate():
            return True

        time.sleep(interval)

    raise AssertionError("condition was not met within %.1fs" % timeout)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeConfig:
    """ConfigManager subset used by CameraManager and VisionService."""

    DEFAULTS = {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30.0,
        "backend": "any",
        "auto_detect": False,
        "probe_max_index": 5,
        "warmup_frames": 2,
        "read_failures_before_reconnect": 15,
        "reconnect_interval_s": 5.0,
        "open_retry_interval_s": 5.0,
        "release_timeout_s": 0.75,
        "frame_stale_after_s": 2.0,
    }

    def __init__(self, **overrides):

        self.values = dict(self.DEFAULTS)

        self.values.update(overrides)

    def get_value(self, file_name, *keys, default=None):

        if file_name != "camera":
            return default

        keys = list(keys)

        # The real ConfigManager is called as
        # get_value("camera", "camera", key), so drop the file-level
        # section key and compare against the flat settings mapping.
        if keys and keys[0] == "camera":
            keys = keys[1:]

        data = self.values

        for key in keys:

            if not isinstance(data, dict) or key not in data:
                return default

            data = data[key]

        return default if data is None else data


class FakeContainer:
    """Container subset: config, logger and event bus."""

    def __init__(self, **overrides):

        self.config = FakeConfig(**overrides)

        self.logger = logging.getLogger("camera-tests")

        self.event_bus = EventBus()


class FakeCapture:
    """cv2.VideoCapture subset with controllable failures."""

    def __init__(
        self,
        frame=None,
        opened=True,
        fail_after=None,
        fail_all=False,
        read_delay=0.0,
        fps=30.0,
        raise_after_release=True,
    ):

        self.frame = (
            frame
            if frame is not None
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )

        self._opened = opened
        self._fail_after = fail_after
        self._fail_all = fail_all
        self._read_delay = read_delay
        self._fps = fps
        self._raise_after_release = raise_after_release

        self.reads = 0
        self.release_count = 0
        self.sets = {}

    def isOpened(self):

        return self._opened and self.release_count == 0

    def set(self, prop, value):

        self.sets[prop] = value

        return True

    def get(self, prop):

        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frame.shape[1]

        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frame.shape[0]

        return 0.0

    def read(self):

        if self.release_count and self._raise_after_release:

            raise RuntimeError("read() called after release()")

        if self._read_delay:

            time.sleep(self._read_delay)

        self.reads += 1

        if self._fail_all:
            return False, None

        if self._fail_after is not None and self.reads > self._fail_after:
            return False, None

        return True, self.frame.copy()

    def release(self):

        self.release_count += 1
        self._opened = False


class CaptureFactory:
    """Creates fake captures and records every attempt."""

    def __init__(self, plan):
        """
        plan: callable(attempt, index) -> FakeCapture, or a list of
        captures reused in order (the last one is reused).
        """

        self._plan = plan
        self.attempts = []

    def __call__(self, index, backend):

        self.attempts.append((index, backend))

        if callable(self._plan):
            return self._plan(len(self.attempts) - 1, index)

        position = min(len(self.attempts) - 1, len(self._plan) - 1)

        return self._plan[position]


class FakeModel:
    """Embedding model stand-in; no InsightFace, no model load."""

    def detect(self, image):
        return []

    def generate(self, image):
        return None


class FakeDatabase:
    """Face database stand-in so no folder scan or cache is written."""

    def __init__(self):
        self.loaded = 0

    def load(self):
        self.loaded += 1

    def get_database(self):
        return {}

    def person_count(self):
        return 0

    def embedding_count(self):
        return 0

    def list_people(self):
        return []


# ----------------------------------------------------------------------
# A. Camera manager
# ----------------------------------------------------------------------

def test_open_and_verify():
    """Opening records what the device really delivers."""

    container = FakeContainer(width=1280, height=720, fps=30)

    capture = FakeCapture(frame=np.zeros((480, 640, 3), dtype=np.uint8))

    factory = CaptureFactory([capture])

    camera = CameraManager(container, capture_factory=factory)

    try:

        assert camera.open() is True

        status = camera.status()

        assert status["actual_width"] == 640, status
        assert status["actual_height"] == 480, status
        assert "640x480" in status["message"], status["message"]
        assert status["open"] is True
        assert camera.is_open()
        assert camera.is_healthy()

    finally:

        camera.release()

    assert capture.release_count == 1, "camera not released exactly once"


def test_discovery_falls_back():
    """A dead configured index must not prevent using a working one."""

    dead = FakeCapture(opened=False)

    working = FakeCapture()

    factory = CaptureFactory([dead, working])

    container = FakeContainer(
        index=3,
        auto_detect=True,
        probe_max_index=4,
    )

    camera = CameraManager(container, capture_factory=factory)

    try:

        assert camera.open() is True

        assert camera.status()["index"] == 0, camera.status()

    finally:

        camera.release()


def test_duplicate_ownership_refused():
    """Two managers must not open the same device index."""

    first_factory = CaptureFactory([FakeCapture()])

    first = CameraManager(
        FakeContainer(index=0, auto_detect=False),
        capture_factory=first_factory,
    )

    second_factory = CaptureFactory([FakeCapture()])

    second = CameraManager(
        FakeContainer(index=0, auto_detect=False),
        capture_factory=second_factory,
    )

    try:

        assert first.open() is True

        assert second.open() is False, "second manager opened the same camera"

        assert second_factory.attempts == [], (
            "second manager touched the device instead of refusing it"
        )

        first.release()

        assert second.open() is True, "device was not usable after release"

    finally:

        first.release()
        second.release()


def test_read_failure_health_and_recovery():
    """Consecutive failed reads mark the camera unhealthy, then recover."""

    capture = FakeCapture()

    camera = CameraManager(
        FakeContainer(read_failures_before_reconnect=2),
        capture_factory=CaptureFactory([capture]),
    )

    try:

        assert camera.open() is True

        # One good read, then the device starts failing.
        capture._fail_after = capture.reads + 1

        assert camera.read() is not None

        assert camera.read() is None
        assert camera.read() is None

        status = camera.status()

        assert status["state"] == "degraded", status
        assert status["consecutive_failures"] >= 2
        assert camera.is_healthy() is False

        capture._fail_after = None

        assert camera.read() is not None

        status = camera.status()

        assert status["state"] == "ready", status
        assert status["consecutive_failures"] == 0
        assert camera.is_healthy() is True

    finally:

        camera.release()


def test_grabber_reconnects():
    """The capture thread reconnects without a second camera manager."""

    first = FakeCapture(fail_after=1)

    second = FakeCapture()

    factory = CaptureFactory([first, second])

    camera = CameraManager(
        FakeContainer(
            read_failures_before_reconnect=2,
            reconnect_interval_s=0.01,
        ),
        capture_factory=factory,
    )

    grabber = FrameGrabber(camera)

    try:

        assert camera.open() is True

        grabber.start()

        wait_for(
            lambda: camera.status()["reconnects"] >= 1
            and grabber.get_frame() is not None
            and grabber.last_timestamp > 0,
            timeout=4.0,
        )

        assert first.release_count == 1, "lost device was not released"
        assert second.release_count == 0, "new device was released too early"
        assert grabber.reconnects >= 1
        assert grabber.is_alive()

    finally:

        assert grabber.stop() is True, "grabber thread did not stop"
        camera.release()

    assert second.release_count == 1


def test_release_waits_for_read():
    """Release waits for an in-flight read instead of racing it."""

    capture = FakeCapture(read_delay=0.20)

    camera = CameraManager(
        FakeContainer(release_timeout_s=1.0),
        capture_factory=CaptureFactory([capture]),
    )

    assert camera.open() is True

    threading.Thread(target=camera.read, daemon=True).start()

    time.sleep(0.05)

    started = time.perf_counter()

    camera.release()

    elapsed = time.perf_counter() - started

    assert elapsed >= 0.10, (
        "release returned after %.2fs, during the read" % elapsed
    )

    assert capture.release_count == 1
    assert camera.is_open() is False


def test_release_abandons_stuck_read():
    """A read that cannot be drained is not released underneath."""

    capture = FakeCapture(read_delay=0.50)

    camera = CameraManager(
        FakeContainer(release_timeout_s=0.10),
        capture_factory=CaptureFactory([capture]),
    )

    assert camera.open() is True

    reader = threading.Thread(target=camera.read, daemon=True)

    reader.start()

    time.sleep(0.05)

    started = time.perf_counter()

    camera.release()

    elapsed = time.perf_counter() - started

    reader.join(timeout=2.0)

    assert elapsed < 0.45, "release blocked on a stuck read"

    assert capture.release_count == 0, (
        "device was released while a read was still using it"
    )

    assert camera.status()["state"] == "release-failed"


def test_grabber_stops_before_release():
    """Shutdown order: capture thread stops, then the device is released."""

    capture = FakeCapture()

    camera = CameraManager(
        FakeContainer(),
        capture_factory=CaptureFactory([capture]),
    )

    grabber = FrameGrabber(camera)

    assert camera.open() is True

    grabber.start()

    wait_for(lambda: grabber.get_frame() is not None)

    assert grabber.stop() is True

    assert grabber.is_alive() is False

    camera.release()

    assert capture.release_count == 1
    assert camera.is_open() is False


# ----------------------------------------------------------------------
# B. Degraded startup
# ----------------------------------------------------------------------

def test_vision_degraded_then_recovers():
    """
    A missing camera must not raise, must mark the app degraded, and
    must start the pipeline once the camera becomes available.
    """

    from app.vision.vision_service import VisionService

    def plan(attempt, index):

        if attempt == 0:
            return FakeCapture(opened=False)

        return FakeCapture()

    factory = CaptureFactory(plan)

    container = FakeContainer(
        auto_detect=False,
        open_retry_interval_s=0.1,
        warmup_frames=1,
    )

    camera = CameraManager(container, capture_factory=factory)

    database = FakeDatabase()

    vision = VisionService(
        container,
        camera=camera,
        embedding_model=FakeModel(),
        database=database,
    )

    events = []

    container.event_bus.subscribe(
        "camera.status",
        lambda **payload: events.append(("camera", payload.get("state"))),
    )

    container.event_bus.subscribe(
        "vision.degraded",
        lambda **payload: events.append(("degraded", True)),
    )

    try:

        assert vision.start() is False, (
            "vision claimed to start without a camera"
        )

        assert vision.is_running() is False
        assert vision.process(annotate=False) is None

        assert ("degraded", True) in events, (
            "degraded mode was not announced"
        )

        assert ("camera", "unavailable") in events, events

        wait_for(vision.is_running, timeout=3.0)

        wait_for(lambda: vision.grabber.get_frame() is not None, timeout=3.0)

        assert ("camera", "ready") in events, events
        assert database.loaded == 1

        health = vision.health()

        assert health["running"] is True
        assert health["camera"]["open"] is True
        assert health["grabber"]["running"] is True

    finally:

        vision.stop()

    assert vision.is_running() is False
    assert vision.camera.is_open() is False
    assert vision.grabber.is_alive() is False


# ----------------------------------------------------------------------
# C. Optional live smoke test
# ----------------------------------------------------------------------

def test_live_camera():
    """Open the real configured camera once. Only with --live."""

    from app.core.container import ServiceContainer

    container = ServiceContainer()

    camera = CameraManager(container)

    try:

        assert camera.open() is True, camera.last_error

        wait_for(lambda: camera.read() is not None, timeout=3.0)

        frame = camera.read()

        assert frame is not None and frame.size > 0

        status = camera.status()

        print(
            "       index=%s actual=%sx%s @ %.1f fps"
            % (
                status["index"],
                status["actual_width"],
                status["actual_height"],
                status["actual_fps"],
            )
        )

    finally:

        camera.release()

    assert camera.is_open() is False


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--live",
        action="store_true",
        help="also open the real camera once",
    )

    args = parser.parse_args(argv)

    print("Greeto camera checks")

    check("A1. open verifies actual resolution", test_open_and_verify)

    check("A2. discovery falls back to a working index", test_discovery_falls_back)

    check("A3. duplicate device ownership refused", test_duplicate_ownership_refused)

    check(
        "A4. read failures mark unhealthy, then recover",
        test_read_failure_health_and_recovery,
    )

    check("A5. capture thread reconnects after loss", test_grabber_reconnects)

    check("A6. release waits for an in-flight read", test_release_waits_for_read)

    check(
        "A7. stuck read is not released underneath",
        test_release_abandons_stuck_read,
    )

    check(
        "A8. capture thread stops before release",
        test_grabber_stops_before_release,
    )

    check(
        "B1. voice-only startup recovers when camera returns",
        test_vision_degraded_then_recovers,
    )

    if args.live:

        check("C1. live camera open/read/release", test_live_camera)

    else:

        print("[SKIP] C1. live camera (use --live)")

    print("-" * 74)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))

        return 1

    print("RESULT: ALL CHECKS PASSED")

    return 0


if __name__ == "__main__":

    sys.exit(main())
