"""
Camera Manager.

Single owner of the webcam device for the whole application.

Responsibilities
----------------
- Open the configured camera (DirectShow on Windows, V4L2 on Linux).
- Fall back to scanning other indices when the configured device is
  unavailable and ``auto_detect`` is enabled.
- Configure, then *verify* the resolution/FPS actually delivered.
- Detect a lost camera through consecutive failed reads.
- Reconnect without creating a second camera manager or capture object.
- Refuse to open a device another CameraManager in this process owns.
- Release the device safely, never underneath an in-flight read.

It never touches the GUI and never publishes biometric data.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from app.core.container import ServiceContainer


CaptureFactory = Callable[[int, Optional[int]], object]


# Indices currently owned by any CameraManager in this process. Opening
# the same device twice is the usual cause of a black preview and a
# crash on shutdown.
_REGISTRY_LOCK = threading.Lock()
_OPEN_INDICES = set()


def _acquire_index(index: int) -> bool:
    with _REGISTRY_LOCK:
        if index in _OPEN_INDICES:
            return False
        _OPEN_INDICES.add(index)
        return True


def _release_index(index: int) -> None:
    with _REGISTRY_LOCK:
        _OPEN_INDICES.discard(index)


@dataclass
class CameraStatus:
    """Serialisable snapshot of the camera state."""

    state: str = "closed"
    index: Optional[int] = None
    message: str = ""
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: float = 0.0
    actual_width: int = 0
    actual_height: int = 0
    actual_fps: float = 0.0
    frames: int = 0
    read_failures: int = 0
    consecutive_failures: int = 0
    reconnects: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class CameraManager:
    """
    Handles all camera operations.

    ``capture_factory`` allows the camera behaviour to be tested
    without hardware. It receives ``(index, backend)`` and returns an
    object with the cv2.VideoCapture subset used here: ``isOpened``,
    ``set``, ``get``, ``read`` and ``release``.
    """

    def __init__(
        self,
        container: ServiceContainer,
        capture_factory: CaptureFactory | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:

        self.container = container
        self.logger = container.logger
        self.config = container.config

        self._capture_factory = capture_factory
        self._clock = clock or time.perf_counter

        self._capture = None
        self._index: Optional[int] = None

        # Serialises reads against release.
        self._io_lock = threading.Lock()

        # Protects capture, index and status fields.
        self._state_lock = threading.Lock()

        # Only one reconnect at a time.
        self._reconnect_lock = threading.Lock()

        self._opened_at = 0.0
        self._last_frame_at = 0.0

        self.frame_count = 0
        self.start_time = 0.0

        self._last_error = ""
        self._status = CameraStatus()

        self._load_settings()

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    def _get(self, key, default):

        value = self.config.get_value(
            "camera", "camera", key, default=default
        )

        return default if value is None else value

    def _get_int(self, key, default):
        try:
            return int(self._get(key, default))
        except (TypeError, ValueError):
            return default

    def _get_float(self, key, default):
        try:
            return float(self._get(key, default))
        except (TypeError, ValueError):
            return default

    def _get_bool(self, key, default):
        value = self._get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _load_settings(self) -> None:

        self._requested_index = self._get_int("index", 0)
        self._width = self._get_int("width", 1280)
        self._height = self._get_int("height", 720)
        self._fps = self._get_float("fps", 30.0)
        self._backend = str(self._get("backend", "auto")).strip().lower()

        self._auto_detect = self._get_bool("auto_detect", True)
        self._probe_max_index = max(
            0, self._get_int("probe_max_index", 5)
        )
        self._warmup_frames = max(0, self._get_int("warmup_frames", 5))

        self._failure_threshold = max(
            1, self._get_int("read_failures_before_reconnect", 15)
        )
        self._reconnect_interval = max(
            0.0, self._get_float("reconnect_interval_s", 5.0)
        )
        self._open_retry_interval = max(
            0.1, self._get_float("open_retry_interval_s", 5.0)
        )
        self._release_timeout = max(
            0.0, self._get_float("release_timeout_s", 0.75)
        )
        self._stale_after = max(
            0.1, self._get_float("frame_stale_after_s", 2.0)
        )

        with self._state_lock:
            self._status.requested_width = self._width
            self._status.requested_height = self._height
            self._status.requested_fps = self._fps

    # Read-only settings used by FrameGrabber.
    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def reconnect_interval(self) -> float:
        return self._reconnect_interval

    @property
    def open_retry_interval(self) -> float:
        return self._open_retry_interval

    @property
    def last_error(self) -> str:
        return self._last_error

    # -----------------------------------------------------
    # Public state
    # -----------------------------------------------------

    def is_open(self) -> bool:
        return self._capture is not None

    def is_healthy(self) -> bool:

        if self._capture is None:
            return False

        with self._state_lock:
            failures = self._status.consecutive_failures

        if failures >= self._failure_threshold:
            return False

        age = self.frame_age()

        return age is None or age <= self._stale_after

    def frame_age(self) -> Optional[float]:
        """Seconds since the last successful read."""

        if self._last_frame_at <= 0:
            return None

        return max(0.0, self._clock() - self._last_frame_at)

    def status(self) -> dict:
        """Snapshot safe to expose without holding internal locks."""

        with self._state_lock:
            snapshot = self._status.to_dict()
            snapshot["index"] = self._index
            opened = self._capture is not None

        snapshot["open"] = opened
        snapshot["healthy"] = opened and self.is_healthy()
        snapshot["frame_age_s"] = self.frame_age()

        return snapshot

    def health(self) -> dict:
        return self.status()

    # -----------------------------------------------------
    # Open / discovery
    # -----------------------------------------------------

    def open(self) -> bool:
        """
        Open a camera and verify that frames can actually be read.

        Tries the configured index first, then (with auto_detect) the
        remaining indices up to ``probe_max_index``.
        """

        if self._capture is not None:
            return True

        self._set_state("opening", "Opening camera...")

        for index in self._candidate_indices():

            capture = self._open_index(index)

            if capture is None:
                continue

            with self._state_lock:
                self._capture = capture
                self._index = index
                self._status.state = "ready"
                self._status.index = index
                self._opened_at = self._clock()
                self._last_frame_at = self._clock()
                actual_width = self._status.actual_width
                actual_height = self._status.actual_height
                actual_fps = self._status.actual_fps

            self._last_error = ""

            self.logger.info(
                "Camera ready (index %s, requested %sx%s @ %.1f FPS; "
                "actual %sx%s @ %.1f FPS)",
                index, self._width, self._height, self._fps,
                actual_width, actual_height, actual_fps,
            )

            self._publish_status("ready")

            return True

        self._last_error = (
            "No usable camera could be opened. Check the connection, "
            "the device index in config/camera.yaml and whether another "
            "application is using the camera."
        )

        self.logger.error(self._last_error)
        self._set_state("unavailable", self._last_error)

        return False

    def _candidate_indices(self) -> list:

        candidates = [self._requested_index]

        if self._auto_detect:
            for index in range(0, self._probe_max_index + 1):
                if index not in candidates:
                    candidates.append(index)

        return candidates

    def _backend_candidates(self) -> list:

        if self._backend == "dshow":
            return [cv2.CAP_DSHOW]

        if self._backend == "v4l2":
            return [cv2.CAP_V4L2]

        if self._backend == "any":
            return [cv2.CAP_ANY]

        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_ANY]

        return [cv2.CAP_V4L2, cv2.CAP_ANY]

    def _create_capture(self, index: int, backend):

        if self._capture_factory is not None:
            return self._capture_factory(index, backend)

        if backend is None:
            return cv2.VideoCapture(index)

        return cv2.VideoCapture(index, backend)

    def _open_index(self, index: int):
        """
        Try one device index. Returns a configured, verified capture
        object or None. The device registry is always released when
        the attempt fails.
        """

        if not _acquire_index(index):
            self.logger.warning(
                "Camera index %s is already owned by another "
                "CameraManager in this process; skipping it.", index
            )
            return None

        capture = None

        try:

            for backend in self._backend_candidates():

                capture = self._create_capture(index, backend)

                if capture is not None and capture.isOpened():
                    break

                if capture is not None:
                    capture.release()

                capture = None

            if capture is None:
                self.logger.warning(
                    "Camera index %s could not be opened.", index
                )
                _release_index(index)
                return None

            self._configure_capture(capture)

            frame = self._warm_up(capture)

            if frame is None:
                self.logger.warning(
                    "Camera index %s opened but delivered no frame.",
                    index,
                )
                capture.release()
                _release_index(index)
                return None

            self._record_actual_properties(capture, frame)

            return capture

        except Exception as error:

            self.logger.exception(error)

            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

            _release_index(index)

            return None

    def _configure_capture(self, capture) -> None:

        # Keep only the newest frame. Not every backend supports this.
        self._safe_set(capture, cv2.CAP_PROP_BUFFERSIZE, 1)

        # MJPG keeps USB bandwidth and latency manageable at 720p.
        self._safe_set(
            capture,
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        self._safe_set(capture, cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._safe_set(capture, cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._safe_set(capture, cv2.CAP_PROP_FPS, self._fps)

    @staticmethod
    def _safe_set(capture, prop, value) -> None:
        try:
            capture.set(prop, value)
        except Exception:
            pass

    def _warm_up(self, capture):

        frame = None

        for _ in range(self._warmup_frames):

            success, candidate = capture.read()

            if success and candidate is not None:
                frame = candidate

        if frame is None:

            success, candidate = capture.read()

            if not success or candidate is None:
                return None

            frame = candidate

        return frame

    def _record_actual_properties(self, capture, frame) -> None:

        height = int(frame.shape[0]) if hasattr(frame, "shape") else 0
        width = int(frame.shape[1]) if hasattr(frame, "shape") else 0

        try:
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            reported_fps = 0.0

        message = ""

        if width and height and (
            width != self._width or height != self._height
        ):

            message = (
                "Camera delivers %sx%s although %sx%s was requested."
                % (width, height, self._width, self._height)
            )

            self.logger.warning(message)

        if reported_fps > 0 and self._fps > 0:

            if abs(reported_fps - self._fps) > max(5.0, self._fps * 0.25):

                self.logger.warning(
                    "Camera reports %.1f FPS although %.1f FPS was "
                    "requested.", reported_fps, self._fps
                )

        with self._state_lock:
            self._status.actual_width = width
            self._status.actual_height = height
            self._status.actual_fps = reported_fps
            self._status.message = message

    # -----------------------------------------------------
    # Read
    # -----------------------------------------------------

    def read(self) -> np.ndarray | None:
        """
        Read the newest frame.

        Read and release are serialised, so a release waits for an
        in-flight read instead of pulling the device out from under it.
        """

        with self._io_lock:

            capture = self._capture

            if capture is None:
                return None

            success, frame = capture.read()

        if not success or frame is None:
            self._note_read_failure()
            return None

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self._note_read_failure()
            return None

        self._note_read_success(frame)

        return frame

    def _note_read_success(self, frame) -> None:

        with self._state_lock:

            # A read that completes after release must not resurrect
            # the camera status.
            if self._capture is None:
                return

            recovered = (
                self._status.consecutive_failures
                >= self._failure_threshold
            )

            if recovered:
                self._status.state = "ready"
                self._status.message = ""

            self._status.consecutive_failures = 0
            self._status.frames += 1
            self._status.actual_width = int(frame.shape[1])
            self._status.actual_height = int(frame.shape[0])

            self._last_frame_at = self._clock()
            self.frame_count += 1

        if recovered:
            self._publish_status("ready")

    def _note_read_failure(self) -> None:

        with self._state_lock:

            if self._capture is None:
                return

            self._status.read_failures += 1
            self._status.consecutive_failures += 1

            tripped = (
                self._status.consecutive_failures
                >= self._failure_threshold
                and self._status.state != "degraded"
            )

            if tripped:
                self._status.state = "degraded"
                self._status.message = (
                    "Camera stopped delivering frames (%s consecutive "
                    "read failures)."
                    % self._status.consecutive_failures
                )

            failures = self._status.consecutive_failures

        if tripped:

            self.logger.warning(
                "Camera delivered %s consecutive failed reads; the "
                "capture thread will reconnect.", failures
            )

            self._publish_status("degraded")

    # -----------------------------------------------------
    # Reconnect
    # -----------------------------------------------------

    def reconnect(self) -> bool:
        """
        Reopen the camera after it was lost.

        Returns False when another reconnect is running or no device
        could be opened. The caller decides how long to wait before
        the next attempt.
        """

        if not self._reconnect_lock.acquire(blocking=False):
            return False

        try:

            self.logger.warning("Attempting camera reconnect...")

            with self._state_lock:
                self._status.reconnects += 1
                self._status.state = "reconnecting"
                self._status.message = "Reconnecting to the camera..."

            self._publish_status("reconnecting")

            self._close_capture("reconnect")

            opened = self.open()

            if opened:
                self.logger.info("Camera reconnect succeeded.")
            else:
                self.logger.error("Camera reconnect failed.")

            return opened

        finally:
            self._reconnect_lock.release()

    # -----------------------------------------------------
    # Release
    # -----------------------------------------------------

    def release(self) -> None:
        """Release the camera. Idempotent."""

        if self._capture is None:
            if self._status.state != "released":
                self._set_state("released", "Camera released.")
            return

        self.logger.info("Releasing camera...")

        if self._close_capture("release"):
            self.logger.info("Camera released.")
            self._set_state("released", "Camera released.")
        else:
            self.logger.error(
                "Camera could not be released safely because a frame "
                "read is still in progress; leaving the device to the "
                "operating system."
            )
            self._set_state(
                "release-failed",
                "Camera read still in progress; device left to the OS.",
            )

    def _close_capture(self, reason: str) -> bool:
        """
        Detach the capture object, then release it once no read is
        using it. Returns False when the device had to be abandoned.
        """

        with self._state_lock:
            capture = self._capture
            index = self._index
            self._capture = None
            self._index = None

        if index is not None:
            _release_index(index)

        if capture is None:
            return True

        acquired = self._io_lock.acquire(timeout=self._release_timeout)

        if not acquired:

            self.logger.error(
                "Timed out waiting for an in-flight frame read (%s) "
                "before releasing the camera.", reason
            )

            return False

        try:

            try:
                capture.release()
            except Exception as error:
                self.logger.exception(error)

        finally:
            self._io_lock.release()

        return True

    # -----------------------------------------------------
    # Metrics and events
    # -----------------------------------------------------

    def get_capture_fps(self) -> float:
        """Actual capture FPS since the camera was opened."""

        elapsed = self._clock() - self._opened_at

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    def _set_state(self, state: str, message: str = "") -> None:

        with self._state_lock:

            self._status.state = state

            if message:
                self._status.message = message

            self._status.index = self._index

        self._publish_status(state)

    def _publish_status(self, state: str) -> None:
        """Publish a camera status event. Never raises."""

        bus = getattr(self.container, "event_bus", None)

        if bus is None:
            return

        try:

            with self._state_lock:
                self._status.state = state
                self._status.index = self._index

            # The snapshot already carries "state"; passing it again
            # would be a duplicate keyword argument.
            bus.publish("camera.status", **self.status())

        except Exception as error:
            self.logger.exception(error)
