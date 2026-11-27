"""
Frame Grabber.

Continuously captures the newest camera frame in a background thread,
so the processing thread always receives the latest frame instead of a
backlog. Also owns the "camera lost" reaction: after a run of failed
reads it asks the CameraManager to reconnect (the manager remains the
only owner of the device).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from app.vision.camera_manager import CameraManager


class FrameGrabber:
    """
    Background frame capture service.
    """

    def __init__(
        self,
        camera: CameraManager,
        failure_threshold: int | None = None,
        reconnect_interval: float | None = None,
    ) -> None:

        self.camera = camera

        self._failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else getattr(camera, "failure_threshold", 15)
        )

        self._reconnect_interval = (
            reconnect_interval
            if reconnect_interval is not None
            else getattr(camera, "reconnect_interval", 5.0)
        )

        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()

        self._latest_frame: np.ndarray | None = None

        self._frame_counter = 0

        self._start_time = 0.0

        self._last_timestamp = 0.0

        self.reconnects = 0

        self._last_error = ""

    # -----------------------------------------------------

    def start(self) -> None:
        """
        Start the capture thread. Idempotent.
        """

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        self._frame_counter = 0

        self._start_time = time.perf_counter()

        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="FrameGrabber",
        )

        self._thread.start()

    # -----------------------------------------------------

    def stop(self, timeout: float = 3.0) -> bool:
        """
        Stop the capture thread.

        Returns True only when the thread has actually finished. The
        caller must not release the camera before that, otherwise a
        read could still be using it.
        """

        self._stop_event.set()

        thread = self._thread

        if thread is not None:

            thread.join(timeout=timeout)

            if thread.is_alive():

                self._last_error = (
                    "Frame capture thread did not stop within %.1fs."
                    % timeout
                )

                return False

            self._thread = None

        return True

    # -----------------------------------------------------

    def is_alive(self) -> bool:

        return self._thread is not None and self._thread.is_alive()

    # -----------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        """
        Return the newest available frame.

        Returns a copy so the processing pipeline cannot modify the
        shared frame.
        """

        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    # -----------------------------------------------------

    def get_fps(self) -> float:
        """Actual capture FPS."""

        elapsed = time.perf_counter() - self._start_time

        if elapsed <= 0:
            return 0.0

        with self._lock:
            frames = self._frame_counter

        return frames / elapsed

    # -----------------------------------------------------

    @property
    def last_timestamp(self) -> float:
        """Timestamp of the newest frame."""

        with self._lock:
            return self._last_timestamp

    @property
    def last_error(self) -> str:
        return self._last_error

    # -----------------------------------------------------

    def health(self) -> dict:
        """Small diagnostics snapshot."""

        age = None

        if self._last_timestamp > 0:

            age = max(0.0, time.perf_counter() - self._last_timestamp)

        return {
            "running": self.is_alive(),
            "frames": self._frame_counter,
            "fps": self.get_fps(),
            "last_frame_age_s": age,
            "reconnects": self.reconnects,
            "last_error": self._last_error,
        }

    # -----------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Continuously capture frames.

        Old frames are discarded automatically. When the device stops
        delivering frames the manager is asked to reconnect; the loop
        keeps running either way so recovery never needs a new thread.
        """

        failures = 0

        while not self._stop_event.is_set():

            if not self.camera.is_open():

                # The VisionService supervisor owns the initial
                # "camera unavailable" retry loop.
                self._stop_event.wait(0.05)

                continue

            frame = self.camera.read()

            if frame is None:

                failures += 1

                if failures < self._failure_threshold:

                    self._stop_event.wait(0.01)

                    continue

                failures = 0

                if self.camera.reconnect():

                    self.reconnects += 1
                    self._last_error = ""

                else:

                    self._last_error = (
                        self.camera.last_error
                        or "Camera reconnect failed."
                    )

                    self._stop_event.wait(self._reconnect_interval)

                continue

            failures = 0

            with self._lock:

                self._latest_frame = frame

                self._last_timestamp = time.perf_counter()

                self._frame_counter += 1
