"""
Frame Grabber.

Continuously captures the newest camera frame in a
background thread.

The processing thread always receives the latest frame,
preventing camera lag caused by queued frames.
"""

from __future__ import annotations

import time
from threading import Lock, Thread

import numpy as np

from app.vision.camera_manager import CameraManager


class FrameGrabber:
    """
    Background frame capture service.
    """

    def __init__(
        self,
        camera: CameraManager,
    ) -> None:

        self.camera = camera

        self._running = False

        self._thread: Thread | None = None

        self._lock = Lock()

        self._latest_frame: np.ndarray | None = None

        self._frame_counter = 0

        self._start_time = 0.0

        self._last_timestamp = 0.0

    # -----------------------------------------------------

    def start(self) -> None:
        """
        Start capture thread.
        """

        if self._running:
            return

        self._running = True

        self._frame_counter = 0

        self._start_time = time.perf_counter()

        self._thread = Thread(
            target=self._capture_loop,
            daemon=True,
            name="FrameGrabber",
        )

        self._thread.start()

    # -----------------------------------------------------

    def stop(self) -> None:
        """
        Stop capture thread.
        """

        self._running = False

        if self._thread is not None:

            self._thread.join(timeout=2.0)

            self._thread = None

    # -----------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        """
        Return the newest available frame.

        Returns a copy so the processing pipeline cannot
        accidentally modify the shared frame.
        """

        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    # -----------------------------------------------------

    def get_fps(self) -> float:
        """
        Return actual capture FPS.
        """

        elapsed = time.perf_counter() - self._start_time

        if elapsed <= 0:
            return 0.0

        return self._frame_counter / elapsed

    # -----------------------------------------------------

    @property
    def last_timestamp(self) -> float:
        """
        Timestamp of newest frame.
        """

        return self._last_timestamp

    # -----------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Continuously capture frames.

        Old frames are discarded automatically.
        """

        while self._running:

            frame = self.camera.read()

            if frame is None:

                time.sleep(0.002)

                continue

            with self._lock:

                self._latest_frame = frame

                self._last_timestamp = time.perf_counter()

            self._frame_counter += 1