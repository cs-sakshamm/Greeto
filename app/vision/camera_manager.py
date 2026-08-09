"""
Camera Manager.

Provides a low-latency interface for webcam capture.
Optimized for Windows using the DirectShow backend.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.core.container import ServiceContainer


class CameraManager:
    """
    Handles all camera operations.
    """

    def __init__(
        self,
        container: ServiceContainer,
    ) -> None:

        self.container = container

        self.logger = container.logger
        self.config = container.config

        self.camera: cv2.VideoCapture | None = None

        self.width = 640
        self.height = 480
        self.fps = 30

        self.frame_count = 0
        self.start_time = 0.0

    # -----------------------------------------------------

    def open(self) -> bool:
        """
        Open camera with low-latency configuration.
        """

        camera_index = self.config.get_value(
            "camera",
            "camera",
            "index",
        )

        self.width = self.config.get_value(
            "camera",
            "camera",
            "width",
        )

        self.height = self.config.get_value(
            "camera",
            "camera",
            "height",
        )

        self.fps = self.config.get_value(
            "camera",
            "camera",
            "fps",
        )

        self.logger.info("Opening camera...")

        #
        # DirectShow gives lower latency on Windows
        #
        self.camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW,
        )

        if not self.camera.isOpened():

            self.logger.error(
                "Unable to open camera."
            )

            return False

        #
        # Keep only newest frame
        #
        self.camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        #
        # Request MJPEG stream
        #
        self.camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        self.camera.set(
            cv2.CAP_PROP_FPS,
            self.fps,
        )

        #
        # Warm-up frames
        #
        for _ in range(5):
            self.camera.read()

        self.start_time = time.perf_counter()

        self.logger.info(
            f"Camera ready ({self.width}x{self.height} @ {self.fps} FPS)"
        )

        return True

    # -----------------------------------------------------

    def read(self) -> np.ndarray | None:
        """
        Read newest frame.
        """

        if self.camera is None:
            return None

        success, frame = self.camera.read()

        if not success:

            self.logger.warning(
                "Camera frame grab failed."
            )

            return None

        self.frame_count += 1

        return frame

    # -----------------------------------------------------

    def get_capture_fps(self) -> float:
        """
        Actual camera FPS.
        """

        elapsed = time.perf_counter() - self.start_time

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    # -----------------------------------------------------

    def release(self) -> None:
        """
        Release camera.
        """

        if self.camera is not None:

            self.logger.info(
                "Releasing camera..."
            )

            self.camera.release()

            self.camera = None

            self.logger.info(
                "Camera released."
            )