"""
Vision pump.

Runs the existing vision pipeline on a worker thread and discards the
frames. The kiosk never displays the camera, so the GUI thread must not
spend time decoding, annotating or scaling frames.

Events (visitor detected / unknown / left) are published by VisionService
through the thread-safe EventBus, which the presenter forwards to the GUI.
"""

from __future__ import annotations

import logging
import threading
import time

from PyQt5.QtCore import QThread


logger = logging.getLogger(__name__)


class VisionPump(QThread):
    """Drives VisionService.process() off the GUI thread."""

    def __init__(self, vision, fps=30, parent=None):

        super().__init__(parent)

        self._vision = vision

        self._interval = 1.0 / max(1, int(fps))

        self._stop_event = threading.Event()

        self.frames = 0

        self.errors = 0

    # -----------------------------------------------------

    def run(self):

        while not self._stop_event.is_set():

            started = time.perf_counter()

            self._process_once()

            remaining = self._interval - (time.perf_counter() - started)

            if remaining > 0:

                self._stop_event.wait(remaining)

    def _process_once(self):

        try:

            # Frames are not needed by the kiosk; skip annotation and copy.
            self._vision.process(annotate=False)

            self.frames += 1

            return

        except TypeError:

            # Older VisionService signatures do not accept `annotate`.
            pass

        except Exception:

            self.errors += 1

            logger.exception("Vision processing failed.")

            return

        try:

            self._vision.process()

            self.frames += 1

        except Exception:

            self.errors += 1

            logger.exception("Vision processing failed.")

    # -----------------------------------------------------

    def stop(self):

        self._stop_event.set()

        if self.isRunning():

            self.wait(3000)
