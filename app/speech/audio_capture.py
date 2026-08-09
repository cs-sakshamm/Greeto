"""
Continuous microphone capture.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Continuously captures microphone audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ) -> None:

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        self.running = False

        self.thread: threading.Thread | None = None


    def start(self) -> None:
        """
        Start microphone capture.
        """

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.thread.start()


    def stop(self) -> None:
        """
        Stop microphone capture.
        """

        self.running = False


    def get_audio(
        self,
        timeout: float = 1.0,
    ) -> np.ndarray | None:
        """
        Get audio chunk.
        """

        try:

            return self.audio_queue.get(
                timeout=timeout
            )

        except queue.Empty:

            return None


    def _capture_loop(self) -> None:
        """
        Background microphone thread.
        """

        def callback(
            indata,
            frames,
            time,
            status,
        ):

            if status:
                print(status)

            audio = (
                indata[:, 0]
                .copy()
            )

            self.audio_queue.put(audio)


        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=self.chunk_size,
        ):

            while self.running:
                pass