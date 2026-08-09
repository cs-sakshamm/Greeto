"""
Speech To Text.

Offline Whisper transcription using pywhispercpp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pywhispercpp.model import Model


class SpeechToText:
    """
    Offline Whisper Speech Recognition.
    """

    def __init__(
        self,
        model_path: str = "models/whisper/ggml-tiny.en.bin",
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Whisper model not found: {self.model_path}"
            )

        self.model = Model(
    str(self.model_path),
    n_threads=8,
            print_realtime=False,
            print_progress=False,
        )


    def transcribe(
        self,
        audio: np.ndarray,
    ) -> str:
        """
        Convert speech audio into text.

        Args:
            audio:
                Float32 mono audio at 16kHz.

        Returns:
            Recognized text.
        """

        if audio is None or len(audio) == 0:

            return ""


        segments = self.model.transcribe(
            audio,
        )


        text = []

        for segment in segments:

            text.append(
                segment.text
            )


        return " ".join(text).strip()