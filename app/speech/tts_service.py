"""
Text To Speech Service.

Converts AI responses into speech.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from app.core.container import ServiceContainer


class TTSService:
    """
    Offline Piper TTS wrapper.
    """


    def __init__(
        self,
        container: ServiceContainer,
    ) -> None:

        self.container = container

        self.logger = container.logger


        self.model_path = Path(
            "models/tts/en_US-lessac-medium.onnx"
        )


        if not self.model_path.exists():

            raise FileNotFoundError(
                f"TTS model missing: {self.model_path}"
            )


    # -------------------------------------------------

    def speak(
        self,
        text: str,
    ) -> None:
        """
        Convert text into speech.
        """

        if not text:
            return


        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as audio_file:

                wav_path = audio_file.name



            command = [
                "piper",
                "--model",
                str(self.model_path),
                "--output_file",
                wav_path,
            ]


            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                text=True,
            )


            process.communicate(
                text
            )


            #
            # Play audio
            #

            self._play_audio(
                wav_path
            )


        except Exception as error:

            self.logger.exception(
                error
            )


    # -------------------------------------------------

    def _play_audio(
        self,
        path: str,
    ) -> None:
        """
        Play generated WAV file.
        """

        import winsound

        winsound.PlaySound(
            path,
            winsound.SND_FILENAME,
        )