"""
Speech Service.

Continuous listening pipeline.

Microphone
    ↓
VAD
    ↓
Audio Buffer
    ↓
Whisper
    ↓
EventBus
"""

from __future__ import annotations

import threading
import time

import numpy as np

from app.core.container import ServiceContainer

from app.speech.audio_capture import AudioCapture
from app.speech.vad import VoiceActivityDetector
from app.speech.stt import SpeechToText


class SpeechService:
    """
    Continuous speech recognition service.
    """

    def __init__(
        self,
        container: ServiceContainer,
    ) -> None:

        self.container = container

        self.logger = container.logger

        self.event_bus = getattr(
            container,
            "event_bus",
            None,
        )

        self.audio_capture = AudioCapture()

        self.vad = VoiceActivityDetector()

        self.stt = SpeechToText()

        self.running = False

        self.thread = None

        # Speech handling
        self.audio_buffer: list[np.ndarray] = []

        self.silence_counter = 0


        # Number of silent chunks before processing
        self.max_silence_chunks = 7


        # Minimum audio duration
        self.min_audio_length = 16000 * 0.3



    def start(self):

        if self.running:
            return

        self.logger.info(
            "Starting Speech Service..."
        )

        self.audio_capture.start()

        self.running = True

        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
        )

        self.thread.start()

        self.logger.info(
            "Speech Service started."
        )



    def stop(self):

        self.running = False

        self.audio_capture.stop()

        self.logger.info(
            "Speech Service stopped."
        )



    def _listen_loop(self):

        while self.running:

            audio = self.audio_capture.get_audio()


            if audio is None:
                continue



            speaking = self.vad.is_speech(
                audio
            )


            if speaking:

                self.logger.info(
                    "Speech detected."
                )

                self.audio_buffer.append(
                    audio
                )

                self.silence_counter = 0



            else:

                if self.audio_buffer:

                    self.silence_counter += 1



                    if (
                        self.silence_counter
                        >= self.max_silence_chunks
                    ):

                        self._process_audio()

                        self.audio_buffer.clear()

                        self.silence_counter = 0



            time.sleep(0.01)



    def _process_audio(self):

        audio = np.concatenate(
            self.audio_buffer
        )


        # Ignore very small audio
        if len(audio) < self.min_audio_length:

            self.logger.info(
                "Audio too short. Ignoring."
            )

            return



        self.logger.info(
            "Transcribing..."
        )


        text = self.stt.transcribe(
            audio
        )


        if not text:
            return



        self.logger.info(
            f"Recognized: {text}"
        )


        if self.event_bus:

            self.event_bus.publish(
    "speech_recognized",
    text=text,
)