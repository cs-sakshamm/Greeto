"""
Voice Activity Detection.

Detects whether incoming audio contains speech.
"""

from __future__ import annotations

import collections

import webrtcvad
import numpy as np


class VoiceActivityDetector:
    """
    Detects human speech from audio frames.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
    ) -> None:

        self.sample_rate = sample_rate

        self.vad = webrtcvad.Vad(
            aggressiveness
        )

        self.speech_frames = collections.deque(
            maxlen=20
        )

    def is_speech(
        self,
        audio: np.ndarray,
    ) -> bool:
        """
        Check if audio contains speech.

        Parameters:
            audio:
                Float32 mono audio.

        Returns:
            True if speech detected.
        """

        # Convert float32 to int16
        pcm_audio = (
            audio * 32768
        ).astype(
            np.int16
        )

        frame_duration = 30

        frame_size = int(
            self.sample_rate
            *
            frame_duration
            /
            1000
        )

        for start in range(
            0,
            len(pcm_audio),
            frame_size,
        ):

            frame = pcm_audio[
                start:start + frame_size
            ]

            if len(frame) != frame_size:
                continue

            if self.vad.is_speech(
                frame.tobytes(),
                self.sample_rate,
            ):
                return True

        return False