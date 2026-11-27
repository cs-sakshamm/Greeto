"""
Voice Activity Detection.

Distinguishes human speech from silence and background noise using
WebRTC VAD when available, with an adaptive audio energy fallback.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Optional, Tuple

import numpy as np

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Detects voice activity in audio chunks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        energy_threshold: float = 0.01,
    ) -> None:
        self.sample_rate = sample_rate
        self.aggressiveness = max(0, min(3, int(aggressiveness)))
        self.frame_ms = frame_ms if frame_ms in (10, 20, 30) else 30
        self.energy_threshold = float(energy_threshold)

        self.vad = None
        if webrtcvad is not None and sample_rate in (8000, 16000, 32000, 48000):
            try:
                self.vad = webrtcvad.Vad(self.aggressiveness)
            except Exception as error:  # noqa: BLE001
                logger.warning("Failed to initialize webrtcvad: %s", error)
                self.vad = None

        self._frame_size = int(sample_rate * self.frame_ms / 1000)
        self.history = collections.deque(maxlen=20)

        # Latency instrumentation
        self._first_speech_ms: Optional[float] = None
        self._last_analysis_ms: Optional[float] = None

    @staticmethod
    def calculate_energy(audio: Optional[np.ndarray]) -> float:
        """
        Compute root-mean-square (RMS) energy of the audio.
        """
        if audio is None or len(audio) == 0:
            return 0.0
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))

    def is_speech(self, audio: Optional[np.ndarray]) -> bool:
        """
        Determine if the audio chunk contains human speech.
        """
        if audio is None or len(audio) == 0:
            return False

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return False

        # Energy gate: filter out low-energy ambient room noise/silence
        rms = self.calculate_energy(arr)
        if rms < self.energy_threshold:
            return False

        # If webrtcvad is unavailable, energy threshold is the fallback
        if self.vad is None:
            return True

        # Process frames with webrtcvad
        pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        speech_frames = 0
        total_frames = 0

        for start in range(0, len(pcm) - self._frame_size + 1, self._frame_size):
            frame = pcm[start : start + self._frame_size]
            if len(frame) != self._frame_size:
                continue
            total_frames += 1
            try:
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    speech_frames += 1
            except Exception:  # noqa: BLE001
                continue

        if total_frames == 0:
            return rms >= self.energy_threshold

        # Consider it speech if at least one 30ms sub-frame is voice
        return speech_frames > 0

    def analyze_chunk(self, audio: Optional[np.ndarray]) -> Tuple[bool, float, int, int]:
        """
        Detailed analysis: returns (is_speech, rms_energy, speech_frames, total_frames).
        """
        if audio is None or len(audio) == 0:
            return False, 0.0, 0, 0

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return False, 0.0, 0, 0

        rms = self.calculate_energy(arr)
        if rms < self.energy_threshold:
            return False, rms, 0, 0

        chunk_ms = time.monotonic() * 1000
        self._last_analysis_ms = chunk_ms

        if self.vad is None:
            if self._first_speech_ms is None:
                self._first_speech_ms = chunk_ms
            return True, rms, 1, 1

        pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        speech_frames = 0
        total_frames = 0

        for start in range(0, len(pcm) - self._frame_size + 1, self._frame_size):
            frame = pcm[start : start + self._frame_size]
            if len(frame) != self._frame_size:
                continue
            total_frames += 1
            try:
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    speech_frames += 1
            except Exception:  # noqa: BLE001
                continue

        if self._first_speech_ms is None and speech_frames > 0:
            self._first_speech_ms = chunk_ms

        return (speech_frames > 0), rms, speech_frames, total_frames

    def get_vad_metrics(self) -> Dict[str, Any]:
        """
        Return timing metrics for this VAD session.
        """
        return {
            "first_speech_ms": self._first_speech_ms,
            "last_analysis_ms": self._last_analysis_ms,
        }

    def get_audio_quality_metrics(self) -> Dict[str, Any]:
        """
        Return audio quality metrics for this VAD session.
        """
        return {
            "frame_ms": self.frame_ms,
            "aggressiveness": self.aggressiveness,
            "energy_threshold": self.energy_threshold,
        }
