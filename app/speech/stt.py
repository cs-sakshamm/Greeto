"""Speech To Text.

Offline Whisper transcription using pywhispercpp with college domain adaptation,
hallucination suppression and text validation.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COLLEGE_PROMPT = (
    "Hello, hi, greetings, I.T.S Engineering College, Greater Noida, "
    "APJ Abdul Kalam Technical University, AKTU, B.Tech, CSE, Computer Science, "
    "ECE, Electronics, ME, Mechanical, Civil, MBA, MCA, admission, fees, "
    "eligibility, hostel, placement, registrar, scholarship, campus, Greeto."
)

# Hallucination/noise tokens that should be filtered from STT output.
# Note: "you", "thank you", "bye" are valid user utterances and are NOT
# included here - they are handled by normal text validation.
KNOWN_HALLUCINATIONS = {
    "[blank_audio]",
    "blank_audio",
    "[silence]",
    "silence",
    "[applause]",
    "(applause)",
    "[cheering]",
    "(cheering)",
    "[laughter]",
    "(laughter)",
    "[music]",
    "(music)",
    "thank you.",
    "thank you",
    "thanks for watching!",
    "thanks for watching.",
    "please subscribe.",
    "subscribe!",
}

KNOWN_SHORT_PHRASES = {
    # These are valid utterances and should NOT be filtered as hallucinations
    "you",
    "hello",
    "hi",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
}


class SpeechToText:
    """Offline Whisper transcription manager."""

    def __init__(
        self,
        model_path: str = "models/whisper/ggml-tiny.en.bin",
        n_threads: int = 4,
        language: Optional[str] = "en",
        initial_prompt: Optional[str] = DEFAULT_COLLEGE_PROMPT,
        min_text_length: int = 3,
        max_text_length: int = 500,
    ) -> None:
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = PROJECT_ROOT / model_path_obj
        self.model_path = model_path_obj
        self.n_threads = max(1, int(n_threads))
        self.language = language or "en"
        self.initial_prompt = initial_prompt
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        self._lock = threading.Lock()
        self.model = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")

        try:
            from pywhispercpp.model import Model
        except ImportError as error:
            raise RuntimeError(f"pywhispercpp unavailable: {error}") from error

        try:
            params = dict(
                n_threads=self.n_threads,
                language=self.language if self.language != "auto" else "",
                redirect_whispercpp_logs_to=None,
                suppress_blank=True,
                temperature=0.0,
            )
            if self.initial_prompt:
                params["initial_prompt"] = self.initial_prompt

            self.model = Model(str(self.model_path), **params)
            logger.info(
                "Whisper model loaded from %s (threads=%d, lang=%s).",
                self.model_path.name,
                self.n_threads,
                self.language,
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize Whisper model: %s", error)
            raise

    def is_available(self) -> bool:
        """Return True if model is loaded and ready."""
        return self.model is not None

    def clean_text(self, raw_text: str) -> str:
        """Normalize transcribed text and filter out hallucinations or noise tokens."""
        if not raw_text:
            return ""

        text = raw_text.strip()

        # Remove bracketed/parenthetical tokens like [BLANK_AUDIO] or (music)
        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
        lower_cleaned = cleaned.lower()

        # Check known hallucinations
        if text.lower() in KNOWN_HALLUCINATIONS or lower_cleaned in KNOWN_HALLUCINATIONS:
            return ""

        # Do not filter valid short phrases like "you", "thank you", "bye"
        # These are legitimate user utterances. Only filter truly irrelevant tokens.

        # Remove duplicate whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Check for repetitive loops (e.g. "you you you you")
        words = cleaned.split()
        if len(words) >= 4 and len(set(words)) == 1:
            return ""

        # Remove trailing and leading punctuation-only strings
        if not re.search(r"\w", cleaned):
            return ""

        # Length validation
        if len(cleaned) < self.min_text_length:
            return ""

        if len(cleaned) > self.max_text_length:
            cleaned = cleaned[: self.max_text_length].strip()

        return cleaned

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio array to text.

        Parameters:
            audio: 1D float32 array at 16000Hz.

        Returns:
            Validated transcription string or empty string.
        """
        if audio is None or len(audio) == 0:
            return ""

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return ""

        if self.model is None:
            raise RuntimeError("Whisper model is not initialized.")

        with self._lock:
            try:
                segments = self.model.transcribe(arr)
            except Exception as error:  # noqa: BLE001
                logger.exception("Whisper inference failed: %s", error)
                raise

        text_pieces = []
        for segment in segments or []:
            t = getattr(segment, "text", None)
            if t is None:
                t = str(segment)
            if t:
                text_pieces.append(t)

        raw = " ".join(text_pieces).strip()
        return self.clean_text(raw)