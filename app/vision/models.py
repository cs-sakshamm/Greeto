"""
Shared Vision Data Models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Recognition statuses. Kept as strings so they are readable in logs and
# event payloads without importing an extra enum.
NOT_PROCESSED = "not_processed"
RECOGNIZED = "recognized"
UNKNOWN = "unknown"
LOW_QUALITY = "low_quality"
INVALID_EMBEDDING = "invalid_embedding"
NO_KNOWN_FACES = "no_known_faces"


@dataclass(frozen=True)
class RecognitionResult:
    """
    Structured outcome of one recognition attempt.

    ``score`` is the best cosine similarity found, or ``None`` when the
    face was not compared at all (quality gate or invalid embedding).
    ``candidate`` is the closest known identity even when it fell below
    the threshold, which makes threshold tuning possible.
    """

    status: str = NOT_PROCESSED

    name: str = "Unknown"

    recognized: bool = False

    score: float | None = None

    threshold: float = 0.55

    candidate: str | None = None

    reason: str = ""


@dataclass
class Face:
    """
    Represents one detected face throughout the entire vision pipeline.
    """

    # Bounding box
    x: int
    y: int
    width: int
    height: int

    # InsightFace embedding. Normalised at ingestion when possible.
    embedding: np.ndarray

    # Detection confidence reported by InsightFace. Recognition never
    # overwrites this value; the similarity lives in ``recognition``.
    confidence: float

    # Recognition
    name: str = "Unknown"
    recognized: bool = False
    recognition: RecognitionResult | None = None

    # Tracking
    visitor_id: int | None = None

    @property
    def recognition_score(self) -> float | None:
        """Best similarity score from the last recognition attempt."""

        if self.recognition is None:
            return None

        return self.recognition.score

    @property
    def recognition_status(self) -> str:
        """Status string from the last recognition attempt."""

        if self.recognition is None:
            return NOT_PROCESSED

        return self.recognition.status

    @property
    def center(self) -> tuple[int, int]:
        """
        Center of the face bounding box.
        """

        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )
