"""
Shared Vision Data Models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Face:
    """
    Represents one detected face throughout the entire vision pipeline.
    """

    # Bounding box
    x: int
    y: int
    width: int
    height: int

    # InsightFace embedding
    embedding: np.ndarray

    # Detection confidence
    confidence: float

    # Recognition
    name: str = "Unknown"
    recognized: bool = False

    # Tracking
    visitor_id: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        """
        Center of the face bounding box.
        """

        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )