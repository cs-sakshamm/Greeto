"""
Similarity utilities.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(
    vector1: np.ndarray,
    vector2: np.ndarray,
) -> float:
    """
    Compute cosine similarity.
    """

    numerator = np.dot(
        vector1,
        vector2,
    )

    denominator = (
        np.linalg.norm(vector1)
        * np.linalg.norm(vector2)
    )

    return float(
        numerator / denominator
    )