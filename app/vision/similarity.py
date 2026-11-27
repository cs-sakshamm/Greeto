"""
Similarity and embedding utilities.

All functions are strict: invalid vectors raise ``ValueError`` instead of
silently producing ``nan`` or a scalar that later crashes the caller.
"""

from __future__ import annotations

import numpy as np


def normalize_embedding(
    embedding,
    expected_size: int | None = None,
    min_norm: float = 1e-6,
    normalize: bool = True,
) -> np.ndarray:
    """
    Validate an embedding and return a 1-D float32 vector.

    Args:
        embedding: Array-like vector.
        expected_size: Required length, or ``None`` to accept any length.
        min_norm: Reject vectors whose L2 norm is at or below this value.
        normalize: Divide by the L2 norm when True.

    Raises:
        ValueError: If the input is empty, multi-dimensional, contains
            non-finite values, has the wrong length or is effectively
            zero-length.
    """

    try:

        vector = np.asarray(embedding, dtype=np.float32)

    except (TypeError, ValueError) as error:

        raise ValueError("embedding is not numeric") from error

    if vector.ndim != 1:

        raise ValueError(
            "embedding must be one-dimensional, got shape %s" % (vector.shape,)
        )

    if vector.size == 0:

        raise ValueError("embedding is empty")

    if expected_size is not None and vector.size != int(expected_size):

        raise ValueError(
            "embedding has %d values, expected %d"
            % (vector.size, int(expected_size))
        )

    if not np.all(np.isfinite(vector)):

        raise ValueError("embedding contains non-finite values")

    norm = float(np.linalg.norm(vector))

    if norm <= float(min_norm):

        raise ValueError(
            "embedding norm %.6g is at or below the minimum %.6g"
            % (norm, float(min_norm))
        )

    if normalize:

        vector = (vector / norm).astype(np.float32, copy=False)

    return vector


def cosine_similarity(
    vector1,
    vector2,
) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]. Both vectors must be non-empty, finite,
    one-dimensional and of the same length, otherwise ``ValueError`` is
    raised. Normalisation is not required; the denominator does it.
    """

    first = np.asarray(vector1, dtype=np.float64)

    second = np.asarray(vector2, dtype=np.float64)

    if first.ndim != 1 or second.ndim != 1:

        raise ValueError("cosine similarity expects one-dimensional vectors")

    if first.size == 0 or second.size == 0:

        raise ValueError("cosine similarity expects non-empty vectors")

    if first.size != second.size:

        raise ValueError(
            "vector sizes differ: %d and %d" % (first.size, second.size)
        )

    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):

        raise ValueError("cosine similarity received non-finite values")

    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))

    if denominator <= 0.0:

        raise ValueError("cosine similarity received a zero-length vector")

    score = float(np.dot(first, second) / denominator)

    # Floating point error can push a perfect match slightly above 1.0.
    return max(-1.0, min(1.0, score))
