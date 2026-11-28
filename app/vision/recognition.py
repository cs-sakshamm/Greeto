"""
Recognition configuration and threshold calibration.

The recogniser itself lives in ``face_recognizer.py``. This module holds
the tunable knobs, their safe defaults, and the pure calibration routine
used by ``scripts/calibrate_threshold.py`` and the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


DEFAULT_THRESHOLD = 0.55

# Required gap between the best and the second-best enrolled identity.
# Zero disables the check (the previous behaviour); the kiosk config uses
# a small positive margin so a near-tie between two enrolled people is
# reported as unknown instead of greeting the wrong person.
DEFAULT_DECISION_MARGIN = 0.0

DEFAULT_MIN_DETECTION_SCORE = 0.55
DEFAULT_MIN_FACE_SIZE = 60
DEFAULT_EMBEDDING_SIZE = 512
DEFAULT_MIN_EMBEDDING_NORM = 0.5


@dataclass(frozen=True)
class RecognitionConfig:
    """Validated recognition settings."""

    threshold: float = DEFAULT_THRESHOLD

    decision_margin: float = DEFAULT_DECISION_MARGIN

    min_detection_score: float = DEFAULT_MIN_DETECTION_SCORE

    min_face_size: int = DEFAULT_MIN_FACE_SIZE

    expected_embedding_size: int = DEFAULT_EMBEDDING_SIZE

    min_embedding_norm: float = DEFAULT_MIN_EMBEDDING_NORM

    normalize_embeddings: bool = True

    # Human-readable notes about values that were rejected and replaced
    # by defaults. Not part of the configuration identity.
    warnings: tuple[str, ...] = field(default=(), compare=False)

    @classmethod
    def from_config(cls, config_manager=None) -> "RecognitionConfig":
        """
        Read ``vision.recognition`` from a ConfigManager-like object.

        Any missing, empty or invalid value falls back to the default
        and is reported in ``warnings``; a bad config file must never
        prevent recognition from running.
        """

        mapping = {}

        if config_manager is not None:

            try:

                mapping = config_manager.get_value(
                    "vision",
                    "recognition",
                    default={},
                )

            except Exception:

                mapping = {}

        return cls.from_mapping(mapping)

    @classmethod
    def from_mapping(cls, mapping) -> "RecognitionConfig":
        """Build a config from an arbitrary mapping."""

        warnings: list[str] = []

        if not isinstance(mapping, dict):

            if mapping not in (None, {}):

                warnings.append(
                    "vision.recognition is not a mapping; defaults used"
                )

            mapping = {}

        def number(key, default, low, high):

            raw = mapping.get(key, default)

            try:

                value = float(raw)

            except (TypeError, ValueError):

                warnings.append(
                    "%s=%r is not a number; using %.3g" % (key, raw, default)
                )

                return float(default)

            if not math.isfinite(value) or value < low or value > high:

                warnings.append(
                    "%s=%r is outside [%g, %g]; using %.3g"
                    % (key, raw, low, high, default)
                )

                return float(default)

            return value

        def integer(key, default, low):

            raw = mapping.get(key, default)

            try:

                value = int(raw)

            except (TypeError, ValueError):

                warnings.append(
                    "%s=%r is not an integer; using %d" % (key, raw, default)
                )

                return int(default)

            if value < low:

                warnings.append(
                    "%s=%r is below %d; using %d" % (key, raw, low, default)
                )

                return int(default)

            return value

        threshold = number("threshold", DEFAULT_THRESHOLD, -1.0, 1.0)

        decision_margin = number(
            "decision_margin",
            DEFAULT_DECISION_MARGIN,
            0.0,
            2.0,
        )

        min_detection = number(
            "min_detection_score",
            DEFAULT_MIN_DETECTION_SCORE,
            0.0,
            1.0,
        )

        min_face_size = integer("min_face_size", DEFAULT_MIN_FACE_SIZE, 1)

        embedding_size = integer(
            "expected_embedding_size",
            DEFAULT_EMBEDDING_SIZE,
            1,
        )

        min_norm = number(
            "min_embedding_norm",
            DEFAULT_MIN_EMBEDDING_NORM,
            1e-9,
            1e9,
        )

        normalize = mapping.get("normalize_embeddings", True)

        if isinstance(normalize, str):

            lowered = normalize.strip().lower()

            if lowered in ("true", "yes", "1", "on"):

                normalize = True

            elif lowered in ("false", "no", "0", "off"):

                normalize = False

        if isinstance(normalize, (int, float)) and not isinstance(normalize, bool):

            normalize = bool(normalize)

        if not isinstance(normalize, bool):

            warnings.append(
                "normalize_embeddings=%r is not a boolean; using True"
                % (normalize,)
            )

            normalize = True

        return cls(
            threshold=threshold,
            decision_margin=decision_margin,
            min_detection_score=min_detection,
            min_face_size=min_face_size,
            expected_embedding_size=embedding_size,
            min_embedding_norm=min_norm,
            normalize_embeddings=normalize,
            warnings=tuple(warnings),
        )

    def with_threshold(self, threshold: float) -> "RecognitionConfig":
        """Return a copy with an explicit threshold, keeping validation."""

        value = float(threshold)

        if not math.isfinite(value) or value < -1.0 or value > 1.0:

            raise ValueError(
                "threshold must be a finite number in [-1, 1], got %r"
                % (threshold,)
            )

        from dataclasses import replace

        return replace(self, threshold=value)


@dataclass(frozen=True)
class ThresholdCalibration:
    """Result of calibrating a cosine-similarity threshold."""

    threshold: float

    false_accept_rate: float

    false_reject_rate: float

    balanced_accuracy: float

    genuine_count: int

    impostor_count: int


def calibrate_threshold(
    genuine_scores,
    impostor_scores,
    low: float = 0.0,
    high: float = 1.0,
    steps: int = 1001,
) -> ThresholdCalibration | None:
    """
    Choose the threshold with the best balanced accuracy.

    ``genuine_scores`` are similarities between images of the same
    person; ``impostor_scores`` are similarities between different
    people. Both are required: without impostors there is no way to
    estimate false accepts, so ``None`` is returned.

    Ties are broken towards the widest margin between the highest
    impostor score and the lowest genuine score.
    """

    genuine = np.asarray(
        [float(value) for value in genuine_scores if math.isfinite(float(value))],
        dtype=np.float64,
    )

    impostor = np.asarray(
        [float(value) for value in impostor_scores if math.isfinite(float(value))],
        dtype=np.float64,
    )

    if genuine.size == 0 or impostor.size == 0:

        return None

    if not math.isfinite(low) or not math.isfinite(high) or low >= high:

        raise ValueError("calibration range must be finite and increasing")

    steps = max(2, int(steps))

    highest_impostor = float(np.max(impostor))

    lowest_genuine = float(np.min(genuine))

    best = None

    for threshold in np.linspace(low, high, steps):

        true_accept = float(np.count_nonzero(genuine >= threshold)) / genuine.size

        true_reject = float(np.count_nonzero(impostor < threshold)) / impostor.size

        balanced = 0.5 * (true_accept + true_reject)

        margin = min(
            lowest_genuine - float(threshold),
            float(threshold) - highest_impostor,
        )

        key = (balanced, margin)

        if best is None or key > best[0]:

            best = (key, float(threshold), true_accept, true_reject, balanced)

    _, threshold, true_accept, true_reject, balanced = best

    return ThresholdCalibration(
        threshold=threshold,
        false_accept_rate=1.0 - true_reject,
        false_reject_rate=1.0 - true_accept,
        balanced_accuracy=balanced,
        genuine_count=int(genuine.size),
        impostor_count=int(impostor.size),
    )
