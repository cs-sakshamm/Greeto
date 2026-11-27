"""
Face Recognizer.

Compares validated, normalised embeddings against the known face
database. Every attempt produces a structured ``RecognitionResult`` so
callers can distinguish a genuine unknown face from a low-quality
detection, an invalid embedding, or an empty database.
"""

from __future__ import annotations

import math

from app.vision.models import (
    INVALID_EMBEDDING,
    LOW_QUALITY,
    NO_KNOWN_FACES,
    RECOGNIZED,
    UNKNOWN,
    Face,
    RecognitionResult,
)
from app.vision.recognition import RecognitionConfig
from app.vision.similarity import cosine_similarity, normalize_embedding


class FaceRecognizer:
    """
    Recognizes detected faces using cosine similarity.
    """

    def __init__(
        self,
        database,
        threshold: float | None = None,
        config: RecognitionConfig | None = None,
    ) -> None:

        if config is None:

            config = RecognitionConfig()

        if not isinstance(config, RecognitionConfig):

            raise TypeError("config must be a RecognitionConfig")

        if threshold is not None:

            config = config.with_threshold(threshold)

        self._database = database

        self._config = config

        # Normalised known vectors, built once from the database.
        self._known: tuple[tuple[str, object], ...] = ()

        self._known_ready = False

        self._invalid_known = 0

        self._last_result: RecognitionResult | None = None

    # -----------------------------------------------------
    # Configuration and diagnostics
    # -----------------------------------------------------

    @property
    def config(self) -> RecognitionConfig:
        """Effective configuration."""

        return self._config

    @property
    def threshold(self) -> float:
        """Effective similarity threshold."""

        return self._config.threshold

    def refresh(self) -> tuple[int, int, int]:
        """
        Rebuild the normalised known-vector cache.

        Returns ``(identities, embeddings, rejected)``. Call this after
        the face database has been loaded or changed.
        """

        database = self._database.get_database()

        if not isinstance(database, dict):

            raise ValueError("face database did not return a mapping")

        known: list[tuple[str, object]] = []

        rejected = 0

        for name, embeddings in database.items():

            if not isinstance(name, str) or not name.strip():

                rejected += 1

                continue

            try:

                candidates = list(embeddings)

            except TypeError:

                rejected += 1

                continue

            for embedding in candidates:

                try:

                    vector = normalize_embedding(
                        embedding,
                        expected_size=self._config.expected_embedding_size,
                        min_norm=self._config.min_embedding_norm,
                        normalize=self._config.normalize_embeddings,
                    )

                except ValueError:

                    rejected += 1

                    continue

                known.append((name, vector))

        self._known = tuple(known)

        self._invalid_known = rejected

        self._known_ready = True

        return (
            len({name for name, _ in self._known}),
            len(self._known),
            rejected,
        )

    def stats(self) -> dict:
        """Diagnostics for the kiosk and logs."""

        if not self._known_ready:

            try:

                self.refresh()

            except Exception:

                pass

        last = self._last_result

        return {
            "threshold": self._config.threshold,
            "known_identities": len({name for name, _ in self._known}),
            "known_embeddings": len(self._known),
            "rejected_embeddings": self._invalid_known,
            "last_status": last.status if last is not None else None,
            "last_score": last.score if last is not None else None,
            "last_candidate": last.candidate if last is not None else None,
        }

    # -----------------------------------------------------
    # Recognition
    # -----------------------------------------------------

    def recognize(self, face: Face) -> Face:
        """
        Recognize a detected face and store the structured result on it.

        ``face.confidence`` (the detector score) is never overwritten.
        """

        result = self.recognize_result(face)

        face.recognition = result

        face.name = result.name

        face.recognized = result.recognized

        return face

    def recognize_result(self, face: Face) -> RecognitionResult:
        """Return the structured outcome without mutating the face."""

        reason = self._quality_issue(face)

        if reason:

            return self._remember(
                RecognitionResult(
                    status=LOW_QUALITY,
                    name="Unknown",
                    recognized=False,
                    score=None,
                    threshold=self._config.threshold,
                    reason=reason,
                )
            )

        try:

            query = normalize_embedding(
                face.embedding,
                expected_size=self._config.expected_embedding_size,
                min_norm=self._config.min_embedding_norm,
                normalize=self._config.normalize_embeddings,
            )

        except ValueError as error:

            return self._remember(
                RecognitionResult(
                    status=INVALID_EMBEDDING,
                    name="Unknown",
                    recognized=False,
                    score=None,
                    threshold=self._config.threshold,
                    reason=str(error),
                )
            )

        try:

            self._ensure_known()

        except Exception as error:

            return self._remember(
                RecognitionResult(
                    status=NO_KNOWN_FACES,
                    name="Unknown",
                    recognized=False,
                    score=None,
                    threshold=self._config.threshold,
                    reason="face database unavailable: %s" % error,
                )
            )

        best_name: str | None = None

        best_score = -1.0

        for name, known in self._known:

            try:

                score = cosine_similarity(query, known)

            except ValueError:

                continue

            if score > best_score:

                best_score = score

                best_name = name

        if best_name is None:

            return self._remember(
                RecognitionResult(
                    status=NO_KNOWN_FACES,
                    name="Unknown",
                    recognized=False,
                    score=None,
                    threshold=self._config.threshold,
                    reason="no comparable known embeddings",
                )
            )

        if best_score >= self._config.threshold:

            return self._remember(
                RecognitionResult(
                    status=RECOGNIZED,
                    name=best_name,
                    recognized=True,
                    score=best_score,
                    threshold=self._config.threshold,
                    candidate=best_name,
                )
            )

        return self._remember(
            RecognitionResult(
                status=UNKNOWN,
                name="Unknown",
                recognized=False,
                score=best_score,
                threshold=self._config.threshold,
                candidate=best_name,
                reason="best score %.3f is below threshold %.3f"
                % (best_score, self._config.threshold),
            )
        )

    def recognize_all(
        self,
        faces: list[Face],
    ) -> list[Face]:
        """
        Recognize multiple faces.

        One bad face never aborts the batch.
        """

        return [
            self.recognize(face)
            for face in faces
        ]

    # -----------------------------------------------------
    # Internals
    # -----------------------------------------------------

    def _quality_issue(self, face: Face) -> str | None:
        """Return a human-readable quality rejection, or None."""

        try:

            width = int(face.width)

            height = int(face.height)

            detection = float(face.confidence)

        except (TypeError, ValueError, AttributeError):

            return "face has no usable bounding box or detection score"

        if width <= 0 or height <= 0:

            return "invalid bounding box %dx%d" % (width, height)

        if not math.isfinite(detection):

            return "detection score is not finite"

        if detection < self._config.min_detection_score:

            return "detection score %.3f is below %.3f" % (
                detection,
                self._config.min_detection_score,
            )

        if min(width, height) < self._config.min_face_size:

            return "face is %dx%d, minimum side is %d" % (
                width,
                height,
                self._config.min_face_size,
            )

        return None

    def _ensure_known(self) -> None:

        if not self._known_ready:

            self.refresh()

    def _remember(self, result: RecognitionResult) -> RecognitionResult:

        self._last_result = result

        return result
