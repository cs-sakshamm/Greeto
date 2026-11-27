"""
Embedding Model.

Uses InsightFace for:
1. Face Detection
2. Face Embedding Generation

Embeddings are validated and L2-normalised at ingestion so every
consumer (recognition, enrolment and the cache) sees the same
well-defined vector. Non-finite or zero-length embeddings are dropped
instead of poisoning the database.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.vision.models import Face
from app.vision.similarity import normalize_embedding


class EmbeddingModel:
    """
    Wrapper around InsightFace FaceAnalysis.

    Responsibilities:
    - Detect faces
    - Generate valid, normalised embeddings
    """

    def __init__(self) -> None:
        self._model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )

        self._model.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

    def detect(self, image: np.ndarray) -> List[Face]:
        """
        Detect all faces in an image.

        Args:
            image: BGR image from OpenCV.

        Returns:
            List of Face objects with validated unit-norm embeddings.
        """

        insight_faces = self._model.get(image)

        detected_faces: List[Face] = []

        for detected in insight_faces:

            x1, y1, x2, y2 = map(int, detected.bbox)

            try:

                embedding = normalize_embedding(detected.embedding)

            except ValueError:

                # A malformed embedding cannot be recognised or stored.
                # Dropping the face is safer than propagating an array
                # that would produce nan scores downstream.
                continue

            detected_faces.append(
                Face(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    embedding=embedding,
                    confidence=float(detected.det_score),
                )
            )

        return detected_faces

    def generate(self, image: np.ndarray) -> np.ndarray | None:
        """
        Generate a normalised embedding for the first usable face.

        Used by FaceDatabase while building the embeddings cache.

        Args:
            image: BGR image.

        Returns:
            Unit-norm embedding or None.
        """

        faces = self.detect(image)

        if not faces:
            return None

        return faces[0].embedding

    @staticmethod
    def crop_face(
        image: np.ndarray,
        face: Face,
        padding: int = 10,
    ) -> np.ndarray:
        """
        Crop a detected face from an image.

        Useful for debugging or future emotion detection.

        Args:
            image: Original BGR image.
            face: Face object.
            padding: Extra pixels around the face.

        Returns:
            Cropped face image.
        """

        h, w = image.shape[:2]

        x1 = max(face.x - padding, 0)
        y1 = max(face.y - padding, 0)

        x2 = min(face.x + face.width + padding, w)
        y2 = min(face.y + face.height + padding, h)

        return image[y1:y2, x1:x2]

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: List[Face],
    ) -> np.ndarray:
        """
        Draw bounding boxes for debugging.

        Args:
            image: Input image.
            faces: Detected faces.

        Returns:
            Annotated image.
        """

        frame = image.copy()

        for face in faces:

            cv2.rectangle(
                frame,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{face.confidence:.2f}",
                (face.x, face.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame
