"""
Face Recognizer.

Matches face embeddings against the known face database.
"""

from __future__ import annotations

from app.vision.face_database import FaceDatabase
from app.vision.models import Face
from app.vision.similarity import cosine_similarity


class FaceRecognizer:
    """
    Recognizes detected faces using cosine similarity.
    """

    def __init__(
        self,
        database: FaceDatabase,
        threshold: float = 0.55,
    ) -> None:

        self._database = database
        self._threshold = threshold

    def recognize(self, face: Face) -> Face:
        """
        Recognize a detected face.

        Args:
            face: Face object containing an embedding.

        Returns:
            The same Face object with recognition fields updated.
        """

        best_name = "Unknown"
        best_score = -1.0

        database = self._database.get_database()

        for person_name, embeddings in database.items():

            for known_embedding in embeddings:

                score = cosine_similarity(
                    face.embedding,
                    known_embedding,
                )

                if score > best_score:
                    best_score = score
                    best_name = person_name

        if best_score >= self._threshold:

            face.name = best_name
            face.recognized = True

        else:

            face.name = "Unknown"
            face.recognized = False

        face.confidence = best_score

        return face

    def recognize_all(
        self,
        faces: list[Face],
    ) -> list[Face]:
        """
        Recognize multiple faces.

        Args:
            faces: List of detected faces.

        Returns:
            Updated list with names and recognition status.
        """

        return [
            self.recognize(face)
            for face in faces
        ]