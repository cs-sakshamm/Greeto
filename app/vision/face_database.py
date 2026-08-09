"""
Face Database.

Loads known face images, generates embeddings,
and caches them for faster startup.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.vision.embedding_model import EmbeddingModel
from app.vision.face_cache import FaceCache


class FaceDatabase:
    """
    Stores embeddings for known identities.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        database_path: str = "models/faces",
    ) -> None:

        self.embedding_model = embedding_model

        self.database_path = Path(database_path)

        self.cache = FaceCache()

        self._database: dict[str, list[np.ndarray]] = {}

    def load(self) -> None:
        """
        Load the database.
        """

        # -----------------------------
        # Load from cache if available
        # -----------------------------
        if self.cache.exists():

            print("[FaceDatabase] Loading cache...")

            self._database = self.cache.load()

            print("[FaceDatabase] Cache loaded successfully.")

            return

        # -----------------------------
        # Otherwise generate embeddings
        # -----------------------------
        print("[FaceDatabase] Generating embeddings...")

        self._database.clear()

        if not self.database_path.exists():
            print("[FaceDatabase] Face folder not found.")
            return

        for person_folder in sorted(self.database_path.iterdir()):

            if not person_folder.is_dir():
                continue

            person_name = person_folder.name

            embeddings: list[np.ndarray] = []

            print(f"Loading {person_name}...")

            image_files = sorted(
                list(person_folder.glob("*.jpg"))
                + list(person_folder.glob("*.jpeg"))
                + list(person_folder.glob("*.png"))
            )

            for image_path in image_files:

                image = cv2.imread(str(image_path))

                if image is None:
                    continue

                embedding = self.embedding_model.generate(image)

                if embedding is None:
                    print(f"  No face found in {image_path.name}")
                    continue

                embeddings.append(embedding)

                print(f"  Loaded {image_path.name}")

            if embeddings:
                self._database[person_name] = embeddings

        # -----------------------------
        # Save cache
        # -----------------------------
        self.cache.save(self._database)

        print("[FaceDatabase] Cache saved.")

        print(
            f"[FaceDatabase] Loaded {self.person_count()} people."
        )

        print(
            f"[FaceDatabase] Total embeddings: {self.embedding_count()}"
        )

    def get_database(self) -> dict[str, list[np.ndarray]]:
        return self._database

    def get_embeddings(
        self,
        person_name: str,
    ) -> list[np.ndarray]:

        return self._database.get(person_name, [])

    def has_person(
        self,
        person_name: str,
    ) -> bool:

        return person_name in self._database

    def person_count(self) -> int:

        return len(self._database)

    def embedding_count(self) -> int:

        total = 0

        for embeddings in self._database.values():
            total += len(embeddings)

        return total

    def list_people(self) -> list[str]:

        return sorted(self._database.keys())