"""
Face Database.

Loads known face images, generates embeddings, and caches them for
faster startup.

Cache entries are validated before use. A corrupt cache, an empty cache,
or one containing malformed vectors falls back to regenerating from the
face folders instead of disabling recognition.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.vision.embedding_model import EmbeddingModel
from app.vision.face_cache import FaceCache
from app.vision.similarity import normalize_embedding


# Repository root, independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sanitize_embeddings(data):
    """
    Validate a cached ``{name: [embedding, ...]}`` mapping.

    Returns ``(cleaned_mapping, rejected_count)``. Malformed identities,
    non-iterable entry lists and invalid vectors are dropped. Vectors are
    not normalised here; the recogniser normalises both sides.
    """

    cleaned: dict[str, list[np.ndarray]] = {}

    rejected = 0

    if not isinstance(data, dict):

        return cleaned, 1

    for name, embeddings in data.items():

        if not isinstance(name, str) or not name.strip():

            rejected += 1

            continue

        if isinstance(embeddings, np.ndarray):

            embeddings = [embeddings]

        try:

            candidates = list(embeddings)

        except TypeError:

            rejected += 1

            continue

        valid: list[np.ndarray] = []

        for embedding in candidates:

            try:

                valid.append(
                    normalize_embedding(
                        embedding,
                        normalize=False,
                    )
                )

            except ValueError:

                rejected += 1

        if valid:

            cleaned[name] = valid

    return cleaned, rejected


class FaceDatabase:
    """
    Stores embeddings for known identities.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        database_path: str = "models/faces",
        cache: FaceCache | None = None,
    ) -> None:

        self.embedding_model = embedding_model

        database_path = Path(database_path)

        if not database_path.is_absolute():

            database_path = PROJECT_ROOT / database_path

        self.database_path = database_path

        self.cache = cache if cache is not None else FaceCache()

        self._database: dict[str, list[np.ndarray]] = {}

    def load(self) -> None:
        """
        Load the database, preferring a valid cache.
        """

        if self._load_cache():

            return

        # -----------------------------
        # Generate embeddings from images
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

                    print(f"  Could not read {image_path.name}")

                    continue

                embedding = self.embedding_model.generate(image)

                if embedding is None:

                    print(f"  No usable face found in {image_path.name}")

                    continue

                try:

                    embedding = normalize_embedding(
                        embedding,
                        normalize=False,
                    )

                except ValueError:

                    print(f"  Invalid embedding in {image_path.name}")

                    continue

                embeddings.append(embedding)

                print(f"  Loaded {image_path.name}")

            if embeddings:

                self._database[person_name] = embeddings

        if self._database:

            self.cache.save(self._database)

            print("[FaceDatabase] Cache saved.")

        print(
            f"[FaceDatabase] Loaded {self.person_count()} people."
        )

        print(
            f"[FaceDatabase] Total embeddings: {self.embedding_count()}"
        )

    def _load_cache(self) -> bool:
        """Load and sanitise the cache. Returns True when usable."""

        if not self.cache.exists():

            return False

        print("[FaceDatabase] Loading cache...")

        try:

            cached = self.cache.load()

        except Exception as error:

            print(
                "[FaceDatabase] Cache could not be read (%s); "
                "rebuilding from face images." % error
            )

            return False

        cleaned, rejected = sanitize_embeddings(cached)

        if not cleaned:

            print(
                "[FaceDatabase] Cache contains no usable embeddings; "
                "rebuilding from face images."
            )

            return False

        self._database = cleaned

        if rejected:

            print(
                "[FaceDatabase] Ignored %d invalid cached embedding(s)."
                % rejected
            )

            # The cache is derived data; keep it clean for the next run.
            self.cache.save(cleaned)

        print("[FaceDatabase] Cache loaded successfully.")

        print(
            f"[FaceDatabase] Loaded {self.person_count()} people."
        )

        print(
            f"[FaceDatabase] Total embeddings: {self.embedding_count()}"
        )

        return True

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
