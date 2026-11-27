"""
Face Embedding Cache.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


# Repository root, independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FaceCache:
    """
    Saves and loads face embeddings.
    """

    def __init__(
        self,
        cache_file: str = "models/cache/face_embeddings.pkl",
    ) -> None:

        cache_file = Path(cache_file)

        if not cache_file.is_absolute():

            cache_file = PROJECT_ROOT / cache_file

        self.cache_file = cache_file

        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def exists(self) -> bool:
        return self.cache_file.exists()

    def save(
        self,
        database: dict[str, list[np.ndarray]],
    ) -> None:

        with open(self.cache_file, "wb") as file:
            pickle.dump(database, file)

    def load(
        self,
    ) -> dict[str, list[np.ndarray]]:

        with open(self.cache_file, "rb") as file:
            return pickle.load(file)

    def delete(self) -> None:

        if self.cache_file.exists():
            self.cache_file.unlink()