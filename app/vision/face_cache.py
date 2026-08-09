"""
Face Embedding Cache.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class FaceCache:
    """
    Saves and loads face embeddings.
    """

    def __init__(
        self,
        cache_file: str = "models/cache/face_embeddings.pkl",
    ) -> None:

        self.cache_file = Path(cache_file)

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