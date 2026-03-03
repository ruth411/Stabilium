from __future__ import annotations

import hashlib
import itertools
import re

import numpy as np
from numpy.typing import NDArray

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class CrossModelDisagreement:
    """Computes pairwise semantic divergence across model outputs."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            msg = "dimension must be >= 1"
            raise ValueError(msg)
        self._dimension = dimension

    def score(self, model_outputs: dict[str, str]) -> float:
        if len(model_outputs) < 2:
            msg = "at least two model outputs are required"
            raise ValueError(msg)

        embeddings = {name: self._embed(text) for name, text in model_outputs.items()}
        distances = [
            self._cosine_distance(embeddings[a], embeddings[b])
            for a, b in itertools.combinations(model_outputs.keys(), 2)
        ]
        return float(np.mean(distances))

    def _embed(self, text: str) -> NDArray[np.float64]:
        vector: NDArray[np.float64] = np.zeros(self._dimension, dtype=np.float64)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def _cosine_distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        # Cosine similarity in [-1, 1], convert to bounded [0, 1] divergence.
        sim = float(np.dot(a, b))
        divergence = (1.0 - sim) / 2.0
        return min(max(divergence, 0.0), 1.0)
