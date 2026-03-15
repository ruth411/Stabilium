from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_stability_engine.engine.embeddings import EmbeddingProvider, TextEmbedder, build_embedder


@dataclass(frozen=True)
class VarianceResult:
    raw_variance: float
    normalized_variance: float


class EmbeddingVarianceScorer:
    """Computes text-embedding variance across repeated outputs."""

    def __init__(
        self,
        expected_max_variance: float = 0.5,
        embedder: TextEmbedder | None = None,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
        embedding_model: str | None = None,
        embedding_openai_api_key: str | None = None,
    ) -> None:
        if expected_max_variance <= 0:
            msg = "expected_max_variance must be > 0"
            raise ValueError(msg)
        if embedder is None:
            resolved_embedder, resolved_provider = build_embedder(
                embedding_provider,
                model_name=embedding_model,
                openai_api_key=embedding_openai_api_key,
            )
            self._embedder = resolved_embedder
            self._embedding_provider = resolved_provider
        else:
            self._embedder = embedder
            self._embedding_provider = embedding_provider
        self._expected_max_variance = expected_max_variance

    def score(self, texts: list[str]) -> VarianceResult:
        if len(texts) < 2:
            msg = "at least two texts are required"
            raise ValueError(msg)

        matrix = self._embedder.encode(texts)
        if matrix.ndim != 2:
            msg = "embedder output must be rank-2 [num_texts, embedding_dim]"
            raise ValueError(msg)
        if matrix.shape[0] != len(texts):
            msg = "embedder output row count must match number of texts"
            raise ValueError(msg)
        # Mean pairwise squared distance — correctly captures bimodal distributions
        # e.g. [A, A, A, B, B, B] scores HIGH variance, not LOW like centroid-distance would
        n = matrix.shape[0]
        raw_variance = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = matrix[i] - matrix[j]
                raw_variance += float(np.dot(diff, diff))
                count += 1
        raw_variance = raw_variance / count if count > 0 else 0.0
        normalized = min(raw_variance / self._expected_max_variance, 1.0)
        return VarianceResult(raw_variance=raw_variance, normalized_variance=normalized)

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider
