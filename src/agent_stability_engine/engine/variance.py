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
        centroid = np.mean(matrix, axis=0)
        distances = np.sum((matrix - centroid) ** 2, axis=1)
        raw_variance = float(np.mean(distances))
        normalized = min(raw_variance / self._expected_max_variance, 1.0)
        return VarianceResult(raw_variance=raw_variance, normalized_variance=normalized)

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider
