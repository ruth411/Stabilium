from __future__ import annotations

import numpy as np

from agent_stability_engine.engine.embeddings import EmbeddingProvider, build_embedder


class CorrectnessScorer:
    """Measures how correct a model output is relative to an expected answer.

    Uses embedding cosine similarity between the model output and the expected
    answer. Returns an incorrectness score in [0, 1] where 0 = fully correct
    and 1 = completely wrong.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
        embedding_model: str | None = None,
        embedding_openai_api_key: str | None = None,
    ) -> None:
        self._embedder, _ = build_embedder(
            embedding_provider,
            model_name=embedding_model,
            openai_api_key=embedding_openai_api_key,
        )

    def score(self, output: str, expected: str) -> float:
        """Return incorrectness in [0, 1]. 0 = matches expected, 1 = completely wrong."""
        matrix = self._embedder.encode([output, expected])
        a, b = matrix[0], matrix[1]
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        cosine_similarity = float(np.dot(a, b) / (norm_a * norm_b))
        # Clip to [0, 1] — negative cosine means completely unrelated
        cosine_similarity = max(0.0, min(1.0, cosine_similarity))
        return 1.0 - cosine_similarity
