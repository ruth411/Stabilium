from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from enum import Enum
from typing import Any, cast
from urllib import error, request

import numpy as np
from numpy.typing import NDArray

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


class EmbeddingProvider(str, Enum):
    AUTO = "auto"
    HASH = "hash"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class TextEmbedder:
    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        raise NotImplementedError


class HashTextEmbedder(TextEmbedder):
    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            msg = "dimension must be >= 1"
            raise ValueError(msg)
        self._dimension = dimension

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        return np.array([self._embed(text) for text in texts], dtype=np.float64)

    def _embed(self, text: str) -> NDArray[np.float64]:
        vector: NDArray[np.float64] = np.zeros(self._dimension, dtype=np.float64)
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / math.sqrt(float(norm))


class SentenceTransformerEmbedder(TextEmbedder):
    _MODEL_CACHE: dict[str, Any] = {}

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        model = self._load_model(self._model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    @classmethod
    def _load_model(cls, model_name: str) -> Any:
        cached = cls._MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            msg = (
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
            raise RuntimeError(msg) from exc

        sentence_transformer_cls = getattr(module, "SentenceTransformer", None)
        if sentence_transformer_cls is None:
            msg = "SentenceTransformer class not found in sentence_transformers package"
            raise RuntimeError(msg)

        model = sentence_transformer_cls(model_name)
        cls._MODEL_CACHE[model_name] = model
        return model


class OpenAIEmbeddingEmbedder(TextEmbedder):
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            msg = "OPENAI_API_KEY is required for OpenAI embedding backend"
            raise ValueError(msg)
        self._timeout_seconds = timeout_seconds

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        payload = {
            "model": self._model,
            "input": texts,
        }
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            _OPENAI_EMBEDDINGS_URL,
            method="POST",
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                loaded = json.loads(raw)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"OpenAI embeddings HTTP error {exc.code}: {body}"
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"OpenAI embeddings request error: {exc.reason}"
            raise RuntimeError(msg) from exc

        if not isinstance(loaded, dict):
            msg = "OpenAI embeddings response must be a JSON object"
            raise RuntimeError(msg)

        data_array = loaded.get("data")
        if not isinstance(data_array, list):
            msg = "OpenAI embeddings response missing data list"
            raise RuntimeError(msg)

        vectors: list[list[float]] = []
        for item in data_array:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                continue
            vec: list[float] = []
            for element in embedding:
                if isinstance(element, (int, float)):
                    vec.append(float(element))
            if vec:
                vectors.append(vec)

        if len(vectors) != len(texts):
            msg = "OpenAI embeddings response size mismatch"
            raise RuntimeError(msg)

        matrix = np.asarray(vectors, dtype=np.float64)
        return _row_normalize(matrix)


def build_embedder(
    provider: EmbeddingProvider,
    *,
    model_name: str | None = None,
    openai_api_key: str | None = None,
) -> tuple[TextEmbedder, EmbeddingProvider]:
    if provider is EmbeddingProvider.HASH:
        return HashTextEmbedder(), EmbeddingProvider.HASH

    if provider is EmbeddingProvider.SENTENCE_TRANSFORMERS:
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return (
            SentenceTransformerEmbedder(model_name=model),
            EmbeddingProvider.SENTENCE_TRANSFORMERS,
        )

    if provider is EmbeddingProvider.OPENAI:
        model = model_name or "text-embedding-3-small"
        return (
            OpenAIEmbeddingEmbedder(model=model, api_key=openai_api_key),
            EmbeddingProvider.OPENAI,
        )

    if provider is EmbeddingProvider.AUTO:
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        try:
            embedder = SentenceTransformerEmbedder(model_name=model)
            # quick smoke call to ensure dependency is available
            embedder.encode(["probe"])
            return embedder, EmbeddingProvider.SENTENCE_TRANSFORMERS
        except RuntimeError:
            return HashTextEmbedder(), EmbeddingProvider.HASH

    msg = f"unsupported embedding provider: {provider.value}"
    raise ValueError(msg)


def _row_normalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return cast(NDArray[np.float64], matrix / safe_norms)
