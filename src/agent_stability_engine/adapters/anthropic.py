from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable
from urllib import error, request

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

# Estimated USD per 1M tokens (input/output) for common models.
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}


@dataclass
class _UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    retries: int = 0
    estimated_cost_usd: float = 0.0


class AnthropicChatAdapter:
    """Callable adapter for Anthropic Messages API with retry and usage tracking."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        min_interval_seconds: float = 0.5,
        base_backoff_seconds: float = 0.5,
        rate_limit_backoff_seconds: float = 12.0,
        jitter_seconds: float = 0.5,
        sender: Callable[[dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        if not model:
            msg = "model must be non-empty"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        if temperature is not None and not (0.0 <= temperature <= 1.0):
            msg = "temperature must be between 0.0 and 1.0 for Anthropic models"
            raise ValueError(msg)
        if max_tokens < 1:
            msg = "max_tokens must be >= 1"
            raise ValueError(msg)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            msg = "ANTHROPIC_API_KEY is required for Anthropic adapter"
            raise ValueError(msg)

        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._min_interval_seconds = min_interval_seconds
        self._base_backoff_seconds = base_backoff_seconds
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._sender = sender or self._default_sender
        self._usage = _UsageTotals()
        self._last_request_monotonic = 0.0

    def __call__(self, prompt: str, rng: random.Random | None = None) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        attempts = self._max_retries + 1
        for attempt in range(attempts):
            self._respect_rate_limit()
            try:
                response = self._sender(payload)
                input_tokens, output_tokens = _extract_usage(response)
                self._track_usage(input_tokens, output_tokens)
                return _extract_text(response)
            except RuntimeError as exc:
                if attempt >= self._max_retries:
                    raise
                self._usage.retries += 1
                if "429" in str(exc):
                    self._sleep_rate_limit(attempt, rng)
                else:
                    self._sleep_backoff(attempt, rng)

        msg = "unreachable retry state"
        raise RuntimeError(msg)

    def usage_snapshot(self) -> dict[str, object]:
        return {
            "provider": "anthropic",
            "model": self._model,
            "requests": self._usage.requests,
            "retries": self._usage.retries,
            "prompt_tokens": self._usage.prompt_tokens,
            "completion_tokens": self._usage.completion_tokens,
            "total_tokens": self._usage.total_tokens,
            "estimated_cost_usd": round(self._usage.estimated_cost_usd, 8),
        }

    def _default_sender(self, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            _ANTHROPIC_MESSAGES_URL,
            method="POST",
            data=data,
            headers={
                "x-api-key": self._api_key,  # type: ignore[dict-item]
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as resp:
                response_data = resp.read().decode("utf-8")
                loaded = json.loads(response_data)
                if not isinstance(loaded, dict):
                    msg = "Anthropic response must be a JSON object"
                    raise ValueError(msg)
                return loaded
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"Anthropic HTTP error {exc.code}: {body}"
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"Anthropic request error: {exc.reason}"
            raise RuntimeError(msg) from exc
        except TimeoutError as exc:
            msg = f"Anthropic request timed out after {self._timeout_seconds}s"
            raise RuntimeError(msg) from exc

    def _track_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._usage.requests += 1
        self._usage.prompt_tokens += input_tokens
        self._usage.completion_tokens += output_tokens
        self._usage.total_tokens += input_tokens + output_tokens
        in_price, out_price = _MODEL_PRICING_PER_1M.get(self._model, (0.0, 0.0))
        cost = ((input_tokens / 1_000_000) * in_price) + ((output_tokens / 1_000_000) * out_price)
        self._usage.estimated_cost_usd += cost

    def _respect_rate_limit(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_monotonic
        remaining = self._min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_monotonic = time.monotonic()

    def _sleep_rate_limit(self, attempt: int, rng: random.Random | None) -> None:
        """Longer backoff specifically for 429 rate-limit responses."""
        delay = self._rate_limit_backoff_seconds * (attempt + 1)
        if self._jitter_seconds > 0:
            jitter_source = rng.random() if rng is not None else random.random()
            delay += jitter_source * self._jitter_seconds
        time.sleep(delay)

    def _sleep_backoff(self, attempt: int, rng: random.Random | None) -> None:
        delay = self._base_backoff_seconds * (2**attempt)
        if self._jitter_seconds > 0:
            jitter_source = rng.random() if rng is not None else random.random()
            delay += jitter_source * self._jitter_seconds
        if delay > 0:
            time.sleep(delay)


def _extract_text(response: dict[str, object]) -> str:
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    return text
    msg = "unable to extract text from Anthropic response"
    raise ValueError(msg)


def _extract_usage(response: dict[str, object]) -> tuple[int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)
    input_tokens = _to_int(usage.get("input_tokens"))
    output_tokens = _to_int(usage.get("output_tokens"))
    return (input_tokens, output_tokens)


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
