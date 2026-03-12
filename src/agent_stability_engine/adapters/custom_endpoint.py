from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Callable
from urllib import error, request

from agent_stability_engine.security import (
    assert_public_endpoint_host,
    validate_custom_endpoint_url,
)


@dataclass
class _UsageTotals:
    requests: int = 0
    retries: int = 0


class CustomEndpointAdapter:
    """Callable adapter for custom HTTP inference endpoints."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.5,
        jitter_seconds: float = 0.2,
        sender: Callable[[dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        endpoint_url = validate_custom_endpoint_url(endpoint_url)
        if not model:
            msg = "model must be non-empty"
            raise ValueError(msg)
        if not api_key:
            msg = "api_key must be non-empty"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)

        self._endpoint_url = endpoint_url
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._sender = sender or self._default_sender
        self._usage = _UsageTotals()

    def __call__(self, prompt: str, rng: random.Random | None = None) -> str:
        payload: dict[str, object] = {"model": self._model, "prompt": prompt}
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._sender(payload)
                self._usage.requests += 1
                return _extract_text(response)
            except Exception:
                if attempt >= self._max_retries:
                    raise
                self._usage.retries += 1
                delay = self._base_backoff_seconds * (2**attempt)
                if self._jitter_seconds > 0:
                    jitter_source = rng.random() if rng is not None else random.random()
                    delay += jitter_source * self._jitter_seconds
                if delay > 0:
                    time.sleep(delay)

        msg = "unreachable retry state"
        raise RuntimeError(msg)

    def call_messages(
        self,
        messages: list[dict[str, str]],
        rng: random.Random | None = None,
    ) -> str:
        payload: dict[str, object] = {"model": self._model, "messages": messages}
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._sender(payload)
                self._usage.requests += 1
                return _extract_text_or_join_messages(response)
            except Exception:
                if attempt >= self._max_retries:
                    raise
                self._usage.retries += 1
                delay = self._base_backoff_seconds * (2**attempt)
                if self._jitter_seconds > 0:
                    jitter_source = rng.random() if rng is not None else random.random()
                    delay += jitter_source * self._jitter_seconds
                if delay > 0:
                    time.sleep(delay)

        msg = "unreachable retry state"
        raise RuntimeError(msg)

    def usage_snapshot(self) -> dict[str, object]:
        return {
            "provider": "custom",
            "model": self._model,
            "endpoint_url": self._endpoint_url,
            "requests": self._usage.requests,
            "retries": self._usage.retries,
        }

    def _default_sender(self, payload: dict[str, object]) -> dict[str, object]:
        assert_public_endpoint_host(self._endpoint_url)
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self._endpoint_url,
            method="POST",
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                response_data = response.read().decode("utf-8")
                loaded = json.loads(response_data)
                if not isinstance(loaded, dict):
                    msg = "custom endpoint response must be a JSON object"
                    raise ValueError(msg)
                return loaded
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"Custom endpoint HTTP error {exc.code}: {body}"
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"Custom endpoint request error: {exc.reason}"
            raise RuntimeError(msg) from exc


def _extract_text(response: dict[str, object]) -> str:
    for key in ("output", "text", "response", "result", "message"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content

    msg = "unable to extract text from custom endpoint response"
    raise ValueError(msg)


def _extract_text_or_join_messages(response: dict[str, object]) -> str:
    try:
        return _extract_text(response)
    except ValueError:
        messages_obj = response.get("messages")
        if isinstance(messages_obj, list):
            parts: list[str] = []
            for item in messages_obj:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        parts.append(content.strip())
            if parts:
                return "\n".join(parts)
        msg = "unable to extract text from custom endpoint message response"
        raise ValueError(msg) from None
