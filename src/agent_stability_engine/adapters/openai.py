from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable
from urllib import error, request

_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# Estimated USD per 1M tokens (input/output) for common models.
# Unknown models default to zero-cost estimation.
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass
class _UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    retries: int = 0
    estimated_cost_usd: float = 0.0


class OpenAIChatAdapter:
    """Callable adapter for OpenAI Responses API with retry and usage tracking."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        min_interval_seconds: float = 0.0,
        base_backoff_seconds: float = 0.5,
        jitter_seconds: float = 0.1,
        sender: Callable[[dict[str, object]], dict[str, object]] | None = None,
        chat_sender: Callable[[dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        if not model:
            msg = "model must be non-empty"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            msg = "temperature must be between 0.0 and 2.0"
            raise ValueError(msg)
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            msg = "OPENAI_API_KEY is required for OpenAI adapter"
            raise ValueError(msg)

        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._min_interval_seconds = min_interval_seconds
        self._base_backoff_seconds = base_backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._sender = sender or self._default_sender
        self._chat_sender = chat_sender or self._default_chat_sender

        self._usage = _UsageTotals()
        self._last_request_monotonic = 0.0

    def __call__(self, prompt: str, rng: random.Random | None = None) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "input": prompt,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        attempts = self._max_retries + 1
        for attempt in range(attempts):
            self._respect_rate_limit()
            try:
                response = self._sender(payload)
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
                self._track_usage(prompt_tokens, completion_tokens, total_tokens)
                text = _extract_text(response)
                return text
            except Exception:
                if attempt >= self._max_retries:
                    raise
                self._usage.retries += 1
                self._sleep_backoff(attempt, rng)

        msg = "unreachable retry state"
        raise RuntimeError(msg)

    def call_messages(
        self,
        messages: list[dict[str, str]],
        rng: random.Random | None = None,
    ) -> str:
        payload: dict[str, object] = {"model": self._model, "messages": messages}
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        attempts = self._max_retries + 1
        for attempt in range(attempts):
            self._respect_rate_limit()
            try:
                response = self._chat_sender(payload)
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
                self._track_usage(prompt_tokens, completion_tokens, total_tokens)
                return _extract_text(response)
            except Exception:
                if attempt >= self._max_retries:
                    raise
                self._usage.retries += 1
                self._sleep_backoff(attempt, rng)

        msg = "unreachable retry state"
        raise RuntimeError(msg)

    def call_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        rng: random.Random | None = None,
    ) -> tuple[list[dict[str, object]], str | None]:
        """One Chat Completions call with tool definitions.

        Returns:
            (tool_call_dicts, None)  — model wants to invoke tools.
            ([], final_text)         — model produced a final answer.
        """
        openai_tools = [{"type": "function", "function": t} for t in tools]
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "tools": openai_tools,
            "tool_choice": "auto",
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        attempts = self._max_retries + 1
        for attempt in range(attempts):
            self._respect_rate_limit()
            try:
                response = self._chat_sender(payload)
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
                self._track_usage(prompt_tokens, completion_tokens, total_tokens)
                return _extract_tool_calls_or_text(response)
            except Exception:
                if attempt >= self._max_retries:
                    raise
                self._usage.retries += 1
                self._sleep_backoff(attempt, rng)

        msg = "unreachable retry state"
        raise RuntimeError(msg)

    def usage_snapshot(self) -> dict[str, object]:
        return {
            "provider": "openai",
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
            _OPENAI_RESPONSES_URL,
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
                    msg = "OpenAI response must be a JSON object"
                    raise ValueError(msg)
                return loaded
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"OpenAI HTTP error {exc.code}: {body}"
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"OpenAI request error: {exc.reason}"
            raise RuntimeError(msg) from exc

    def _default_chat_sender(self, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            _OPENAI_CHAT_URL,
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
                    msg = "OpenAI chat response must be a JSON object"
                    raise ValueError(msg)
                return loaded
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"OpenAI HTTP error {exc.code}: {body}"
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"OpenAI request error: {exc.reason}"
            raise RuntimeError(msg) from exc

    def _track_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self._usage.requests += 1
        self._usage.prompt_tokens += prompt_tokens
        self._usage.completion_tokens += completion_tokens
        self._usage.total_tokens += total_tokens
        input_price, output_price = _MODEL_PRICING_PER_1M.get(self._model, (0.0, 0.0))
        cost = ((prompt_tokens / 1_000_000) * input_price) + (
            (completion_tokens / 1_000_000) * output_price
        )
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

    def _sleep_backoff(self, attempt: int, rng: random.Random | None) -> None:
        delay = self._base_backoff_seconds * (2**attempt)
        if self._jitter_seconds > 0:
            jitter_source = rng.random() if rng is not None else random.random()
            delay += jitter_source * self._jitter_seconds
        if delay > 0:
            time.sleep(delay)


def _extract_text(response: dict[str, object]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text")
                    if isinstance(text, str):
                        return text

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content

    msg = "unable to extract text from OpenAI response"
    raise ValueError(msg)


def _extract_usage(response: dict[str, object]) -> tuple[int, int, int]:
    usage_obj = response.get("usage")
    if not isinstance(usage_obj, dict):
        return (0, 0, 0)

    prompt_tokens = _to_int(usage_obj.get("input_tokens"))
    completion_tokens = _to_int(usage_obj.get("output_tokens"))
    total_tokens = _to_int(usage_obj.get("total_tokens"))

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        prompt_tokens = _to_int(usage_obj.get("prompt_tokens"))
        completion_tokens = _to_int(usage_obj.get("completion_tokens"))
        total_tokens = _to_int(usage_obj.get("total_tokens"))

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return (prompt_tokens, completion_tokens, total_tokens)


def _extract_tool_calls_or_text(
    response: dict[str, object],
) -> tuple[list[dict[str, object]], str | None]:
    """Parse a Chat Completions response into (tool_calls, None) or ([], text)."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "No choices in OpenAI tool response"
        raise ValueError(msg)
    first = choices[0]
    if not isinstance(first, dict):
        msg = "Unexpected choices format in OpenAI tool response"
        raise ValueError(msg)
    message = first.get("message", {})
    if not isinstance(message, dict):
        msg = "Unexpected message format in OpenAI tool response"
        raise ValueError(msg)
    finish_reason = first.get("finish_reason")
    tool_calls = message.get("tool_calls")
    if finish_reason == "tool_calls" and isinstance(tool_calls, list):
        return (tool_calls, None)
    content = message.get("content")
    if isinstance(content, str):
        return ([], content)
    msg = "Cannot extract tool calls or text from OpenAI response"
    raise ValueError(msg)


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
