from __future__ import annotations

import random

from agent_stability_engine.adapters.openai import OpenAIChatAdapter


def test_openai_adapter_returns_text_and_tracks_usage() -> None:
    def sender(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "output_text": "adapter output",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }

    adapter = OpenAIChatAdapter(
        model="gpt-4o-mini",
        api_key="test-key",
        sender=sender,
        max_retries=0,
    )

    result = adapter("hello", random.Random(0))
    usage = adapter.usage_snapshot()

    assert result == "adapter output"
    assert usage["requests"] == 1
    assert usage["total_tokens"] == 15
    assert usage["estimated_cost_usd"] > 0


def test_openai_adapter_retries_and_succeeds() -> None:
    calls = 0

    def sender(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return {
            "output_text": "ok",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        }

    adapter = OpenAIChatAdapter(
        model="gpt-4o-mini",
        api_key="test-key",
        sender=sender,
        max_retries=1,
        base_backoff_seconds=0.0,
        jitter_seconds=0.0,
    )

    result = adapter("hello", random.Random(0))
    usage = adapter.usage_snapshot()

    assert result == "ok"
    assert calls == 2
    assert usage["retries"] == 1
