from __future__ import annotations

import random

from agent_stability_engine.adapters.custom_endpoint import CustomEndpointAdapter


def test_custom_endpoint_adapter_returns_text_and_tracks_usage() -> None:
    def sender(_payload: dict[str, object]) -> dict[str, object]:
        return {"output": "custom output"}

    adapter = CustomEndpointAdapter(
        endpoint_url="https://example.com/infer",
        model="customer-agent-v1",
        api_key="test-key",
        sender=sender,
        max_retries=0,
    )

    result = adapter("hello", random.Random(0))
    usage = adapter.usage_snapshot()

    assert result == "custom output"
    assert usage["provider"] == "custom"
    assert usage["requests"] == 1
    assert usage["retries"] == 0


def test_custom_endpoint_adapter_retries_and_succeeds() -> None:
    calls = 0

    def sender(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary custom endpoint failure")
        return {"text": "ok"}

    adapter = CustomEndpointAdapter(
        endpoint_url="https://example.com/infer",
        model="customer-agent-v1",
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
