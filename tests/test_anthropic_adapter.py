from __future__ import annotations

import random

from agent_stability_engine.adapters.anthropic import AnthropicChatAdapter


def test_anthropic_adapter_returns_text_and_tracks_usage() -> None:
    def sender(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "content": [{"type": "text", "text": "anthropic output"}],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
            },
        }

    adapter = AnthropicChatAdapter(
        model="claude-haiku-4-5",
        api_key="test-key",
        sender=sender,
        max_retries=0,
    )

    result = adapter("hello", random.Random(0))
    usage = adapter.usage_snapshot()

    assert result == "anthropic output"
    assert usage["requests"] == 1
    assert usage["total_tokens"] == 20
    assert usage["estimated_cost_usd"] > 0


def test_anthropic_adapter_retries_429_and_succeeds() -> None:
    calls = 0

    def sender(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Anthropic HTTP error 429: rate limit")
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    adapter = AnthropicChatAdapter(
        model="claude-haiku-4-5",
        api_key="test-key",
        sender=sender,
        max_retries=1,
        base_backoff_seconds=0.0,
        rate_limit_backoff_seconds=0.0,
        jitter_seconds=0.0,
    )

    result = adapter("hello", random.Random(0))
    usage = adapter.usage_snapshot()

    assert result == "ok"
    assert calls == 2
    assert usage["retries"] == 1


def test_anthropic_adapter_call_messages_extracts_system_prompt() -> None:
    captured_payload: dict[str, object] = {}

    def sender(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "content": [{"type": "text", "text": "chat output"}],
            "usage": {"input_tokens": 4, "output_tokens": 3},
        }

    adapter = AnthropicChatAdapter(
        model="claude-haiku-4-5",
        api_key="test-key",
        sender=sender,
        max_retries=0,
    )

    result = adapter.call_messages(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell me a joke"},
        ],
        random.Random(0),
    )
    usage = adapter.usage_snapshot()

    assert result == "chat output"
    assert usage["requests"] == 1
    assert usage["total_tokens"] == 7
    assert captured_payload["system"] == "Be concise."
    messages = captured_payload["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1


def test_anthropic_adapter_call_with_tools_returns_tool_use_blocks() -> None:
    def sender(payload: dict[str, object]) -> dict[str, object]:
        tools = payload.get("tools")
        assert isinstance(tools, list)
        return {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search_web",
                    "input": {"query": "AAPL"},
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }

    adapter = AnthropicChatAdapter(
        model="claude-haiku-4-5",
        api_key="test-key",
        sender=sender,
        max_retries=0,
    )

    tool_calls, text = adapter.call_with_tools(
        [{"role": "user", "content": "Find AAPL"}],
        [
            {
                "name": "search_web",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        random.Random(0),
    )
    usage = adapter.usage_snapshot()

    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    assert text is None
    assert usage["requests"] == 1
    assert usage["total_tokens"] == 9


def test_anthropic_adapter_call_with_tools_returns_text() -> None:
    def sender(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "No tool needed."}],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

    adapter = AnthropicChatAdapter(
        model="claude-haiku-4-5",
        api_key="test-key",
        sender=sender,
        max_retries=0,
    )

    tool_calls, text = adapter.call_with_tools(
        [{"role": "user", "content": "Say hi"}],
        [],
        random.Random(1),
    )

    assert tool_calls == []
    assert text == "No tool needed."
