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


def test_openai_adapter_call_messages_uses_chat_sender() -> None:
    def chat_sender(payload: dict[str, object]) -> dict[str, object]:
        messages = payload.get("messages")
        assert isinstance(messages, list)
        return {
            "choices": [{"message": {"content": "chat output"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    adapter = OpenAIChatAdapter(
        model="gpt-4o-mini",
        api_key="test-key",
        sender=lambda _payload: {"output_text": "unused"},
        chat_sender=chat_sender,
        max_retries=0,
    )

    result = adapter.call_messages(
        [{"role": "user", "content": "Hello"}],
        random.Random(0),
    )
    usage = adapter.usage_snapshot()

    assert result == "chat output"
    assert usage["requests"] == 1
    assert usage["total_tokens"] == 5


def test_openai_adapter_call_with_tools_returns_tool_calls() -> None:
    def chat_sender(payload: dict[str, object]) -> dict[str, object]:
        assert payload["tool_choice"] == "auto"
        tools = payload.get("tools")
        assert isinstance(tools, list)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "search_web",
                                    "arguments": '{"query":"AAPL"}',
                                },
                            }
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        }

    adapter = OpenAIChatAdapter(
        model="gpt-4o-mini",
        api_key="test-key",
        sender=lambda _payload: {"output_text": "unused"},
        chat_sender=chat_sender,
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
    assert usage["total_tokens"] == 10


def test_openai_adapter_call_with_tools_returns_text() -> None:
    def chat_sender(_payload: dict[str, object]) -> dict[str, object]:
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "No tool needed."},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    adapter = OpenAIChatAdapter(
        model="gpt-4o-mini",
        api_key="test-key",
        sender=lambda _payload: {"output_text": "unused"},
        chat_sender=chat_sender,
        max_retries=0,
    )

    tool_calls, text = adapter.call_with_tools(
        [{"role": "user", "content": "Say hi"}],
        [],
        random.Random(1),
    )

    assert tool_calls == []
    assert text == "No tool needed."
