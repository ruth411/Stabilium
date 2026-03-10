from __future__ import annotations

import random

import pytest

from agent_stability_engine.traces.sandbox import SandboxExecutor
from agent_stability_engine.traces.schema import AgentTask


def _task(
    *,
    sandbox_responses: dict[str, str] | None = None,
    tool_endpoints: dict[str, str] | None = None,
) -> AgentTask:
    return AgentTask(
        id="task-001",
        difficulty="medium",
        goal="Find AAPL and decide above 200",
        tools=[],
        reference_trajectory=["search_web"],
        expected_answer="above",
        max_steps=5,
        timeout_seconds=30,
        sandbox_responses=sandbox_responses or {},
        tool_endpoints=tool_endpoints or {},
    )


def test_sandbox_executor_rejects_invalid_fault_rate() -> None:
    with pytest.raises(ValueError, match="fault_rate"):
        SandboxExecutor(task=_task(), fault_rate=1.1)


def test_sandbox_executor_fault_injection_wins() -> None:
    executor = SandboxExecutor(
        task=_task(sandbox_responses={"search_web": "AAPL is 227.5"}),
        fault_rate=1.0,
        rng=random.Random(0),
    )

    call = executor.execute("search_web", {"query": "AAPL"}, "call-1")
    assert call.is_fault_injected is True
    assert call.result is None
    assert call.error is not None
    assert call.duration_ms >= 0


def test_sandbox_executor_uses_sandbox_when_no_endpoint() -> None:
    executor = SandboxExecutor(
        task=_task(sandbox_responses={"search_web": "AAPL is 227.5"}),
        fault_rate=0.0,
        rng=random.Random(0),
    )

    call = executor.execute("search_web", {"query": "AAPL"}, "call-2")
    assert call.is_fault_injected is False
    assert call.result == "AAPL is 227.5"
    assert call.error is None


def test_sandbox_executor_returns_error_when_tool_unconfigured() -> None:
    executor = SandboxExecutor(task=_task(), fault_rate=0.0, rng=random.Random(0))

    call = executor.execute("missing_tool", {"x": 1}, "call-3")
    assert call.result is None
    assert call.error is not None
    assert "No sandbox_response or tool_endpoint configured" in call.error


def test_sandbox_executor_prefers_endpoint_over_sandbox() -> None:
    class _EndpointExecutor(SandboxExecutor):
        def _call_endpoint(self, url: str, tool_name: str, arguments: dict[str, object]) -> str:
            assert url == "https://example.test/tool"
            assert tool_name == "search_web"
            assert arguments == {"query": "AAPL"}
            return "LIVE: AAPL is 227.5"

    executor = _EndpointExecutor(
        task=_task(
            sandbox_responses={"search_web": "SANDBOX: AAPL is 100.0"},
            tool_endpoints={"search_web": "https://example.test/tool"},
        ),
        fault_rate=0.0,
        rng=random.Random(0),
    )

    call = executor.execute("search_web", {"query": "AAPL"}, "call-4")
    assert call.result == "LIVE: AAPL is 227.5"
    assert call.error is None


def test_sandbox_executor_falls_back_when_endpoint_fails() -> None:
    class _EndpointFailingExecutor(SandboxExecutor):
        def _call_endpoint(self, url: str, tool_name: str, arguments: dict[str, object]) -> str:
            _ = (url, tool_name, arguments)
            raise RuntimeError("network down")

    executor = _EndpointFailingExecutor(
        task=_task(
            sandbox_responses={"search_web": "SANDBOX: AAPL is 227.5"},
            tool_endpoints={"search_web": "https://example.test/tool"},
        ),
        fault_rate=0.0,
        rng=random.Random(0),
    )

    call = executor.execute("search_web", {"query": "AAPL"}, "call-5")
    assert call.result == "SANDBOX: AAPL is 227.5"
    assert call.error is None
