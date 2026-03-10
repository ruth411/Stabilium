from __future__ import annotations

import pytest

from agent_stability_engine.engine.trajectory import (
    compute_trace_metrics,
    parameter_fidelity,
    trajectory_consistency,
)
from agent_stability_engine.traces.schema import AgentTask, AgentTrace, ToolCall


def _tool_call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        tool_call_id=f"call-{name}",
        tool_name=name,
        arguments=arguments,
        result="ok",
        is_fault_injected=False,
        error=None,
        duration_ms=1,
    )


def _trace(run_index: int, tools: list[ToolCall], success: bool) -> AgentTrace:
    return AgentTrace(
        trace_id=f"trace-{run_index}",
        task_id="task-001",
        run_index=run_index,
        goal="Find AAPL and decide above 200",
        tool_calls=tools,
        final_answer="AAPL is above 200" if success else "Could not complete task",
        success=success,
        total_steps=len(tools),
        duration_ms=20,
        timed_out=False,
    )


def _task() -> AgentTask:
    return AgentTask(
        id="task-001",
        difficulty="medium",
        goal="Find AAPL and decide above 200",
        tools=[
            {
                "name": "search_web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "extract_number",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "target": {"type": "string"}},
                    "required": ["text", "target"],
                },
            },
        ],
        reference_trajectory=["search_web", "extract_number"],
        expected_answer="above 200",
        max_steps=6,
        timeout_seconds=60,
        sandbox_responses={},
        tool_endpoints={},
    )


def test_trajectory_consistency_is_one_for_identical_sequences() -> None:
    traces = [
        _trace(0, [_tool_call("search_web", {"query": "AAPL"})], success=True),
        _trace(1, [_tool_call("search_web", {"query": "AAPL"})], success=True),
    ]
    assert trajectory_consistency(traces) == pytest.approx(1.0)


def test_trajectory_consistency_drops_for_different_sequences() -> None:
    traces = [
        _trace(0, [_tool_call("search_web", {"query": "AAPL"})], success=True),
        _trace(
            1,
            [
                _tool_call("extract_number", {"text": "AAPL 227", "target": "AAPL"}),
                _tool_call("search_web", {"query": "AAPL"}),
            ],
            success=True,
        ),
    ]
    assert trajectory_consistency(traces) < 1.0


def test_parameter_fidelity_penalizes_missing_required_arguments() -> None:
    traces = [
        _trace(0, [_tool_call("search_web", {"query": "AAPL"})], success=True),
        _trace(1, [_tool_call("extract_number", {"text": "AAPL 227"})], success=True),
    ]
    score = parameter_fidelity(traces, _task().tools)
    assert 0.0 <= score < 1.0


def test_compute_trace_metrics_returns_expected_shape() -> None:
    task = _task()
    normal_traces = [
        _trace(
            0,
            [
                _tool_call("search_web", {"query": "AAPL"}),
                _tool_call("extract_number", {"text": "AAPL 227.5", "target": "AAPL"}),
            ],
            success=True,
        ),
        _trace(
            1,
            [
                _tool_call("search_web", {"query": "AAPL"}),
                _tool_call("extract_number", {"text": "AAPL 228.0", "target": "AAPL"}),
            ],
            success=True,
        ),
    ]
    fault_traces = [
        _trace(
            2,
            [_tool_call("search_web", {"query": "AAPL"})],
            success=False,
        )
    ]

    metrics = compute_trace_metrics(normal_traces, task, fault_traces=fault_traces)

    expected_keys = {
        "trajectory_consistency",
        "tool_selection_accuracy",
        "step_efficiency",
        "goal_completion_rate",
        "parameter_fidelity",
        "fault_robustness",
        "trace_asi",
    }
    assert set(metrics.keys()) == expected_keys
    assert 0.0 <= metrics["trajectory_consistency"] <= 1.0
    assert 0.0 <= metrics["tool_selection_accuracy"] <= 1.0
    assert 0.0 <= metrics["step_efficiency"] <= 1.0
    assert 0.0 <= metrics["goal_completion_rate"] <= 1.0
    assert 0.0 <= metrics["parameter_fidelity"] <= 1.0
    assert 0.0 <= metrics["fault_robustness"] <= 1.0
    assert 0.0 <= metrics["trace_asi"] <= 100.0
