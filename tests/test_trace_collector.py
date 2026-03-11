from __future__ import annotations

from agent_stability_engine.engine.trajectory import compute_trace_metrics
from agent_stability_engine.traces import TraceCollector
from agent_stability_engine.traces.schema import AgentTask


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


def _run_custom_agent(collector: TraceCollector, run_index: int) -> None:
    with collector.trace(
        task_id="task-001",
        goal="Find AAPL and decide above 200",
        run_index=run_index,
        expected_answer="above 200",
    ) as trace:
        with trace.tool_span("search_web", {"query": "AAPL price"}) as search_span:
            search_span.result = "AAPL is 227.5"
        with trace.tool_span(
            "extract_number",
            {"text": search_span.result, "target": "AAPL"},
        ) as extract_span:
            extract_span.result = "227.5"
        trace.final_answer = f"AAPL is above 200 at ${extract_span.result}"


def test_trace_collector_integration_with_compute_trace_metrics() -> None:
    collector = TraceCollector()
    for run_index in range(3):
        _run_custom_agent(collector, run_index)

    traces = collector.get_traces()
    assert len(traces) == 3
    assert [trace.trace_id for trace in traces] == [
        "trace-deeb65133142833b",
        "trace-ccc3e72dcc54ef1b",
        "trace-afc32857e6efbd56",
    ]
    assert all(trace.success for trace in traces)
    assert all(trace.total_steps == 2 for trace in traces)

    metrics = compute_trace_metrics(traces, _task())
    assert metrics["trajectory_consistency"] == 1.0
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["step_efficiency"] == 1.0
    assert metrics["goal_completion_rate"] == 1.0
    assert metrics["parameter_fidelity"] == 1.0
    assert metrics["fault_robustness"] == 1.0
    assert metrics["trace_asi"] == 100.0


def test_trace_collector_expected_answer_sets_success_flag() -> None:
    collector = TraceCollector()
    with collector.trace(
        task_id="task-002",
        goal="Simple expected answer match",
        run_index=0,
        expected_answer="expected",
    ) as trace:
        trace.final_answer = "This output contains EXPECTED token."

    with collector.trace(
        task_id="task-002",
        goal="Simple expected answer mismatch",
        run_index=1,
        expected_answer="expected",
    ) as trace:
        trace.final_answer = "No required token here."

    traces = collector.get_traces()
    assert traces[0].success is True
    assert traces[1].success is False


def test_trace_collector_clear_removes_all_traces() -> None:
    collector = TraceCollector()
    with collector.trace(task_id="task-003", run_index=0) as trace:
        trace.final_answer = "ok"

    assert len(collector.get_traces()) == 1
    collector.clear()
    assert collector.get_traces() == []
