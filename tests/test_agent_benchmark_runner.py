from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agent_stability_engine.runners.agent_benchmark import run_agent_benchmark_suite


class _ToolUsingAdapter:
    def call_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        rng: random.Random | None = None,
    ) -> tuple[list[dict[str, object]], str | None]:
        _ = tools
        _ = rng
        tool_results = [msg for msg in messages if msg.get("role") == "tool"]
        if not tool_results:
            return (
                [
                    {
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "AAPL price today"}),
                        },
                    }
                ],
                None,
            )

        result_text = str(tool_results[-1].get("content", ""))
        if "227.50" in result_text:
            return ([], "AAPL is above $200.")
        return ([], "Unable to determine current AAPL price.")


def _write_agent_suite(path: Path) -> None:
    suite: dict[str, Any] = {
        "name": "agent_suite_test",
        "type": "agent",
        "tasks": [
            {
                "id": "search-price-001",
                "difficulty": "easy",
                "goal": "Find AAPL price and tell me if it is above $200",
                "tools": [
                    {
                        "name": "search_web",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
                "reference_trajectory": ["search_web"],
                "expected_answer": "above $200",
                "max_steps": 3,
                "timeout_seconds": 30,
                "sandbox_responses": {"search_web": "AAPL is trading at $227.50"},
            },
            {
                "id": "search-price-002",
                "difficulty": "easy",
                "goal": "Find AAPL price and tell me if it is above $200",
                "tools": [
                    {
                        "name": "search_web",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
                "reference_trajectory": ["search_web"],
                "expected_answer": "above $200",
                "max_steps": 3,
                "timeout_seconds": 30,
                "sandbox_responses": {"search_web": "AAPL is trading at $227.50"},
            },
        ],
    }
    path.write_text(json.dumps(suite), encoding="utf-8")


def test_agent_benchmark_runner_outputs_expected_report_shape(tmp_path: Path) -> None:
    suite_path = tmp_path / "agent_suite.json"
    _write_agent_suite(suite_path)

    result = run_agent_benchmark_suite(
        suite_path=suite_path,
        adapter=_ToolUsingAdapter(),
        run_count=2,
        seed=9,
    )
    report = result.report
    assert str(report["benchmark_id"]).startswith("agent-bench-")
    assert report["suite_name"] == "agent_suite_test"
    assert report["job_type"] == "agent_benchmark"
    assert report["num_cases"] == 2
    assert report["fault_rate"] == 0.0

    assert isinstance(report["mean_trace_asi"], float)
    assert report["mean_asi"] == report["mean_trace_asi"]

    stats = report["trace_asi_statistics"]
    assert isinstance(stats, dict)
    assert stats["sample_size"] == 2
    assert stats["ci_low"] <= report["mean_trace_asi"] <= stats["ci_high"]

    case_values = report["case_trace_asi_values"]
    assert isinstance(case_values, list)
    assert len(case_values) == 2

    cases = report["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 2
    first_case = cases[0]
    assert isinstance(first_case, dict)
    assert "task_sha256" in first_case
    task_report = first_case["report"]
    assert isinstance(task_report, dict)
    metrics = task_report["metrics"]
    assert isinstance(metrics, dict)
    assert "trace_asi" in metrics

    traces = result.traces
    assert len(traces) == 4
    assert all(trace.task_id.startswith("search-price-") for trace in traces)


def test_agent_benchmark_runner_parallel_uses_agent_factory(tmp_path: Path) -> None:
    suite_path = tmp_path / "agent_suite.json"
    _write_agent_suite(suite_path)
    created = 0

    class _FactoryAdapter(_ToolUsingAdapter):
        def __init__(self, marker: int) -> None:
            self._marker = marker

    def make_adapter() -> object:
        nonlocal created
        created += 1
        return _FactoryAdapter(created)

    result = run_agent_benchmark_suite(
        suite_path=suite_path,
        adapter=_ToolUsingAdapter(),
        agent_factory=make_adapter,
        run_count=2,
        seed=3,
        workers=2,
    )
    report = result.report
    assert report["num_cases"] == 2
    assert created == 2


def test_agent_benchmark_runner_fault_rate_impacts_fault_robustness(tmp_path: Path) -> None:
    suite_path = tmp_path / "agent_suite.json"
    _write_agent_suite(suite_path)

    result = run_agent_benchmark_suite(
        suite_path=suite_path,
        adapter=_ToolUsingAdapter(),
        run_count=2,
        seed=5,
        fault_rate=1.0,
    )
    report = result.report
    assert report["fault_rate"] == 1.0
    cases = report["cases"]
    assert isinstance(cases, list)
    assert cases
    first = cases[0]
    assert isinstance(first, dict)
    task_report = first["report"]
    assert isinstance(task_report, dict)
    metrics = task_report["metrics"]
    assert isinstance(metrics, dict)
    fault_robustness = metrics.get("fault_robustness")
    assert isinstance(fault_robustness, float)
    assert 0.0 <= fault_robustness <= 1.0
