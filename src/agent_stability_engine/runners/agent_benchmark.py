from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from agent_stability_engine.engine.asi import AgentWeights
from agent_stability_engine.engine.stats import summarize_mean_confidence
from agent_stability_engine.engine.trajectory import compute_trace_metrics
from agent_stability_engine.traces.sandbox import SandboxExecutor
from agent_stability_engine.traces.schema import AgentTask, AgentTrace, ToolCall


@dataclass(frozen=True)
class AgentBenchmarkResult:
    report: dict[str, object]
    traces: list[AgentTrace]


def run_agent_benchmark_suite(
    suite_path: Path,
    adapter: Any,
    run_count: int,
    seed: int,
    timestamp_utc: str | None = None,  # noqa: ARG001
    weights: AgentWeights | None = None,
    fault_rate: float = 0.0,
    max_tasks: int | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
    agent_factory: Callable[[], Any] | None = None,
) -> AgentBenchmarkResult:
    if run_count < 1:
        msg = "run_count must be >= 1"
        raise ValueError(msg)
    if not (0.0 <= fault_rate <= 1.0):
        msg = "fault_rate must be in [0.0, 1.0]"
        raise ValueError(msg)

    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    canonical_suite = json.dumps(suite_data, sort_keys=True, separators=(",", ":"))
    suite_sha256 = hashlib.sha256(canonical_suite.encode()).hexdigest()
    tasks = _load_tasks(suite_data)
    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    shared_adapter_lock = threading.Lock()
    shared_adapter = adapter

    class _LockedAdapter:
        def call_with_tools(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]],
            rng: random.Random | None = None,
        ) -> tuple[list[dict[str, object]], str | None]:
            with shared_adapter_lock:
                return _call_with_tools(
                    adapter=shared_adapter,
                    messages=messages,
                    tools=tools,
                    rng=rng,
                )

    def _make_task_adapter() -> Any:
        if agent_factory is not None:
            return agent_factory()
        if workers > 1:
            return _LockedAdapter()
        return shared_adapter

    def _evaluate_task(task: AgentTask) -> tuple[dict[str, object], list[AgentTrace]]:
        task_adapter = _make_task_adapter()
        normal_traces = [
            _run_agent_task(
                task=task,
                adapter=task_adapter,
                run_index=i,
                seed=seed,
                fault_rate=0.0,
            )
            for i in range(run_count)
        ]
        fault_traces = (
            [
                _run_agent_task(
                    task=task,
                    adapter=task_adapter,
                    run_index=i,
                    seed=seed + 100_003,
                    fault_rate=fault_rate,
                )
                for i in range(run_count)
            ]
            if fault_rate > 0
            else []
        )
        metrics = compute_trace_metrics(
            normal_traces,
            task,
            fault_traces=fault_traces,
            weights=weights,
        )
        task_hash = hashlib.sha256(
            json.dumps(asdict(task), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        case_report: dict[str, object] = {
            "task_id": task.id,
            "task_sha256": task_hash,
            "report": {
                "metrics": metrics,
                "run_count": run_count,
                "fault_rate": fault_rate,
                "normal_trace_count": len(normal_traces),
                "fault_trace_count": len(fault_traces),
            },
        }
        return (case_report, normal_traces)

    case_reports: list[dict[str, object]] = []
    all_traces: list[AgentTrace] = []
    total = len(tasks)
    completed_count = 0
    progress_lock = threading.Lock()

    def _evaluate_and_track(task: AgentTask) -> tuple[dict[str, object], list[AgentTrace]]:
        nonlocal completed_count
        result = _evaluate_task(task)
        with progress_lock:
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, total, task.id)
        return result

    if workers > 1:
        ordered: dict[int, tuple[dict[str, object], list[AgentTrace]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_evaluate_and_track, task): i for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        ordered_items = [ordered[i] for i in range(len(tasks))]
    else:
        ordered_items = [_evaluate_and_track(task) for task in tasks]

    for case_report, traces in ordered_items:
        case_reports.append(case_report)
        all_traces.extend(traces)

    trace_asi_values: list[float] = []
    domain_trace_asi: dict[str, list[float]] = {}
    for case in case_reports:
        task_id = str(case.get("task_id", ""))
        domain = task_id.split("-")[0] if "-" in task_id else "general"
        report = case.get("report")
        if not isinstance(report, dict):
            continue
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            continue
        raw_trace_asi = metrics.get("trace_asi")
        if isinstance(raw_trace_asi, (int, float)):
            trace_asi = float(raw_trace_asi)
            trace_asi_values.append(trace_asi)
            domain_trace_asi.setdefault(domain, []).append(trace_asi)

    mean_trace_asi = sum(trace_asi_values) / len(trace_asi_values) if trace_asi_values else 0.0
    trace_asi_statistics = (
        summarize_mean_confidence(trace_asi_values).to_dict() if trace_asi_values else None
    )
    domain_scores = {
        domain: round(sum(values) / len(values), 2)
        for domain, values in sorted(domain_trace_asi.items())
    }

    benchmark_id_seed = json.dumps(
        {
            "suite_sha256": suite_sha256,
            "run_count": run_count,
            "seed": seed,
            "job_type": "agent_benchmark",
            "fault_rate": fault_rate,
        },
        sort_keys=True,
    )
    benchmark_id = hashlib.sha256(benchmark_id_seed.encode()).hexdigest()[:16]

    aggregate: dict[str, object] = {
        "benchmark_id": f"agent-bench-{benchmark_id}",
        "suite_name": suite_data.get("name", "unnamed_agent_suite"),
        "suite_sha256": suite_sha256,
        "run_count": run_count,
        "seed": seed,
        "job_type": "agent_benchmark",
        "fault_rate": fault_rate,
        "num_cases": len(tasks),
        "mean_trace_asi": mean_trace_asi,
        # Compatibility for consumers expecting "mean_asi".
        "mean_asi": mean_trace_asi,
        "trace_asi_statistics": trace_asi_statistics,
        "case_trace_asi_values": trace_asi_values,
        "domain_scores": domain_scores,
        "cases": case_reports,
    }
    return AgentBenchmarkResult(report=aggregate, traces=all_traces)


def _run_agent_task(
    *,
    task: AgentTask,
    adapter: Any,
    run_index: int,
    seed: int,
    fault_rate: float,
) -> AgentTrace:
    task_seed = int(hashlib.sha256(task.id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + run_index * 997 + task_seed)
    executor = SandboxExecutor(task, fault_rate=fault_rate, rng=rng)

    messages: list[dict[str, object]] = [{"role": "user", "content": task.goal}]
    all_tool_calls: list[ToolCall] = []
    final_answer: str | None = None
    start_ms = int(time.monotonic() * 1000)
    deadline_ms = start_ms + (task.timeout_seconds * 1000)
    timed_out = False

    for _ in range(task.max_steps):
        if int(time.monotonic() * 1000) >= deadline_ms:
            timed_out = True
            break
        tool_call_dicts, text = _call_with_tools(
            adapter=adapter,
            messages=messages,
            tools=task.tools,
            rng=rng,
        )
        if text is not None:
            final_answer = text
            break
        if not tool_call_dicts:
            break

        for raw_tool_call in tool_call_dicts:
            tool_name, arguments, call_id, mode, assistant_tool_call = _parse_tool_call(
                raw_tool_call
            )
            tool_call = executor.execute(tool_name, arguments, call_id)
            all_tool_calls.append(tool_call)
            _append_tool_roundtrip_messages(
                messages=messages,
                mode=mode,
                assistant_tool_call=assistant_tool_call,
                tool_call=tool_call,
            )
    else:
        timed_out = True

    success = False
    if final_answer and task.expected_answer:
        success = task.expected_answer.lower() in final_answer.lower()
    elif final_answer and not task.expected_answer:
        success = True

    duration_ms = int(time.monotonic() * 1000) - start_ms
    trace_seed = f"{task.id}-{run_index}-{seed}-{fault_rate:.6f}"
    trace_id = f"trace-{hashlib.sha256(trace_seed.encode()).hexdigest()[:16]}"
    return AgentTrace(
        trace_id=trace_id,
        task_id=task.id,
        run_index=run_index,
        goal=task.goal,
        tool_calls=all_tool_calls,
        final_answer=final_answer,
        success=success,
        total_steps=len(all_tool_calls),
        duration_ms=duration_ms,
        timed_out=timed_out,
    )


def _call_with_tools(
    *,
    adapter: Any,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]],
    rng: random.Random | None,
) -> tuple[list[dict[str, object]], str | None]:
    call_with_tools = getattr(adapter, "call_with_tools", None)
    if not callable(call_with_tools):
        msg = "adapter must implement call_with_tools(messages, tools, rng)"
        raise ValueError(msg)
    result = call_with_tools(messages, tools, rng)
    if not isinstance(result, tuple) or len(result) != 2:
        msg = "adapter.call_with_tools must return (tool_calls, text)"
        raise ValueError(msg)
    tool_calls, text = result
    if not isinstance(tool_calls, list):
        msg = "adapter.call_with_tools must return a list for tool_calls"
        raise ValueError(msg)
    if text is not None and not isinstance(text, str):
        msg = "adapter.call_with_tools must return str | None for text"
        raise ValueError(msg)

    parsed_calls: list[dict[str, object]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            parsed_calls.append(call)
    return (parsed_calls, text)


def _parse_tool_call(
    raw_tool_call: dict[str, object],
) -> tuple[str, dict[str, object], str, str, dict[str, object]]:
    openai_function = raw_tool_call.get("function")
    if isinstance(openai_function, dict):
        tool_name = str(openai_function.get("name", "")).strip() or "unknown_tool"
        arguments = _parse_openai_arguments(openai_function.get("arguments"))
        call_id = _tool_call_id(raw_tool_call, "call")
        normalized: dict[str, object] = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, sort_keys=True),
            },
        }
        return (tool_name, arguments, call_id, "openai", normalized)

    if raw_tool_call.get("type") == "tool_use" or "input" in raw_tool_call:
        tool_name = str(raw_tool_call.get("name", "")).strip() or "unknown_tool"
        arguments_obj = raw_tool_call.get("input")
        arguments = arguments_obj if isinstance(arguments_obj, dict) else {}
        call_id = _tool_call_id(raw_tool_call, "toolu")
        normalized = {
            "type": "tool_use",
            "id": call_id,
            "name": tool_name,
            "input": arguments,
        }
        return (tool_name, arguments, call_id, "anthropic", normalized)

    msg = "unsupported tool call format from adapter"
    raise ValueError(msg)


def _tool_call_id(raw_tool_call: dict[str, object], prefix: str) -> str:
    raw_id = raw_tool_call.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    digest = hashlib.sha256(
        json.dumps(raw_tool_call, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"{prefix}_{digest[:8]}"


def _parse_openai_arguments(arguments: object) -> dict[str, object]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _append_tool_roundtrip_messages(
    *,
    messages: list[dict[str, object]],
    mode: str,
    assistant_tool_call: dict[str, object],
    tool_call: ToolCall,
) -> None:
    tool_output = _tool_output(tool_call)
    if mode == "openai":
        messages.append({"role": "assistant", "content": None, "tool_calls": [assistant_tool_call]})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.tool_call_id,
                "content": tool_output,
            }
        )
        return

    messages.append({"role": "assistant", "content": [assistant_tool_call]})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.tool_call_id,
                    "content": tool_output,
                }
            ],
        }
    )


def _tool_output(tool_call: ToolCall) -> str:
    if tool_call.result is not None:
        return tool_call.result
    if tool_call.error:
        return f"TOOL_ERROR: {tool_call.error}"
    return "TOOL_ERROR: unknown"


def _load_tasks(suite_data: dict[str, object]) -> list[AgentTask]:
    raw_tasks = suite_data.get("tasks")
    if not isinstance(raw_tasks, list):
        msg = "agent suite must contain a 'tasks' list"
        raise ValueError(msg)

    tasks: list[AgentTask] = []
    for raw in raw_tasks:
        if isinstance(raw, dict):
            tasks.append(_parse_task(raw))
    return tasks


def _parse_task(raw: dict[str, object]) -> AgentTask:
    task_id = str(raw.get("id", "")).strip()
    if not task_id:
        msg = "each task must include a non-empty id"
        raise ValueError(msg)

    goal = str(raw.get("goal", "")).strip()
    if not goal:
        msg = f"task '{task_id}' must include a non-empty goal"
        raise ValueError(msg)

    tools_raw = raw.get("tools")
    tools: list[dict[str, object]] = []
    if isinstance(tools_raw, list):
        for tool in tools_raw:
            if isinstance(tool, dict):
                tools.append(tool)

    reference_raw = raw.get("reference_trajectory")
    reference_trajectory: list[str] = []
    if isinstance(reference_raw, list):
        for item in reference_raw:
            if isinstance(item, str) and item.strip():
                reference_trajectory.append(item.strip())

    sandbox_raw = raw.get("sandbox_responses")
    sandbox_responses: dict[str, str] = {}
    if isinstance(sandbox_raw, dict):
        for key, value in sandbox_raw.items():
            if isinstance(key, str) and isinstance(value, str):
                sandbox_responses[key] = value

    endpoints_raw = raw.get("tool_endpoints")
    tool_endpoints: dict[str, str] = {}
    if isinstance(endpoints_raw, dict):
        for key, value in endpoints_raw.items():
            if isinstance(key, str) and isinstance(value, str):
                tool_endpoints[key] = value

    expected_answer_raw = raw.get("expected_answer")
    expected_answer = (
        str(expected_answer_raw).strip()
        if isinstance(expected_answer_raw, str) and expected_answer_raw.strip()
        else None
    )

    return AgentTask(
        id=task_id,
        difficulty=str(raw.get("difficulty", "unknown")),
        goal=goal,
        tools=tools,
        reference_trajectory=reference_trajectory,
        expected_answer=expected_answer,
        max_steps=_positive_int(raw.get("max_steps"), default=6),
        timeout_seconds=_positive_int(raw.get("timeout_seconds"), default=60),
        sandbox_responses=sandbox_responses,
        tool_endpoints=tool_endpoints,
    )


def _positive_int(raw_value: object, *, default: int) -> int:
    if isinstance(raw_value, bool):
        return default
    if isinstance(raw_value, int) and raw_value > 0:
        return raw_value
    if isinstance(raw_value, float) and raw_value > 0:
        return int(raw_value)
    return default
