from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from agent_stability_engine.traces.schema import AgentTrace, ToolCall


class _ToolSpanContext:
    """Context manager for a single tool call within a trace."""

    def __init__(self, trace: _TraceContext, tool_name: str, arguments: dict[str, object]) -> None:
        self._trace = trace
        self._tool_name = tool_name
        self._arguments = arguments
        self._start_ms = int(time.monotonic() * 1000)
        self.result: str | None = None
        self.error: str | None = None

    def __enter__(self) -> _ToolSpanContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        duration_ms = int(time.monotonic() * 1000) - self._start_ms
        tc = ToolCall(
            tool_call_id=str(uuid.uuid4())[:8],
            tool_name=self._tool_name,
            arguments=self._arguments,
            result=self.result,
            is_fault_injected=False,
            error=str(exc_val) if exc_type is not None else self.error,
            duration_ms=duration_ms,
        )
        self._trace._tool_calls.append(tc)


class _TraceContext:
    """Context for a single agent run. Yield from TraceCollector.trace()."""

    def __init__(self, task_id: str, goal: str, run_index: int) -> None:
        self._task_id = task_id
        self._goal = goal
        self._run_index = run_index
        self._tool_calls: list[ToolCall] = []
        self._start_ms = int(time.monotonic() * 1000)
        self.final_answer: str | None = None

    def tool_span(
        self,
        tool_name: str,
        arguments: dict[str, object] | None = None,
    ) -> _ToolSpanContext:
        """Open a context manager for one tool call. Set span.result before exiting."""
        return _ToolSpanContext(self, tool_name, arguments or {})

    def _to_trace(self, expected_answer: str | None = None) -> AgentTrace:
        success = False
        if self.final_answer is not None:
            if expected_answer:
                success = expected_answer.lower() in self.final_answer.lower()
            else:
                success = True
        trace_seed = f"{self._task_id}-{self._run_index}"
        trace_id = "trace-" + hashlib.sha256(trace_seed.encode()).hexdigest()[:16]
        return AgentTrace(
            trace_id=trace_id,
            task_id=self._task_id,
            run_index=self._run_index,
            goal=self._goal,
            tool_calls=list(self._tool_calls),
            final_answer=self.final_answer,
            success=success,
            total_steps=len(self._tool_calls),
            duration_ms=int(time.monotonic() * 1000) - self._start_ms,
            timed_out=False,
        )


class TraceCollector:
    """Collect agent traces from custom user-built agents (Mode 2 SDK).

    Usage::

        collector = TraceCollector()
        for i in range(3):
            with collector.trace("task-001", goal="Find AAPL price", run_index=i) as trace:
                with trace.tool_span("search_web", {"query": "AAPL price"}) as span:
                    span.result = my_search_fn("AAPL price")
                trace.final_answer = f"AAPL is at ${span.result}"

        traces = collector.get_traces()
    """

    def __init__(self) -> None:
        self._collected: list[AgentTrace] = []

    @contextmanager
    def trace(
        self,
        task_id: str,
        goal: str = "",
        run_index: int = 0,
        expected_answer: str | None = None,
    ) -> Iterator[_TraceContext]:
        """Context manager for one agent run. Appends the completed trace on exit."""
        ctx = _TraceContext(task_id, goal, run_index)
        yield ctx
        self._collected.append(ctx._to_trace(expected_answer=expected_answer))

    def get_traces(self) -> list[AgentTrace]:
        """Return all collected traces (in insertion order)."""
        return list(self._collected)

    def clear(self) -> None:
        """Remove all collected traces."""
        self._collected.clear()
