from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool invocation within an agent trace."""

    tool_call_id: str  # "call_abc123" (OpenAI) or "toolu_abc123" (Anthropic)
    tool_name: str
    arguments: dict[str, object]
    result: str | None  # None if not yet executed or fault-injected
    is_fault_injected: bool
    error: str | None
    duration_ms: int


@dataclass
class AgentTrace:
    """One complete run of an agent task."""

    trace_id: str  # "trace-{sha256[:16]}"
    task_id: str
    run_index: int  # 0-based index within k runs
    goal: str
    tool_calls: list[ToolCall]
    final_answer: str | None
    success: bool
    total_steps: int
    duration_ms: int
    timed_out: bool


@dataclass
class AgentTask:
    """Definition of a single agent task (loaded from JSON suite)."""

    id: str
    difficulty: str
    goal: str
    tools: list[dict[str, object]]  # OpenAI function schema format
    reference_trajectory: list[str]  # ordered list of expected tool names
    expected_answer: str | None
    max_steps: int
    timeout_seconds: int
    sandbox_responses: dict[str, str]  # {tool_name: mock_response_string}
    tool_endpoints: dict[str, str] = field(default_factory=dict)  # {tool_name: http_url}
