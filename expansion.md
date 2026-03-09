# Stabilium: Level 2 + Level 3 — Complete Architecture Plan

## Confirmed Scope
- **Level 2**: Full conversation stability (context retention + self-consistency + instruction adherence + behavioral drift)
- **Level 3**: Agentic trace evaluation — Mode 1 (native API tool-calling: GPT-4/Claude) + Mode 2 (custom Python agent SDK)
- **Tool results**: Real tools with sandbox fallback (try real HTTP endpoint first; use mock if fails or not provided)
- **Delivery**: Web UI + FastAPI + Python SDK — all three
- **Build order**: Level 2 and Level 3 together (share the same adapter foundation work)

---

## What Already Exists (must not break)

| File | What it does | Lines to know |
|------|-------------|---------------|
| `adapters/openai.py` | Responses API (`/v1/responses`), `{"input": prompt}`, injectable `sender` | 1–232 |
| `adapters/anthropic.py` | Messages API, `{"messages": [{"role":"user","content":prompt}]}`, injectable `sender` | 1–218 |
| `engine/asi.py` | `ASIWeights`, `ASIProfile` (BALANCED/SAFETY_STRICT/REASONING_FOCUS), `ASICalculator.calculate(sv, cr, md, cm, tmf, ir)` | 1–97 |
| `runners/benchmark.py` | `run_benchmark_suite()`, `BenchmarkResult`, reads `case["prompt"]` | 1–170 |
| `engine/evaluator.py` | `StabilityEvaluator.evaluate(prompt, agent_fn, run_count, seed, ...)` | — |

**Critical constraints:**
- All existing `__call__(prompt: str, rng) -> str` interfaces must remain unchanged
- `ASIWeights` is a frozen dataclass — cannot change its fields (add new dataclasses instead)
- `run_benchmark_suite()` must continue reading `case["prompt"]` for existing suites
- `_default_sender` in both adapters uses `urllib` (no external dependencies) — maintain this pattern

---

---

# LEVEL 2: MULTI-TURN CONVERSATION STABILITY

## Test Case JSON Schema

Cases with `"type": "conversation"` have `messages` instead of `prompt`.
Single-turn cases (`"type": "single"` or absent) are unchanged.

```json
{
  "name": "conversation_suite_v1",
  "cases": [
    {
      "id": "conv-memory-001",
      "difficulty": "medium",
      "type": "conversation",
      "messages": [
        {"role": "system",    "content": "You are a helpful assistant. Always be concise."},
        {"role": "user",      "content": "My name is Alex and I work at a startup called Novex."},
        {"role": "assistant", "content": "__AGENT__"},
        {"role": "user",      "content": "What do you think are the biggest challenges for a startup CEO?"},
        {"role": "assistant", "content": "__AGENT__"},
        {"role": "user",      "content": "By the way, what was the name of my company again?"},
        {"role": "assistant", "content": "__AGENT__"}
      ],
      "expected_final": "Novex",
      "eval_turns": [-1],
      "constraints": [
        "must mention Novex in final response",
        "must not confuse the company name"
      ]
    }
  ]
}
```

**Field definitions:**
- `type`: `"conversation"` activates multi-turn runner; `"single"` or absent uses existing single-turn runner
- `messages`: List of `{role, content}`. `"__AGENT__"` marks turns the model fills in at eval time
- `expected_final`: Optional — what the final `__AGENT__` response should contain (for context_failure_rate)
- `eval_turns`: Which `__AGENT__` positions to evaluate. `-1` = last one only. `[0, -1]` = first and last
- `constraints`: Plain-English rules; each checked with keyword/substring matching in final response

---

## Level 2: New Metric Dataclass in `engine/asi.py`

Add alongside (not replacing) existing `ASIWeights`:

```python
@dataclass(frozen=True)
class ConversationWeights:
    """All metrics are failure rates (0.0 = perfect, 1.0 = worst)."""
    cross_run_variance: float        # embedding variance of final responses across k runs
    turn_contradiction_rate: float   # contradictions detected between turns in same run
    context_failure_rate: float      # 1 - context_retention (failed to recall expected)
    constraint_violation_rate: float # fraction of constraints violated
    drift_rate: float                # per-turn embedding drift from run 1's first response

    def as_tuple(self) -> tuple[float, ...]:
        return (self.cross_run_variance, self.turn_contradiction_rate,
                self.context_failure_rate, self.constraint_violation_rate, self.drift_rate)

DEFAULT_CONVERSATION_WEIGHTS = ConversationWeights(0.30, 0.25, 0.20, 0.15, 0.10)

class ConvASICalculator:
    def __init__(self, weights: ConversationWeights = DEFAULT_CONVERSATION_WEIGHTS) -> None:
        total = sum(weights.as_tuple())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("ConversationWeights must sum to 1.0")
        self._weights = weights

    def calculate(
        self,
        cross_run_variance: float,
        turn_contradiction_rate: float,
        context_failure_rate: float,
        constraint_violation_rate: float,
        drift_rate: float,
    ) -> float:
        metrics = (cross_run_variance, turn_contradiction_rate,
                   context_failure_rate, constraint_violation_rate, drift_rate)
        penalty = sum(w * m for w, m in zip(self._weights.as_tuple(), metrics))
        return min(max(100.0 * (1.0 - penalty), 0.0), 100.0)
```

---

## Level 2: Adapter Changes

### `adapters/openai.py` — add `call_messages()`

OpenAI has two APIs:
- Responses API (`/v1/responses`) — current, used for single-turn
- Chat Completions (`/v1/chat/completions`) — add for multi-turn + tool-calling

Add new constant: `_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"`

New method (does NOT change `__call__`):

```python
def call_messages(
    self,
    messages: list[dict[str, str]],
    rng: random.Random | None = None,
) -> str:
    """Send a multi-turn message list via OpenAI Chat Completions API."""
    payload: dict[str, object] = {"model": self._model, "messages": messages}
    if self._temperature is not None:
        payload["temperature"] = self._temperature

    attempts = self._max_retries + 1
    for attempt in range(attempts):
        self._respect_rate_limit()
        try:
            response = self._chat_sender(payload)
            # Reuse existing _extract_usage() — Chat Completions returns same usage keys
            pt, ct, tt = _extract_usage(response)
            self._track_usage(pt, ct, tt)
            return _extract_text(response)  # existing fn handles choices[0].message.content
        except Exception:
            if attempt >= self._max_retries:
                raise
            self._usage.retries += 1
            self._sleep_backoff(attempt, rng)
    raise RuntimeError("unreachable retry state")

def _chat_sender(self, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        _OPENAI_CHAT_URL,
        method="POST",
        data=data,
        headers={
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(http_request, timeout=self._timeout_seconds) as resp:
            loaded = json.loads(resp.read().decode("utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("OpenAI chat response must be a JSON object")
            return loaded
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP error {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request error: {exc.reason}") from exc
```

### `adapters/anthropic.py` — add `call_messages()`

Anthropic Messages API natively supports multi-turn. `system` role must be extracted separately.

```python
def call_messages(
    self,
    messages: list[dict[str, str]],
    rng: random.Random | None = None,
) -> str:
    """Send a multi-turn message list via Anthropic Messages API."""
    system_prompt: str | None = None
    filtered: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            filtered.append(msg)

    payload: dict[str, object] = {
        "model": self._model,
        "max_tokens": self._max_tokens,
        "messages": filtered,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if self._temperature is not None:
        payload["temperature"] = self._temperature

    attempts = self._max_retries + 1
    for attempt in range(attempts):
        self._respect_rate_limit()
        try:
            response = self._sender(payload)  # reuse existing _default_sender
            input_tokens, output_tokens = _extract_usage(response)
            self._track_usage(input_tokens, output_tokens)
            return _extract_text(response)  # existing fn works
        except RuntimeError as exc:
            if attempt >= self._max_retries:
                raise
            self._usage.retries += 1
            if "429" in str(exc):
                self._sleep_rate_limit(attempt, rng)
            else:
                self._sleep_backoff(attempt, rng)
    raise RuntimeError("unreachable retry state")
```

### `adapters/custom_endpoint.py` — add `call_messages()`

Post `{"messages": [...]}` to custom endpoint; join as text if endpoint returns non-standard response.

---

## Level 2: New File — `engine/conversation.py`

```python
from __future__ import annotations

import copy
from dataclasses import dataclass

from agent_stability_engine.engine.asi import ConvASICalculator, ConversationWeights
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.variance import EmbeddingVarianceScorer

@dataclass(frozen=True)
class ConversationEvaluation:
    report: dict[str, object]

class ConversationEvaluator:
    def __init__(
        self,
        weights: ConversationWeights | None = None,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
        embedding_openai_api_key: str | None = None,
    ) -> None:
        self._calculator = ConvASICalculator(weights or ConversationWeights(0.30, 0.25, 0.20, 0.15, 0.10))
        self._variance_scorer = EmbeddingVarianceScorer(
            provider=embedding_provider,
            openai_api_key=embedding_openai_api_key,
        )

    def evaluate(
        self,
        case: dict,
        adapter,
        run_count: int,
        seed: int,
    ) -> ConversationEvaluation:
        messages = case["messages"]
        expected_final: str | None = case.get("expected_final")
        constraints: list[str] = case.get("constraints", [])
        eval_turns: list[int] = case.get("eval_turns", [-1])

        # Find positions of "__AGENT__" turns
        agent_positions = [i for i, m in enumerate(messages) if m.get("content") == "__AGENT__"]
        if not agent_positions:
            raise ValueError(f"Case {case['id']} has no __AGENT__ positions")

        all_run_traces: list[list[str]] = []  # all_run_traces[run_i][turn_i] = response

        for run_i in range(run_count):
            history = copy.deepcopy(messages)
            run_responses: list[str] = []
            for pos in agent_positions:
                # Build message history up to (not including) this position
                context = [m for m in history[:pos] if m.get("content") != "__AGENT__"]
                response = adapter.call_messages(context)
                history[pos] = {"role": "assistant", "content": response}
                run_responses.append(response)
            all_run_traces.append(run_responses)

        # Compute eval turns — collect final responses (last eval turn) across k runs
        final_responses = [trace[-1] for trace in all_run_traces]

        # Metric 1: cross_run_variance — embedding variance of final responses
        variance_result = self._variance_scorer.score(final_responses)
        cross_run_variance = variance_result.normalized

        # Metric 2: turn_contradiction_rate — per-run contradiction detection across turns
        turn_contradiction_rate = self._compute_turn_contradictions(all_run_traces)

        # Metric 3: context_failure_rate — how often final response fails to contain expected
        if expected_final:
            context_failure_rate = self._compute_context_failure(final_responses, expected_final)
        else:
            context_failure_rate = 0.0

        # Metric 4: constraint_violation_rate — keyword matching of constraints
        if constraints:
            constraint_violation_rate = self._compute_constraint_violations(
                final_responses, constraints
            )
        else:
            constraint_violation_rate = 0.0

        # Metric 5: drift_rate — per-turn embedding drift
        drift_rate = self._compute_drift(all_run_traces)

        conv_asi = self._calculator.calculate(
            cross_run_variance,
            turn_contradiction_rate,
            context_failure_rate,
            constraint_violation_rate,
            drift_rate,
        )

        return ConversationEvaluation(report={
            "case_id": case["id"],
            "num_turns": len(agent_positions),
            "run_count": run_count,
            "metrics": {
                "cross_run_variance": round(cross_run_variance, 4),
                "turn_contradiction_rate": round(turn_contradiction_rate, 4),
                "context_failure_rate": round(context_failure_rate, 4),
                "constraint_violation_rate": round(constraint_violation_rate, 4),
                "drift_rate": round(drift_rate, 4),
                "conv_asi": round(conv_asi, 2),
            },
            "artifacts": {
                "final_responses": final_responses,
            }
        })

    # Private helpers defined below (use EmbeddingVarianceScorer and existing contradiction logic)
    def _compute_turn_contradictions(...) -> float: ...
    def _compute_context_failure(...) -> float: ...
    def _compute_constraint_violations(...) -> float: ...
    def _compute_drift(...) -> float: ...
```

---

## Level 2: New File — `runners/conversation_benchmark.py`

Mirrors `benchmark.py` exactly, but reads `case["messages"]` and routes to `ConversationEvaluator`.

```python
def run_conversation_benchmark_suite(
    suite_path: Path,
    adapter,                         # must have call_messages() method
    run_count: int,
    seed: int,
    weights: ConversationWeights | None = None,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
    embedding_openai_api_key: str | None = None,
    max_cases: int | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
    agent_factory: Callable[[], Any] | None = None,
) -> BenchmarkResult:
    # Same ThreadPoolExecutor pattern as benchmark.py
    # Returns BenchmarkResult with mean_conv_asi, domain_scores, etc.
```

**Report key difference:** Uses `conv_asi` instead of `agent_stability_index` in case reports. Aggregate key: `mean_conv_asi`.

---

## Level 2: Example Conversation Suite

`examples/benchmarks/conversation_suite.json` — 40 cases across 4 domains:
- **memory** (10 cases): User introduces names, numbers, preferences in early turns; later turn asks to recall
- **instruction_following** (10 cases): System prompt gives rules; later user tries to violate; model must hold
- **self_consistency** (10 cases): User asks equivalent questions two turns apart; model should not contradict itself
- **context_reasoning** (10 cases): Each turn builds on previous — model must chain information across turns

---

---

# LEVEL 3: AGENTIC TRACE EVALUATION

## Agent Task JSON Schema

```json
{
  "name": "agent_task_suite_v1",
  "type": "agent",
  "tasks": [
    {
      "id": "agent-search-001",
      "difficulty": "medium",
      "goal": "Find the current price of AAPL stock and tell me if it is above $200.",
      "tools": [
        {
          "name": "search_web",
          "description": "Search the web for a query and return relevant text.",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
          }
        },
        {
          "name": "extract_number",
          "description": "Extract a numeric value from text.",
          "parameters": {
            "type": "object",
            "properties": {
              "text":   {"type": "string"},
              "target": {"type": "string", "description": "What number to extract"}
            },
            "required": ["text", "target"]
          }
        }
      ],
      "reference_trajectory": ["search_web", "extract_number"],
      "expected_answer": "above $200",
      "max_steps": 8,
      "timeout_seconds": 60,
      "sandbox_responses": {
        "search_web":     "AAPL is trading at $227.50 as of today.",
        "extract_number": "227.50"
      },
      "tool_endpoints": {
        "search_web": "https://my-search-api.example.com/search"
      }
    }
  ]
}
```

**Schema fields:**
- `tools`: List of tool definitions — same schema as OpenAI function calling (`name`, `description`, `parameters` as JSON Schema)
- `reference_trajectory`: Ordered list of tool names representing the ideal path
- `expected_answer`: Substring the final answer must contain for `success = True` (case-insensitive match)
- `max_steps`: Maximum tool calls before declaring failure (prevents infinite loops)
- `timeout_seconds`: Per-run time limit
- `sandbox_responses`: `{tool_name: mock_response_string}` — deterministic fallback for every tool
- `tool_endpoints`: Optional `{tool_name: url}` — real HTTP endpoint; tried first, falls back to sandbox

**Anthropic tool format** (auto-converted from the above schema internally):
```json
{"name": "search_web", "description": "...", "input_schema": {...}}
```

---

## Level 3: New Metric Dataclass in `engine/asi.py`

```python
@dataclass(frozen=True)
class AgentWeights:
    """All metrics are failure rates (0.0 = perfect, 1.0 = worst).
    Note: trajectory_consistency, goal_completion_rate etc. are SUCCESS rates,
    so stored here as (1 - success_rate) = failure rate before being passed to calculate()."""
    trajectory_inconsistency: float  # 1 - trajectory_consistency
    goal_failure_rate: float         # 1 - goal_completion_rate
    tool_error_rate: float           # 1 - tool_selection_accuracy
    path_waste_rate: float           # 1 - step_efficiency
    parameter_error_rate: float      # 1 - parameter_fidelity
    fault_failure_rate: float        # 1 - fault_robustness

    def as_tuple(self) -> tuple[float, ...]:
        return (self.trajectory_inconsistency, self.goal_failure_rate,
                self.tool_error_rate, self.path_waste_rate,
                self.parameter_error_rate, self.fault_failure_rate)

DEFAULT_AGENT_WEIGHTS = AgentWeights(0.25, 0.25, 0.20, 0.15, 0.10, 0.05)

class AgentASICalculator:
    def __init__(self, weights: AgentWeights = DEFAULT_AGENT_WEIGHTS) -> None:
        total = sum(weights.as_tuple())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("AgentWeights must sum to 1.0")
        self._weights = weights

    def calculate(
        self,
        trajectory_inconsistency: float,
        goal_failure_rate: float,
        tool_error_rate: float,
        path_waste_rate: float,
        parameter_error_rate: float,
        fault_failure_rate: float,
    ) -> float:
        metrics = (trajectory_inconsistency, goal_failure_rate, tool_error_rate,
                   path_waste_rate, parameter_error_rate, fault_failure_rate)
        penalty = sum(w * m for w, m in zip(self._weights.as_tuple(), metrics))
        return min(max(100.0 * (1.0 - penalty), 0.0), 100.0)
```

---

## Level 3: Trace Data Model — `traces/schema.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    tool_call_id: str          # e.g. "call_abc123" (OpenAI) or "toolu_abc123" (Anthropic)
    tool_name: str             # e.g. "search_web"
    arguments: dict[str, object]  # parsed from JSON
    result: str | None         # tool output string; None if not yet executed
    is_fault_injected: bool    # True if this call was deliberately failed
    error: str | None          # error message if execution failed
    duration_ms: int           # how long execution took

@dataclass
class AgentTrace:
    trace_id: str              # e.g. "trace-{sha256[:16]}"
    task_id: str               # matches AgentTask.id
    run_index: int             # 0-based index of this run within k runs
    goal: str                  # the task goal
    tool_calls: list[ToolCall] # ordered list of all tool calls in this run
    final_answer: str | None   # model's final text response
    success: bool              # final_answer contains expected_answer
    total_steps: int           # len(tool_calls)
    duration_ms: int           # wall clock time for this run
    timed_out: bool            # True if hit timeout_seconds limit

@dataclass
class AgentTask:
    id: str
    difficulty: str
    goal: str
    tools: list[dict]                   # tool definitions (OpenAI schema)
    reference_trajectory: list[str]     # ordered list of expected tool names
    expected_answer: str | None
    max_steps: int
    timeout_seconds: int
    sandbox_responses: dict[str, str]   # {tool_name: mock_response}
    tool_endpoints: dict[str, str]      # {tool_name: http_url} (optional real tools)
```

---

## Level 3: Tool Execution — `traces/sandbox.py`

```python
class SandboxExecutor:
    """Execute a tool call. Priority: real HTTP endpoint → sandbox response → error."""

    def __init__(
        self,
        task: AgentTask,
        fault_rate: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self._task = task
        self._fault_rate = fault_rate
        self._rng = rng or random.Random()

    def execute(self, tool_name: str, arguments: dict, call_id: str) -> ToolCall:
        start = time.monotonic()
        is_fault = self._fault_rate > 0 and self._rng.random() < self._fault_rate

        if is_fault:
            return ToolCall(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                is_fault_injected=True,
                error=f"Simulated tool failure for '{tool_name}'",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        # Try real endpoint first
        endpoint = self._task.tool_endpoints.get(tool_name)
        if endpoint:
            try:
                result = self._call_endpoint(endpoint, tool_name, arguments)
                return ToolCall(tool_call_id=call_id, tool_name=tool_name,
                                arguments=arguments, result=result, is_fault_injected=False,
                                error=None, duration_ms=int((time.monotonic() - start) * 1000))
            except Exception as exc:
                # Fall through to sandbox
                pass

        # Sandbox fallback
        sandbox = self._task.sandbox_responses.get(tool_name)
        if sandbox is not None:
            return ToolCall(tool_call_id=call_id, tool_name=tool_name,
                            arguments=arguments, result=sandbox, is_fault_injected=False,
                            error=None, duration_ms=int((time.monotonic() - start) * 1000))

        # Neither configured
        return ToolCall(tool_call_id=call_id, tool_name=tool_name,
                        arguments=arguments, result=None, is_fault_injected=False,
                        error=f"No sandbox_response or tool_endpoint configured for '{tool_name}'",
                        duration_ms=int((time.monotonic() - start) * 1000))

    def _call_endpoint(self, url: str, tool_name: str, arguments: dict) -> str:
        """HTTP POST to real tool endpoint. Timeout: 30s."""
        data = json.dumps({"tool": tool_name, "arguments": arguments}).encode("utf-8")
        req = urllib_request.Request(url, method="POST", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib_request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
```

---

## Level 3: Adapter Changes — Mode 1 Tool-Calling

### `adapters/openai.py` — add `call_with_tools()`

Uses Chat Completions API (`/v1/chat/completions`) with `tools` parameter.
The main loop is managed by the agent_benchmark runner, NOT inside this method.
This method makes ONE Chat Completions call and returns either:
- `(tool_calls: list[dict], text: None)` — model wants to call tools
- `(tool_calls: [], text: str)` — model gave final answer

```python
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

def call_with_tools(
    self,
    messages: list[dict],
    tools: list[dict],
    rng: random.Random | None = None,
) -> tuple[list[dict], str | None]:
    """One-shot Chat Completions call with tools.
    Returns (tool_calls, None) if model wants tools, or ([], final_text) if done."""
    # Convert to OpenAI format: tools need type="function" wrapper
    openai_tools = [{"type": "function", "function": t} for t in tools]
    payload = {
        "model": self._model,
        "messages": messages,
        "tools": openai_tools,
        "tool_choice": "auto",
    }
    if self._temperature is not None:
        payload["temperature"] = self._temperature

    attempts = self._max_retries + 1
    for attempt in range(attempts):
        self._respect_rate_limit()
        try:
            response = self._chat_sender(payload)
            pt, ct, tt = _extract_usage(response)
            self._track_usage(pt, ct, tt)
            return _extract_tool_calls_or_text(response)
        except Exception:
            if attempt >= self._max_retries:
                raise
            self._usage.retries += 1
            self._sleep_backoff(attempt, rng)
    raise RuntimeError("unreachable retry state")
```

New module-level helper:
```python
def _extract_tool_calls_or_text(response: dict) -> tuple[list[dict], str | None]:
    """Returns (tool_calls, None) or ([], text)."""
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("No choices in OpenAI response")
    message = choices[0].get("message", {})
    finish_reason = choices[0].get("finish_reason")

    tool_calls = message.get("tool_calls")
    if finish_reason == "tool_calls" and isinstance(tool_calls, list):
        return (tool_calls, None)

    content = message.get("content")
    if isinstance(content, str):
        return ([], content)

    raise ValueError("Cannot extract tool calls or text from OpenAI response")
```

### `adapters/anthropic.py` — add `call_with_tools()`

Uses Anthropic Messages API with `tools` parameter. Same one-shot pattern.

```python
def call_with_tools(
    self,
    messages: list[dict],
    tools: list[dict],
    rng: random.Random | None = None,
) -> tuple[list[dict], str | None]:
    """One-shot Anthropic Messages call with tools.
    Returns (tool_use_blocks, None) or ([], text) depending on stop_reason."""
    # Anthropic format: tools use "input_schema" instead of "parameters"
    anthropic_tools = [
        {"name": t["name"], "description": t.get("description", ""),
         "input_schema": t.get("parameters", {})}
        for t in tools
    ]
    system_prompt = None
    filtered_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered_messages.append(m)

    payload: dict = {
        "model": self._model,
        "max_tokens": self._max_tokens,
        "messages": filtered_messages,
        "tools": anthropic_tools,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if self._temperature is not None:
        payload["temperature"] = self._temperature

    attempts = self._max_retries + 1
    for attempt in range(attempts):
        self._respect_rate_limit()
        try:
            response = self._sender(payload)
            it, ot = _extract_usage(response)
            self._track_usage(it, ot)
            return _extract_anthropic_tool_calls_or_text(response)
        except RuntimeError as exc:
            if attempt >= self._max_retries:
                raise
            self._usage.retries += 1
            if "429" in str(exc):
                self._sleep_rate_limit(attempt, rng)
            else:
                self._sleep_backoff(attempt, rng)
    raise RuntimeError("unreachable retry state")
```

New helper:
```python
def _extract_anthropic_tool_calls_or_text(response: dict) -> tuple[list[dict], str | None]:
    content = response.get("content", [])
    stop_reason = response.get("stop_reason")

    if stop_reason == "tool_use":
        tool_use_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        return (tool_use_blocks, None)

    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return ([], block.get("text", ""))

    raise ValueError("Cannot extract tool calls or text from Anthropic response")
```

---

## Level 3: Trajectory Metrics — `engine/trajectory.py`

All functions are pure Python — no external dependencies.

```python
from __future__ import annotations

def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """Standard Levenshtein distance between two sequences of strings."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def trajectory_consistency(traces: list[AgentTrace]) -> float:
    """1.0 = all runs took identical tool sequence. 0.0 = completely different.
    Method: average pairwise (1 - normalized_levenshtein) across all run pairs."""
    sequences = [[tc.tool_name for tc in t.tool_calls] for t in traces]
    if len(sequences) < 2:
        return 1.0
    sims = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            dist = _levenshtein(sequences[i], sequences[j])
            max_len = max(len(sequences[i]), len(sequences[j]), 1)
            sims.append(1.0 - dist / max_len)
    return sum(sims) / len(sims)

def tool_selection_accuracy(traces: list[AgentTrace], reference: list[str]) -> float:
    """Jaccard overlap: |actual ∩ reference| / |actual ∪ reference|, averaged across runs."""
    if not reference:
        return 1.0
    ref_set = set(reference)
    accs = []
    for t in traces:
        actual_set = {tc.tool_name for tc in t.tool_calls}
        intersection = len(actual_set & ref_set)
        union = len(actual_set | ref_set)
        accs.append(intersection / union if union > 0 else 1.0)
    return sum(accs) / len(accs) if accs else 0.0

def step_efficiency(traces: list[AgentTrace], reference_steps: int) -> float:
    """min(reference_steps / mean_actual_steps, 1.0). Penalizes extra steps."""
    if reference_steps <= 0 or not traces:
        return 1.0
    mean_actual = sum(len(t.tool_calls) for t in traces) / len(traces)
    return min(reference_steps / mean_actual, 1.0) if mean_actual > 0 else 1.0

def goal_completion_rate(traces: list[AgentTrace]) -> float:
    """Fraction of runs where trace.success is True."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.success) / len(traces)

def parameter_fidelity(traces: list[AgentTrace], tool_schemas: list[dict]) -> float:
    """Fraction of tool calls where all required parameters were present."""
    schema_by_name = {s["name"]: s.get("parameters", {}) for s in tool_schemas}
    total, valid = 0, 0
    for t in traces:
        for tc in t.tool_calls:
            total += 1
            required = schema_by_name.get(tc.tool_name, {}).get("required", [])
            if all(r in tc.arguments for r in required):
                valid += 1
    return valid / total if total > 0 else 1.0

def fault_robustness(
    fault_traces: list[AgentTrace],
    normal_traces: list[AgentTrace],
) -> float:
    """1 - degradation_from_faults. 1.0 = agent handles failures perfectly."""
    if not fault_traces or not normal_traces:
        return 1.0
    normal_rate = goal_completion_rate(normal_traces)
    fault_rate_val = goal_completion_rate(fault_traces)
    return max(0.0, 1.0 - max(0.0, normal_rate - fault_rate_val))
```

---

## Level 3: Agent Benchmark Runner — `runners/agent_benchmark.py`

The `_run_agent_task()` function implements the tool-calling loop:

```python
def _run_agent_task(
    task: AgentTask,
    adapter,              # OpenAIChatAdapter or AnthropicChatAdapter
    run_index: int,
    seed: int,
    fault_rate: float,
) -> AgentTrace:
    rng = random.Random(seed + run_index * 997)
    executor = SandboxExecutor(task, fault_rate=fault_rate, rng=rng)

    messages = [{"role": "user", "content": task.goal}]
    all_tool_calls: list[ToolCall] = []
    final_answer: str | None = None
    start_ms = int(time.monotonic() * 1000)
    timed_out = False

    for step in range(task.max_steps):
        tool_call_dicts, text = adapter.call_with_tools(messages, task.tools, rng=rng)

        if text is not None:
            # Model is done
            final_answer = text
            break

        # Process each tool call
        for tc_dict in tool_call_dicts:
            # Handle OpenAI format: tc_dict["function"]["name"] / tc_dict["function"]["arguments"]
            # Handle Anthropic format: tc_dict["name"] / tc_dict["input"]
            tool_name, arguments, call_id = _parse_tool_call(tc_dict)
            tool_call = executor.execute(tool_name, arguments, call_id)
            all_tool_calls.append(tool_call)

            # Append to message history so model sees the result
            # OpenAI format — tool result message:
            messages.append(_make_assistant_tool_msg(tc_dict))  # adapter-specific
            messages.append(_make_tool_result_msg(tool_call, adapter))  # adapter-specific
    else:
        timed_out = True  # hit max_steps without final answer

    success = False
    if final_answer and task.expected_answer:
        success = task.expected_answer.lower() in final_answer.lower()
    elif final_answer and not task.expected_answer:
        success = True  # no expected answer = success if any answer given

    duration_ms = int(time.monotonic() * 1000) - start_ms
    return AgentTrace(
        trace_id=f"trace-{hashlib.sha256(f'{task.id}-{run_index}-{seed}'.encode()).hexdigest()[:16]}",
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
```

**Message format helpers** (handle OpenAI vs Anthropic differences):

For OpenAI:
```python
# Assistant message with tool call:
{"role": "assistant", "content": null, "tool_calls": [{...}]}
# Tool result message:
{"role": "tool", "tool_call_id": "call_abc123", "content": "AAPL at $227.50"}
```

For Anthropic:
```python
# Assistant message with tool use:
{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_abc", "name": "search_web", "input": {...}}]}
# Tool result message:
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "AAPL at $227.50"}]}
```

The runner detects adapter type and formats messages accordingly.

**Top-level function:**

```python
def run_agent_benchmark_suite(
    suite_path: Path,
    adapter,
    run_count: int,
    seed: int,
    weights: AgentWeights | None = None,
    fault_rate: float = 0.0,
    max_tasks: int | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
    agent_factory: Callable[[], Any] | None = None,
) -> BenchmarkResult:
    # Load tasks from JSON
    # For each task: run it run_count times → collect AgentTrace list
    # Also run with fault_rate if > 0 (separate set of traces for fault_robustness)
    # Compute all 6 trajectory metrics
    # Compute TraceASI via AgentASICalculator
    # Return BenchmarkResult with aggregate report
```

---

## Level 3: Mode 2 — Custom Agent SDK — `traces/collector.py`

Simple context manager. No monkey-patching. User wraps tool calls explicitly.

```python
from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

class _ToolSpanContext:
    def __init__(self, trace: "_TraceContext", tool_name: str, arguments: dict) -> None:
        self._trace = trace
        self._tool_name = tool_name
        self._arguments = arguments
        self._start_ms = int(time.monotonic() * 1000)
        self.result: str | None = None
        self.error: str | None = None

    def __enter__(self) -> _ToolSpanContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = int(time.monotonic() * 1000) - self._start_ms
        tc = ToolCall(
            tool_call_id=str(uuid.uuid4())[:8],
            tool_name=self._tool_name,
            arguments=self._arguments,
            result=self.result,
            is_fault_injected=False,
            error=str(exc_val) if exc_type else self.error,
            duration_ms=duration_ms,
        )
        self._trace._tool_calls.append(tc)


class _TraceContext:
    def __init__(self, task_id: str, goal: str, run_index: int) -> None:
        self._task_id = task_id
        self._goal = goal
        self._run_index = run_index
        self._tool_calls: list[ToolCall] = []
        self._start_ms = int(time.monotonic() * 1000)
        self.final_answer: str | None = None

    def tool_span(self, tool_name: str, arguments: dict | None = None) -> _ToolSpanContext:
        return _ToolSpanContext(self, tool_name, arguments or {})

    def _to_trace(self, expected_answer: str | None = None) -> AgentTrace:
        success = False
        if self.final_answer and expected_answer:
            success = expected_answer.lower() in self.final_answer.lower()
        elif self.final_answer:
            success = True
        trace_seed = f"{self._task_id}-{self._run_index}"
        return AgentTrace(
            trace_id=f"trace-{hashlib.sha256(trace_seed.encode()).hexdigest()[:16]}",
            task_id=self._task_id,
            run_index=self._run_index,
            goal=self._goal,
            tool_calls=self._tool_calls,
            final_answer=self.final_answer,
            success=success,
            total_steps=len(self._tool_calls),
            duration_ms=int(time.monotonic() * 1000) - self._start_ms,
            timed_out=False,
        )


class TraceCollector:
    """Collect agent traces from custom user-built agents."""

    def __init__(self) -> None:
        self._collected_traces: list[AgentTrace] = []

    @contextmanager
    def trace(self, task_id: str, goal: str = "", run_index: int = 0):
        ctx = _TraceContext(task_id, goal, run_index)
        yield ctx
        self._collected_traces.append(ctx._to_trace())

    def get_traces(self) -> list[AgentTrace]:
        return list(self._collected_traces)

    def clear(self) -> None:
        self._collected_traces.clear()
```

**Usage example for user's custom agent:**
```python
from agent_stability_engine.traces import TraceCollector

collector = TraceCollector()
for run in range(3):
    with collector.trace("task-001", goal="Find AAPL price", run_index=run) as trace:
        with trace.tool_span("search_web", {"query": "AAPL price"}) as span:
            span.result = my_search("AAPL price")
        with trace.tool_span("extract_number", {"text": span.result, "target": "price"}) as span:
            span.result = my_extract(span.result)
        trace.final_answer = f"AAPL is at ${span.result}"

traces = collector.get_traces()
# Now pass traces to compute_trace_metrics(traces, task) from engine/trajectory.py
```

---

## Level 3: Compute Metrics from Traces — New Entry Point

```python
def compute_trace_metrics(
    traces: list[AgentTrace],
    task: AgentTask,
    fault_traces: list[AgentTrace] | None = None,
    weights: AgentWeights | None = None,
) -> dict[str, float]:
    """Compute all 6 TraceASI metrics from collected traces."""
    tc = trajectory_consistency(traces)
    tsa = tool_selection_accuracy(traces, task.reference_trajectory)
    se = step_efficiency(traces, len(task.reference_trajectory))
    gcr = goal_completion_rate(traces)
    pf = parameter_fidelity(traces, task.tools)
    fr = fault_robustness(fault_traces or [], traces)

    calc = AgentASICalculator(weights or DEFAULT_AGENT_WEIGHTS)
    trace_asi = calc.calculate(
        trajectory_inconsistency=1.0 - tc,
        goal_failure_rate=1.0 - gcr,
        tool_error_rate=1.0 - tsa,
        path_waste_rate=1.0 - se,
        parameter_error_rate=1.0 - pf,
        fault_failure_rate=1.0 - fr,
    )

    return {
        "trajectory_consistency": round(tc, 4),
        "tool_selection_accuracy": round(tsa, 4),
        "step_efficiency": round(se, 4),
        "goal_completion_rate": round(gcr, 4),
        "parameter_fidelity": round(pf, 4),
        "fault_robustness": round(fr, 4),
        "trace_asi": round(trace_asi, 2),
    }
```

---

## Web API Changes (`api/main.py`)

### New job types

```python
class JobCreateRequest(BaseModel):
    # Existing fields:
    provider: str
    model: str
    api_key: str
    suite: str
    run_count: int = Field(default=3, ge=1, le=20)
    workers: int = Field(default=3, ge=1, le=10)
    # New fields:
    job_type: Literal["benchmark", "conversation_benchmark", "agent_benchmark"] = "benchmark"
    fault_rate: float = Field(default=0.0, ge=0.0, le=0.5)  # Level 3 only
```

### New DB table (add to `_init_db()`):

```sql
CREATE TABLE IF NOT EXISTS agent_traces (
    id          SERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    task_id     TEXT NOT NULL,
    run_index   INTEGER NOT NULL,
    trace_json  TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

### New endpoint:

```
GET /jobs/{job_id}/traces
→ returns list of AgentTrace JSON objects for that job
→ 404 if job doesn't exist, 400 if job_type != "agent_benchmark"
```

### Routing in `_run_benchmark_report()`:

```python
if job_type == "benchmark":
    result = run_benchmark_suite(...)
elif job_type == "conversation_benchmark":
    result = run_conversation_benchmark_suite(...)
elif job_type == "agent_benchmark":
    result = run_agent_benchmark_suite(...)
    # Also save raw traces to agent_traces table
```

---

## Frontend Changes

Three tabs in the benchmark form:
1. **"Single-turn"** — existing form (no changes)
2. **"Conversation"** — conversation suite upload, same fields + type=conversation hint
3. **"Agent"** — agent task suite upload + fault rate slider (0–50%, default 0%)

Results display:

- **Level 2**: Show ConvASI + bar chart of per-metric scores (cross_run_variance, turn_contradiction_rate, etc.)
- **Level 3**: Show TraceASI + trajectory timeline (per-task: which tools were called in which order across k runs, color-coded by consistency)

New frontend route: `/jobs/[id]/traces` — shows raw trace table with tool names, arguments, results per run.

---

## Sample Agent Tasks — `examples/agent_tasks/sample_tasks.json`

20 tasks across 5 domains:

| Domain | Count | Description |
|--------|-------|-------------|
| `search` | 5 | Web search + extract — tests tool selection and parameter correctness |
| `calculation` | 4 | Multi-step math via tools — tests step efficiency |
| `comparison` | 4 | Retrieve multiple things, compare — tests trajectory consistency |
| `decision` | 4 | Use tools to gather data, then decide — tests goal completion |
| `error_recovery` | 3 | One tool has unreliable sandbox (empty string) — tests fault robustness |

All tasks include `sandbox_responses` and `reference_trajectory`. Domain is encoded in `id`: `search-price-001`, `calc-compound-001`, etc.

---

## Complete File Table

| File | Action | What changes |
|------|--------|-------------|
| `adapters/openai.py` | Modify | Add `call_messages()`, `call_with_tools()`, `_chat_sender()`, `_extract_tool_calls_or_text()`, `_OPENAI_CHAT_URL` constant |
| `adapters/anthropic.py` | Modify | Add `call_messages()`, `call_with_tools()`, `_extract_anthropic_tool_calls_or_text()` |
| `adapters/custom_endpoint.py` | Modify | Add `call_messages()` |
| `adapters/__init__.py` | Modify | Export `TraceCollector` |
| `engine/asi.py` | Modify | Add `ConversationWeights`, `ConvASICalculator`, `AgentWeights`, `AgentASICalculator`, `DEFAULT_CONVERSATION_WEIGHTS`, `DEFAULT_AGENT_WEIGHTS` |
| `engine/conversation.py` | CREATE | `ConversationEvaluator`, `ConversationEvaluation`, all conversation metrics |
| `engine/trajectory.py` | CREATE | `trajectory_consistency`, `tool_selection_accuracy`, `step_efficiency`, `goal_completion_rate`, `parameter_fidelity`, `fault_robustness`, `compute_trace_metrics` |
| `traces/__init__.py` | CREATE | Package init, exports |
| `traces/schema.py` | CREATE | `ToolCall`, `AgentTrace`, `AgentTask` dataclasses |
| `traces/sandbox.py` | CREATE | `SandboxExecutor` — real tool + sandbox fallback + fault injection |
| `traces/collector.py` | CREATE | `TraceCollector`, `_TraceContext`, `_ToolSpanContext` |
| `runners/conversation_benchmark.py` | CREATE | `run_conversation_benchmark_suite()` |
| `runners/agent_benchmark.py` | CREATE | `run_agent_benchmark_suite()`, `_run_agent_task()`, message format helpers |
| `__init__.py` | Modify | Export `ConversationEvaluator`, `ConversationEvaluation`, `AgentTrace`, `AgentTask`, `TraceCollector`, `compute_trace_metrics`, `run_conversation_benchmark_suite`, `run_agent_benchmark_suite` |
| `examples/benchmarks/conversation_suite.json` | CREATE | 40 multi-turn conversation cases |
| `examples/agent_tasks/sample_tasks.json` | CREATE | 20 agent tasks with sandbox_responses |
| `api/main.py` | Modify | New `job_type`, `fault_rate` fields; conversation/agent routing; `agent_traces` table; `GET /jobs/{id}/traces` endpoint |
| Frontend `page.tsx` | Modify | Three tabs (Single-turn / Conversation / Agent), fault rate slider |
| Frontend job detail page | Modify | ConvASI display, trajectory timeline, raw traces link |

**Total: 8 modified files + 9 new files**

---

## Phased Delivery

### Phase 1 — Shared Foundation + Level 2 (Weeks 1–3)

**Week 1:**
- Add `call_messages()` to all three adapters
- Create `engine/conversation.py` (ConversationEvaluator + all 5 metrics)
- Add `ConversationWeights`, `ConvASICalculator` to `engine/asi.py`

**Week 2:**
- Create `runners/conversation_benchmark.py`
- Create `examples/benchmarks/conversation_suite.json` (40 cases)
- Update `api/main.py` — add `conversation_benchmark` job type

**Week 3:**
- Frontend: add Conversation tab
- End-to-end test: run 40 conversation cases with 2 models, compare ConvASI
- Run pytest + mypy

### Phase 2 — Level 3 Mode 1 (API Tool-Calling) (Weeks 4–7)

**Week 4:**
- Create `traces/schema.py` (ToolCall, AgentTrace, AgentTask)
- Create `traces/sandbox.py` (SandboxExecutor: real tool + sandbox fallback + fault injection)
- Create `engine/trajectory.py` (all 6 metric functions)

**Week 5:**
- Add `call_with_tools()` to OpenAI adapter (with Chat Completions endpoint)
- Add `call_with_tools()` to Anthropic adapter (with tool_use format)
- Add message format helpers (OpenAI/Anthropic tool result messages)
- Add `AgentWeights`, `AgentASICalculator` to `engine/asi.py`

**Week 6:**
- Create `runners/agent_benchmark.py` (main loop + parallel execution)
- Create `examples/agent_tasks/sample_tasks.json` (20 tasks)
- Update `api/main.py` — agent_benchmark job type + agent_traces table + /traces endpoint

**Week 7:**
- Frontend: Agent tab + trajectory timeline display
- End-to-end test: run 20 tasks with gpt-4o-mini vs claude-haiku-4-5, compare TraceASI
- Run pytest + mypy

### Phase 3 — Level 3 Mode 2 (Custom Agent SDK) (Weeks 8–9)

**Week 8:**
- Create `traces/collector.py` (TraceCollector + context managers)
- Export from `__init__.py`

**Week 9:**
- Integration test: wrap a simple custom Python agent with TraceCollector
- Document Mode 2 usage in README
- Final pytest + mypy pass

---

## Verification Commands

```bash
# === PHASE 1: Level 2 Conversation ===

# Check adapter call_messages works
python -c "
from agent_stability_engine.adapters import AnthropicChatAdapter
import os
a = AnthropicChatAdapter(model='claude-haiku-4-5', api_key=os.getenv('ANTHROPIC_API_KEY'))
msgs = [{'role': 'user', 'content': 'My name is Alex.'}, {'role': 'assistant', 'content': 'Nice to meet you, Alex!'}, {'role': 'user', 'content': 'What is my name?'}]
print(a.call_messages(msgs))
"

# Run conversation benchmark
python scripts/validate_models.py \
  --models claude-haiku-4-5 gpt-4o-mini \
  --suite examples/benchmarks/conversation_suite.json \
  --mode conversation \
  --run-count 3 \
  --embedding-provider hash \
  --output-dir out/conv-test

# Check ConvASI reported
cat out/conv-test/results.json | python -c "import json,sys; r=json.load(sys.stdin); print(r['mean_conv_asi'])"


# === PHASE 2: Level 3 Agent ===

# Check adapter call_with_tools works
python -c "
from agent_stability_engine.adapters import AnthropicChatAdapter
import os
a = AnthropicChatAdapter(model='claude-haiku-4-5', api_key=os.getenv('ANTHROPIC_API_KEY'))
tools = [{'name': 'calculator', 'description': 'Calculate math', 'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}]
msgs = [{'role': 'user', 'content': 'What is 2 + 2?'}]
tool_calls, text = a.call_with_tools(msgs, tools)
print('tool_calls:', tool_calls, 'text:', text)
"

# Run agent benchmark with sandbox only
python scripts/validate_models.py \
  --models gpt-4o-mini \
  --suite examples/agent_tasks/sample_tasks.json \
  --mode agent \
  --run-count 3 \
  --output-dir out/agent-test

# Run with fault injection
python scripts/validate_models.py \
  --models gpt-4o-mini claude-haiku-4-5 \
  --suite examples/agent_tasks/sample_tasks.json \
  --mode agent \
  --fault-rate 0.2 \
  --run-count 5 \
  --output-dir out/agent-fault-test


# === PHASE 3: Custom Agent SDK ===

# Test TraceCollector context manager
python -c "
from agent_stability_engine.traces import TraceCollector
from agent_stability_engine.engine.trajectory import compute_trace_metrics
from agent_stability_engine.traces.schema import AgentTask

collector = TraceCollector()
task = AgentTask(id='test', difficulty='easy', goal='Test', tools=[], reference_trajectory=['tool_a'], expected_answer='done', max_steps=5, timeout_seconds=30, sandbox_responses={}, tool_endpoints={})

for i in range(3):
    with collector.trace('test', 'Test goal', run_index=i) as t:
        with t.tool_span('tool_a', {'arg': 'val'}) as s:
            s.result = 'result_a'
        t.final_answer = 'done'

metrics = compute_trace_metrics(collector.get_traces(), task)
print(metrics)
"

# Run full test suite
python -m pytest tests/ -v
python -m mypy src
```

---

## Scope Limits (do NOT add in this phase)

- **Multi-agent evaluation** (agents talking to each other) — Level 4, future
- **Real-time streaming** in adapters — adds complexity, no user request
- **Automatic OpenTelemetry export** — users can export traces manually if needed
- **LangChain callback auto-instrumentation** — Mode 2 uses explicit context managers; no monkey-patching
- **Human-in-the-loop evaluation** — manual review UI is separate product
- **Long-horizon tasks** (>50 steps) — not needed for MVP; max_steps = 8 default
