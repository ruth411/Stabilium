from __future__ import annotations

from agent_stability_engine.engine.asi import (
    DEFAULT_AGENT_WEIGHTS,
    AgentASICalculator,
    AgentWeights,
)
from agent_stability_engine.traces.schema import AgentTask, AgentTrace

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """Standard Levenshtein edit distance between two string sequences."""
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


# ---------------------------------------------------------------------------
# Metric functions (all return values in [0.0, 1.0])
# ---------------------------------------------------------------------------


def trajectory_consistency(traces: list[AgentTrace]) -> float:
    """Fraction of similarity across all run pairs based on tool-call sequences.

    1.0 = all runs took identical tool sequences. 0.0 = completely different.
    Method: average pairwise (1 - normalised Levenshtein) across all pairs.
    """
    sequences = [[tc.tool_name for tc in t.tool_calls] for t in traces]
    if len(sequences) < 2:
        return 1.0
    sims: list[float] = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            dist = _levenshtein(sequences[i], sequences[j])
            max_len = max(len(sequences[i]), len(sequences[j]), 1)
            sims.append(1.0 - dist / max_len)
    return sum(sims) / len(sims)


def tool_selection_accuracy(traces: list[AgentTrace], reference: list[str]) -> float:
    """Jaccard overlap between actual tool set and reference tool set, averaged across runs.

    |actual ∩ reference| / |actual ∪ reference|
    """
    if not reference:
        return 1.0
    ref_set = set(reference)
    accs: list[float] = []
    for t in traces:
        actual_set = {tc.tool_name for tc in t.tool_calls}
        intersection = len(actual_set & ref_set)
        union = len(actual_set | ref_set)
        accs.append(intersection / union if union > 0 else 1.0)
    return sum(accs) / len(accs) if accs else 0.0


def step_efficiency(traces: list[AgentTrace], reference_steps: int) -> float:
    """Ratio of reference steps to mean actual steps, capped at 1.0.

    Penalises runs that use more steps than the reference trajectory.
    """
    if reference_steps <= 0 or not traces:
        return 1.0
    mean_actual = sum(len(t.tool_calls) for t in traces) / len(traces)
    return min(reference_steps / mean_actual, 1.0) if mean_actual > 0 else 1.0


def goal_completion_rate(traces: list[AgentTrace]) -> float:
    """Fraction of runs where trace.success is True."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.success) / len(traces)


def parameter_fidelity(traces: list[AgentTrace], tool_schemas: list[dict[str, object]]) -> float:
    """Fraction of tool calls where all required parameters were present."""
    schema_by_name: dict[str, dict[str, object]] = {}
    for schema in tool_schemas:
        name = schema.get("name")
        if isinstance(name, str):
            params = schema.get("parameters", {})
            schema_by_name[name] = params if isinstance(params, dict) else {}

    total = 0
    valid = 0
    for t in traces:
        for tc in t.tool_calls:
            total += 1
            schema = schema_by_name.get(tc.tool_name, {})
            required = schema.get("required", [])
            if isinstance(required, list) and all(r in tc.arguments for r in required):
                valid += 1
    return valid / total if total > 0 else 1.0


def fault_robustness(
    fault_traces: list[AgentTrace],
    normal_traces: list[AgentTrace],
) -> float:
    """Measures how well the agent handles tool failures.

    1.0 = no degradation when faults are injected.
    0.0 = complete failure under fault conditions.
    """
    if not fault_traces or not normal_traces:
        return 1.0
    normal_rate = goal_completion_rate(normal_traces)
    fault_rate_val = goal_completion_rate(fault_traces)
    degradation = max(0.0, normal_rate - fault_rate_val)
    return max(0.0, 1.0 - degradation)


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------


def compute_trace_metrics(
    traces: list[AgentTrace],
    task: AgentTask,
    fault_traces: list[AgentTrace] | None = None,
    weights: AgentWeights | None = None,
) -> dict[str, float]:
    """Compute all 6 trajectory metrics plus trace_asi from collected traces.

    Args:
        traces: Collected AgentTrace objects (normal runs).
        task: The AgentTask definition (for reference_trajectory and tool schemas).
        fault_traces: Optional traces from fault-injected runs (for fault_robustness).
        weights: Custom AgentWeights; defaults to DEFAULT_AGENT_WEIGHTS.

    Returns:
        Dict with keys: trajectory_consistency, tool_selection_accuracy, step_efficiency,
        goal_completion_rate, parameter_fidelity, fault_robustness, trace_asi.
    """
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
