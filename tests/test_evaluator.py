from __future__ import annotations

import random

import pytest

from agent_stability_engine.engine.alignment import GoalSpec
from agent_stability_engine.engine.evaluator import StabilityEvaluator


def _agent(prompt: str, rng: random.Random) -> str:
    options = ["stable", "variant", "alt"]
    return f"{prompt}:{options[rng.randrange(len(options))]}"


def test_evaluator_emits_full_metrics() -> None:
    evaluator = StabilityEvaluator()
    result = evaluator.evaluate(
        prompt="Evaluate resilience of this answer",
        agent_fn=_agent,
        run_count=5,
        seed=123,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    report = result.report
    assert report["schema_version"] == "0.1.0"
    assert report["timestamp_utc"] == "2026-03-02T00:00:00Z"

    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert "semantic_variance" in metrics
    assert "contradiction_rate" in metrics
    assert "mutation_degradation" in metrics
    assert "cross_model_disagreement" in metrics
    assert "tool_misuse_frequency" in metrics
    assert "goal_misalignment_rate" in metrics
    assert "behavior_drift_score" in metrics
    assert "agent_stability_index" in metrics
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    notes = artifacts["notes"]
    assert isinstance(notes, str)
    assert "primary_failure=" in notes
    assert "failure_severity=" in notes
    assert "asi_profile=" in notes


def test_evaluator_run_id_reproducible_for_same_inputs() -> None:
    evaluator = StabilityEvaluator()
    result_a = evaluator.evaluate(
        prompt="same prompt",
        agent_fn=_agent,
        run_count=3,
        seed=99,
        timestamp_utc="2026-03-02T00:00:00Z",
    )
    result_b = evaluator.evaluate(
        prompt="same prompt",
        agent_fn=_agent,
        run_count=3,
        seed=99,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    assert result_a.report["run_id"] == result_b.report["run_id"]


def test_evaluator_mutation_degradation_high_for_brittle_agent() -> None:
    evaluator = StabilityEvaluator()

    def brittle_agent(prompt: str, _rng: random.Random) -> str:
        if prompt == "base prompt":
            return "baseline answer"
        return "completely unrelated output"

    result = evaluator.evaluate(
        prompt="base prompt",
        agent_fn=brittle_agent,
        run_count=2,
        seed=5,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    metrics = result.report["metrics"]
    assert isinstance(metrics, dict)
    degradation = metrics["mutation_degradation"]
    assert isinstance(degradation, float)
    assert degradation > 0.8


def test_evaluator_uses_cross_model_arbitration_when_shadows_present() -> None:
    evaluator = StabilityEvaluator()

    def primary(prompt: str, _rng: random.Random) -> str:
        return f"{prompt} stable summary"

    def shadow_a(prompt: str, _rng: random.Random) -> str:
        return f"{prompt} stable summary with detail"

    def shadow_b(_prompt: str, _rng: random.Random) -> str:
        return "unrelated SQL optimization answer"

    result = evaluator.evaluate(
        prompt="explain checksum value",
        agent_fn=primary,
        run_count=3,
        seed=8,
        shadow_agents={"shadow_a": shadow_a, "shadow_b": shadow_b},
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    metrics = result.report["metrics"]
    artifacts = result.report["artifacts"]
    assert isinstance(metrics, dict)
    assert isinstance(artifacts, dict)
    disagreement = metrics["cross_model_disagreement"]
    assert isinstance(disagreement, float)
    assert disagreement > 0
    notes = artifacts["notes"]
    assert isinstance(notes, str)
    assert "arbitration=consensus:" in notes


def test_evaluator_mutation_limit_caps_agent_calls() -> None:
    call_count = 0

    def counting_agent(_prompt: str, _rng: random.Random) -> str:
        nonlocal call_count
        call_count += 1
        return "deterministic output"

    evaluator = StabilityEvaluator(mutation_sample_limit=4)
    evaluator.evaluate(
        prompt="counting prompt",
        agent_fn=counting_agent,
        run_count=3,
        seed=2,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    # run_count baseline calls + mutation-limited degradation calls
    assert call_count == 7


def test_evaluator_goal_alignment_and_drift_integration() -> None:
    evaluator = StabilityEvaluator()
    goal = GoalSpec(required_keywords=("checksum",), forbidden_patterns=(r"drop\s+table",))
    baseline_report = {
        "metrics": {
            "semantic_variance": {"raw": 0.01, "normalized": 0.1},
            "contradiction_rate": 0.0,
            "mutation_degradation": 0.1,
            "cross_model_disagreement": 0.0,
            "tool_misuse_frequency": 0.0,
            "goal_misalignment_rate": 0.0,
            "behavior_drift_score": 0.0,
            "agent_stability_index": 90.0,
        }
    }

    def misaligned_agent(_prompt: str, _rng: random.Random) -> str:
        return "drop table accounts immediately"

    result = evaluator.evaluate(
        prompt="Explain checksum validation",
        agent_fn=misaligned_agent,
        run_count=3,
        seed=4,
        timestamp_utc="2026-03-02T00:00:00Z",
        goal_spec=goal,
        baseline_reports=[baseline_report],
    )

    metrics = result.report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["goal_misalignment_rate"] == pytest.approx(1.0)
    drift_score = metrics["behavior_drift_score"]
    assert isinstance(drift_score, float)
    assert drift_score >= 0.0


def test_evaluator_includes_agent_usage_when_available() -> None:
    class UsageAwareAgent:
        def __init__(self) -> None:
            self._calls = 0

        def __call__(self, _prompt: str, _rng: random.Random) -> str:
            self._calls += 1
            return "stable response"

        def usage_snapshot(self) -> dict[str, object]:
            return {
                "provider": "mock",
                "model": "mock-model",
                "requests": self._calls,
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "estimated_cost_usd": 0.001,
            }

    agent = UsageAwareAgent()
    evaluator = StabilityEvaluator(mutation_sample_limit=1)
    result = evaluator.evaluate(
        prompt="explain checksum",
        agent_fn=agent,
        run_count=2,
        seed=1,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    artifacts = result.report["artifacts"]
    assert isinstance(artifacts, dict)
    usage = artifacts["usage"]
    assert isinstance(usage, dict)
    assert usage["provider"] == "mock"
