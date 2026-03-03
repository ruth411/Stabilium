from __future__ import annotations

import random
from pathlib import Path

from agent_stability_engine.runners.regression import run_benchmark_regression


def _agent(prompt: str, rng: random.Random) -> str:
    variants = ["stable", "stable", "stable-ish"]
    return f"{prompt}:{variants[rng.randrange(len(variants))]}"


def test_regression_runner_passes_with_permissive_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "pass.baseline.json"
    baseline.write_text(
        '{"suite_name":"reasoning_suite_v1","minimum_mean_asi":0.0,"allowed_drop":0.0}',
        encoding="utf-8",
    )
    result = run_benchmark_regression(
        suite_path=Path("examples/benchmarks/reasoning_suite.json"),
        baseline_path=baseline,
        agent_fn=_agent,
        run_count=3,
        seed=7,
        timestamp_utc="2026-03-02T00:00:00Z",
    )
    report = result.report
    assert report["suite_name_match"] is True
    assert report["passed"] is True
    assert "observed_mean_asi" in report


def test_regression_runner_fails_for_strict_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "strict.baseline.json"
    baseline.write_text(
        '{"suite_name":"reasoning_suite_v1","minimum_mean_asi":99.9,"allowed_drop":0.0}',
        encoding="utf-8",
    )
    result = run_benchmark_regression(
        suite_path=Path("examples/benchmarks/reasoning_suite.json"),
        baseline_path=baseline,
        agent_fn=_agent,
        run_count=3,
        seed=7,
        timestamp_utc="2026-03-02T00:00:00Z",
    )
    assert result.report["passed"] is False
