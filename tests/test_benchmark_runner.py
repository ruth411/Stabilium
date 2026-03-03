from __future__ import annotations

import random
from pathlib import Path

from agent_stability_engine.runners.benchmark import run_benchmark_suite


def _agent(prompt: str, rng: random.Random) -> str:
    variants = ["x", "y", "z"]
    return f"{prompt}:{variants[rng.randrange(len(variants))]}"


def test_benchmark_runner_executes_all_cases() -> None:
    suite = Path("examples/benchmarks/default_suite.json")
    result = run_benchmark_suite(
        suite_path=suite,
        agent_fn=_agent,
        run_count=4,
        seed=7,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    report = result.report
    assert str(report["benchmark_id"]).startswith("bench-")
    assert report["suite_name"] == "default_week3_suite"
    assert len(str(report["suite_sha256"])) == 64
    assert report["asi_profile"] == "balanced"
    assert report["mutation_limit"] is None
    assert report["num_cases"] == 3
    assert "mean_asi" in report

    cases = report["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 3
    first_case = cases[0]
    assert isinstance(first_case, dict)
    assert "prompt_sha256" in first_case
