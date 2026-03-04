from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_stability_engine.engine.asi import ASIProfile
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.stats import (
    one_sample_threshold_significance,
    summarize_mean_confidence,
)
from agent_stability_engine.runners.benchmark import run_benchmark_suite


@dataclass(frozen=True)
class BenchmarkRegressionResult:
    report: dict[str, object]


def run_benchmark_regression(
    suite_path: Path,
    baseline_path: Path,
    agent_fn: Callable[..., str],
    run_count: int,
    seed: int,
    timestamp_utc: str | None = None,
    asi_profile: ASIProfile = ASIProfile.BALANCED,
    mutation_sample_limit: int | None = None,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
    embedding_model: str | None = None,
    embedding_openai_api_key: str | None = None,
) -> BenchmarkRegressionResult:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    minimum_mean_asi = _read_number(baseline, "minimum_mean_asi")
    allowed_drop = _read_number(baseline, "allowed_drop", default=0.0)
    expected_suite_name = _read_string(baseline, "suite_name", default="")
    require_significance = _read_bool(baseline, "require_significance", default=False)
    significance_alpha = _read_number(baseline, "significance_alpha", default=0.05)

    benchmark = run_benchmark_suite(
        suite_path=suite_path,
        agent_fn=agent_fn,
        run_count=run_count,
        seed=seed,
        timestamp_utc=timestamp_utc,
        asi_profile=asi_profile,
        mutation_sample_limit=mutation_sample_limit,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_openai_api_key=embedding_openai_api_key,
    )
    benchmark_report = benchmark.report
    observed_mean_asi = _read_number(benchmark_report, "mean_asi")
    observed_case_asi = _read_number_list(benchmark_report, "case_asi_values")
    if not observed_case_asi:
        observed_case_asi = [observed_mean_asi]
    observed_asi_statistics = summarize_mean_confidence(observed_case_asi)

    effective_min = minimum_mean_asi - allowed_drop
    passed = observed_mean_asi >= effective_min
    threshold_significance = one_sample_threshold_significance(
        observed_asi_statistics,
        effective_min,
        alpha=significance_alpha,
    )
    if require_significance:
        passed = passed and bool(threshold_significance["significant_pass"])

    suite_name = _read_string(benchmark_report, "suite_name", default="")
    suite_name_match = (not expected_suite_name) or (suite_name == expected_suite_name)
    passed = passed and suite_name_match

    regression_report: dict[str, object] = {
        "suite": str(suite_path),
        "baseline": str(baseline_path),
        "suite_name": suite_name,
        "expected_suite_name": expected_suite_name,
        "suite_name_match": suite_name_match,
        "minimum_mean_asi": minimum_mean_asi,
        "allowed_drop": allowed_drop,
        "effective_minimum_mean_asi": effective_min,
        "observed_mean_asi": observed_mean_asi,
        "observed_asi_statistics": observed_asi_statistics.to_dict(),
        "threshold_significance": threshold_significance,
        "require_significance": require_significance,
        "significance_alpha": significance_alpha,
        "delta_vs_minimum": observed_mean_asi - effective_min,
        "passed": passed,
        "benchmark_report": benchmark_report,
    }
    return BenchmarkRegressionResult(report=regression_report)


def _read_number(payload: dict[str, object], key: str, default: float | None = None) -> float:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if default is None:
        msg = f"missing numeric field: {key}"
        raise ValueError(msg)
    return default


def _read_string(payload: dict[str, object], key: str, default: str | None = None) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    if default is None:
        msg = f"missing string field: {key}"
        raise ValueError(msg)
    return default


def _read_bool(payload: dict[str, object], key: str, default: bool | None = None) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if default is None:
        msg = f"missing boolean field: {key}"
        raise ValueError(msg)
    return default


def _read_number_list(payload: dict[str, object], key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    parsed: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            parsed.append(float(item))
    return parsed
