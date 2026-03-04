from __future__ import annotations

from agent_stability_engine.engine.stats import (
    compare_sample_means,
    one_sample_threshold_significance,
    summarize_mean_confidence,
)


def test_summarize_mean_confidence_includes_ci() -> None:
    summary = summarize_mean_confidence([80.0, 82.0, 84.0, 86.0])
    assert summary.sample_size == 4
    assert summary.mean == 83.0
    assert summary.ci_low <= summary.mean <= summary.ci_high
    assert summary.std_dev > 0


def test_one_sample_threshold_significance_detects_pass() -> None:
    summary = summarize_mean_confidence([80.0, 82.0, 84.0, 86.0])
    result = one_sample_threshold_significance(summary, threshold=70.0, alpha=0.05)
    assert result["significant_pass"] is True
    assert result["p_value"] < 0.05


def test_compare_sample_means_flags_significant_difference() -> None:
    left = [90.0, 91.0, 92.0, 93.0]
    right = [80.0, 81.0, 82.0, 83.0]
    result = compare_sample_means(left, right, alpha=0.05)
    assert result["better_sample"] == "left"
    assert result["significant_difference"] is True
    assert result["p_value"] < 0.05
