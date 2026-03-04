from __future__ import annotations

from agent_stability_engine.report.export import build_export_bundle


def _benchmark_report(mean_asi: float, timestamp: str) -> dict[str, object]:
    return {
        "suite_name": "reasoning_suite_v1",
        "mean_asi": mean_asi,
        "asi_statistics": {
            "sample_size": 3,
            "mean": mean_asi,
            "std_dev": 1.0,
            "std_error": 0.577,
            "confidence_level": 0.95,
            "ci_low": mean_asi - 1.0,
            "ci_high": mean_asi + 1.0,
            "method": "normal_approx",
        },
        "timestamp_utc": timestamp,
    }


def test_build_export_bundle_includes_summary_trend_and_attestation() -> None:
    current = _benchmark_report(84.0, "2026-03-04T00:00:00Z")
    history = [
        _benchmark_report(80.0, "2026-03-01T00:00:00Z"),
        _benchmark_report(82.0, "2026-03-03T00:00:00Z"),
    ]
    bundle = build_export_bundle(
        input_report=current,
        history_reports=history,
        timestamp_utc="2026-03-04T12:00:00Z",
        signing_key="test-key",
    )

    assert bundle["bundle_version"] == "0.1.0"
    assert bundle["created_at_utc"] == "2026-03-04T12:00:00Z"
    summary = bundle["summary"]
    assert isinstance(summary, dict)
    assert summary["report_type"] == "benchmark"
    assert summary["asi_score"] == 84.0

    metrics = bundle["metrics"]
    assert isinstance(metrics, dict)
    trend = metrics["trend"]
    assert isinstance(trend, dict)
    assert trend["direction"] == "up"
    assert trend["delta_vs_previous"] == 2.0

    attestation = bundle["attestation"]
    assert isinstance(attestation, dict)
    assert attestation["signed"] is True
    assert isinstance(attestation["signature_hmac_sha256"], str)


def test_build_export_bundle_for_regression_uses_nested_benchmark() -> None:
    regression = {
        "suite_name": "reasoning_suite_v1",
        "observed_mean_asi": 83.0,
        "observed_asi_statistics": {
            "sample_size": 3,
            "mean": 83.0,
            "std_dev": 1.0,
            "std_error": 0.577,
            "confidence_level": 0.95,
            "ci_low": 82.0,
            "ci_high": 84.0,
            "method": "normal_approx",
        },
        "threshold_significance": {
            "method": "one_sided_z_approx",
            "alpha": 0.05,
            "threshold": 75.0,
            "p_value": 0.001,
            "significant_pass": True,
        },
        "benchmark_report": _benchmark_report(83.0, "2026-03-04T00:00:00Z"),
    }
    bundle = build_export_bundle(
        input_report=regression,
        history_reports=[],
        timestamp_utc="2026-03-04T12:00:00Z",
    )
    summary = bundle["summary"]
    assert isinstance(summary, dict)
    assert summary["report_type"] == "regression"
    assert summary["asi_score"] == 83.0
