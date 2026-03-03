from __future__ import annotations

import pytest

from agent_stability_engine.engine.drift import DriftTracker, metrics_from_report


def test_metrics_from_report_extracts_vector() -> None:
    report = {
        "metrics": {
            "semantic_variance": {"raw": 0.1, "normalized": 0.2},
            "contradiction_rate": 0.1,
            "mutation_degradation": 0.2,
            "cross_model_disagreement": 0.3,
            "tool_misuse_frequency": 0.4,
            "goal_misalignment_rate": 0.6,
            "behavior_drift_score": 0.2,
            "agent_stability_index": 80,
        }
    }
    metrics = metrics_from_report(report)
    assert metrics["semantic_variance"] == pytest.approx(0.2)
    assert metrics["goal_misalignment_rate"] == pytest.approx(0.6)
    assert metrics["behavior_drift_score"] == pytest.approx(0.2)
    assert metrics["agent_stability_index"] == pytest.approx(80.0)


def test_metrics_from_report_defaults_new_fields_for_old_report() -> None:
    old_report = {
        "metrics": {
            "semantic_variance": {"raw": 0.1, "normalized": 0.1},
            "contradiction_rate": 0.0,
            "mutation_degradation": 0.0,
            "cross_model_disagreement": 0.0,
            "tool_misuse_frequency": 0.0,
            "agent_stability_index": 95.0,
        }
    }
    metrics = metrics_from_report(old_report)
    assert metrics["goal_misalignment_rate"] == pytest.approx(0.0)
    assert metrics["behavior_drift_score"] == pytest.approx(0.0)


def test_drift_tracker_detects_shift_from_baseline() -> None:
    tracker = DriftTracker(detection_threshold=0.1)
    current = {
        "semantic_variance": 0.6,
        "contradiction_rate": 0.4,
        "mutation_degradation": 0.5,
        "cross_model_disagreement": 0.5,
        "tool_misuse_frequency": 0.3,
        "agent_stability_index": 45,
    }
    baseline = [
        {
            "semantic_variance": 0.2,
            "contradiction_rate": 0.1,
            "mutation_degradation": 0.2,
            "cross_model_disagreement": 0.2,
            "tool_misuse_frequency": 0.1,
            "agent_stability_index": 80,
        }
    ]
    analysis = tracker.compare(current, baseline)
    assert analysis.drift_detected is True
    assert analysis.drift_score > 0.1
    assert "agent_stability_index" in analysis.metric_deltas
