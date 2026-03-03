from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftAnalysis:
    drift_score: float
    drift_detected: bool
    metric_deltas: dict[str, float]


class DriftTracker:
    """Compares current metric vector against historical baseline vectors."""

    def __init__(self, detection_threshold: float = 0.15) -> None:
        self._detection_threshold = detection_threshold

    def compare(
        self,
        current_metrics: dict[str, float],
        baseline_metrics: list[dict[str, float]],
    ) -> DriftAnalysis:
        if not baseline_metrics:
            return DriftAnalysis(drift_score=0.0, drift_detected=False, metric_deltas={})

        keys = sorted(current_metrics.keys())
        baseline_means: dict[str, float] = {}
        for key in keys:
            values = [metrics.get(key, 0.0) for metrics in baseline_metrics]
            baseline_means[key] = sum(values) / len(values)

        deltas = {key: current_metrics[key] - baseline_means[key] for key in keys}
        normalized_deltas: list[float] = []
        for key, delta in deltas.items():
            if key == "agent_stability_index":
                normalized_deltas.append(abs(delta) / 100.0)
            else:
                normalized_deltas.append(abs(delta))

        drift_score = sum(normalized_deltas) / len(normalized_deltas) if normalized_deltas else 0.0
        return DriftAnalysis(
            drift_score=min(max(drift_score, 0.0), 1.0),
            drift_detected=drift_score >= self._detection_threshold,
            metric_deltas=deltas,
        )


def metrics_from_report(report: dict[str, object]) -> dict[str, float]:
    metrics_obj = report.get("metrics")
    if not isinstance(metrics_obj, dict):
        return {}

    semantic_obj = metrics_obj.get("semantic_variance")
    semantic = 0.0
    if isinstance(semantic_obj, dict):
        normalized = semantic_obj.get("normalized")
        if isinstance(normalized, (int, float)):
            semantic = float(normalized)

    def _num(name: str) -> float:
        value = metrics_obj.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    return {
        "semantic_variance": semantic,
        "contradiction_rate": _num("contradiction_rate"),
        "mutation_degradation": _num("mutation_degradation"),
        "cross_model_disagreement": _num("cross_model_disagreement"),
        "tool_misuse_frequency": _num("tool_misuse_frequency"),
        "goal_misalignment_rate": _num("goal_misalignment_rate"),
        "behavior_drift_score": _num("behavior_drift_score"),
        "agent_stability_index": _num("agent_stability_index"),
    }
