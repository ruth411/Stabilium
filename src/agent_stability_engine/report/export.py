from __future__ import annotations

from datetime import datetime, timezone

from agent_stability_engine.report.attestation import build_attestation, sha256_payload


def build_export_bundle(
    *,
    input_report: dict[str, object],
    history_reports: list[dict[str, object]] | None = None,
    timestamp_utc: str | None = None,
    methodology_version: str = "asi-methodology-0.2.0",
    signing_key: str | None = None,
) -> dict[str, object]:
    created_at = timestamp_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    history = history_reports or []
    report_type = _report_type(input_report)
    model_info = _model_info(input_report)
    current_asi = _extract_asi(input_report)
    current_confidence = _extract_confidence(input_report)
    trend = _build_trend(current_report=input_report, history_reports=history)

    summary: dict[str, object] = {
        "report_type": report_type,
        "model_provider": model_info.get("provider"),
        "model_name": model_info.get("model"),
        "suite_name": _extract_suite_name(input_report),
        "asi_score": current_asi,
        "asi_confidence": current_confidence,
    }
    metrics: dict[str, object] = {
        "agent_stability_index": current_asi,
        "agent_stability_index_confidence": current_confidence,
        "trend": trend,
    }
    source: dict[str, object] = {
        "input_report_sha256": sha256_payload(input_report),
        "history_report_count": len(history),
    }
    methodology: dict[str, object] = {
        "version": methodology_version,
        "confidence_method": _extract_confidence_method(current_confidence),
        "significance_method": _extract_significance_method(input_report),
    }

    body: dict[str, object] = {
        "bundle_version": "0.1.0",
        "created_at_utc": created_at,
        "summary": summary,
        "metrics": metrics,
        "methodology": methodology,
        "source": source,
    }
    attestation = build_attestation(
        payload=body,
        created_at_utc=created_at,
        methodology_version=methodology_version,
        signing_key=signing_key,
    )
    body["attestation"] = attestation
    return body


def _report_type(report: dict[str, object]) -> str:
    if "benchmark_report" in report:
        return "regression"
    if "mean_asi" in report:
        return "benchmark"
    if "metrics" in report and "inputs" in report:
        return "evaluation"
    return "unknown"


def _extract_asi(report: dict[str, object]) -> float:
    if "benchmark_report" in report and isinstance(report.get("benchmark_report"), dict):
        nested = report["benchmark_report"]
        if isinstance(nested, dict):
            return _extract_asi(nested)
    mean_asi = report.get("mean_asi")
    if isinstance(mean_asi, (int, float)):
        return float(mean_asi)
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        asi = metrics.get("agent_stability_index")
        if isinstance(asi, (int, float)):
            return float(asi)
    return 0.0


def _extract_confidence(report: dict[str, object]) -> dict[str, object] | None:
    if "benchmark_report" in report and isinstance(report.get("benchmark_report"), dict):
        nested = report["benchmark_report"]
        if isinstance(nested, dict):
            nested_conf = _extract_confidence(nested)
            if nested_conf is not None:
                return nested_conf

    stats = report.get("asi_statistics")
    if isinstance(stats, dict):
        return dict(stats)

    observed_stats = report.get("observed_asi_statistics")
    if isinstance(observed_stats, dict):
        return dict(observed_stats)

    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        confidence = metrics.get("agent_stability_index_confidence")
        if isinstance(confidence, dict):
            return dict(confidence)

    return None


def _extract_suite_name(report: dict[str, object]) -> str | None:
    suite_name = report.get("suite_name")
    if isinstance(suite_name, str):
        return suite_name
    benchmark_report = report.get("benchmark_report")
    if isinstance(benchmark_report, dict):
        nested_suite_name = benchmark_report.get("suite_name")
        if isinstance(nested_suite_name, str):
            return nested_suite_name
    return None


def _model_info(report: dict[str, object]) -> dict[str, str | None]:
    # For benchmark/regression reports, attempt extraction from first case usage artifact.
    report_for_scan = report
    nested = report.get("benchmark_report")
    if isinstance(nested, dict):
        report_for_scan = nested

    cases = report_for_scan.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            case_report = case.get("report")
            if not isinstance(case_report, dict):
                continue
            artifacts = case_report.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            usage = artifacts.get("usage")
            if not isinstance(usage, dict):
                continue
            provider = usage.get("provider")
            model = usage.get("model")
            return {
                "provider": str(provider) if isinstance(provider, str) else None,
                "model": str(model) if isinstance(model, str) else None,
            }
    return {"provider": None, "model": None}


def _extract_timestamp(report: dict[str, object]) -> str | None:
    timestamp = report.get("timestamp_utc")
    if isinstance(timestamp, str):
        return timestamp
    created = report.get("created_at_utc")
    if isinstance(created, str):
        return created
    return None


def _build_trend(
    *,
    current_report: dict[str, object],
    history_reports: list[dict[str, object]],
) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for report in history_reports:
        points.append(
            {
                "timestamp_utc": _extract_timestamp(report),
                "asi_score": _extract_asi(report),
            }
        )
    points.append(
        {
            "timestamp_utc": _extract_timestamp(current_report),
            "asi_score": _extract_asi(current_report),
        }
    )

    sorted_points = sorted(
        points,
        key=lambda point: str(point.get("timestamp_utc") or ""),
    )
    if len(sorted_points) >= 2:
        previous = sorted_points[-2]["asi_score"]
        current = sorted_points[-1]["asi_score"]
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            delta = float(current) - float(previous)
        else:
            delta = None
    else:
        delta = None

    if isinstance(delta, float):
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
        else:
            direction = "flat"
    else:
        direction = "flat"

    return {
        "points": sorted_points,
        "delta_vs_previous": delta,
        "direction": direction,
    }


def _extract_confidence_method(confidence: dict[str, object] | None) -> str:
    if not isinstance(confidence, dict):
        return "unknown"
    method = confidence.get("method")
    if isinstance(method, str):
        return method
    return "unknown"


def _extract_significance_method(report: dict[str, object]) -> str:
    threshold = report.get("threshold_significance")
    if not isinstance(threshold, dict):
        return "unknown"
    method = threshold.get("method")
    if isinstance(method, str):
        return method
    return "unknown"
