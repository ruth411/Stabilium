from __future__ import annotations

import hashlib

import pytest
from jsonschema import ValidationError

from agent_stability_engine.report.schema import REPORT_SCHEMA_VERSION, validate_report


def _valid_report() -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": "run-0001",
        "timestamp_utc": "2026-03-02T21:00:00Z",
        "inputs": {
            "run_count": 5,
            "seed": 42,
            "prompt_hash": hashlib.sha256(b"test prompt").hexdigest(),
        },
        "metrics": {
            "semantic_variance": {"raw": 0.12, "normalized": 0.24},
        },
        "artifacts": {
            "outputs": ["A", "B", "C"],
        },
    }


def test_report_schema_accepts_valid_payload() -> None:
    report = _valid_report()
    validate_report(report)


def test_report_schema_rejects_invalid_payload() -> None:
    report = _valid_report()
    report["metrics"] = {"semantic_variance": {"raw": -1, "normalized": 2}}

    with pytest.raises(ValidationError):
        validate_report(report)
