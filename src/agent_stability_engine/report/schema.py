from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

REPORT_SCHEMA_VERSION = "0.1.0"

REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ase.dev/schemas/report-0.1.0.json",
    "title": "Agent Stability Report",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "run_id", "timestamp_utc", "inputs", "metrics", "artifacts"],
    "properties": {
        "schema_version": {"type": "string", "const": REPORT_SCHEMA_VERSION},
        "run_id": {"type": "string", "minLength": 1},
        "timestamp_utc": {"type": "string", "format": "date-time"},
        "inputs": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_count", "seed", "prompt_hash"],
            "properties": {
                "run_count": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer"},
                "prompt_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            },
        },
        "metrics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["semantic_variance"],
            "properties": {
                "semantic_variance": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["raw", "normalized"],
                    "properties": {
                        "raw": {"type": "number", "minimum": 0},
                        "normalized": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "contradiction_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "mutation_degradation": {"type": "number", "minimum": 0, "maximum": 1},
                "cross_model_disagreement": {"type": "number", "minimum": 0, "maximum": 1},
                "tool_misuse_frequency": {"type": "number", "minimum": 0, "maximum": 1},
                "incorrectness_rate": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "goal_misalignment_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "behavior_drift_score": {"type": "number", "minimum": 0, "maximum": 1},
                "long_horizon_instability": {"type": "number", "minimum": 0, "maximum": 1},
                "agent_stability_index": {"type": "number", "minimum": 0, "maximum": 100},
                "agent_stability_index_confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "sample_size",
                        "mean",
                        "std_dev",
                        "std_error",
                        "confidence_level",
                        "ci_low",
                        "ci_high",
                        "method",
                    ],
                    "properties": {
                        "sample_size": {"type": "integer", "minimum": 1},
                        "mean": {"type": "number"},
                        "std_dev": {"type": "number", "minimum": 0},
                        "std_error": {"type": "number", "minimum": 0},
                        "confidence_level": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "exclusiveMaximum": 1,
                        },
                        "ci_low": {"type": "number"},
                        "ci_high": {"type": "number"},
                        "method": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "artifacts": {
            "type": "object",
            "additionalProperties": False,
            "required": ["outputs"],
            "properties": {
                "outputs": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "usage": {"type": "object"},
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(REPORT_SCHEMA)


def validate_report(report: dict[str, Any]) -> None:
    """Raises jsonschema.ValidationError if invalid."""
    _VALIDATOR.validate(report)
