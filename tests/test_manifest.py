from __future__ import annotations

from pathlib import Path

from agent_stability_engine.report.manifest import build_manifest


def test_manifest_hashes_are_deterministic() -> None:
    payload = {"k": "v", "nested": {"a": 1}}
    config = {"seed": 7, "run_count": 5}

    manifest_a = build_manifest(
        command="evaluate",
        output_path=Path("out/eval.json"),
        output_payload=payload,
        input_config=config,
        timestamp_utc="2026-03-02T00:00:00Z",
    )
    manifest_b = build_manifest(
        command="evaluate",
        output_path=Path("out/eval.json"),
        output_payload=payload,
        input_config=config,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    assert manifest_a["output_sha256"] == manifest_b["output_sha256"]
    assert manifest_a["input_config_sha256"] == manifest_b["input_config_sha256"]
    assert manifest_a["manifest_version"] == "0.1.0"
