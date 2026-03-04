from __future__ import annotations

import json
from pathlib import Path

from agent_stability_engine.cli import main


def test_cli_evaluate_writes_output(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "evaluate.json"
    manifest = tmp_path / "evaluate.manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "evaluate",
            "--prompt",
            "hello world",
            "--run-count",
            "3",
            "--seed",
            "5",
            "--asi-profile",
            "reasoning_focus",
            "--mutation-limit",
            "4",
            "--fixed-timestamp",
            "2026-03-02T00:00:00Z",
            "--output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1.0"
    notes = payload["artifacts"]["notes"]
    assert "asi_profile=reasoning_focus" in notes
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["command"] == "evaluate"
    assert manifest_payload["output_path"] == str(output)
    assert len(manifest_payload["output_sha256"]) == 64
    assert manifest_payload["input_config"]["mutation_limit"] == 4


def test_cli_benchmark_writes_output(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "benchmark.json"
    manifest = tmp_path / "benchmark.manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "benchmark",
            "--suite",
            "examples/benchmarks/default_suite.json",
            "--run-count",
            "2",
            "--seed",
            "3",
            "--asi-profile",
            "safety_strict",
            "--mutation-limit",
            "6",
            "--fixed-timestamp",
            "2026-03-02T00:00:00Z",
            "--output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["suite_name"] == "default_week3_suite"
    assert payload["asi_profile"] == "safety_strict"
    assert payload["mutation_limit"] == 6
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["command"] == "benchmark"


def test_cli_drift_writes_output(tmp_path: Path, monkeypatch: object) -> None:
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "drift.json"

    current.write_text(
        json.dumps(
            {
                "metrics": {
                    "semantic_variance": {"raw": 0.5, "normalized": 0.5},
                    "contradiction_rate": 0.3,
                    "mutation_degradation": 0.4,
                    "cross_model_disagreement": 0.4,
                    "tool_misuse_frequency": 0.2,
                    "agent_stability_index": 55.0,
                }
            }
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "metrics": {
                    "semantic_variance": {"raw": 0.1, "normalized": 0.1},
                    "contradiction_rate": 0.0,
                    "mutation_degradation": 0.1,
                    "cross_model_disagreement": 0.1,
                    "tool_misuse_frequency": 0.0,
                    "agent_stability_index": 90.0,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "drift",
            "--current-report",
            str(current),
            "--baseline-report",
            str(baseline),
            "--output",
            str(output),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "drift_score" in payload
    assert payload["drift_score"] > 0


def test_cli_horizon_writes_output(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "horizon.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "horizon",
            "--prompt",
            "plan migration safely",
            "--horizon",
            "4",
            "--run-count",
            "3",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["horizon"] == 4
    assert payload["run_count"] == 3
    assert "long_horizon_instability" in payload


def test_cli_heal_writes_output(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "heal.json"
    manifest = tmp_path / "heal.manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "heal",
            "--prompt",
            "provide triage steps",
            "--run-count",
            "3",
            "--seed",
            "4",
            "--max-attempts",
            "2",
            "--asi-profile",
            "balanced",
            "--output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "attempted" in payload
    assert "asi_before" in payload
    assert "asi_after" in payload
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["command"] == "heal"


def test_cli_regress_writes_output(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "regress.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "regress",
            "--suite",
            "examples/benchmarks/reasoning_suite.json",
            "--baseline",
            "examples/baselines/reasoning_suite.baseline.json",
            "--run-count",
            "3",
            "--seed",
            "2",
            "--output",
            str(output),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "passed" in payload
    assert payload["suite_name"] == "reasoning_suite_v1"


def test_cli_demo_writes_bundle(tmp_path: Path, monkeypatch: object) -> None:
    output_dir = tmp_path / "demo_bundle"
    manifest = tmp_path / "demo.manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "demo",
            "--output-dir",
            str(output_dir),
            "--run-count",
            "2",
            "--seed",
            "5",
            "--horizon",
            "3",
            "--manifest-output",
            str(manifest),
        ],
    )

    exit_code = main()
    assert exit_code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    for key in [
        "baseline_eval",
        "eval",
        "benchmark",
        "regression",
        "drift",
        "horizon",
        "heal",
        "summary",
    ]:
        path = artifacts[key]
        assert isinstance(path, str)
        assert Path(path).exists()

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["command"] == "demo"


def test_cli_evaluate_uses_resolved_agent(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "eval-openai-route.json"
    used_provider: list[str] = []

    def fake_resolve(args: object) -> object:
        used_provider.append(str(args.agent_provider))

        def fake_agent(prompt: str, _rng: object) -> str:
            return f"fake::{prompt}"

        return fake_agent

    monkeypatch.setattr("agent_stability_engine.cli._resolve_agent", fake_resolve)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "evaluate",
            "--agent-provider",
            "openai",
            "--prompt",
            "hello",
            "--run-count",
            "2",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    assert used_provider == ["openai"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1.0"


def test_cli_evaluate_accepts_anthropic_provider(tmp_path: Path, monkeypatch: object) -> None:
    output = tmp_path / "eval-anthropic-route.json"
    used_provider: list[str] = []

    def fake_resolve(args: object) -> object:
        used_provider.append(str(args.agent_provider))

        def fake_agent(prompt: str, _rng: object) -> str:
            return f"fake::{prompt}"

        return fake_agent

    monkeypatch.setattr("agent_stability_engine.cli._resolve_agent", fake_resolve)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "evaluate",
            "--agent-provider",
            "anthropic",
            "--agent-model",
            "claude-haiku-4-5",
            "--prompt",
            "hello",
            "--run-count",
            "2",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
    )

    exit_code = main()
    assert exit_code == 0
    assert used_provider == ["anthropic"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1.0"


def test_cli_export_writes_bundle_and_pdf(tmp_path: Path, monkeypatch: object) -> None:
    input_report = tmp_path / "benchmark.json"
    input_report.write_text(
        json.dumps(
            {
                "suite_name": "reasoning_suite_v1",
                "mean_asi": 83.9,
                "asi_statistics": {
                    "sample_size": 3,
                    "mean": 83.9,
                    "std_dev": 1.0,
                    "std_error": 0.577,
                    "confidence_level": 0.95,
                    "ci_low": 82.9,
                    "ci_high": 84.9,
                    "method": "normal_approx",
                },
                "timestamp_utc": "2026-03-04T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    bundle_output = tmp_path / "bundle.json"
    pdf_output = tmp_path / "bundle.pdf"

    monkeypatch.setenv("ASE_SIGNING_KEY", "unit-test-signing-key")
    monkeypatch.setattr(
        "sys.argv",
        [
            "ase",
            "export",
            "--input-report",
            str(input_report),
            "--bundle-output",
            str(bundle_output),
            "--pdf-output",
            str(pdf_output),
            "--fixed-timestamp",
            "2026-03-04T12:00:00Z",
        ],
    )

    exit_code = main()
    assert exit_code == 0

    bundle_payload = json.loads(bundle_output.read_text(encoding="utf-8"))
    assert bundle_payload["bundle_version"] == "0.1.0"
    attestation = bundle_payload["attestation"]
    assert isinstance(attestation, dict)
    assert attestation["signed"] is True
    assert isinstance(attestation["signature_hmac_sha256"], str)
    assert pdf_output.exists()
    assert pdf_output.read_bytes().startswith(b"%PDF-1.4")
