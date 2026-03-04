from __future__ import annotations

from pathlib import Path

from agent_stability_engine.report.pdf_renderer import write_compliance_pdf


def test_write_compliance_pdf_writes_pdf_file(tmp_path: Path) -> None:
    bundle = {
        "created_at_utc": "2026-03-04T00:00:00Z",
        "summary": {
            "report_type": "benchmark",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "suite_name": "reasoning_suite_v1",
            "asi_score": 83.9,
            "asi_confidence": {
                "sample_size": 100,
                "ci_low": 81.8,
                "ci_high": 86.0,
            },
        },
        "metrics": {
            "trend": {
                "direction": "up",
                "delta_vs_previous": 1.2,
            }
        },
        "methodology": {
            "version": "asi-methodology-0.2.0",
            "confidence_method": "normal_approx",
            "significance_method": "one_sided_z_approx",
        },
        "attestation": {
            "report_sha256": "a" * 64,
            "signature_hmac_sha256": "b" * 64,
            "signed": True,
        },
    }

    pdf_path = tmp_path / "report.pdf"
    write_compliance_pdf(bundle, pdf_path)

    assert pdf_path.exists()
    payload = pdf_path.read_bytes()
    assert payload.startswith(b"%PDF-1.4")
    assert len(payload) > 200
