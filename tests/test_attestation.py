from __future__ import annotations

from agent_stability_engine.report.attestation import build_attestation


def test_attestation_signing_is_deterministic() -> None:
    payload = {"a": 1, "b": "two"}
    attestation_a = build_attestation(
        payload=payload,
        created_at_utc="2026-03-04T00:00:00Z",
        signing_key="test-key",
    )
    attestation_b = build_attestation(
        payload=payload,
        created_at_utc="2026-03-04T00:00:00Z",
        signing_key="test-key",
    )

    assert attestation_a["report_sha256"] == attestation_b["report_sha256"]
    assert attestation_a["signature_hmac_sha256"] == attestation_b["signature_hmac_sha256"]
    assert attestation_a["signed"] is True


def test_attestation_signature_changes_with_payload() -> None:
    base = build_attestation(
        payload={"v": 1},
        created_at_utc="2026-03-04T00:00:00Z",
        signing_key="test-key",
    )
    changed = build_attestation(
        payload={"v": 2},
        created_at_utc="2026-03-04T00:00:00Z",
        signing_key="test-key",
    )
    assert base["signature_hmac_sha256"] != changed["signature_hmac_sha256"]


def test_attestation_without_signing_key_is_unsigned() -> None:
    attestation = build_attestation(
        payload={"x": "y"},
        created_at_utc="2026-03-04T00:00:00Z",
        signing_key=None,
    )
    assert attestation["signed"] is False
    assert attestation["signature_hmac_sha256"] is None
