from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

_DEFAULT_METHODOLOGY_VERSION = "asi-methodology-0.2.0"


def canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(payload: dict[str, object]) -> str:
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_attestation(
    *,
    payload: dict[str, object],
    created_at_utc: str | None = None,
    methodology_version: str = _DEFAULT_METHODOLOGY_VERSION,
    signing_key: str | None = None,
) -> dict[str, object]:
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report_sha256 = sha256_payload(payload)
    key = signing_key or os.getenv("ASE_SIGNING_KEY")
    signed = bool(key)
    signature = _hmac_sha256(report_sha256, key) if key else None

    return {
        "report_sha256": report_sha256,
        "created_at_utc": timestamp,
        "methodology_version": methodology_version,
        "tool_version": _tool_version(),
        "signature_algorithm": "hmac_sha256" if signed else None,
        "signature_hmac_sha256": signature,
        "signed": signed,
    }


def _hmac_sha256(message: str, key: str) -> str:
    digest = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def _tool_version() -> str:
    for distribution_name in ("agent-stability-engine", "agent_stability_engine"):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    return "0+unknown"
