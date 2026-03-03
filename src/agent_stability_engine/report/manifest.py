from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    command: str,
    output_path: Path,
    output_payload: dict[str, Any],
    input_config: dict[str, Any],
    timestamp_utc: str | None,
) -> dict[str, Any]:
    canonical_output = json.dumps(output_payload, sort_keys=True, separators=(",", ":"))
    canonical_config = json.dumps(input_config, sort_keys=True, separators=(",", ":"))

    output_hash = hashlib.sha256(canonical_output.encode()).hexdigest()
    config_hash = hashlib.sha256(canonical_config.encode()).hexdigest()

    return {
        "manifest_version": "0.1.0",
        "command": command,
        "created_at_utc": timestamp_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_path": str(output_path),
        "output_sha256": output_hash,
        "input_config": input_config,
        "input_config_sha256": config_hash,
    }
