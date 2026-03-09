from __future__ import annotations

from pathlib import Path

import pytest

from agent_stability_engine.contracts import (
    DEFAULT_SUITE,
    resolve_suite_path,
    validate_job_contract,
)


def test_resolve_suite_path_uses_default_suite() -> None:
    base_dir = Path.cwd()
    resolved = resolve_suite_path(base_dir=base_dir, suite=None)
    assert resolved == (base_dir / DEFAULT_SUITE).resolve()


def test_resolve_suite_path_rejects_non_json_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "suite.txt"
    bad_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="\\.json"):
        resolve_suite_path(base_dir=tmp_path, suite=str(bad_file.relative_to(tmp_path)))


def test_validate_job_contract_rejects_fault_rate_for_non_agent() -> None:
    with pytest.raises(ValueError, match="fault_rate"):
        validate_job_contract(job_type="benchmark", fault_rate=0.1)


def test_validate_job_contract_accepts_agent_fault_rate() -> None:
    validate_job_contract(job_type="agent_benchmark", fault_rate=0.2)
