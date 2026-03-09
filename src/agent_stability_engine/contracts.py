from __future__ import annotations

from pathlib import Path

SUPPORTED_JOB_TYPES = ("benchmark", "conversation_benchmark", "agent_benchmark")
DEFAULT_JOB_TYPE = "benchmark"
DEFAULT_SUITE = "examples/benchmarks/large_suite.json"


def resolve_suite_path(*, base_dir: Path, suite: str | None) -> Path:
    raw = (suite or DEFAULT_SUITE).strip()
    if not raw:
        raw = DEFAULT_SUITE

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    repo_root = base_dir.resolve()
    if repo_root not in candidate.parents and candidate != repo_root:
        msg = "suite path must be inside repository root"
        raise ValueError(msg)
    if candidate.suffix.lower() != ".json":
        msg = "suite file must be a .json file"
        raise ValueError(msg)
    if not candidate.exists():
        msg = f"suite file not found: {candidate}"
        raise ValueError(msg)
    return candidate


def validate_job_contract(*, job_type: str, fault_rate: float) -> None:
    if job_type not in SUPPORTED_JOB_TYPES:
        msg = f"unsupported job_type: {job_type}"
        raise ValueError(msg)
    if fault_rate > 0.0 and job_type != "agent_benchmark":
        msg = "fault_rate is only valid when job_type is agent_benchmark"
        raise ValueError(msg)
