from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from agent_stability_engine.engine.sampling import MultiRunSampler
from agent_stability_engine.engine.variance import EmbeddingVarianceScorer
from agent_stability_engine.report.schema import REPORT_SCHEMA_VERSION, validate_report


@dataclass(frozen=True)
class EvaluationResult:
    report: dict[str, object]


def evaluate_prompt(
    prompt: str,
    agent_fn: Callable[[str], str],
    run_count: int,
    seed: int = 0,
    expected_max_variance: float = 0.5,
) -> EvaluationResult:
    sampler = MultiRunSampler[str, str](seed=seed)

    def wrapped(payload: str, _rng: random.Random) -> str:
        return agent_fn(payload)

    sampled = sampler.run(wrapped, prompt, run_count)
    outputs = sampled.outputs

    variance = EmbeddingVarianceScorer(expected_max_variance=expected_max_variance).score(outputs)

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": f"run-{uuid4()}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "run_count": run_count,
            "seed": seed,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "metrics": {
            "semantic_variance": {
                "raw": variance.raw_variance,
                "normalized": variance.normalized_variance,
            }
        },
        "artifacts": {
            "outputs": outputs,
            "notes": "Week 1 phase integrated evaluator",
        },
    }

    validate_report(report)
    return EvaluationResult(report=report)
