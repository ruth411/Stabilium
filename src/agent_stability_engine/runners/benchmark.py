from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_stability_engine.engine.asi import ASIProfile
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.evaluator import StabilityEvaluator


@dataclass(frozen=True)
class BenchmarkResult:
    report: dict[str, object]


def run_benchmark_suite(
    suite_path: Path,
    agent_fn: Callable[..., str],
    run_count: int,
    seed: int,
    timestamp_utc: str | None = None,
    asi_profile: ASIProfile = ASIProfile.BALANCED,
    mutation_sample_limit: int | None = None,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
    embedding_model: str | None = None,
    embedding_openai_api_key: str | None = None,
    max_cases: int | None = None,
    workers: int = 1,
) -> BenchmarkResult:
    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    canonical_suite = json.dumps(suite_data, sort_keys=True, separators=(",", ":"))
    suite_sha256 = hashlib.sha256(canonical_suite.encode()).hexdigest()
    cases = suite_data["cases"]
    if max_cases is not None and max_cases > 0:
        cases = cases[:max_cases]

    evaluator = StabilityEvaluator(
        asi_profile=asi_profile,
        mutation_sample_limit=mutation_sample_limit,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_openai_api_key=embedding_openai_api_key,
    )

    def _evaluate_case(case: dict[str, object]) -> dict[str, object]:
        case_prompt = str(case["prompt"])
        case_id = str(case["id"])
        evaluation = evaluator.evaluate(
            prompt=case_prompt,
            agent_fn=agent_fn,
            run_count=run_count,
            seed=seed,
            timestamp_utc=timestamp_utc,
        )
        report = evaluation.report
        return {
            "case_id": case_id,
            "prompt_sha256": hashlib.sha256(case_prompt.encode()).hexdigest(),
            "report": report,
        }

    case_reports: list[dict[str, object]] = []
    asi_values: list[float] = []

    if workers > 1:
        ordered: dict[int, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_evaluate_case, case): i for i, case in enumerate(cases)}
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        case_reports = [ordered[i] for i in range(len(cases))]
    else:
        case_reports = [_evaluate_case(case) for case in cases]

    for entry in case_reports:
        report = entry["report"]
        if isinstance(report, dict):
            metrics = report.get("metrics")
            if isinstance(metrics, dict):
                asi = metrics.get("agent_stability_index")
                if isinstance(asi, (int, float)):
                    asi_values.append(float(asi))

    mean_asi = sum(asi_values) / len(asi_values) if asi_values else 0.0

    benchmark_id_seed = json.dumps(
        {
            "suite_sha256": suite_sha256,
            "run_count": run_count,
            "seed": seed,
        },
        sort_keys=True,
    )
    benchmark_id = hashlib.sha256(benchmark_id_seed.encode()).hexdigest()[:16]

    aggregate = {
        "benchmark_id": f"bench-{benchmark_id}",
        "suite_name": suite_data.get("name", "unnamed_suite"),
        "suite_sha256": suite_sha256,
        "run_count": run_count,
        "seed": seed,
        "asi_profile": asi_profile.value,
        "mutation_limit": mutation_sample_limit,
        "embedding_provider": embedding_provider.value,
        "embedding_model": embedding_model,
        "num_cases": len(cases),
        "mean_asi": mean_asi,
        "cases": case_reports,
    }
    return BenchmarkResult(report=aggregate)
