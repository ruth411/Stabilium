from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_stability_engine.engine.asi import ASIProfile
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.evaluator import StabilityEvaluator
from agent_stability_engine.engine.stats import summarize_mean_confidence


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
    progress_callback: Callable[[int, int, str], None] | None = None,
    agent_factory: Callable[[], Callable[..., str]] | None = None,
) -> BenchmarkResult:
    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    canonical_suite = json.dumps(suite_data, sort_keys=True, separators=(",", ":"))
    suite_sha256 = hashlib.sha256(canonical_suite.encode()).hexdigest()
    cases = suite_data["cases"]
    if max_cases is not None and max_cases > 0:
        cases = cases[:max_cases]

    def _make_evaluator() -> StabilityEvaluator:
        return StabilityEvaluator(
            asi_profile=asi_profile,
            mutation_sample_limit=mutation_sample_limit,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_openai_api_key=embedding_openai_api_key,
        )

    shared_agent_lock = threading.Lock()
    shared_agent_fn = agent_fn

    def _locked_agent(prompt: str, rng: object) -> str:
        with shared_agent_lock:
            return _invoke_agent(shared_agent_fn, prompt, rng)

    def _make_case_agent() -> Callable[..., str]:
        if agent_factory is not None:
            return agent_factory()
        if workers > 1:
            # Avoid concurrent access to mutable adapters when one shared agent
            # instance is used with multi-worker benchmarking.
            return _locked_agent
        return shared_agent_fn

    def _evaluate_case(case: dict[str, object]) -> dict[str, object]:
        case_prompt = str(case["prompt"])
        case_id = str(case["id"])
        case_expected = case.get("expected")
        expected_str = str(case_expected) if case_expected is not None else None
        evaluator = _make_evaluator()
        case_agent_fn = _make_case_agent()
        evaluation = evaluator.evaluate(
            prompt=case_prompt,
            agent_fn=case_agent_fn,
            run_count=run_count,
            seed=seed,
            timestamp_utc=timestamp_utc,
            expected=expected_str,
        )
        report = evaluation.report
        return {
            "case_id": case_id,
            "prompt_sha256": hashlib.sha256(case_prompt.encode()).hexdigest(),
            "report": report,
        }

    case_reports: list[dict[str, object]] = []
    asi_values: list[float] = []
    total = len(cases)
    completed_count = 0
    _lock = threading.Lock()

    def _evaluate_and_track(case: dict[str, object]) -> dict[str, object]:
        nonlocal completed_count
        result = _evaluate_case(case)
        with _lock:
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, total, str(case.get("id", "")))
        return result

    if workers > 1:
        ordered: dict[int, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_evaluate_and_track, case): i for i, case in enumerate(cases)}
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        case_reports = [ordered[i] for i in range(len(cases))]
    else:
        case_reports = [_evaluate_and_track(case) for case in cases]

    domain_asi: dict[str, list[float]] = defaultdict(list)
    for entry in case_reports:
        case_id = str(entry.get("case_id", ""))
        domain = case_id.split("-")[0] if "-" in case_id else "general"
        report = entry["report"]
        if isinstance(report, dict):
            metrics = report.get("metrics")
            if isinstance(metrics, dict):
                asi = metrics.get("agent_stability_index")
                if isinstance(asi, (int, float)):
                    asi_values.append(float(asi))
                    domain_asi[domain].append(float(asi))

    mean_asi = sum(asi_values) / len(asi_values) if asi_values else 0.0
    asi_statistics = summarize_mean_confidence(asi_values).to_dict() if asi_values else None
    domain_scores = {d: round(sum(v) / len(v), 2) for d, v in sorted(domain_asi.items())}

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
        "asi_statistics": asi_statistics,
        "case_asi_values": asi_values,
        "domain_scores": domain_scores,
        "cases": case_reports,
    }
    return BenchmarkResult(report=aggregate)


def _invoke_agent(fn: Callable[..., str], prompt: str, rng: object) -> str:
    try:
        return fn(prompt, rng)
    except TypeError:
        return fn(prompt)
