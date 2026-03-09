from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from agent_stability_engine.engine.conversation import ConversationEvaluator
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.stats import summarize_mean_confidence
from agent_stability_engine.runners.benchmark import BenchmarkResult


def run_conversation_benchmark_suite(
    suite_path: Path,
    adapter: Any,
    run_count: int,
    seed: int,
    timestamp_utc: str | None = None,  # noqa: ARG001
    embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
    embedding_model: str | None = None,  # noqa: ARG001
    embedding_openai_api_key: str | None = None,
    max_cases: int | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
    agent_factory: Callable[[], Any] | None = None,
) -> BenchmarkResult:
    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    canonical_suite = json.dumps(suite_data, sort_keys=True, separators=(",", ":"))
    suite_sha256 = hashlib.sha256(canonical_suite.encode()).hexdigest()
    raw_cases = suite_data["cases"]
    cases = [case for case in raw_cases if isinstance(case, dict)]
    if max_cases is not None and max_cases > 0:
        cases = cases[:max_cases]

    def _make_evaluator() -> ConversationEvaluator:
        return ConversationEvaluator(
            embedding_provider=embedding_provider,
            embedding_openai_api_key=embedding_openai_api_key,
        )

    shared_adapter_lock = threading.Lock()
    shared_adapter = adapter

    def _locked_call_messages(
        messages: list[dict[str, str]],
        rng: object | None = None,
    ) -> str:
        with shared_adapter_lock:
            call = getattr(shared_adapter, "call_messages", None)
            if not callable(call):
                msg = "adapter must implement call_messages(messages, rng)"
                raise ValueError(msg)
            response = call(messages, rng)
            if not isinstance(response, str):
                msg = "adapter.call_messages must return a string response"
                raise ValueError(msg)
            return response

    class _LockedAdapter:
        def call_messages(self, messages: list[dict[str, str]], rng: object | None = None) -> str:
            return _locked_call_messages(messages, rng)

    def _make_case_adapter() -> Any:
        if agent_factory is not None:
            return agent_factory()
        if workers > 1:
            return _LockedAdapter()
        return shared_adapter

    def _evaluate_case(case: dict[str, object]) -> dict[str, object]:
        case_id = str(case.get("id", ""))
        evaluator = _make_evaluator()
        case_adapter = _make_case_adapter()
        evaluation = evaluator.evaluate(
            case=case,
            adapter=case_adapter,
            run_count=run_count,
            seed=seed,
        )
        report = evaluation.report
        case_hash = hashlib.sha256(
            json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "case_id": case_id,
            "case_sha256": case_hash,
            "report": report,
        }

    case_reports: list[dict[str, object]] = []
    conv_asi_values: list[float] = []
    total = len(cases)
    completed_count = 0
    lock = threading.Lock()

    def _evaluate_and_track(case: dict[str, object]) -> dict[str, object]:
        nonlocal completed_count
        result = _evaluate_case(case)
        with lock:
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

    domain_conv_asi: dict[str, list[float]] = defaultdict(list)
    for entry in case_reports:
        case_id = str(entry.get("case_id", ""))
        domain = case_id.split("-")[0] if "-" in case_id else "general"
        report = entry.get("report")
        if isinstance(report, dict):
            metrics = report.get("metrics")
            if isinstance(metrics, dict):
                conv_asi = metrics.get("conv_asi")
                if isinstance(conv_asi, (int, float)):
                    value = float(conv_asi)
                    conv_asi_values.append(value)
                    domain_conv_asi[domain].append(value)

    mean_conv_asi = sum(conv_asi_values) / len(conv_asi_values) if conv_asi_values else 0.0
    conv_asi_statistics = (
        summarize_mean_confidence(conv_asi_values).to_dict() if conv_asi_values else None
    )
    domain_scores = {
        domain: round(sum(values) / len(values), 2)
        for domain, values in sorted(domain_conv_asi.items())
    }

    benchmark_id_seed = json.dumps(
        {
            "suite_sha256": suite_sha256,
            "run_count": run_count,
            "seed": seed,
            "job_type": "conversation_benchmark",
        },
        sort_keys=True,
    )
    benchmark_id = hashlib.sha256(benchmark_id_seed.encode()).hexdigest()[:16]

    aggregate: dict[str, object] = {
        "benchmark_id": f"conv-bench-{benchmark_id}",
        "suite_name": suite_data.get("name", "unnamed_conversation_suite"),
        "suite_sha256": suite_sha256,
        "run_count": run_count,
        "seed": seed,
        "job_type": "conversation_benchmark",
        "embedding_provider": embedding_provider.value,
        "num_cases": len(cases),
        "mean_conv_asi": mean_conv_asi,
        # Compatibility for existing consumers expecting mean_asi.
        "mean_asi": mean_conv_asi,
        "conv_asi_statistics": conv_asi_statistics,
        "case_conv_asi_values": conv_asi_values,
        "domain_scores": domain_scores,
        "cases": case_reports,
    }
    return BenchmarkResult(report=aggregate)
