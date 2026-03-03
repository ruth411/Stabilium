from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable

from agent_stability_engine.adapters.openai import OpenAIChatAdapter
from agent_stability_engine.engine.alignment import GoalSpec
from agent_stability_engine.engine.asi import ASIProfile
from agent_stability_engine.engine.drift import DriftTracker, metrics_from_report
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.evaluator import StabilityEvaluator
from agent_stability_engine.engine.self_healing import SelfHealingEngine
from agent_stability_engine.report.manifest import build_manifest
from agent_stability_engine.runners.benchmark import run_benchmark_suite
from agent_stability_engine.runners.horizon import LongHorizonStabilityRunner
from agent_stability_engine.runners.regression import run_benchmark_regression


def _demo_agent(prompt: str, rng: random.Random) -> str:
    suffixes = [
        "answer remains stable",
        "answer slightly reframed",
        "answer with concise detail",
    ]
    suffix = suffixes[rng.randrange(len(suffixes))]
    return f"{prompt.strip()} :: {suffix}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"expected top-level JSON object in {path}"
        raise ValueError(msg)
    return loaded


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-provider",
        default="demo",
        choices=["demo", "openai"],
        help="Agent backend provider used for generation commands",
    )
    parser.add_argument(
        "--agent-model",
        default="gpt-4o-mini",
        help="Model name for external providers (for example OpenAI model id)",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the OpenAI API key",
    )
    parser.add_argument("--openai-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--openai-max-retries", type=int, default=2)
    parser.add_argument("--openai-min-interval-seconds", type=float, default=0.0)


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        default=EmbeddingProvider.AUTO.value,
        choices=[provider.value for provider in EmbeddingProvider],
        help="Embedding backend used for semantic variance scoring",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Optional embedding model id (provider-specific)",
    )


def _resolve_agent(args: argparse.Namespace) -> Callable[..., str]:
    if args.agent_provider == "demo":
        return _demo_agent

    if args.agent_provider == "openai":
        env_name = args.openai_api_key_env
        api_key = os.getenv(env_name)
        if not api_key:
            msg = f"missing OpenAI API key in environment variable: {env_name}"
            raise ValueError(msg)
        return OpenAIChatAdapter(
            model=args.agent_model,
            api_key=api_key,
            timeout_seconds=args.openai_timeout_seconds,
            max_retries=args.openai_max_retries,
            min_interval_seconds=args.openai_min_interval_seconds,
        )

    msg = f"unsupported agent provider: {args.agent_provider}"
    raise ValueError(msg)


def _agent_usage_snapshot(agent_fn: Callable[..., str]) -> dict[str, object] | None:
    usage_fn = getattr(agent_fn, "usage_snapshot", None)
    if not callable(usage_fn):
        return None
    usage = usage_fn()
    if isinstance(usage, dict):
        return usage
    return None


def _resolve_embedding_provider(args: argparse.Namespace) -> EmbeddingProvider:
    return EmbeddingProvider(args.embedding_provider)


def _resolve_embedding_model(args: argparse.Namespace) -> str | None:
    raw_model = getattr(args, "embedding_model", None)
    if not isinstance(raw_model, str):
        return None
    model = raw_model.strip()
    if not model:
        return None
    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ase", description="Agent Stability Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run stability evaluation for one prompt")
    _add_agent_args(eval_parser)
    _add_embedding_args(eval_parser)
    eval_parser.add_argument("--prompt", required=True, help="Prompt to evaluate")
    eval_parser.add_argument("--run-count", type=int, default=5)
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument(
        "--baseline-report",
        action="append",
        default=[],
        help="Optional historical report JSON path (repeatable) for drift scoring",
    )
    eval_parser.add_argument(
        "--goal-required",
        action="append",
        default=[],
        help="Required keyword for goal-alignment checking (repeatable)",
    )
    eval_parser.add_argument(
        "--goal-forbidden",
        action="append",
        default=[],
        help="Forbidden regex pattern for goal-alignment checking (repeatable)",
    )
    eval_parser.add_argument(
        "--mutation-limit",
        type=int,
        default=None,
        help="Optional cap on number of mutation prompts evaluated",
    )
    eval_parser.add_argument(
        "--asi-profile",
        default=ASIProfile.BALANCED.value,
        choices=[profile.value for profile in ASIProfile],
    )
    eval_parser.add_argument("--fixed-timestamp", default=None)
    eval_parser.add_argument("--output", required=True, help="Output JSON file path")
    eval_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for reproducibility metadata",
    )

    heal_parser = subparsers.add_parser("heal", help="Run self-healing remediation on one prompt")
    _add_agent_args(heal_parser)
    _add_embedding_args(heal_parser)
    heal_parser.add_argument("--prompt", required=True, help="Prompt to evaluate and remediate")
    heal_parser.add_argument("--run-count", type=int, default=5)
    heal_parser.add_argument("--seed", type=int, default=0)
    heal_parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Number of healing attempts to run",
    )
    heal_parser.add_argument(
        "--baseline-report",
        action="append",
        default=[],
        help="Optional historical report JSON path (repeatable) for drift scoring",
    )
    heal_parser.add_argument(
        "--goal-required",
        action="append",
        default=[],
        help="Required keyword for goal-alignment checking (repeatable)",
    )
    heal_parser.add_argument(
        "--goal-forbidden",
        action="append",
        default=[],
        help="Forbidden regex pattern for goal-alignment checking (repeatable)",
    )
    heal_parser.add_argument(
        "--mutation-limit",
        type=int,
        default=None,
        help="Optional cap on number of mutation prompts evaluated",
    )
    heal_parser.add_argument(
        "--asi-profile",
        default=ASIProfile.BALANCED.value,
        choices=[profile.value for profile in ASIProfile],
    )
    heal_parser.add_argument("--fixed-timestamp", default=None)
    heal_parser.add_argument("--output", required=True, help="Output JSON file path")
    heal_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for reproducibility metadata",
    )

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite evaluation")
    _add_agent_args(bench_parser)
    _add_embedding_args(bench_parser)
    bench_parser.add_argument(
        "--suite",
        default="examples/benchmarks/default_suite.json",
        help="Path to benchmark suite JSON",
    )
    bench_parser.add_argument("--run-count", type=int, default=5)
    bench_parser.add_argument("--seed", type=int, default=0)
    bench_parser.add_argument(
        "--mutation-limit",
        type=int,
        default=None,
        help="Optional cap on number of mutation prompts evaluated per case",
    )
    bench_parser.add_argument(
        "--asi-profile",
        default=ASIProfile.BALANCED.value,
        choices=[profile.value for profile in ASIProfile],
    )
    bench_parser.add_argument("--fixed-timestamp", default=None)
    bench_parser.add_argument("--output", required=True, help="Output JSON file path")
    bench_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for reproducibility metadata",
    )

    regress_parser = subparsers.add_parser(
        "regress",
        help="Run benchmark suite and validate against baseline thresholds",
    )
    _add_agent_args(regress_parser)
    _add_embedding_args(regress_parser)
    regress_parser.add_argument("--suite", required=True, help="Path to benchmark suite JSON")
    regress_parser.add_argument("--baseline", required=True, help="Path to baseline threshold JSON")
    regress_parser.add_argument("--run-count", type=int, default=5)
    regress_parser.add_argument("--seed", type=int, default=0)
    regress_parser.add_argument(
        "--mutation-limit",
        type=int,
        default=None,
        help="Optional cap on number of mutation prompts evaluated per case",
    )
    regress_parser.add_argument(
        "--asi-profile",
        default=ASIProfile.BALANCED.value,
        choices=[profile.value for profile in ASIProfile],
    )
    regress_parser.add_argument("--fixed-timestamp", default=None)
    regress_parser.add_argument("--output", required=True, help="Output JSON file path")
    regress_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for reproducibility metadata",
    )

    drift_parser = subparsers.add_parser(
        "drift",
        help="Compare one report against baseline reports for behavior-drift scoring",
    )
    drift_parser.add_argument("--current-report", required=True, help="Current report JSON path")
    drift_parser.add_argument(
        "--baseline-report",
        action="append",
        required=True,
        help="Baseline report JSON path (repeatable)",
    )
    drift_parser.add_argument("--output", required=True, help="Output JSON file path")
    drift_parser.add_argument("--fixed-timestamp", default=None)
    drift_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for reproducibility metadata",
    )

    horizon_parser = subparsers.add_parser(
        "horizon",
        help="Run long-horizon decision stability evaluation",
    )
    horizon_parser.add_argument("--prompt", required=True)
    horizon_parser.add_argument("--horizon", type=int, default=5)
    horizon_parser.add_argument("--run-count", type=int, default=5)
    horizon_parser.add_argument("--seed", type=int, default=0)
    horizon_parser.add_argument("--output", required=True)
    horizon_parser.add_argument("--fixed-timestamp", default=None)
    horizon_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for reproducibility metadata",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Generate a complete demo artifact bundle for release walkthroughs",
    )
    _add_agent_args(demo_parser)
    _add_embedding_args(demo_parser)
    demo_parser.add_argument("--output-dir", required=True, help="Directory for demo artifacts")
    demo_parser.add_argument("--run-count", type=int, default=3)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--horizon", type=int, default=4)
    demo_parser.add_argument(
        "--mutation-limit",
        type=int,
        default=6,
        help="Mutation sample cap used for demo runs",
    )
    demo_parser.add_argument(
        "--asi-profile",
        default=ASIProfile.BALANCED.value,
        choices=[profile.value for profile in ASIProfile],
    )
    demo_parser.add_argument("--fixed-timestamp", default=None)
    demo_parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON path for demo summary reproducibility metadata",
    )

    return parser


def _demo_policy(context: str, step: int, rng: random.Random) -> str:
    decision_pool = [
        "continue with current plan",
        "request verification checkpoint",
        "fallback to conservative path",
    ]
    choice = decision_pool[(step + rng.randrange(len(decision_pool))) % len(decision_pool)]
    return f"{choice} [{len(context) % 7}]"


def _run_demo_bundle(
    *,
    output_dir: Path,
    run_count: int,
    seed: int,
    horizon: int,
    fixed_timestamp: str | None,
    asi_profile: ASIProfile,
    mutation_limit: int | None,
    agent_fn: Callable[..., str],
    agent_provider: str,
    agent_model: str,
    embedding_provider: EmbeddingProvider,
    embedding_model: str | None,
    embedding_openai_api_key: str | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_eval_path = output_dir / "baseline_eval.json"
    eval_path = output_dir / "eval.json"
    benchmark_path = output_dir / "benchmark.json"
    regression_path = output_dir / "regression.json"
    drift_path = output_dir / "drift.json"
    horizon_path = output_dir / "horizon.json"
    heal_path = output_dir / "heal.json"
    summary_path = output_dir / "summary.json"

    evaluator = StabilityEvaluator(
        asi_profile=asi_profile,
        mutation_sample_limit=mutation_limit,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_openai_api_key=embedding_openai_api_key,
    )
    goal_spec = GoalSpec(
        required_keywords=("safe",),
        forbidden_patterns=(r"run_shell\(",),
    )

    baseline_eval = evaluator.evaluate(
        prompt="Provide safe API incident response steps",
        agent_fn=agent_fn,
        run_count=run_count,
        seed=seed + 1,
        timestamp_utc=fixed_timestamp,
    )
    _write_json(baseline_eval_path, baseline_eval.report)

    eval_result = evaluator.evaluate(
        prompt="Provide safe API incident response steps",
        agent_fn=agent_fn,
        run_count=run_count,
        seed=seed,
        timestamp_utc=fixed_timestamp,
        goal_spec=goal_spec,
        baseline_reports=[baseline_eval.report],
    )
    _write_json(eval_path, eval_result.report)

    benchmark_result = run_benchmark_suite(
        suite_path=Path("examples/benchmarks/reasoning_suite.json"),
        agent_fn=agent_fn,
        run_count=run_count,
        seed=seed,
        timestamp_utc=fixed_timestamp,
        asi_profile=asi_profile,
        mutation_sample_limit=mutation_limit,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_openai_api_key=embedding_openai_api_key,
    )
    _write_json(benchmark_path, benchmark_result.report)

    regression_result = run_benchmark_regression(
        suite_path=Path("examples/benchmarks/reasoning_suite.json"),
        baseline_path=Path("examples/baselines/reasoning_suite.baseline.json"),
        agent_fn=agent_fn,
        run_count=run_count,
        seed=seed,
        timestamp_utc=fixed_timestamp,
        asi_profile=asi_profile,
        mutation_sample_limit=mutation_limit,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_openai_api_key=embedding_openai_api_key,
    )
    _write_json(regression_path, regression_result.report)

    tracker = DriftTracker()
    drift_analysis = tracker.compare(
        current_metrics=metrics_from_report(eval_result.report),
        baseline_metrics=[metrics_from_report(baseline_eval.report)],
    )
    drift_payload: dict[str, object] = {
        "current_report": str(eval_path),
        "baseline_reports": [str(baseline_eval_path)],
        "drift_score": drift_analysis.drift_score,
        "drift_detected": drift_analysis.drift_detected,
        "metric_deltas": drift_analysis.metric_deltas,
    }
    _write_json(drift_path, drift_payload)

    horizon_runner = LongHorizonStabilityRunner()
    horizon_result = horizon_runner.run(
        policy_fn=_demo_policy,
        initial_context="Plan safe rollout strategy",
        horizon=horizon,
        run_count=run_count,
        seed=seed,
    )
    horizon_payload: dict[str, object] = {
        "prompt": "Plan safe rollout strategy",
        "horizon": horizon_result.horizon,
        "run_count": horizon_result.run_count,
        "seed": seed,
        "long_horizon_instability": horizon_result.long_horizon_instability,
        "per_step_instability": horizon_result.per_step_instability,
        "trajectories": [run.decisions for run in horizon_result.runs],
    }
    _write_json(horizon_path, horizon_payload)

    healer = SelfHealingEngine(evaluator=evaluator)
    heal_result = healer.heal(
        prompt="Provide triage steps",
        agent_fn=agent_fn,
        run_count=run_count,
        seed=seed,
        timestamp_utc=fixed_timestamp,
        goal_spec=goal_spec,
        baseline_reports=[baseline_eval.report],
        max_attempts=2,
    )
    heal_payload = heal_result.to_dict()
    _write_json(heal_path, heal_payload)

    summary: dict[str, object] = {
        "demo_version": "0.1.0",
        "output_dir": str(output_dir),
        "agent_provider": agent_provider,
        "agent_model": agent_model,
        "embedding_provider": embedding_provider.value,
        "embedding_model": embedding_model,
        "asi_profile": asi_profile.value,
        "run_count": run_count,
        "seed": seed,
        "horizon": horizon,
        "mutation_limit": mutation_limit,
        "artifacts": {
            "baseline_eval": str(baseline_eval_path),
            "eval": str(eval_path),
            "benchmark": str(benchmark_path),
            "regression": str(regression_path),
            "drift": str(drift_path),
            "horizon": str(horizon_path),
            "heal": str(heal_path),
            "summary": str(summary_path),
        },
        "status": {
            "regression_passed": regression_result.report["passed"],
            "drift_detected": drift_analysis.drift_detected,
            "healing_improved": heal_result.improved,
        },
    }
    usage = _agent_usage_snapshot(agent_fn)
    if usage is not None:
        summary["agent_usage"] = usage
    _write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        agent_fn = _resolve_agent(args)
        embedding_provider = _resolve_embedding_provider(args)
        embedding_model = _resolve_embedding_model(args)
        embedding_api_key = os.getenv(args.openai_api_key_env)
        goal_spec = None
        if args.goal_required or args.goal_forbidden:
            goal_spec = GoalSpec(
                required_keywords=tuple(args.goal_required),
                forbidden_patterns=tuple(args.goal_forbidden),
            )
        baseline_reports = [_read_json(Path(path)) for path in args.baseline_report]

        evaluator = StabilityEvaluator(
            asi_profile=ASIProfile(args.asi_profile),
            mutation_sample_limit=args.mutation_limit,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_openai_api_key=embedding_api_key,
        )
        output_path = Path(args.output)
        eval_result = evaluator.evaluate(
            prompt=args.prompt,
            agent_fn=agent_fn,
            run_count=args.run_count,
            seed=args.seed,
            timestamp_utc=args.fixed_timestamp,
            goal_spec=goal_spec,
            baseline_reports=baseline_reports,
        )
        _write_json(output_path, eval_result.report)
        if args.manifest_output:
            manifest = build_manifest(
                command="evaluate",
                output_path=output_path,
                output_payload=eval_result.report,
                input_config={
                    "prompt": args.prompt,
                    "run_count": args.run_count,
                    "seed": args.seed,
                    "agent_provider": args.agent_provider,
                    "agent_model": args.agent_model,
                    "embedding_provider": args.embedding_provider,
                    "embedding_model": args.embedding_model,
                    "asi_profile": args.asi_profile,
                    "mutation_limit": args.mutation_limit,
                    "baseline_report_count": len(args.baseline_report),
                    "goal_required": args.goal_required,
                    "goal_forbidden": args.goal_forbidden,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    if args.command == "heal":
        agent_fn = _resolve_agent(args)
        embedding_provider = _resolve_embedding_provider(args)
        embedding_model = _resolve_embedding_model(args)
        embedding_api_key = os.getenv(args.openai_api_key_env)
        goal_spec = None
        if args.goal_required or args.goal_forbidden:
            goal_spec = GoalSpec(
                required_keywords=tuple(args.goal_required),
                forbidden_patterns=tuple(args.goal_forbidden),
            )
        baseline_reports = [_read_json(Path(path)) for path in args.baseline_report]
        evaluator = StabilityEvaluator(
            asi_profile=ASIProfile(args.asi_profile),
            mutation_sample_limit=args.mutation_limit,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_openai_api_key=embedding_api_key,
        )
        healer = SelfHealingEngine(evaluator=evaluator)
        heal_result = healer.heal(
            prompt=args.prompt,
            agent_fn=agent_fn,
            run_count=args.run_count,
            seed=args.seed,
            timestamp_utc=args.fixed_timestamp,
            goal_spec=goal_spec,
            baseline_reports=baseline_reports,
            max_attempts=args.max_attempts,
        )
        payload = heal_result.to_dict()
        output_path = Path(args.output)
        _write_json(output_path, payload)
        if args.manifest_output:
            manifest = build_manifest(
                command="heal",
                output_path=output_path,
                output_payload=payload,
                input_config={
                    "prompt": args.prompt,
                    "run_count": args.run_count,
                    "seed": args.seed,
                    "agent_provider": args.agent_provider,
                    "agent_model": args.agent_model,
                    "embedding_provider": args.embedding_provider,
                    "embedding_model": args.embedding_model,
                    "max_attempts": args.max_attempts,
                    "asi_profile": args.asi_profile,
                    "mutation_limit": args.mutation_limit,
                    "baseline_report_count": len(args.baseline_report),
                    "goal_required": args.goal_required,
                    "goal_forbidden": args.goal_forbidden,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    if args.command == "benchmark":
        agent_fn = _resolve_agent(args)
        embedding_provider = _resolve_embedding_provider(args)
        embedding_model = _resolve_embedding_model(args)
        embedding_api_key = os.getenv(args.openai_api_key_env)
        output_path = Path(args.output)
        benchmark_result = run_benchmark_suite(
            suite_path=Path(args.suite),
            agent_fn=agent_fn,
            run_count=args.run_count,
            seed=args.seed,
            timestamp_utc=args.fixed_timestamp,
            asi_profile=ASIProfile(args.asi_profile),
            mutation_sample_limit=args.mutation_limit,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_openai_api_key=embedding_api_key,
        )
        _write_json(output_path, benchmark_result.report)
        if args.manifest_output:
            manifest = build_manifest(
                command="benchmark",
                output_path=output_path,
                output_payload=benchmark_result.report,
                input_config={
                    "suite": args.suite,
                    "run_count": args.run_count,
                    "seed": args.seed,
                    "agent_provider": args.agent_provider,
                    "agent_model": args.agent_model,
                    "embedding_provider": args.embedding_provider,
                    "embedding_model": args.embedding_model,
                    "asi_profile": args.asi_profile,
                    "mutation_limit": args.mutation_limit,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    if args.command == "regress":
        agent_fn = _resolve_agent(args)
        embedding_provider = _resolve_embedding_provider(args)
        embedding_model = _resolve_embedding_model(args)
        embedding_api_key = os.getenv(args.openai_api_key_env)
        output_path = Path(args.output)
        regression = run_benchmark_regression(
            suite_path=Path(args.suite),
            baseline_path=Path(args.baseline),
            agent_fn=agent_fn,
            run_count=args.run_count,
            seed=args.seed,
            timestamp_utc=args.fixed_timestamp,
            asi_profile=ASIProfile(args.asi_profile),
            mutation_sample_limit=args.mutation_limit,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_openai_api_key=embedding_api_key,
        )
        payload = regression.report
        _write_json(output_path, payload)
        if args.manifest_output:
            manifest = build_manifest(
                command="regress",
                output_path=output_path,
                output_payload=payload,
                input_config={
                    "suite": args.suite,
                    "baseline": args.baseline,
                    "run_count": args.run_count,
                    "seed": args.seed,
                    "agent_provider": args.agent_provider,
                    "agent_model": args.agent_model,
                    "embedding_provider": args.embedding_provider,
                    "embedding_model": args.embedding_model,
                    "asi_profile": args.asi_profile,
                    "mutation_limit": args.mutation_limit,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    if args.command == "drift":
        current_report = _read_json(Path(args.current_report))
        baseline_reports = [_read_json(Path(path)) for path in args.baseline_report]
        tracker = DriftTracker()
        current_metrics = metrics_from_report(current_report)
        baseline_metrics = [metrics_from_report(report) for report in baseline_reports]
        baseline_metrics = [vector for vector in baseline_metrics if vector]
        analysis = tracker.compare(current_metrics, baseline_metrics)

        drift_payload: dict[str, object] = {
            "current_report": args.current_report,
            "baseline_reports": args.baseline_report,
            "drift_score": analysis.drift_score,
            "drift_detected": analysis.drift_detected,
            "metric_deltas": analysis.metric_deltas,
        }
        output_path = Path(args.output)
        _write_json(output_path, drift_payload)
        if args.manifest_output:
            manifest = build_manifest(
                command="drift",
                output_path=output_path,
                output_payload=drift_payload,
                input_config={
                    "current_report": args.current_report,
                    "baseline_reports": args.baseline_report,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    if args.command == "horizon":
        runner = LongHorizonStabilityRunner()
        horizon_result = runner.run(
            policy_fn=_demo_policy,
            initial_context=args.prompt,
            horizon=args.horizon,
            run_count=args.run_count,
            seed=args.seed,
        )
        horizon_payload: dict[str, object] = {
            "prompt": args.prompt,
            "horizon": horizon_result.horizon,
            "run_count": horizon_result.run_count,
            "seed": args.seed,
            "long_horizon_instability": horizon_result.long_horizon_instability,
            "per_step_instability": horizon_result.per_step_instability,
            "trajectories": [run.decisions for run in horizon_result.runs],
        }
        output_path = Path(args.output)
        _write_json(output_path, horizon_payload)
        if args.manifest_output:
            manifest = build_manifest(
                command="horizon",
                output_path=output_path,
                output_payload=horizon_payload,
                input_config={
                    "prompt": args.prompt,
                    "horizon": args.horizon,
                    "run_count": args.run_count,
                    "seed": args.seed,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    if args.command == "demo":
        agent_fn = _resolve_agent(args)
        embedding_provider = _resolve_embedding_provider(args)
        embedding_model = _resolve_embedding_model(args)
        embedding_api_key = os.getenv(args.openai_api_key_env)
        output_dir = Path(args.output_dir)
        summary = _run_demo_bundle(
            output_dir=output_dir,
            run_count=args.run_count,
            seed=args.seed,
            horizon=args.horizon,
            fixed_timestamp=args.fixed_timestamp,
            asi_profile=ASIProfile(args.asi_profile),
            mutation_limit=args.mutation_limit,
            agent_fn=agent_fn,
            agent_provider=args.agent_provider,
            agent_model=args.agent_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_openai_api_key=embedding_api_key,
        )
        if args.manifest_output:
            manifest = build_manifest(
                command="demo",
                output_path=output_dir / "summary.json",
                output_payload=summary,
                input_config={
                    "output_dir": args.output_dir,
                    "run_count": args.run_count,
                    "seed": args.seed,
                    "horizon": args.horizon,
                    "agent_provider": args.agent_provider,
                    "agent_model": args.agent_model,
                    "embedding_provider": args.embedding_provider,
                    "embedding_model": args.embedding_model,
                    "asi_profile": args.asi_profile,
                    "mutation_limit": args.mutation_limit,
                    "fixed_timestamp": args.fixed_timestamp,
                },
                timestamp_utc=args.fixed_timestamp,
            )
            _write_json(Path(args.manifest_output), manifest)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
