#!/usr/bin/env python3
"""Phase 1 Validation Study: compare ASI scores across multiple LLMs.

Usage (demo, no API key needed):
    python scripts/validate_models.py --models demo

Usage (OpenAI models, requires OPENAI_API_KEY):
    python scripts/validate_models.py --models gpt-4o-mini gpt-4o gpt-4.1-mini

Usage (Anthropic models, requires ANTHROPIC_API_KEY):
    python scripts/validate_models.py --models claude-haiku-4-5 claude-sonnet-4-6

Usage (cross-provider comparison):
    python scripts/validate_models.py --models gpt-4o-mini claude-haiku-4-5

Usage (temperature sensitivity):
    python scripts/validate_models.py --models gpt-4o-mini@0 gpt-4o-mini@1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_stability_engine.adapters.anthropic import AnthropicChatAdapter
from agent_stability_engine.adapters.openai import OpenAIChatAdapter
from agent_stability_engine.engine.asi import ASIProfile
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.engine.stats import compare_sample_means
from agent_stability_engine.runners.benchmark import run_benchmark_suite
from agent_stability_engine.runners.conversation_benchmark import run_conversation_benchmark_suite

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


def _make_progress_callback(model: str, total: int, start_time: float) -> object:
    """Returns a thread-safe callback that prints a live progress bar."""
    bar_width = 28

    def callback(completed: int, _total: int, case_id: str) -> None:
        elapsed = time.monotonic() - start_time
        pct = completed / _total if _total else 0
        eta_str = ""
        if completed > 0:
            eta_secs = int((elapsed / completed) * (_total - completed))
            eta_str = f" | ETA ~{eta_secs // 60}m{eta_secs % 60:02d}s"
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        domain = case_id.split("-")[0] if "-" in case_id else case_id
        elapsed_str = f"{int(elapsed) // 60}m{int(elapsed) % 60:02d}s"
        line = (
            f"\r  [{model}] {bar} {completed}/{_total}"
            f" ({pct:.0%})  {elapsed_str}{eta_str}  [{domain}]   "
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        if completed == _total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return callback


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def _demo_agent(prompt: str, rng: random.Random | None = None) -> str:
    """Deterministic stub — useful for free dry runs."""
    words = prompt.lower().split()
    return (
        f"Based on the request, here is a structured response. "
        f"The key considerations are: {', '.join(words[:4])}. "
        f"This analysis follows best practices and established guidelines."
    )


class _DemoAdapter:
    """Demo adapter that supports both single-turn and conversation interfaces."""

    def __call__(self, prompt: str, rng: random.Random | None = None) -> str:
        return _demo_agent(prompt, rng)

    def call_messages(
        self,
        messages: list[dict[str, str]],
        rng: random.Random | None = None,
    ) -> str:
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        prompt = (
            user_messages[-1] if user_messages else (messages[-1]["content"] if messages else "")
        )
        return _demo_agent(prompt, rng)


def _parse_model_spec(spec: str) -> tuple[str, float | None]:
    """Parse 'gpt-4o-mini@0.5' into ('gpt-4o-mini', 0.5). No '@' → temperature=None."""
    if "@" in spec:
        model_name, temp_str = spec.rsplit("@", 1)
        return model_name, float(temp_str)
    return spec, None


def _build_agent(
    model_spec: str,
    openai_key: str | None,
    anthropic_key: str | None,
) -> Callable[..., str]:
    if model_spec == "demo":
        return _DemoAdapter()
    model_name, temperature = _parse_model_spec(model_spec)
    if model_name.startswith("claude-"):
        return AnthropicChatAdapter(
            model=model_name, api_key=anthropic_key, temperature=temperature
        )
    return OpenAIChatAdapter(model=model_name, api_key=openai_key, temperature=temperature)


def _build_agent_factory(
    model_spec: str,
    openai_key: str | None,
    anthropic_key: str | None,
) -> Callable[[], Callable[..., str]]:
    return lambda: _build_agent(model_spec, openai_key, anthropic_key)


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _extract_metrics(benchmark_report: dict[str, object]) -> dict[str, float]:
    """Pull per-metric means from all cases in a benchmark report."""
    cases = benchmark_report.get("cases", [])
    if not cases:
        return {}

    sums: dict[str, float] = {
        "agent_stability_index": 0.0,
        "semantic_variance": 0.0,
        "contradiction_rate": 0.0,
        "mutation_degradation": 0.0,
        "cross_model_disagreement": 0.0,
        "tool_misuse_frequency": 0.0,
    }
    count = 0
    for case in cases:
        report = case.get("report", {})
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        sums["agent_stability_index"] += float(metrics.get("agent_stability_index", 0))
        sv = metrics.get("semantic_variance", {})
        sums["semantic_variance"] += float(sv.get("normalized", 0) if isinstance(sv, dict) else 0)
        sums["contradiction_rate"] += float(metrics.get("contradiction_rate", 0))
        sums["mutation_degradation"] += float(metrics.get("mutation_degradation", 0))
        sums["cross_model_disagreement"] += float(metrics.get("cross_model_disagreement", 0))
        sums["tool_misuse_frequency"] += float(metrics.get("tool_misuse_frequency", 0))
        count += 1

    if count == 0:
        return {}
    return {k: round(v / count, 4) for k, v in sums.items()}


def _extract_conversation_metrics(benchmark_report: dict[str, object]) -> dict[str, float]:
    """Pull per-metric means from a conversation benchmark report."""
    cases = benchmark_report.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return {}

    sums: dict[str, float] = {
        "agent_stability_index": 0.0,
        "conv_asi": 0.0,
        "cross_run_variance": 0.0,
        "turn_contradiction_rate": 0.0,
        "context_failure_rate": 0.0,
        "constraint_violation_rate": 0.0,
        "drift_rate": 0.0,
    }
    count = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        report = case.get("report", {})
        if not isinstance(report, dict):
            continue
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        conv_asi = float(metrics.get("conv_asi", 0))
        sums["agent_stability_index"] += conv_asi
        sums["conv_asi"] += conv_asi
        sums["cross_run_variance"] += float(metrics.get("cross_run_variance", 0))
        sums["turn_contradiction_rate"] += float(metrics.get("turn_contradiction_rate", 0))
        sums["context_failure_rate"] += float(metrics.get("context_failure_rate", 0))
        sums["constraint_violation_rate"] += float(metrics.get("constraint_violation_rate", 0))
        sums["drift_rate"] += float(metrics.get("drift_rate", 0))
        count += 1

    if count == 0:
        return {}
    return {k: round(v / count, 4) for k, v in sums.items()}


def _extract_mode_metrics(benchmark_report: dict[str, object], mode: str) -> dict[str, float]:
    if mode == "conversation":
        return _extract_conversation_metrics(benchmark_report)
    return _extract_metrics(benchmark_report)


def _extract_case_asi_values(benchmark_report: dict[str, object], mode: str) -> list[float]:
    raw_values = (
        benchmark_report.get("case_conv_asi_values")
        if mode == "conversation"
        else benchmark_report.get("case_asi_values")
    )
    if not isinstance(raw_values, list):
        return []
    values: list[float] = []
    for item in raw_values:
        if isinstance(item, (int, float)):
            values.append(float(item))
    return values


def _pairwise_significance(
    model_case_asi: dict[str, list[float]],
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> list[dict[str, object]]:
    models = sorted(model_case_asi.keys())
    comparisons: list[dict[str, object]] = []
    for left_idx in range(len(models)):
        for right_idx in range(left_idx + 1, len(models)):
            left_model = models[left_idx]
            right_model = models[right_idx]
            left_values = model_case_asi[left_model]
            right_values = model_case_asi[right_model]
            if not left_values or not right_values:
                continue

            result = compare_sample_means(
                left_values,
                right_values,
                alpha=alpha,
                confidence_level=confidence_level,
            )
            comparisons.append(
                {
                    "left_model": left_model,
                    "right_model": right_model,
                    "delta_mean": result["delta_mean"],
                    "delta_ci_low": result["delta_ci_low"],
                    "delta_ci_high": result["delta_ci_high"],
                    "p_value": result["p_value"],
                    "significant_difference": result["significant_difference"],
                    "better_model": (
                        left_model
                        if result["better_sample"] == "left"
                        else right_model if result["better_sample"] == "right" else "tie"
                    ),
                    "alpha": alpha,
                    "confidence_level": confidence_level,
                    "method": result["method"],
                }
            )
    return comparisons


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _generate_markdown(comparison: dict[str, object]) -> str:
    raw_models = comparison.get("models")
    models: dict[str, dict[str, object]] = raw_models if isinstance(raw_models, dict) else {}
    suite = comparison.get("suite")
    run_count = comparison.get("run_count")
    generated_at = comparison.get("generated_at")
    num_cases = comparison.get("num_cases", "?")
    mode = str(comparison.get("mode", "single"))
    is_conversation = mode == "conversation"

    sorted_models = sorted(
        models.items(),
        key=lambda x: float(x[1].get("agent_stability_index", 0.0)),
        reverse=True,
    )

    lines = [
        "# ASE Model Stability Comparison",
        "",
        f"**Suite:** `{suite}`  ",
        f"**Mode:** `{mode}`  ",
        f"**Cases:** {num_cases}  ",
        f"**Runs per case:** {run_count}  ",
        f"**Generated:** {generated_at}",
        "",
        "## Results",
        "",
        (
            "| Rank | Model | ConvASI | ConvASI 95% CI | Cross-Run Variance | "
            "Turn Contradiction | Context Failure | Constraint Violation | Drift |"
            if is_conversation
            else (
                "| Rank | Model | ASI | ASI 95% CI | Variance | "
                "Contradiction | Mutation Δ | Tool Misuse |"
            )
        ),
        (
            "|------|-------|---------|----------------|--------------------|"
            "--------------------|----------------|----------------------|-------|"
            if is_conversation
            else (
                "|------|-------|-----|------------|----------|"
                "---------------|------------|-------------|"
            )
        ),
    ]

    for rank, (model, m) in enumerate(sorted_models, 1):
        asi = float(m.get("agent_stability_index", 0.0))
        ci_low_obj = m.get("asi_ci95_low")
        ci_high_obj = m.get("asi_ci95_high")
        ci_low = float(ci_low_obj) if isinstance(ci_low_obj, (int, float)) else None
        ci_high = float(ci_high_obj) if isinstance(ci_high_obj, (int, float)) else None
        ci_text = "—" if ci_low is None or ci_high is None else f"[{ci_low:.1f}, {ci_high:.1f}]"
        if is_conversation:
            row = (
                f"| {rank} | `{model}` | **{asi:.1f}** | {ci_text} | "
                f"{float(m.get('cross_run_variance', 0.0)):.4f} | "
                f"{float(m.get('turn_contradiction_rate', 0.0)):.4f} | "
                f"{float(m.get('context_failure_rate', 0.0)):.4f} | "
                f"{float(m.get('constraint_violation_rate', 0.0)):.4f} | "
                f"{float(m.get('drift_rate', 0.0)):.4f} |"
            )
        else:
            var = float(m.get("semantic_variance", 0.0))
            contra = float(m.get("contradiction_rate", 0.0))
            mut = float(m.get("mutation_degradation", 0.0))
            tool = float(m.get("tool_misuse_frequency", 0.0))
            row = (
                f"| {rank} | `{model}` | **{asi:.1f}**"
                f" | {ci_text} | {var:.4f} | {contra:.4f} | {mut:.4f} | {tool:.4f} |"
            )
        lines.append(row)

    # --- Domain breakdown ---
    all_domains: set[str] = set()
    for m in models.values():
        domains = m.get("_domains", {})
        if isinstance(domains, dict):
            all_domains.update(domains.keys())

    if all_domains:
        sorted_domains = sorted(all_domains)
        header_cols = " | ".join(f"`{m}`" for m, _ in sorted_models)
        sep_cols = " | ".join("------" for _ in sorted_models)
        lines += [
            "",
            "## Domain Breakdown (ASI per domain)",
            "",
            f"| Domain | {header_cols} |",
            f"|--------|{sep_cols}|",
        ]
        for domain in sorted_domains:
            row_vals = []
            for _model, m in sorted_models:
                domains = m.get("_domains", {})
                score = domains.get(domain, None) if isinstance(domains, dict) else None
                row_vals.append(f"**{score:.1f}**" if score is not None else "—")
            lines.append(f"| {domain} | {' | '.join(row_vals)} |")

    raw_pairwise = comparison.get("pairwise_significance")
    pairwise = raw_pairwise if isinstance(raw_pairwise, list) else []
    if pairwise:
        lines += [
            "",
            "## Model-Swap Significance",
            "",
            "| Comparison | Δ Mean ASI | 95% CI (Δ) | p-value | Significant | Better |",
            "|------------|------------|------------|---------|-------------|--------|",
        ]
        for row in pairwise:
            if not isinstance(row, dict):
                continue
            left_model = str(row.get("left_model", "left"))
            right_model = str(row.get("right_model", "right"))
            delta = float(row.get("delta_mean", 0.0))
            ci_low = float(row.get("delta_ci_low", 0.0))
            ci_high = float(row.get("delta_ci_high", 0.0))
            p_value = float(row.get("p_value", 1.0))
            significant = bool(row.get("significant_difference", False))
            better = str(row.get("better_model", "tie"))
            lines.append(
                f"| `{left_model}` vs `{right_model}` | {delta:.2f} | "
                f"[{ci_low:.2f}, {ci_high:.2f}] | {p_value:.4f} | "
                f"{'yes' if significant else 'no'} | `{better}` |"
            )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "| Metric | Range | Better when |",
        "|--------|-------|-------------|",
    ]
    if is_conversation:
        lines += [
            "| **ConvASI** (Conversation ASI) | 0–100 | Higher |",
            "| **Cross-Run Variance** | 0–1 | Lower — stable final responses across runs |",
            "| **Turn Contradiction Rate** | 0–1 | Lower — fewer contradictions across turns |",
            "| **Context Failure Rate** | 0–1 | Lower — better memory/context retention |",
            "| **Constraint Violation Rate** | 0–1 | Lower — better instruction adherence |",
            "| **Drift Rate** | 0–1 | Lower — less behavior drift over turns |",
        ]
    else:
        lines += [
            "| **ASI** (Agent Stability Index) | 0–100 | Higher |",
            "| **Variance** | 0–1 | Lower — consistent outputs across identical prompts |",
            "| **Contradiction Rate** | 0–1 | Lower — model doesn't contradict itself |",
            "| **Mutation Degradation** | 0–1 | Lower — robust to prompt perturbations |",
            "| **Tool Misuse** | 0–1 | Lower — correct tool usage |",
        ]
    lines += [
        "",
        "## Key Findings",
        "",
    ]

    if sorted_models:
        best_model, best_m = sorted_models[0]
        worst_model, worst_m = sorted_models[-1]
        best_asi = float(best_m.get("agent_stability_index", 0.0))
        worst_asi = float(worst_m.get("agent_stability_index", 0.0))

        lines.append(f"- **Most stable:** `{best_model}` with ASI {best_asi:.1f}")
        if len(sorted_models) > 1:
            lines.append(f"- **Least stable:** `{worst_model}` with ASI {worst_asi:.1f}")
            gap = best_asi - worst_asi
            lines.append(f"- **Stability gap:** {gap:.1f} ASI points between best and worst")

        # Flag any model with high contradiction
        high_contra = [
            (m, d) for m, d in sorted_models if float(d.get("contradiction_rate", 0.0)) > 0.15
        ]
        if high_contra:
            names = ", ".join(f"`{m}`" for m, _ in high_contra)
            lines.append(f"- **High contradiction warning:** {names} (rate > 0.15)")

        # Flag mutation sensitivity
        high_mut = [
            (m, d) for m, d in sorted_models if float(d.get("mutation_degradation", 0.0)) > 0.3
        ]
        if high_mut:
            names = ", ".join(f"`{m}`" for m, _ in high_mut)
            lines.append(f"- **Mutation-sensitive:** {names} (degradation > 0.30)")

    lines += [
        "",
        "---",
        "*Generated by [Agent Stability Engine](https://github.com/ruth411/Stabilium)*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ASE Phase 1 Validation: compare stability across LLMs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["demo"],
        metavar="MODEL",
        help="Models to benchmark. Use 'demo' for free dry run, or e.g. gpt-4o-mini gpt-4o",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Path to benchmark suite JSON (defaults by mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "conversation"],
        default="single",
        help="Validation mode: single-turn ASI or conversation ConvASI",
    )
    parser.add_argument("--run-count", type=int, default=3, help="Runs per case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-limit", type=int, default=6, help="Max mutations per case")
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Cap number of cases (useful for quick checks)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers for case evaluation"
    )
    parser.add_argument(
        "--asi-profile",
        default="balanced",
        choices=["balanced", "safety_strict", "reasoning_focus"],
    )
    parser.add_argument(
        "--embedding-provider",
        default="hash",
        choices=["hash", "openai", "sentence_transformers", "auto"],
        help="Embedding provider for semantic variance. 'hash' is fast but not semantic.",
    )
    parser.add_argument("--output-dir", default="out/validation")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    real_models = [m for m in args.models if m != "demo" and not m.startswith("demo@")]
    openai_models = [m for m in real_models if not _parse_model_spec(m)[0].startswith("claude-")]
    anthropic_models = [m for m in real_models if _parse_model_spec(m)[0].startswith("claude-")]
    if openai_models and not openai_key:
        print("ERROR: OPENAI_API_KEY is required for OpenAI models.")
        print(f"  Models needing key: {openai_models}")
        return 1
    if anthropic_models and not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY is required for Anthropic (Claude) models.")
        print(f"  Models needing key: {anthropic_models}")
        return 1

    profile = ASIProfile(args.asi_profile)
    if args.suite is not None:
        suite_path = Path(args.suite)
    elif args.mode == "conversation":
        suite_path = Path("examples/benchmarks/conversation_suite.json")
    else:
        suite_path = Path("examples/benchmarks/default_suite.json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ASE Model Stability Validation Study")
    print("=" * 60)
    print(f"  Suite:      {suite_path}")
    print(f"  Mode:       {args.mode}")
    print(f"  Models:     {', '.join(args.models)}")
    print(f"  Runs/case:  {args.run_count}")
    print(f"  Profile:    {args.asi_profile}")
    print(f"  Embeddings: {args.embedding_provider}")
    print("=" * 60)

    all_metrics: dict[str, dict[str, object]] = {}
    all_case_asi: dict[str, list[float]] = {}
    num_cases = 0

    for model in args.models:
        print(f"\n[{model}] Starting benchmark...")
        agent_factory = _build_agent_factory(model, openai_key, anthropic_key)
        agent_fn = agent_factory()
        _start = time.monotonic()

        if args.mode == "conversation":
            result = run_conversation_benchmark_suite(
                suite_path=suite_path,
                adapter=agent_fn,
                run_count=args.run_count,
                seed=args.seed,
                embedding_provider=EmbeddingProvider(args.embedding_provider),
                max_cases=args.max_cases,
                workers=args.workers,
                progress_callback=_make_progress_callback(model, 0, _start),
                agent_factory=agent_factory if args.workers > 1 else None,
            )
        else:
            result = run_benchmark_suite(
                suite_path=suite_path,
                agent_fn=agent_fn,
                run_count=args.run_count,
                seed=args.seed,
                asi_profile=profile,
                mutation_sample_limit=args.mutation_limit,
                embedding_provider=EmbeddingProvider(args.embedding_provider),
                max_cases=args.max_cases,
                workers=args.workers,
                progress_callback=_make_progress_callback(model, 0, _start),
                agent_factory=agent_factory if args.workers > 1 else None,
            )

        report = result.report
        num_cases = int(report.get("num_cases", 0))

        # Save per-model full report
        safe_name = model.replace("/", "_").replace("-", "_").replace(".", "_")
        model_path = output_dir / f"{safe_name}.json"
        model_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        metrics = _extract_mode_metrics(report, args.mode)
        asi_statistics = (
            report.get("conv_asi_statistics")
            if args.mode == "conversation"
            else report.get("asi_statistics")
        )
        if isinstance(asi_statistics, dict):
            ci_low = asi_statistics.get("ci_low")
            ci_high = asi_statistics.get("ci_high")
            sample_size = asi_statistics.get("sample_size")
            if isinstance(ci_low, (int, float)):
                metrics["asi_ci95_low"] = float(ci_low)
            if isinstance(ci_high, (int, float)):
                metrics["asi_ci95_high"] = float(ci_high)
            if isinstance(sample_size, int):
                metrics["sample_size"] = sample_size
        domain_scores = report.get("domain_scores", {})
        if isinstance(domain_scores, dict):
            metrics["_domains"] = domain_scores
        all_case_asi[model] = _extract_case_asi_values(report, args.mode)
        all_metrics[model] = metrics

        asi = metrics.get("agent_stability_index", report.get("mean_asi", 0))
        asi_float = float(asi) if isinstance(asi, (int, float)) else 0.0
        label = "ConvASI" if args.mode == "conversation" else "ASI"
        print(
            f"[{model}] {label} = {asi_float:.1f}  |  "
            f"cases={num_cases}  →  saved {model_path.name}"
        )

    pairwise_significance = _pairwise_significance(all_case_asi)

    # Build comparison
    comparison: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite_path),
        "run_count": args.run_count,
        "seed": args.seed,
        "asi_profile": args.asi_profile,
        "mutation_limit": args.mutation_limit,
        "mode": args.mode,
        "num_cases": num_cases,
        "models": all_metrics,
        "pairwise_significance": pairwise_significance,
    }

    comparison_json = output_dir / "model_comparison.json"
    comparison_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    markdown = _generate_markdown(comparison)
    comparison_md = output_dir / "model_comparison.md"
    comparison_md.write_text(markdown, encoding="utf-8")

    print("\n" + "=" * 60)
    print(markdown)
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  {comparison_json.name}")
    print(f"  {comparison_md.name}")
    print("  + one JSON per model")

    return 0


if __name__ == "__main__":
    sys.exit(main())
