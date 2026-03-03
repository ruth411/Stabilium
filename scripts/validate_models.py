#!/usr/bin/env python3
"""Phase 1 Validation Study: compare ASI scores across multiple LLMs.

Usage (demo, no API key needed):
    python scripts/validate_models.py --models demo

Usage (real models, requires OPENAI_API_KEY):
    python scripts/validate_models.py --models gpt-4o-mini gpt-4o gpt-4.1-mini

Usage (mix):
    python scripts/validate_models.py --models demo gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_stability_engine.adapters.openai import OpenAIChatAdapter
from agent_stability_engine.engine.asi import ASIProfile
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.runners.benchmark import run_benchmark_suite


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


def _build_agent(model: str, api_key: str | None) -> object:
    if model == "demo":
        return _demo_agent
    return OpenAIChatAdapter(model=model, api_key=api_key)


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


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _generate_markdown(comparison: dict[str, object]) -> str:
    models: dict[str, dict[str, float]] = comparison["models"]  # type: ignore[assignment]
    suite = comparison["suite"]
    run_count = comparison["run_count"]
    generated_at = comparison["generated_at"]
    num_cases = comparison.get("num_cases", "?")

    sorted_models = sorted(models.items(), key=lambda x: x[1].get("agent_stability_index", 0), reverse=True)

    lines = [
        "# ASE Model Stability Comparison",
        "",
        f"**Suite:** `{suite}`  ",
        f"**Cases:** {num_cases}  ",
        f"**Runs per case:** {run_count}  ",
        f"**Generated:** {generated_at}",
        "",
        "## Results",
        "",
        "| Rank | Model | ASI ↓ better=higher | Variance | Contradiction | Mutation Δ | Tool Misuse |",
        "|------|-------|---------------------|----------|---------------|------------|-------------|",
    ]

    for rank, (model, m) in enumerate(sorted_models, 1):
        asi = m.get("agent_stability_index", 0)
        var = m.get("semantic_variance", 0)
        contra = m.get("contradiction_rate", 0)
        mut = m.get("mutation_degradation", 0)
        tool = m.get("tool_misuse_frequency", 0)
        lines.append(
            f"| {rank} | `{model}` | **{asi:.1f}** | {var:.4f} | {contra:.4f} | {mut:.4f} | {tool:.4f} |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "| Metric | Range | Better when |",
        "|--------|-------|-------------|",
        "| **ASI** (Agent Stability Index) | 0–100 | Higher |",
        "| **Variance** | 0–1 | Lower — consistent outputs across identical prompts |",
        "| **Contradiction Rate** | 0–1 | Lower — model doesn't contradict itself |",
        "| **Mutation Degradation** | 0–1 | Lower — robust to prompt perturbations |",
        "| **Tool Misuse** | 0–1 | Lower — correct tool usage |",
        "",
        "## Key Findings",
        "",
    ]

    if sorted_models:
        best_model, best_m = sorted_models[0]
        worst_model, worst_m = sorted_models[-1]
        best_asi = best_m.get("agent_stability_index", 0)
        worst_asi = worst_m.get("agent_stability_index", 0)

        lines.append(f"- **Most stable:** `{best_model}` with ASI {best_asi:.1f}")
        if len(sorted_models) > 1:
            lines.append(f"- **Least stable:** `{worst_model}` with ASI {worst_asi:.1f}")
            gap = best_asi - worst_asi
            lines.append(f"- **Stability gap:** {gap:.1f} ASI points between best and worst")

        # Flag any model with high contradiction
        high_contra = [(m, d) for m, d in sorted_models if d.get("contradiction_rate", 0) > 0.15]
        if high_contra:
            names = ", ".join(f"`{m}`" for m, _ in high_contra)
            lines.append(f"- **High contradiction warning:** {names} (rate > 0.15)")

        # Flag mutation sensitivity
        high_mut = [(m, d) for m, d in sorted_models if d.get("mutation_degradation", 0) > 0.3]
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
        default="examples/benchmarks/default_suite.json",
        help="Path to benchmark suite JSON",
    )
    parser.add_argument("--run-count", type=int, default=3, help="Runs per case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-limit", type=int, default=6, help="Max mutations per case")
    parser.add_argument("--asi-profile", default="balanced", choices=["balanced", "safety_strict", "reasoning_focus"])
    parser.add_argument("--output-dir", default="out/validation")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    real_models = [m for m in args.models if m != "demo"]
    if real_models and not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required for non-demo models.")
        print(f"  Models needing API key: {real_models}")
        return 1

    profile = ASIProfile(args.asi_profile)
    suite_path = Path(args.suite)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ASE Model Stability Validation Study")
    print("=" * 60)
    print(f"  Suite:      {suite_path}")
    print(f"  Models:     {', '.join(args.models)}")
    print(f"  Runs/case:  {args.run_count}")
    print(f"  Profile:    {args.asi_profile}")
    print("=" * 60)

    all_metrics: dict[str, dict[str, float]] = {}
    num_cases = 0

    for model in args.models:
        print(f"\n[{model}] Starting benchmark...")
        agent_fn = _build_agent(model, api_key)

        result = run_benchmark_suite(
            suite_path=suite_path,
            agent_fn=agent_fn,
            run_count=args.run_count,
            seed=args.seed,
            asi_profile=profile,
            mutation_sample_limit=args.mutation_limit,
            embedding_provider=EmbeddingProvider.HASH,  # fast, no deps
        )

        report = result.report
        num_cases = int(report.get("num_cases", 0))

        # Save per-model full report
        safe_name = model.replace("/", "_").replace("-", "_").replace(".", "_")
        model_path = output_dir / f"{safe_name}.json"
        model_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        metrics = _extract_metrics(report)
        all_metrics[model] = metrics

        asi = metrics.get("agent_stability_index", report.get("mean_asi", 0))
        print(f"[{model}] ASI = {asi:.1f}  |  cases={num_cases}  →  saved {model_path.name}")

    # Build comparison
    comparison: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": str(suite_path),
        "run_count": args.run_count,
        "seed": args.seed,
        "asi_profile": args.asi_profile,
        "mutation_limit": args.mutation_limit,
        "num_cases": num_cases,
        "models": all_metrics,
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
    print(f"  + one JSON per model")

    return 0


if __name__ == "__main__":
    sys.exit(main())
