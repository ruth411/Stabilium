from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from agent_stability_engine.engine.alignment import GoalSpec
from agent_stability_engine.engine.evaluator import StabilityEvaluator
from agent_stability_engine.taxonomy.classifier import FailureType


@dataclass(frozen=True)
class SelfHealingResult:
    attempted: bool
    improved: bool
    primary_failure: str
    applied_strategy: str
    max_attempts: int
    baseline_report: dict[str, object]
    healed_report: dict[str, object]
    healed_prompt: str
    asi_before: float
    asi_after: float
    asi_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "improved": self.improved,
            "primary_failure": self.primary_failure,
            "applied_strategy": self.applied_strategy,
            "max_attempts": self.max_attempts,
            "asi_before": self.asi_before,
            "asi_after": self.asi_after,
            "asi_delta": self.asi_delta,
            "healed_prompt": self.healed_prompt,
            "baseline_report": self.baseline_report,
            "healed_report": self.healed_report,
        }


class SelfHealingEngine:
    """Applies failure-specific remediation prompts and selects the best recovery run."""

    def __init__(self, evaluator: StabilityEvaluator) -> None:
        self._evaluator = evaluator

    def heal(
        self,
        prompt: str,
        agent_fn: Callable[..., str],
        run_count: int,
        seed: int = 0,
        shadow_agents: dict[str, Callable[..., str]] | None = None,
        timestamp_utc: str | None = None,
        goal_spec: GoalSpec | None = None,
        baseline_reports: list[dict[str, object]] | None = None,
        max_attempts: int = 1,
    ) -> SelfHealingResult:
        if max_attempts < 1:
            msg = "max_attempts must be >= 1"
            raise ValueError(msg)

        baseline_eval = self._evaluator.evaluate(
            prompt=prompt,
            agent_fn=agent_fn,
            run_count=run_count,
            seed=seed,
            shadow_agents=shadow_agents,
            timestamp_utc=timestamp_utc,
            goal_spec=goal_spec,
            baseline_reports=baseline_reports,
        )
        baseline_report = baseline_eval.report
        base_asi = _extract_asi(baseline_report)

        primary_failure = self._extract_primary_failure(baseline_report)
        strategy_name = self._strategy_name(primary_failure)
        if strategy_name is None:
            return SelfHealingResult(
                attempted=False,
                improved=False,
                primary_failure=primary_failure,
                applied_strategy="none",
                max_attempts=max_attempts,
                baseline_report=baseline_report,
                healed_report=baseline_report,
                healed_prompt=prompt,
                asi_before=base_asi,
                asi_after=base_asi,
                asi_delta=0.0,
            )

        best_report = baseline_report
        best_prompt = prompt
        best_asi = base_asi

        for attempt in range(max_attempts):
            candidate_prompt = self._apply_strategy(strategy_name, prompt, attempt)
            candidate_eval = self._evaluator.evaluate(
                prompt=candidate_prompt,
                agent_fn=agent_fn,
                run_count=run_count,
                seed=seed,
                shadow_agents=shadow_agents,
                timestamp_utc=timestamp_utc,
                goal_spec=goal_spec,
                baseline_reports=baseline_reports,
            )
            candidate_report = candidate_eval.report
            candidate_asi = _extract_asi(candidate_report)
            if candidate_asi > best_asi:
                best_asi = candidate_asi
                best_report = candidate_report
                best_prompt = candidate_prompt

        improved = best_asi > base_asi
        return SelfHealingResult(
            attempted=True,
            improved=improved,
            primary_failure=primary_failure,
            applied_strategy=strategy_name,
            max_attempts=max_attempts,
            baseline_report=baseline_report,
            healed_report=best_report,
            healed_prompt=best_prompt,
            asi_before=base_asi,
            asi_after=best_asi,
            asi_delta=best_asi - base_asi,
        )

    def _extract_primary_failure(self, report: dict[str, object]) -> str:
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict):
            return FailureType.STABLE.value
        notes = artifacts.get("notes")
        if not isinstance(notes, str):
            return FailureType.STABLE.value
        match = re.search(r"\bprimary_failure=([a-z_]+)\b", notes)
        if match is None:
            return FailureType.STABLE.value
        return match.group(1)

    def _strategy_name(self, primary_failure: str) -> str | None:
        strategies = {
            FailureType.CONTRADICTION.value: "consistency_guard",
            FailureType.MUTATION_DEGRADATION.value: "objective_anchor",
            FailureType.CROSS_MODEL_DISAGREEMENT.value: "consensus_bias",
            FailureType.TOOL_MISUSE.value: "safe_tooling",
            FailureType.HIGH_VARIANCE.value: "deterministic_format",
        }
        return strategies.get(primary_failure)

    def _apply_strategy(self, strategy_name: str, prompt: str, attempt: int) -> str:
        if strategy_name == "consistency_guard":
            return (
                "Respond with logically consistent statements only. "
                "Do not contradict earlier claims.\n"
                f"Task: {prompt}"
            )

        if strategy_name == "objective_anchor":
            return (
                "Preserve the core objective exactly even if wording changes. "
                "Prioritize invariant requirements.\n"
                f"Task: {prompt}"
            )

        if strategy_name == "consensus_bias":
            return (
                "Provide a conservative consensus answer grounded in common facts. "
                "Avoid speculation.\n"
                f"Task: {prompt}"
            )

        if strategy_name == "safe_tooling":
            return (
                "Do not run shell commands or unsafe tools. "
                "Return a safe text-only answer.\n"
                f"Task: {prompt}"
            )

        if strategy_name == "deterministic_format":
            return (
                "Use deterministic structure with stable wording. "
                f"Attempt {attempt + 1}.\n"
                f"Task: {prompt}"
            )

        return prompt


def _extract_asi(report: dict[str, object]) -> float:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    asi = metrics.get("agent_stability_index")
    if isinstance(asi, (int, float)):
        return float(asi)
    return 0.0
