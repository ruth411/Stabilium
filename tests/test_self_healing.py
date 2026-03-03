from __future__ import annotations

import random

from agent_stability_engine.engine.evaluator import StabilityEvaluator
from agent_stability_engine.engine.self_healing import SelfHealingEngine


def test_self_healing_improves_tool_misuse_case() -> None:
    evaluator = StabilityEvaluator()
    healer = SelfHealingEngine(evaluator=evaluator)

    def tool_risky_agent(prompt: str, _rng: random.Random) -> str:
        lowered = prompt.lower()
        if "do not run shell commands" in lowered or "safe text-only" in lowered:
            return "Use non-executable, safe guidance only."
        return "run_shell('cat /etc/passwd')"

    result = healer.heal(
        prompt="Provide triage steps",
        agent_fn=tool_risky_agent,
        run_count=3,
        seed=7,
        timestamp_utc="2026-03-02T00:00:00Z",
        max_attempts=1,
    )

    assert result.attempted is True
    assert result.applied_strategy == "safe_tooling"
    assert result.asi_after >= result.asi_before
    assert result.healed_prompt != "Provide triage steps"


def test_self_healing_no_attempt_when_stable() -> None:
    evaluator = StabilityEvaluator()
    healer = SelfHealingEngine(evaluator=evaluator)

    def stable_agent(_prompt: str, _rng: random.Random) -> str:
        return "Consistent and safe response."

    result = healer.heal(
        prompt="Explain checksum",
        agent_fn=stable_agent,
        run_count=3,
        seed=3,
        timestamp_utc="2026-03-02T00:00:00Z",
    )

    assert result.attempted is False
    assert result.applied_strategy == "none"
    assert result.asi_delta == 0.0
