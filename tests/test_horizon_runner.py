from __future__ import annotations

import random

import pytest

from agent_stability_engine.runners.horizon import LongHorizonStabilityRunner


def test_horizon_runner_reports_instability() -> None:
    runner = LongHorizonStabilityRunner()

    def policy(_context: str, step: int, rng: random.Random) -> str:
        options = ["A", "B", "C"]
        return options[(step + rng.randrange(3)) % 3]

    result = runner.run(policy, "start", horizon=4, run_count=3, seed=1)
    assert result.horizon == 4
    assert result.run_count == 3
    assert len(result.per_step_instability) == 4
    assert 0 <= result.long_horizon_instability <= 1


def test_horizon_runner_rejects_invalid_horizon() -> None:
    runner = LongHorizonStabilityRunner()
    with pytest.raises(ValueError, match="horizon"):
        runner.run(lambda prompt: prompt, "x", horizon=0, run_count=2)
