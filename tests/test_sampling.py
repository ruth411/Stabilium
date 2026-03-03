from __future__ import annotations

import random

import pytest

from agent_stability_engine.engine.sampling import MultiRunSampler


def _toy_agent(prompt: str, rng: random.Random) -> str:
    choices = ["alpha", "beta", "gamma"]
    return f"{prompt}:{choices[rng.randrange(len(choices))]}"


def test_sampling_is_deterministic_for_same_seed() -> None:
    sampler_a = MultiRunSampler[str, str](seed=7)
    sampler_b = MultiRunSampler[str, str](seed=7)

    runs_a = sampler_a.run(_toy_agent, "task", run_count=5)
    runs_b = sampler_b.run(_toy_agent, "task", run_count=5)

    assert runs_a.outputs == runs_b.outputs
    assert runs_a.run_count == 5
    assert runs_a.duration_seconds >= 0


def test_sampling_rejects_non_positive_run_count() -> None:
    sampler = MultiRunSampler[str, str]()

    with pytest.raises(ValueError, match="run_count"):
        sampler.run(_toy_agent, "task", run_count=0)
