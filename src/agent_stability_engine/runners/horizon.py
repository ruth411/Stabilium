from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HorizonRun:
    decisions: list[str]


@dataclass(frozen=True)
class LongHorizonResult:
    horizon: int
    run_count: int
    long_horizon_instability: float
    per_step_instability: list[float]
    runs: list[HorizonRun]


class LongHorizonStabilityRunner:
    """Evaluates decision stability across multi-step trajectories."""

    def run(
        self,
        policy_fn: Callable[..., str],
        initial_context: str,
        horizon: int,
        run_count: int,
        seed: int = 0,
    ) -> LongHorizonResult:
        if horizon <= 0:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        if run_count <= 0:
            msg = "run_count must be >= 1"
            raise ValueError(msg)

        runs: list[HorizonRun] = []
        for run_idx in range(run_count):
            rng = random.Random(seed + run_idx)
            context = initial_context
            decisions: list[str] = []
            for step in range(horizon):
                decision = self._invoke_policy(policy_fn, context, step, rng)
                decisions.append(decision)
                context = f"{context}\nStep {step + 1} decision: {decision}"
            runs.append(HorizonRun(decisions=decisions))

        per_step_instability: list[float] = []
        for step in range(horizon):
            step_decisions = [run.decisions[step] for run in runs]
            unique_count = len(set(step_decisions))
            if run_count == 1:
                instability = 0.0
            else:
                instability = (unique_count - 1) / (run_count - 1)
            per_step_instability.append(instability)

        instability = sum(per_step_instability) / len(per_step_instability)
        return LongHorizonResult(
            horizon=horizon,
            run_count=run_count,
            long_horizon_instability=instability,
            per_step_instability=per_step_instability,
            runs=runs,
        )

    def _invoke_policy(
        self,
        policy_fn: Callable[..., str],
        context: str,
        step: int,
        rng: random.Random,
    ) -> str:
        try:
            return policy_fn(context, step, rng)
        except TypeError:
            try:
                return policy_fn(context, rng)
            except TypeError:
                return policy_fn(context)
