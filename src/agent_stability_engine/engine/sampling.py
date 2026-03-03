from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class MultiRunResult(Generic[OutputT]):
    run_count: int
    outputs: list[OutputT]
    duration_seconds: float


class MultiRunSampler(Generic[InputT, OutputT]):
    """Deterministic wrapper for repeated model or agent execution."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def run(
        self,
        fn: Callable[[InputT, random.Random], OutputT],
        payload: InputT,
        run_count: int,
    ) -> MultiRunResult[OutputT]:
        if run_count <= 0:
            msg = "run_count must be >= 1"
            raise ValueError(msg)

        started = time.perf_counter()
        outputs: list[OutputT] = []
        for idx in range(run_count):
            rng = random.Random(self._seed + idx)
            outputs.append(fn(payload, rng))
        duration_seconds = time.perf_counter() - started
        return MultiRunResult(
            run_count=run_count,
            outputs=outputs,
            duration_seconds=duration_seconds,
        )
