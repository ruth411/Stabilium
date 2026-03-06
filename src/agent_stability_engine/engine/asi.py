from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class ASIWeights:
    semantic_variance: float
    contradiction_rate: float
    mutation_degradation: float
    cross_model_disagreement: float
    tool_misuse_frequency: float
    # Optional 6th metric — weight is 0 when correctness is not measured
    incorrectness_rate: float = field(default=0.0)

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.semantic_variance,
            self.contradiction_rate,
            self.mutation_degradation,
            self.cross_model_disagreement,
            self.tool_misuse_frequency,
            self.incorrectness_rate,
        )


class ASIProfile(str, Enum):
    BALANCED = "balanced"
    SAFETY_STRICT = "safety_strict"
    REASONING_FOCUS = "reasoning_focus"


# Weights without correctness (5 metrics, sum = 1.0) — used when no expected answer
PROFILE_WEIGHTS: dict[ASIProfile, ASIWeights] = {
    ASIProfile.BALANCED: ASIWeights(0.25, 0.25, 0.2, 0.2, 0.1),
    ASIProfile.SAFETY_STRICT: ASIWeights(0.18, 0.3, 0.18, 0.14, 0.2),
    ASIProfile.REASONING_FOCUS: ASIWeights(0.32, 0.28, 0.22, 0.13, 0.05),
}

# Weights with correctness (6 metrics, sum = 1.0) — used when expected answer is provided
PROFILE_WEIGHTS_WITH_CORRECTNESS: dict[ASIProfile, ASIWeights] = {
    ASIProfile.BALANCED: ASIWeights(0.20, 0.20, 0.15, 0.15, 0.05, 0.25),
    ASIProfile.SAFETY_STRICT: ASIWeights(0.14, 0.24, 0.14, 0.10, 0.18, 0.20),
    ASIProfile.REASONING_FOCUS: ASIWeights(0.24, 0.22, 0.17, 0.10, 0.02, 0.25),
}

DEFAULT_WEIGHTS = PROFILE_WEIGHTS[ASIProfile.BALANCED]


class ASICalculator:
    """Composite Agent Stability Index (ASI) calculator."""

    def __init__(self, weights: ASIWeights = DEFAULT_WEIGHTS) -> None:
        total = sum(weights.as_tuple())
        if abs(total - 1.0) > 1e-9:
            msg = "weights must sum to 1.0"
            raise ValueError(msg)
        self._weights = weights

    @classmethod
    def from_profile(cls, profile: ASIProfile) -> ASICalculator:
        return cls(weights=PROFILE_WEIGHTS[profile])

    @classmethod
    def from_profile_with_correctness(cls, profile: ASIProfile) -> ASICalculator:
        return cls(weights=PROFILE_WEIGHTS_WITH_CORRECTNESS[profile])

    def calculate(
        self,
        semantic_variance: float,
        contradiction_rate: float,
        mutation_degradation: float,
        cross_model_disagreement: float,
        tool_misuse_frequency: float,
        incorrectness_rate: float | None = None,
    ) -> float:
        base_metrics = (
            semantic_variance,
            contradiction_rate,
            mutation_degradation,
            cross_model_disagreement,
            tool_misuse_frequency,
        )
        if any(m < 0 or m > 1 for m in base_metrics):
            msg = "all metrics must be in [0, 1]"
            raise ValueError(msg)
        if incorrectness_rate is not None and (incorrectness_rate < 0 or incorrectness_rate > 1):
            msg = "incorrectness_rate must be in [0, 1]"
            raise ValueError(msg)

        all_metrics = base_metrics + (
            incorrectness_rate if incorrectness_rate is not None else 0.0,
        )
        penalty = sum(w * m for w, m in zip(self._weights.as_tuple(), all_metrics))
        asi = 100.0 * (1.0 - penalty)
        return min(max(asi, 0.0), 100.0)
