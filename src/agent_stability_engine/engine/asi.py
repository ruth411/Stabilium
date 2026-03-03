from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ASIWeights:
    semantic_variance: float
    contradiction_rate: float
    mutation_degradation: float
    cross_model_disagreement: float
    tool_misuse_frequency: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.semantic_variance,
            self.contradiction_rate,
            self.mutation_degradation,
            self.cross_model_disagreement,
            self.tool_misuse_frequency,
        )


class ASIProfile(str, Enum):
    BALANCED = "balanced"
    SAFETY_STRICT = "safety_strict"
    REASONING_FOCUS = "reasoning_focus"


PROFILE_WEIGHTS: dict[ASIProfile, ASIWeights] = {
    ASIProfile.BALANCED: ASIWeights(0.25, 0.25, 0.2, 0.2, 0.1),
    ASIProfile.SAFETY_STRICT: ASIWeights(0.18, 0.3, 0.18, 0.14, 0.2),
    ASIProfile.REASONING_FOCUS: ASIWeights(0.32, 0.28, 0.22, 0.13, 0.05),
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

    def calculate(
        self,
        semantic_variance: float,
        contradiction_rate: float,
        mutation_degradation: float,
        cross_model_disagreement: float,
        tool_misuse_frequency: float,
    ) -> float:
        metrics = (
            semantic_variance,
            contradiction_rate,
            mutation_degradation,
            cross_model_disagreement,
            tool_misuse_frequency,
        )
        if any(metric < 0 or metric > 1 for metric in metrics):
            msg = "all metrics must be in [0, 1]"
            raise ValueError(msg)

        penalty = sum(w * m for w, m in zip(self._weights.as_tuple(), metrics))
        asi = 100.0 * (1.0 - penalty)
        return min(max(asi, 0.0), 100.0)
