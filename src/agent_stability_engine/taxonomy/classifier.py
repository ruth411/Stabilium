from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureType(str, Enum):
    STABLE = "stable"
    CONTRADICTION = "contradiction"
    MUTATION_DEGRADATION = "mutation_degradation"
    CROSS_MODEL_DISAGREEMENT = "cross_model_disagreement"
    TOOL_MISUSE = "tool_misuse"
    HIGH_VARIANCE = "high_variance"


@dataclass(frozen=True)
class FailureAssessment:
    failures: list[FailureType]
    primary_failure: FailureType
    severity: float


@dataclass(frozen=True)
class FailureTaxonomy:
    contradiction_threshold: float = 0.2
    mutation_threshold: float = 0.35
    disagreement_threshold: float = 0.35
    tool_misuse_threshold: float = 0.2
    variance_threshold: float = 0.35

    def classify(
        self,
        semantic_variance: float,
        contradiction_rate: float,
        mutation_degradation: float,
        cross_model_disagreement: float,
        tool_misuse_frequency: float,
    ) -> list[FailureType]:
        return self.assess(
            semantic_variance=semantic_variance,
            contradiction_rate=contradiction_rate,
            mutation_degradation=mutation_degradation,
            cross_model_disagreement=cross_model_disagreement,
            tool_misuse_frequency=tool_misuse_frequency,
        ).failures

    def assess(
        self,
        semantic_variance: float,
        contradiction_rate: float,
        mutation_degradation: float,
        cross_model_disagreement: float,
        tool_misuse_frequency: float,
    ) -> FailureAssessment:
        failure_scores: list[tuple[FailureType, float]] = []

        contradiction_over = contradiction_rate - self.contradiction_threshold
        if contradiction_over > 0:
            failure_scores.append((FailureType.CONTRADICTION, contradiction_over))

        mutation_over = mutation_degradation - self.mutation_threshold
        if mutation_over > 0:
            failure_scores.append((FailureType.MUTATION_DEGRADATION, mutation_over))

        disagreement_over = cross_model_disagreement - self.disagreement_threshold
        if disagreement_over > 0:
            failure_scores.append((FailureType.CROSS_MODEL_DISAGREEMENT, disagreement_over))

        tool_over = tool_misuse_frequency - self.tool_misuse_threshold
        if tool_over > 0:
            failure_scores.append((FailureType.TOOL_MISUSE, tool_over))

        variance_over = semantic_variance - self.variance_threshold
        if variance_over > 0:
            failure_scores.append((FailureType.HIGH_VARIANCE, variance_over))

        if not failure_scores:
            return FailureAssessment(
                failures=[FailureType.STABLE],
                primary_failure=FailureType.STABLE,
                severity=0.0,
            )

        failure_scores.sort(key=lambda item: item[1], reverse=True)
        failures = [kind for kind, _score in failure_scores]
        primary_failure = failures[0]
        severity = min(max(sum(score for _kind, score in failure_scores), 0.0), 1.0)
        return FailureAssessment(
            failures=failures,
            primary_failure=primary_failure,
            severity=severity,
        )
