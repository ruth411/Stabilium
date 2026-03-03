from __future__ import annotations

from agent_stability_engine.taxonomy.classifier import FailureTaxonomy, FailureType


def test_taxonomy_returns_stable_when_under_thresholds() -> None:
    taxonomy = FailureTaxonomy()
    failures = taxonomy.classify(0.1, 0.0, 0.1, 0.1, 0.0)
    assert failures == [FailureType.STABLE]


def test_taxonomy_returns_relevant_failure_labels() -> None:
    taxonomy = FailureTaxonomy()
    failures = taxonomy.classify(0.5, 0.3, 0.4, 0.5, 0.3)
    assert FailureType.HIGH_VARIANCE in failures
    assert FailureType.CONTRADICTION in failures
    assert FailureType.MUTATION_DEGRADATION in failures
    assert FailureType.CROSS_MODEL_DISAGREEMENT in failures
    assert FailureType.TOOL_MISUSE in failures


def test_taxonomy_assessment_returns_primary_and_severity() -> None:
    taxonomy = FailureTaxonomy()
    assessment = taxonomy.assess(0.55, 0.25, 0.6, 0.45, 0.1)
    assert assessment.primary_failure == FailureType.MUTATION_DEGRADATION
    assert assessment.severity > 0
    assert FailureType.STABLE not in assessment.failures
