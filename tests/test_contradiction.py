from __future__ import annotations

import pytest

from agent_stability_engine.engine.contradiction import ContradictionDetector


def test_contradiction_rate_detects_simple_conflict() -> None:
    detector = ContradictionDetector()
    texts = [
        "water is wet",
        "water is not wet",
        "earth is round",
    ]
    rate = detector.contradiction_rate(texts)
    assert rate == pytest.approx(1 / 3)


def test_contradiction_rate_zero_without_assertions() -> None:
    detector = ContradictionDetector()
    assert detector.contradiction_rate(["..."]) == pytest.approx(0.0)


def test_contradiction_analysis_includes_counts_and_examples() -> None:
    detector = ContradictionDetector()
    analysis = detector.analyze(
        [
            "fire is hot",
            "fire is not hot",
            "water is wet",
        ]
    )
    assert analysis.assertion_count == 3
    assert analysis.contradiction_count == 1
    assert analysis.contradiction_rate == pytest.approx(1 / 3)
    assert len(analysis.examples) == 1
    assert analysis.examples[0].subject == "fire"
