from __future__ import annotations

import pytest

from agent_stability_engine.arbitration.disagreement import CrossModelDisagreement


def test_disagreement_low_for_identical_outputs() -> None:
    scorer = CrossModelDisagreement()
    score = scorer.score({"m1": "alpha answer", "m2": "alpha answer"})
    assert score == pytest.approx(0.0)


def test_disagreement_higher_for_different_outputs() -> None:
    scorer = CrossModelDisagreement()
    score = scorer.score({"m1": "sun is bright", "m2": "use SQL for query", "m3": "boil water"})
    assert 0 < score <= 1


def test_disagreement_requires_multiple_models() -> None:
    scorer = CrossModelDisagreement()
    with pytest.raises(ValueError, match="at least two"):
        scorer.score({"m1": "single"})
