from __future__ import annotations

import pytest

from agent_stability_engine.engine.variance import EmbeddingVarianceScorer


def test_variance_scores_low_for_identical_outputs() -> None:
    scorer = EmbeddingVarianceScorer(expected_max_variance=0.25)
    result = scorer.score(["same answer", "same answer", "same answer"])

    assert result.raw_variance == pytest.approx(0.0)
    assert result.normalized_variance == pytest.approx(0.0)


def test_variance_increases_for_different_outputs() -> None:
    scorer = EmbeddingVarianceScorer(expected_max_variance=0.1)
    result = scorer.score(["the sky is blue", "2+2=4", "use a wrench"])

    assert result.raw_variance > 0
    assert 0 <= result.normalized_variance <= 1


def test_variance_requires_two_or_more_texts() -> None:
    scorer = EmbeddingVarianceScorer()

    with pytest.raises(ValueError, match="at least two"):
        scorer.score(["single"])
