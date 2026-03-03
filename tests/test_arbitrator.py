from __future__ import annotations

import pytest

from agent_stability_engine.arbitration.arbitrator import CrossModelArbitrator


def test_arbitrator_returns_consensus_and_pairwise() -> None:
    arbitrator = CrossModelArbitrator()
    outputs = {
        "m1": "the sky is blue",
        "m2": "the sky is blue and clear",
        "m3": "database index optimization",
    }

    result = arbitrator.arbitrate(outputs)

    assert 0 <= result.disagreement_score <= 1
    assert result.consensus_model in outputs
    assert result.consensus_output == outputs[result.consensus_model]
    assert len(result.pairwise) == 3
    assert all(0 <= edge.divergence <= 1 for edge in result.pairwise)


def test_arbitrator_requires_at_least_two_models() -> None:
    arbitrator = CrossModelArbitrator()
    with pytest.raises(ValueError, match="at least two"):
        arbitrator.arbitrate({"m1": "single output"})
