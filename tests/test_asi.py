from __future__ import annotations

import pytest

from agent_stability_engine.engine.asi import ASICalculator, ASIProfile, ASIWeights


def test_asi_calculation_bounds_and_value() -> None:
    calc = ASICalculator()
    asi = calc.calculate(0.2, 0.2, 0.2, 0.2, 0.2)
    assert asi == pytest.approx(80.0)


def test_asi_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        ASICalculator(weights=ASIWeights(0.3, 0.3, 0.3, 0.3, 0.3))


def test_asi_metrics_must_be_normalized() -> None:
    calc = ASICalculator()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calc.calculate(1.2, 0.0, 0.0, 0.0, 0.0)


def test_asi_profile_changes_weighting() -> None:
    balanced = ASICalculator.from_profile(ASIProfile.BALANCED)
    strict = ASICalculator.from_profile(ASIProfile.SAFETY_STRICT)
    score_balanced = balanced.calculate(0.2, 0.1, 0.1, 0.1, 0.8)
    score_strict = strict.calculate(0.2, 0.1, 0.1, 0.1, 0.8)
    assert score_strict < score_balanced
