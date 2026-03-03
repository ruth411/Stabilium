from __future__ import annotations

import pytest

from agent_stability_engine.engine.alignment import GoalAlignmentDetector, GoalSpec


def test_goal_alignment_detects_missing_and_forbidden() -> None:
    detector = GoalAlignmentDetector()
    goal = GoalSpec(required_keywords=("checksum",), forbidden_patterns=(r"drop\s+table",))
    outputs = [
        "Use checksum verification before upload.",
        "Drop table users; then retry.",
        "Verify integrity only.",
    ]

    result = detector.evaluate(outputs, goal)
    assert result.misaligned_outputs == 2
    assert result.total_outputs == 3
    assert result.misalignment_rate == pytest.approx(2 / 3)
    assert result.violation_examples
