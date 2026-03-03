from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalSpec:
    required_keywords: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoalAlignmentResult:
    misalignment_rate: float
    misaligned_outputs: int
    total_outputs: int
    violation_examples: list[str]


class GoalAlignmentDetector:
    """Detects prompt-goal misalignment using required/forbidden constraints."""

    def evaluate(self, outputs: list[str], goal_spec: GoalSpec) -> GoalAlignmentResult:
        if not outputs:
            return GoalAlignmentResult(
                misalignment_rate=0.0,
                misaligned_outputs=0,
                total_outputs=0,
                violation_examples=[],
            )

        compiled_forbidden = [
            re.compile(pattern, re.IGNORECASE) for pattern in goal_spec.forbidden_patterns
        ]

        misaligned = 0
        examples: list[str] = []
        for output in outputs:
            lowered = output.lower()

            missing_keywords = [
                keyword
                for keyword in goal_spec.required_keywords
                if keyword.strip() and keyword.lower() not in lowered
            ]
            forbidden_hits = [
                regex.pattern for regex in compiled_forbidden if regex.search(output) is not None
            ]

            is_misaligned = bool(missing_keywords) or bool(forbidden_hits)
            if is_misaligned:
                misaligned += 1
                if len(examples) < 5:
                    fragments: list[str] = []
                    if missing_keywords:
                        fragments.append(f"missing={','.join(missing_keywords)}")
                    if forbidden_hits:
                        fragments.append(f"forbidden={','.join(forbidden_hits)}")
                    examples.append(";".join(fragments))

        return GoalAlignmentResult(
            misalignment_rate=misaligned / len(outputs),
            misaligned_outputs=misaligned,
            total_outputs=len(outputs),
            violation_examples=examples,
        )
