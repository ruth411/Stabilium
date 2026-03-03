from __future__ import annotations

import re
from dataclasses import dataclass

_FACT_RE = re.compile(
    r"\b([a-z][a-z0-9_ ]{1,40})\s+is\s+(not\s+)?([a-z0-9_ ]{1,40})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContradictionExample:
    subject: str
    object_: str
    first_negated: bool
    second_negated: bool


@dataclass(frozen=True)
class ContradictionAnalysis:
    contradiction_rate: float
    contradiction_count: int
    assertion_count: int
    examples: list[ContradictionExample]


class ContradictionDetector:
    """Heuristic contradiction detector for simple factual assertions."""

    def analyze(self, texts: list[str]) -> ContradictionAnalysis:
        if not texts:
            return ContradictionAnalysis(
                contradiction_rate=0.0,
                contradiction_count=0,
                assertion_count=0,
                examples=[],
            )

        assertion_count = 0
        contradiction_count = 0
        seen: dict[tuple[str, str], bool] = {}
        examples: list[ContradictionExample] = []

        for text in texts:
            for match in _FACT_RE.finditer(text):
                subj = match.group(1).strip().lower()
                negated = match.group(2) is not None
                obj = match.group(3).strip().lower()
                assertion_count += 1

                key = (subj, obj)
                existing = seen.get(key)
                if existing is not None and existing != negated:
                    contradiction_count += 1
                    examples.append(
                        ContradictionExample(
                            subject=subj,
                            object_=obj,
                            first_negated=existing,
                            second_negated=negated,
                        )
                    )
                seen[key] = negated

        if assertion_count == 0:
            return ContradictionAnalysis(
                contradiction_rate=0.0,
                contradiction_count=0,
                assertion_count=0,
                examples=[],
            )

        return ContradictionAnalysis(
            contradiction_rate=contradiction_count / assertion_count,
            contradiction_count=contradiction_count,
            assertion_count=assertion_count,
            examples=examples,
        )

    def contradiction_rate(self, texts: list[str]) -> float:
        return self.analyze(texts).contradiction_rate
