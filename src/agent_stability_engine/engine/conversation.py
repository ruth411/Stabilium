from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from agent_stability_engine.engine.asi import (
    DEFAULT_CONVERSATION_WEIGHTS,
    ConvASICalculator,
    ConversationWeights,
)
from agent_stability_engine.engine.contradiction import ContradictionDetector
from agent_stability_engine.engine.embeddings import (
    EmbeddingProvider,
    build_embedder,
)
from agent_stability_engine.engine.variance import EmbeddingVarianceScorer

_AGENT_PLACEHOLDER = "__AGENT__"
_KEYWORD_RE = re.compile(r"[a-z0-9]+")
_CONSTRAINT_STOPWORDS = {
    "a",
    "an",
    "and",
    "be",
    "by",
    "do",
    "final",
    "in",
    "is",
    "it",
    "must",
    "not",
    "or",
    "response",
    "should",
    "the",
    "to",
    "with",
}


class ConversationAdapter(Protocol):
    def call_messages(
        self,
        messages: list[dict[str, str]],
        rng: random.Random | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class ConversationEvaluation:
    report: dict[str, object]


class ConversationEvaluator:
    def __init__(
        self,
        weights: ConversationWeights | None = None,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO,
        embedding_openai_api_key: str | None = None,
    ) -> None:
        embedder, resolved_provider = build_embedder(
            embedding_provider,
            openai_api_key=embedding_openai_api_key,
        )
        self._embedder = embedder
        self._variance_scorer = EmbeddingVarianceScorer(
            embedder=embedder,
            embedding_provider=resolved_provider,
        )
        self._calculator = ConvASICalculator(weights or DEFAULT_CONVERSATION_WEIGHTS)
        self._contradiction_detector = ContradictionDetector()

    def evaluate(
        self,
        case: dict[str, object],
        adapter: ConversationAdapter,
        run_count: int,
        seed: int,
    ) -> ConversationEvaluation:
        if run_count < 1:
            msg = "run_count must be >= 1"
            raise ValueError(msg)
        messages = _extract_messages(case)
        agent_positions = [
            idx for idx, message in enumerate(messages) if message["content"] == _AGENT_PLACEHOLDER
        ]
        if not agent_positions:
            case_id = case.get("id")
            msg = f"conversation case has no {_AGENT_PLACEHOLDER} turns: {case_id}"
            raise ValueError(msg)

        eval_turns_raw = _extract_eval_turns(case)
        eval_turns = _resolve_turn_indexes(eval_turns_raw, len(agent_positions))
        final_eval_turn = eval_turns[-1]

        expected_final = _extract_optional_str(case, "expected_final")
        constraints = _extract_str_list(case, "constraints")

        all_run_traces: list[list[str]] = []
        for run_idx in range(run_count):
            history = copy.deepcopy(messages)
            run_responses: list[str] = []
            run_rng = random.Random(seed + run_idx * 997)
            for position in agent_positions:
                context: list[dict[str, str]] = []
                for message in history[:position]:
                    content = message["content"]
                    if content == _AGENT_PLACEHOLDER:
                        continue
                    context.append({"role": message["role"], "content": content})
                response = adapter.call_messages(context, rng=run_rng)
                history[position] = {"role": "assistant", "content": response}
                run_responses.append(response)
            all_run_traces.append(run_responses)

        final_responses = [trace[final_eval_turn] for trace in all_run_traces]
        if len(final_responses) >= 2:
            variance_result = self._variance_scorer.score(final_responses)
            cross_run_variance = variance_result.normalized_variance
        else:
            cross_run_variance = 0.0
        turn_contradiction_rate = self._compute_turn_contradictions(all_run_traces)
        context_failure_rate = self._compute_context_failure(final_responses, expected_final)
        constraint_violation_rate = self._compute_constraint_violations(
            final_responses, constraints
        )
        drift_rate = self._compute_drift(all_run_traces)

        conv_asi = self._calculator.calculate(
            cross_run_variance=cross_run_variance,
            turn_contradiction_rate=turn_contradiction_rate,
            context_failure_rate=context_failure_rate,
            constraint_violation_rate=constraint_violation_rate,
            drift_rate=drift_rate,
        )
        case_id = case.get("id")
        report: dict[str, object] = {
            "case_id": str(case_id) if isinstance(case_id, str) else "unknown",
            "run_count": run_count,
            "num_turns": len(agent_positions),
            "eval_turns": eval_turns,
            "metrics": {
                "cross_run_variance": round(cross_run_variance, 4),
                "turn_contradiction_rate": round(turn_contradiction_rate, 4),
                "context_failure_rate": round(context_failure_rate, 4),
                "constraint_violation_rate": round(constraint_violation_rate, 4),
                "drift_rate": round(drift_rate, 4),
                "conv_asi": round(conv_asi, 2),
            },
            "artifacts": {
                "final_responses": final_responses,
                "run_traces": all_run_traces,
            },
        }
        return ConversationEvaluation(report=report)

    def _compute_turn_contradictions(self, traces: list[list[str]]) -> float:
        if not traces:
            return 0.0
        rates = [self._contradiction_detector.contradiction_rate(trace) for trace in traces]
        if not rates:
            return 0.0
        return float(sum(rates) / len(rates))

    def _compute_context_failure(
        self, final_responses: list[str], expected_final: str | None
    ) -> float:
        if not expected_final:
            return 0.0
        expected = expected_final.lower()
        if not final_responses:
            return 0.0
        failed = sum(1 for response in final_responses if expected not in response.lower())
        return failed / len(final_responses)

    def _compute_constraint_violations(
        self, final_responses: list[str], constraints: list[str]
    ) -> float:
        if not constraints:
            return 0.0
        checks = [_constraint_check(constraint) for constraint in constraints]
        active_checks = [check for check in checks if check is not None]
        if not active_checks:
            return 0.0
        if not final_responses:
            return 0.0

        total_violations = 0
        total_checks = len(final_responses) * len(active_checks)
        for response in final_responses:
            lowered = response.lower()
            for check in active_checks:
                if check is None:
                    continue
                if not check(lowered):
                    total_violations += 1
        return total_violations / total_checks

    def _compute_drift(self, traces: list[list[str]]) -> float:
        baseline_text = self._baseline_text(traces)
        if baseline_text is None:
            return 0.0

        baseline_matrix = self._embedder.encode([baseline_text])
        if baseline_matrix.shape[0] != 1:
            return 0.0
        baseline_vector = _normalize_vector(np.asarray(baseline_matrix[0], dtype=np.float64))

        drifts: list[float] = []
        for run_idx, trace in enumerate(traces):
            for turn_idx, text in enumerate(trace):
                if run_idx == 0 and turn_idx == 0:
                    continue
                vec_matrix = self._embedder.encode([text])
                if vec_matrix.shape[0] != 1:
                    continue
                vector = _normalize_vector(np.asarray(vec_matrix[0], dtype=np.float64))
                cosine = float(np.clip(np.dot(baseline_vector, vector), -1.0, 1.0))
                drifts.append((1.0 - cosine) / 2.0)

        if not drifts:
            return 0.0
        return float(sum(drifts) / len(drifts))

    def _baseline_text(self, traces: list[list[str]]) -> str | None:
        if not traces:
            return None
        first_trace = traces[0]
        if not first_trace:
            return None
        return first_trace[0]


def _extract_messages(case: dict[str, object]) -> list[dict[str, str]]:
    raw = case.get("messages")
    if not isinstance(raw, list):
        msg = "conversation case must include a messages list"
        raise ValueError(msg)

    messages: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            msg = "conversation messages must be objects with role/content"
            raise ValueError(msg)
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            msg = "conversation messages must have string role/content"
            raise ValueError(msg)
        messages.append({"role": role, "content": content})
    return messages


def _extract_eval_turns(case: dict[str, object]) -> list[int]:
    raw = case.get("eval_turns", [-1])
    if not isinstance(raw, list):
        msg = "eval_turns must be a list of integers"
        raise ValueError(msg)
    out: list[int] = []
    for value in raw:
        if not isinstance(value, int):
            msg = "eval_turns must contain integers"
            raise ValueError(msg)
        out.append(value)
    if not out:
        out = [-1]
    return out


def _resolve_turn_indexes(indexes: list[int], turn_count: int) -> list[int]:
    if turn_count <= 0:
        msg = "turn_count must be >= 1"
        raise ValueError(msg)
    resolved: list[int] = []
    for idx in indexes:
        value = idx if idx >= 0 else turn_count + idx
        if value < 0 or value >= turn_count:
            msg = f"eval_turn index out of range: {idx}"
            raise ValueError(msg)
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        msg = "resolved eval_turns cannot be empty"
        raise ValueError(msg)
    return resolved


def _extract_optional_str(case: dict[str, object], key: str) -> str | None:
    value = case.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_str_list(case: dict[str, object], key: str) -> list[str]:
    raw = case.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = f"{key} must be a list of strings"
        raise ValueError(msg)
    values: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            values.append(item.strip())
    return values


def _constraint_check(constraint: str) -> Callable[[str], bool] | None:
    lowered = constraint.lower()
    quoted = re.findall(r'"([^"]+)"', lowered)
    if quoted:
        needles = [part.strip() for part in quoted if part.strip()]
        if needles:
            return lambda response: all(needle in response for needle in needles)

    keywords = [
        token
        for token in _KEYWORD_RE.findall(lowered)
        if len(token) >= 3 and token not in _CONSTRAINT_STOPWORDS
    ]
    if not keywords:
        return None
    return lambda response: any(keyword in response for keyword in keywords)


def _normalize_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm
