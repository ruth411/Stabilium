from __future__ import annotations

import random

import pytest

from agent_stability_engine.engine.conversation import ConversationEvaluator
from agent_stability_engine.engine.embeddings import EmbeddingProvider


class _ScriptedConversationAdapter:
    def __init__(self, scripted_responses: list[str]) -> None:
        self._scripted = scripted_responses
        self._idx = 0

    def call_messages(
        self,
        messages: list[dict[str, str]],
        rng: random.Random | None = None,
    ) -> str:
        if rng is not None:
            _ = rng.random()
        _ = messages
        if self._idx >= len(self._scripted):
            return "fallback response"
        out = self._scripted[self._idx]
        self._idx += 1
        return out


def test_conversation_evaluator_generates_conv_asi_report() -> None:
    case: dict[str, object] = {
        "id": "conv-001",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "My company is Novex."},
            {"role": "assistant", "content": "__AGENT__"},
            {"role": "user", "content": "What is my company name?"},
            {"role": "assistant", "content": "__AGENT__"},
        ],
        "expected_final": "Novex",
        "constraints": ["must mention Novex in final response"],
        "eval_turns": [-1],
    }
    adapter = _ScriptedConversationAdapter(
        [
            "Noted.",
            "Your company is Novex.",
            "Got it.",
            "You said Novex.",
        ]
    )
    evaluator = ConversationEvaluator(embedding_provider=EmbeddingProvider.HASH)

    result = evaluator.evaluate(case=case, adapter=adapter, run_count=2, seed=42)
    report = result.report
    metrics = report["metrics"]
    assert isinstance(metrics, dict)

    conv_asi = metrics["conv_asi"]
    assert isinstance(conv_asi, float)
    assert 0.0 <= conv_asi <= 100.0
    assert metrics["context_failure_rate"] == 0.0
    assert metrics["constraint_violation_rate"] == 0.0

    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    finals = artifacts["final_responses"]
    assert isinstance(finals, list)
    assert len(finals) == 2


def test_conversation_evaluator_context_failure_rate_detects_missing_expected() -> None:
    case: dict[str, object] = {
        "id": "conv-002",
        "messages": [
            {"role": "user", "content": "Remember: company is Novex."},
            {"role": "assistant", "content": "__AGENT__"},
        ],
        "expected_final": "Novex",
    }
    adapter = _ScriptedConversationAdapter(["I forgot.", "Still unknown."])
    evaluator = ConversationEvaluator(embedding_provider=EmbeddingProvider.HASH)

    result = evaluator.evaluate(case=case, adapter=adapter, run_count=2, seed=7)
    metrics = result.report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["context_failure_rate"] == 1.0


def test_conversation_evaluator_rejects_invalid_eval_turn() -> None:
    case: dict[str, object] = {
        "id": "conv-003",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "__AGENT__"},
        ],
        "eval_turns": [2],
    }
    adapter = _ScriptedConversationAdapter(["Hello"])
    evaluator = ConversationEvaluator(embedding_provider=EmbeddingProvider.HASH)

    with pytest.raises(ValueError, match="eval_turn"):
        evaluator.evaluate(case=case, adapter=adapter, run_count=1, seed=1)
