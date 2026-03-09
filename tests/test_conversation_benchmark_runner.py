from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agent_stability_engine.runners.conversation_benchmark import run_conversation_benchmark_suite


class _DeterministicAdapter:
    def call_messages(
        self,
        messages: list[dict[str, str]],
        rng: random.Random | None = None,
    ) -> str:
        _ = rng
        user_text = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
        lowered = user_text.lower()
        if "novex" in lowered:
            return "The company name is Novex."
        if "zephyr" in lowered:
            return "The project codename is Zephyr."
        return "Not sure."


def _write_suite(path: Path) -> None:
    suite: dict[str, Any] = {
        "name": "conversation_suite_test",
        "cases": [
            {
                "id": "memory-001",
                "type": "conversation",
                "messages": [
                    {"role": "user", "content": "Remember this: Novex."},
                    {"role": "assistant", "content": "__AGENT__"},
                ],
                "expected_final": "Novex",
                "eval_turns": [-1],
            },
            {
                "id": "context_reasoning-001",
                "type": "conversation",
                "messages": [
                    {"role": "user", "content": "Track this codename: Zephyr."},
                    {"role": "assistant", "content": "__AGENT__"},
                ],
                "expected_final": "Zephyr",
                "eval_turns": [-1],
            },
        ],
    }
    path.write_text(json.dumps(suite), encoding="utf-8")


def test_conversation_benchmark_runner_outputs_expected_report_shape(tmp_path: Path) -> None:
    suite_path = tmp_path / "conversation_suite.json"
    _write_suite(suite_path)

    result = run_conversation_benchmark_suite(
        suite_path=suite_path,
        adapter=_DeterministicAdapter(),
        run_count=2,
        seed=7,
    )
    report = result.report
    assert str(report["benchmark_id"]).startswith("conv-bench-")
    assert report["suite_name"] == "conversation_suite_test"
    assert report["job_type"] == "conversation_benchmark"
    assert report["num_cases"] == 2

    assert isinstance(report["mean_conv_asi"], float)
    assert report["mean_asi"] == report["mean_conv_asi"]

    stats = report["conv_asi_statistics"]
    assert isinstance(stats, dict)
    assert stats["sample_size"] == 2
    assert stats["ci_low"] <= report["mean_conv_asi"] <= stats["ci_high"]

    domain_scores = report["domain_scores"]
    assert isinstance(domain_scores, dict)
    assert "memory" in domain_scores
    assert "context_reasoning" in domain_scores

    cases = report["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 2
    first_case = cases[0]
    assert isinstance(first_case, dict)
    assert "case_sha256" in first_case
    assert "report" in first_case


def test_conversation_benchmark_runner_parallel_uses_agent_factory(tmp_path: Path) -> None:
    suite_path = tmp_path / "conversation_suite.json"
    _write_suite(suite_path)
    created = 0

    class _FactoryAdapter:
        def __init__(self, marker: int) -> None:
            self._marker = marker

        def call_messages(
            self,
            messages: list[dict[str, str]],
            rng: random.Random | None = None,
        ) -> str:
            _ = rng
            _ = messages
            return f"factory-{self._marker}-Novex"

    def make_adapter() -> object:
        nonlocal created
        created += 1
        return _FactoryAdapter(created)

    result = run_conversation_benchmark_suite(
        suite_path=suite_path,
        adapter=_DeterministicAdapter(),
        agent_factory=make_adapter,
        run_count=2,
        seed=3,
        workers=2,
    )
    report = result.report
    assert report["num_cases"] == 2
    assert created == 2
