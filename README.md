# Agent Stability Engine (ASE)

Week 1-12 implementation for ASE core metrics, mutation stress testing, arbitration, contradiction analysis, taxonomy severity scoring, drift tracking, long-horizon stability, self-healing remediation, benchmark runner, regression gating, release/demo packaging, and CLI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m pytest
python -m ruff check .
python -m black --check .
python -m mypy src
```

## CLI

```bash
export OPENAI_API_KEY="..."

python -m agent_stability_engine.cli evaluate --prompt "Explain checksums" --run-count 5 --seed 42 --asi-profile reasoning_focus --mutation-limit 6 --output out/eval.json --manifest-output out/eval.manifest.json
python -m agent_stability_engine.cli evaluate --agent-provider openai --agent-model gpt-4o-mini --prompt "Explain checksums" --run-count 3 --seed 42 --output out/eval-openai.json
python -m agent_stability_engine.cli benchmark --suite examples/benchmarks/default_suite.json --run-count 5 --seed 42 --asi-profile safety_strict --mutation-limit 6 --output out/bench.json --manifest-output out/bench.manifest.json
python -m agent_stability_engine.cli regress --suite examples/benchmarks/reasoning_suite.json --baseline examples/baselines/reasoning_suite.baseline.json --run-count 3 --seed 42 --output out/regress-reasoning.json
python -m agent_stability_engine.cli drift --current-report out/eval.json --baseline-report out/baseline_eval.json --output out/drift.json
python -m agent_stability_engine.cli horizon --prompt "Plan migration strategy" --horizon 6 --run-count 5 --seed 42 --output out/horizon.json
python -m agent_stability_engine.cli heal --prompt "Provide triage steps" --run-count 5 --seed 42 --max-attempts 2 --output out/heal.json --manifest-output out/heal.manifest.json
python -m agent_stability_engine.cli demo --output-dir out/demo --run-count 3 --seed 42 --horizon 4 --manifest-output out/demo.manifest.json
```

## GitHub Action (Regression Gate)

Use the bundled action to gate PRs on benchmark ASI regressions.

```yaml
name: ASE Regression Gate

on:
  pull_request:

jobs:
  ase-regress:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ruthwikdovala/Stabilium@main
        with:
          suite: examples/benchmarks/reasoning_suite.json
          baseline: examples/baselines/reasoning_suite.baseline.json
          run-count: "3"
          seed: "42"
```

For OpenAI-backed runs, add:

```yaml
      - uses: ruthwikdovala/Stabilium@main
        with:
          suite: examples/benchmarks/reasoning_suite.json
          baseline: examples/baselines/reasoning_suite.baseline.json
          agent-provider: openai
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

## Release Docs

- `docs/RELEASE_CHECKLIST.md`
- `docs/DEMO_RUNBOOK.md`
