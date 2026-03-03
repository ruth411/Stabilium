# ASE Demo Runbook (Week 12)

## Goal

Generate a deterministic artifact bundle showing end-to-end ASE capabilities:

1. Stability evaluation
2. Benchmark evaluation
3. Regression gate
4. Drift comparison
5. Long-horizon stability
6. Self-healing remediation

## Command

```bash
python -m agent_stability_engine.cli demo --output-dir out/demo --run-count 3 --seed 42 --horizon 4
```

## Expected Outputs

The demo writes a `summary.json` plus these files:

- `baseline_eval.json`
- `eval.json`
- `benchmark.json`
- `regression.json`
- `drift.json`
- `horizon.json`
- `heal.json`

## Verification

- `summary.json` exists
- `summary.json` has `regression_passed`
- `summary.json` has all artifact paths
- Each artifact file is valid JSON
