# ASE Release Checklist (Week 12)

## Quality Gate

- [ ] `python -m ruff check .`
- [ ] `python -m black --check .`
- [ ] `python -m mypy src`
- [ ] `python -m pytest`
- [ ] Benchmark regression checks pass for all curated suites

## Artifact Gate

- [ ] Demo bundle generated via `ase demo`
- [ ] Reproducibility manifests generated for key reports
- [ ] Benchmark baselines reviewed and versioned in `examples/baselines/`

## Documentation Gate

- [ ] README command examples verified
- [ ] Demo runbook updated
- [ ] Release notes/changelog drafted

## Packaging Gate

- [ ] `pyproject.toml` version updated
- [ ] Clean install in fresh virtual environment verified
- [ ] CLI entrypoint `ase` works after install

## Pre-Publication Gate

- [ ] Security/safety assumptions reviewed (tool misuse and safe-mode behavior)
- [ ] Known limitations documented (heuristic contradiction logic, drift simplifications)
- [ ] Tag/release candidate prepared
