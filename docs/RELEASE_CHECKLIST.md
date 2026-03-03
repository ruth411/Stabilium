# ASE Release Checklist (Week 12)

## Interpreter And Tooling Policy

- [ ] Release interpreter is pinned to Python 3.11 (`python3.11 -m venv .venv`)
- [ ] Release commands use module invocation (`python -m pip`, `python -m build`, `python -m twine`)
- [ ] Release commands are not run with Xcode system Python (`/Applications/Xcode.app/.../python3`)

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
- [ ] Build artifacts generated via `python -m build` with no errors
- [ ] Upload command validated from same env (`python -m twine check dist/*` and/or `python -m twine upload dist/*`)
- [ ] Clean install in fresh virtual environment verified
- [ ] CLI entrypoint `ase` works after install
- [ ] Import check passes: `python -c "import agent_stability_engine; print(agent_stability_engine.__version__)"`

## Canonical Command Sequence

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build twine
python -m build
python -m twine upload dist/*
```

## Install Verification Sequence

```bash
python3.11 -m venv .venv-install-check
source .venv-install-check/bin/activate
python -m pip install --upgrade pip
python -m pip install -U agent-stability-engine
ase --help
python -c "import agent_stability_engine; print(agent_stability_engine.__version__)"
```

## PATH Policy

- [ ] User-site path export is only used when intentionally doing `--user` installs
- [ ] Preferred invocation is from active venv entrypoint (`.venv/bin/ase` or `ase` after activation)

## Pre-Publication Gate

- [ ] Security/safety assumptions reviewed (tool misuse and safe-mode behavior)
- [ ] Known limitations documented (heuristic contradiction logic, drift simplifications)
- [ ] Tag/release candidate prepared
