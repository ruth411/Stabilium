# ASE Build, Publish, Install Workflow

This document standardizes ASE packaging and release operations to a single interpreter path.

## Release Environment Standard

- Use Python 3.11 for release operations.
- Use one active virtual environment for the full build and upload flow.
- Use module invocation (`python -m ...`) for packaging tools.

## Canonical Build + Publish Commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## Canonical Install Verification Commands

```bash
python3.11 -m venv .venv-install-check
source .venv-install-check/bin/activate
python -m pip install --upgrade pip
python -m pip install -U agent-stability-engine
ase --help
python -c "import agent_stability_engine; print(agent_stability_engine.__version__)"
```

## PATH Policy

- If you install with `python -m pip install --user ...`, add user-site bin to PATH.
- If you use a virtual environment, prefer the env-local CLI entrypoint (`ase` after activation).

## Common Failure Modes

- `No module named build`:
  - Install build in active env: `python -m pip install build`.
- `No module named twine`:
  - Install twine in active env: `python -m pip install twine`.
- `python -m pip3 ...` fails:
  - Use `python -m pip ...` (`pip3` is a shell command name, not a Python module).
- `ase: command not found`:
  - Ensure active venv is selected, or run full path to the script in the target env.
- Interpreter drift:
  - Avoid mixing Xcode Python, Homebrew Python, and venv Python in one release session.
