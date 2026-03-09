# Agent Stability Engine (ASE)

A rigorous benchmarking and stability testing framework for AI agents. ASE measures how consistently an agent responds across semantic mutations, prompt variations, and adversarial perturbations — producing a single **Agent Stability Index (ASI)** score per model.

Live platform: [stabilium.ruthwikdovala.com](https://stabilium.ruthwikdovala.com)

---

## What's inside

| Path | Purpose |
|---|---|
| `src/agent_stability_engine/` | Core engine (evaluator, mutations, embeddings, stats) |
| `api/main.py` | FastAPI REST API (auth, jobs, benchmark runner) |
| `examples/benchmarks/` | Benchmark suites (100-case `large_suite.json`) |
| `examples/baselines/` | Baseline results for regression gating |
| `scripts/validate_models.py` | CLI script to compare models head-to-head |

---

## Quickstart (engine)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
python -m pytest
python -m ruff check .
python -m black --check .
python -m mypy src
```

---

## CLI

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Single prompt evaluation
python -m agent_stability_engine.cli evaluate \
  --prompt "Explain checksums" \
  --run-count 5 --seed 42 \
  --asi-profile reasoning_focus \
  --mutation-limit 6 \
  --output out/eval.json

# Evaluate with a specific provider/model
python -m agent_stability_engine.cli evaluate \
  --agent-provider openai --agent-model gpt-4o-mini \
  --prompt "Explain checksums" --run-count 3 --seed 42 \
  --output out/eval-openai.json

python -m agent_stability_engine.cli evaluate \
  --agent-provider anthropic --agent-model claude-haiku-4-5 \
  --prompt "Explain checksums" --run-count 3 --seed 42 \
  --output out/eval-anthropic.json

# Benchmark suite
python -m agent_stability_engine.cli benchmark \
  --suite examples/benchmarks/large_suite.json \
  --run-count 5 --seed 42 \
  --output out/bench.json

# Regression gate
python -m agent_stability_engine.cli regress \
  --suite examples/benchmarks/reasoning_suite.json \
  --baseline examples/baselines/reasoning_suite.baseline.json \
  --run-count 3 --seed 42 --output out/regress.json

# Drift, horizon, heal, export
python -m agent_stability_engine.cli drift \
  --current-report out/eval.json --baseline-report out/baseline_eval.json \
  --output out/drift.json

python -m agent_stability_engine.cli horizon \
  --prompt "Plan migration strategy" \
  --horizon 6 --run-count 5 --seed 42 --output out/horizon.json

python -m agent_stability_engine.cli heal \
  --prompt "Provide triage steps" \
  --run-count 5 --seed 42 --max-attempts 2 \
  --output out/heal.json

python -m agent_stability_engine.cli export \
  --input-report out/bench.json \
  --history-report out/bench_prev.json \
  --bundle-output out/compliance.bundle.json \
  --pdf-output out/compliance.pdf
```

---

## REST API

The API server powers the [Stabilium web platform](https://stabilium.ruthwikdovala.com). Run it locally:

```bash
cd api
uvicorn api.main:app --reload --port 8000
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (Railway sets this automatically) |
| `DATABASE_URL` | _(required)_ | PostgreSQL DSN used by the API service |
| `ASE_API_SESSION_TTL_HOURS` | `168` | Session expiry (7 days) |
| `ASE_WATCHDOG_TIMEOUT_SECONDS` | `3600` | Max seconds a benchmark job may run |
| `ASE_ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated, or `*`) |

### Auth endpoints

```
POST /auth/register   { name, business_name, email, password }  → { token, user }
POST /auth/login      { email, password }                        → { token, user }
GET  /auth/me                                                    → UserPublic
POST /auth/logout
```

Passwords are hashed with PBKDF2-SHA256 (310,000 iterations). Sessions are stored in PostgreSQL and expire after 7 days.

### Job endpoints

```
POST   /jobs               Submit a new benchmark evaluation (async, returns 202)
GET    /jobs               List all jobs for the authenticated user
GET    /jobs/{id}          Get a single job (includes completed_cases for progress)
GET    /jobs/{id}/report   Get the full JSON report once completed
GET    /jobs/{id}/report/pdf  Download a PDF compliance report
```

**Job request body:**
```json
{
  "provider": "openai | anthropic | custom",
  "model": "gpt-4o-mini",
  "suite": "examples/benchmarks/large_suite.json",
  "job_type": "benchmark | conversation_benchmark | agent_benchmark",
  "fault_rate": 0.0,
  "workers": 3,
  "api_key": "sk-...",
  "run_count": 3,
  "max_cases": 20,
  "seed": 42
}
```

Jobs run in a background subprocess. The `completed_cases` field updates after each case finishes — poll `GET /jobs` every few seconds to show a progress bar.

Stage 0 contract notes:
- Existing clients remain compatible: `suite`, `job_type`, `fault_rate`, and `workers` are optional.
- `job_type` defaults to `benchmark`.
- `conversation_benchmark` and `agent_benchmark` are contract-locked but not yet enabled in Stage 0 (returns `501`).
- `fault_rate` is validated and only allowed for `agent_benchmark`.
- `suite` must be a JSON file path within the repository root.

### Live demo endpoint

```
POST /evaluate   Synchronous evaluation (no auth required, capped at 100 cases)
```

---

## Database schema

PostgreSQL. Three tables:

```sql
users   (id, name, business_name, email, password_salt, password_hash, created_at)
sessions (token, user_id, created_at, expires_at)
jobs    (id, user_id, provider, model, suite, job_type, fault_rate, workers,
         run_count, max_cases, seed,
         status, completed_cases, created_at, updated_at,
         started_at, finished_at, error_message, result_json)
```

---

## GitHub Action (Regression Gate)

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
          agent-provider: openai
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

---

## Build & publish

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip build twine
python -m build
python -m twine upload dist/*
```

Install verification:

```bash
python3.11 -m venv .venv-install-check
source .venv-install-check/bin/activate
pip install -U agent-stability-engine
ase --help
python -c "import agent_stability_engine; print(agent_stability_engine.__version__)"
```

---

## Docs

- `docs/BUILD_PUBLISH_INSTALL.md`
- `docs/COMPLIANCE_EXPORT.md`
- `docs/RELEASE_CHECKLIST.md`
- `docs/DEMO_RUNBOOK.md`
- `docs/STAGE0_CONTRACT.md`
