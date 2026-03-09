# Stage 0 Contract Lock (Level 2 + Level 3 Foundation)

This document locks the interface and persistence contracts before Level 2/3 feature delivery.

## Goals
- Freeze request/response shapes so frontend and SDK integration do not churn.
- Keep backward compatibility with existing `benchmark` clients.
- Add DB columns now so later phases are additive.

## Job API Contract

`POST /jobs` now accepts:

- `provider`: `openai | anthropic | custom`
- `model`: model identifier
- `custom_endpoint`: required when `provider=custom`
- `api_key`: one-time key (never persisted)
- `suite`: JSON suite path (default: `examples/benchmarks/large_suite.json`)
- `job_type`: `benchmark | conversation_benchmark | agent_benchmark` (default: `benchmark`)
- `fault_rate`: `0.0..0.5` (default: `0.0`)
- `workers`: `1..10` (default: `3`)
- `run_count`, `max_cases`, `seed`

Stage 0 behavior:
- `benchmark` is enabled.
- `conversation_benchmark` and `agent_benchmark` are contract-locked but not enabled (`501`).
- `fault_rate` is validated and only valid for `agent_benchmark`.
- `suite` must resolve to an existing `.json` file inside repository root.

## Evaluate Contract

`POST /evaluate` remains synchronous and backward compatible, with optional `suite`.

## Backward Compatibility Rules

- Old clients that only send `provider/model/api_key/run_count/max_cases/seed` continue to work.
- New fields are optional with safe defaults.
- `run_benchmark_suite()` behavior is unchanged for single-turn prompt suites.

## Database Contract

`jobs` table has forward-compatible columns:

- `suite TEXT NOT NULL DEFAULT 'examples/benchmarks/large_suite.json'`
- `job_type TEXT NOT NULL DEFAULT 'benchmark'`
- `fault_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0`
- `workers INTEGER NOT NULL DEFAULT 1`
- `completed_cases INTEGER NOT NULL DEFAULT 0` (existing progress field)

All additions are idempotent migrations via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

## Non-Goals in Stage 0

- No conversation runner or agent runner execution yet.
- No trace storage/`/jobs/{id}/traces` endpoint yet.
- No frontend multi-tab UX changes yet.
