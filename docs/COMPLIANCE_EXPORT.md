# ASE Compliance Export

Generate a compliance-ready PDF and a signed JSON evidence bundle from ASE reports.

## Command

```bash
export ASE_SIGNING_KEY="replace-with-long-random-secret"

python -m agent_stability_engine.cli export \
  --input-report out/bench.json \
  --history-report out/bench_prev.json \
  --bundle-output out/compliance.bundle.json \
  --pdf-output out/compliance.pdf
```

## Inputs

- `--input-report`: benchmark, regression, or evaluation JSON report.
- `--history-report`: optional repeatable input used for trend deltas.
- `--bundle-output`: machine-readable compliance evidence bundle.
- `--pdf-output`: human-readable compliance PDF.
- `--signing-key-env`: env var name for HMAC key (`ASE_SIGNING_KEY` by default).

## Outputs

`bundle.json` includes:

- summary (`report_type`, `model`, `suite`, `asi_score`)
- confidence block
- trend points and delta vs previous report
- methodology metadata
- attestation:
  - `report_sha256`
  - `created_at_utc`
  - `tool_version`
  - `signature_hmac_sha256` (if signing key is configured)

## Security Notes

- Keep signing keys in secure secret stores (GitHub Actions secrets, vaults).
- Rotate signing keys regularly.
- Never hardcode signing keys in repository files.
