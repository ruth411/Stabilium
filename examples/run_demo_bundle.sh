#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-out/demo}"
SEED="${2:-42}"
RUN_COUNT="${3:-3}"
HORIZON="${4:-4}"

python -m agent_stability_engine.cli demo \
  --output-dir "$OUT_DIR" \
  --seed "$SEED" \
  --run-count "$RUN_COUNT" \
  --horizon "$HORIZON" \
  --manifest-output "$OUT_DIR/demo.manifest.json"

echo "Demo bundle written to: $OUT_DIR"
