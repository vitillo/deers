#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/tmp/deers-verify.log}"
FILTER="${2:-}"
if [ -n "$FILTER" ]; then
  cargo test --locked "$FILTER" 2>&1 | tee "$OUT"
else
  cargo test --locked 2>&1 | tee "$OUT"
fi
echo "evidence: $OUT"
