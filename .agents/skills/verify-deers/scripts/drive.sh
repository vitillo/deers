#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/tmp/deers-verify.log}"
FILTER="${2:-}"
if [ "$(uname -s)" != "Darwin" ] && ! command -v nvcc >/dev/null 2>&1 && [ "${DEERS_VERIFY_CPU_OK:-}" != "1" ]; then
  echo "drive.sh: CPU-only host without nvcc. cargo test needs the CUDA toolkit through the candle-core cuda dev-dep. Run cargo build --locked plus cargo clippy --lib instead. Set DEERS_VERIFY_CPU_OK=1 to attempt anyway." | tee "$OUT"
  exit 2
fi
if [ -n "$FILTER" ]; then
  cargo test --locked "$FILTER" 2>&1 | tee "$OUT"
else
  cargo test --locked 2>&1 | tee "$OUT"
fi
echo "evidence: $OUT"
