#!/usr/bin/env bash
set -euo pipefail
echo "== deers doctor =="
cargo --version
rustc --version
uname -s
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi: absent"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  echo "nvidia-smi: no GPU"
else
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -5
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -1
else
  echo "nvcc: absent"
fi
ls /dev/nvidia* 2>/dev/null || echo "/dev/nvidia*: absent"
TIER="cpu-only"
if [ "$(uname -s)" = "Darwin" ]; then
  TIER="metal"
elif command -v nvcc >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  TIER="cuda"
fi
echo "tier: $TIER"
cargo build --locked 2>&1 | tail -5
echo "doctor: build ok tier=$TIER"
