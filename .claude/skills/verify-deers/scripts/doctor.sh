#!/usr/bin/env bash
set -euo pipefail
echo "== deers doctor =="
cargo --version
rustc --version
cargo build --locked 2>&1 | tail -5
echo "doctor: build ok"
