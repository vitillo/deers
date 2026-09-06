# NN modules

## Sub-features

Linear, Embedding, RMSNorm, Sequential, CausalSelfAttention, MLP, GPTBlock, GPT.

## How to get to it (user POV)

Build modules via `deers::nn`, run `forward`, move with `Module::to_device`.

## Driving it with cargo test

```sh
cargo test --locked nn 2>&1 | tail -20
cargo test --locked gpt 2>&1 | tail -20
```

Proof is exit code 0 and passing candle parity checks for the module
(logits and grads vs the candle reference at 1e-4 on CPU, 2e-3 on
accelerators, following `tests/gpt.rs`).

## Gotchas

Device movement is explicit. Keep work on-device and avoid extra CPU roundtrips in perf-sensitive checks.
