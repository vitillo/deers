# Tensor ops

## Sub-features

Element-wise ops (add, mul, relu, silu, gelu, powf, log, sqrt, exp, sin, cos), matmul, reductions (sum, mean, max, softmax, log_sum_exp), shape ops (reshape, permute, narrow, broadcast, transpose, compact), indexing (gather, index_add, topk, argmax), and the einops dialect (rearrange, reduce, repeat, einsum in `src/einops.rs`).

## How to get to it (user POV)

Call `deers::Tensor` constructors and methods from Rust code. Tests live in `tests/tensor.rs` (113 tests) and `tests/einops.rs` (80 tests).

## Driving it with cargo test

```sh
cargo test --locked tensor 2>&1 | tail -20
cargo test --locked --test einops 2>&1 | tail -20
```

Proof is exit code 0 plus passing forward and backward candle parity
assertions for the touched op (same inputs in deers and candle, compared
with `assert_close` at 1e-4 for f32). On CPU-only hosts use the CPU tier
from the skill body. The full test needs `nvcc`.

## Gotchas

Shape and dtype mismatches panic by design. A panic with a shape message is expected behavior, not a crash to work around.
