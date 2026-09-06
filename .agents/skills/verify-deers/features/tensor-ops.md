# Tensor ops

## Sub-features

Element-wise ops (add, mul, relu), matmul, reductions (sum, mean, softmax), shape ops (reshape, permute, narrow).

## How to get to it (user POV)

Call `deers::Tensor` constructors and methods from Rust code. Tests live in `tests/tensor.rs`.

## Driving it with cargo test

```sh
cargo test --locked tensor 2>&1 | tail -20
```

Proof is exit code 0 plus passing forward and backward candle parity
assertions for the touched op (same inputs in deers and candle, compared
with `assert_close` at 1e-4 for f32).

## Gotchas

Shape and dtype mismatches panic by design. A panic with a shape message is expected behavior, not a crash to work around.
