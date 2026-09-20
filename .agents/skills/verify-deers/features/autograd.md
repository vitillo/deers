# Autograd

## Sub-features

Forward graph construction, `.backward()`, gradient accumulation, `.attach()`, `nn::Parameter`, and the `no_grad` scope / `NoGradGuard`. (`Var` is candle's handle in parity tests, not a deers type.)

## How to get to it (user POV)

Build ops on an attached Tensor, call `.backward()`, read grads from the `GradientStore`. `Parameter::new` attaches for you. Tests live in `tests/tensor.rs` (`*_backward_conforms`), `tests/einops.rs` (`*_gradients_*`), and `tests/no_grad.rs`.

## Driving it with cargo test

```sh
cargo test --locked backward 2>&1 | tail -20
cargo test --locked grad 2>&1 | tail -20
```

Proof is exit code 0 and passing gradient parity checks against
candle's `GradStore` on the same graph (sum the outputs, call backward in
both frameworks, compare each input grad at 1e-4). On CPU-only hosts use
the CPU tier from the skill body. The full test needs `nvcc`.

## Gotchas

Gradients accumulate across backward calls. Reset or account for accumulation when asserting values.
