# NN modules

## Sub-features

Linear, Embedding, RMSNorm, LayerNorm, ReLU, GELU, SiLU, SwiGLU, Dropout, Sequential, `Parameter` / `ParamStore` / `ParamBuilder`, `nn::functional` (dropout, causal_mask), CausalSelfAttention (MHA/GQA, optional Qwen3.5-style output gate), MLP, GPT Block, GPT, KvCache, Qwen3Block, Qwen3 (incl. QK-Norm and RoPE scaling), MnistMLP.

## How to get to it (user POV)

Build modules via `deers::nn`, run `forward`, move with `Module::to_device`.

## Driving it with cargo test

```sh
cargo test --locked --test nn --test gpt 2>&1 | tail -20
cargo test --locked --test qwen3 --test qwen3_block --test qwen3_candle_parity 2>&1 | tail -20
cargo test --locked --test gqa --test kvcache --test qknorm --test silu_swiglu --test rope_scaling --test output_gate 2>&1 | tail -20
```

(Use the `--test <target>` form: bare name filters match test function
names, so `cargo test --locked kvcache` matches nothing while
`--test kvcache` drives `tests/kvcache.rs`.)

Proof is exit code 0 and passing candle parity checks for the module
(logits and grads vs the candle reference at 1e-4 on CPU, following
`tests/gpt.rs` and `tests/qwen3_candle_parity.rs`). Accelerator numbers
count only when the run used that hardware. On CPU-only hosts use the
CPU tier from the skill body.

## Gotchas

Device movement is explicit. Keep work on-device and avoid extra CPU roundtrips in perf-sensitive checks.
