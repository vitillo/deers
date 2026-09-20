# Training

## Sub-features

SGD and AdamW (incl. AdamW state save/load), WarmupWarmdown LR schedule, clip_grad_norm, cross_entropy and nll_loss, safetensors checkpoints (incl. sharded Qwen loads), MNIST and TinyStories example trainers, the `qwen3_run` inference example, `eval` perplexity scoring, `sample` (temperature/top-k/top-p), the `tokenizer` family (GPT-2, Qwen3, Qwen3.5), and `dataset` loaders.

## How to get to it (user POV)

Train via `examples/mnist_train.rs` or `examples/tinystories_train.rs`; run inference via `examples/qwen3_run.rs` (needs downloaded weights). Unit coverage lives in `tests/nn.rs`, `tests/gpt.rs`, `tests/qwen3*.rs`, `tests/eval.rs`, and unit tests in `src/optim.rs`, `src/loss.rs`, `src/sample.rs`, and `src/eval.rs`.

## Driving it with cargo test

```sh
cargo test --locked 2>&1 | tail -20
cargo build --locked --example mnist_train 2>&1 | tail -5
```

Proof is a green suite plus loss and gradient parity with candle on a
small fixed batch (see `tests/gpt.rs` loss_and_grads pattern). Run a full training loop only when the change touches the training path, since it downloads data and takes much longer. Building an example (`cargo build --locked --example mnist_train`) counts as smoke only where the skill body allows it; on hosts where the skill body says to skip the example build, stay with the library build and clippy proof.

## Gotchas

Training examples auto-download datasets into `data/` on first run. That is network access, not an offline check. Keep data downloads out of routine proofs.
