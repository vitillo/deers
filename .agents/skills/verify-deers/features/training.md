# Training

## Sub-features

SGD and AdamW, LR schedules, cross_entropy and nll_loss, safetensors checkpoints, MNIST and TinyStories example trainers.

## How to get to it (user POV)

Train via `examples/mnist_train.rs` or `examples/tinystories_train.rs`. Unit coverage lives in `tests/nn.rs` and `tests/gpt.rs`.

## Driving it with cargo test

```sh
cargo test --locked 2>&1 | tail -20
cargo build --locked --example mnist_train 2>&1 | tail -5
```

Proof is a green suite plus loss and gradient parity with candle on a
small fixed batch (see `tests/gpt.rs` loss_and_grads pattern). Run a full training loop only when the change touches the training path, since it downloads data and takes much longer.

## Gotchas

Training examples auto-download datasets into `data/` on first run. That is network access, not an offline check. Keep data downloads out of routine proofs.
