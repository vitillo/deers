# Deers

A minimal deep learning framework in Rust. Built for understanding, not production: think "PyTorch from scratch" in a small codebase you can actually read.

Deers implements reverse-mode automatic differentiation over a define-by-run computation graph: operations build the graph during the forward pass, and `.backward()` traverses it in reverse to compute gradients.

The project favors readability and explicit tensor construction over production-grade robustness.
Most `Tensor` methods return values directly and may panic on invalid inputs (shape or device mismatches), so beginners get a minimal learning path while keeping the call sites compact.

## Training a GPT on TinyStories

Deers can train a small GPT language model from scratch. Here's the core of the training loop from [`examples/tinystories_train.rs`](examples/tinystories_train.rs):

```rust
use deers::models::gpt::{GPT, GPTConfig};
use deers::nn::{ParamStore, Module};
use deers::optim::{AdamW, AdamWConfig, WarmupWarmdown, clip_grad_norm};
use deers::dataset::TokenBinDataset;
use deers::tokenizer::Tokenizer;
use deers::{Device, loss};

// Tokenizer and data
let tokenizer = Tokenizer::gpt2();
let dataset = TokenBinDataset::load("data/tinystories_gpt2/train.bin", 256).unwrap();

// Build a 6-layer GPT
let config = GPTConfig {
    vocab_size: tokenizer.vocab_size(),
    sequence_len: 256, n_layer: 6, n_head: 6, n_embd: 192,
    mlp_hidden_dim: 768, rms_norm_eps: 1e-5, rope_base: 10_000.0,
};
let store = ParamStore::new();
let mut model = GPT::new(config, store.root());
model.to_device(Device::Mps).unwrap();

// AdamW with warmup/warmdown LR schedule
let params = store.parameters();
let mut opt = AdamWConfig::new(5e-4)
    .weight_decay(0.1)
    .build(params.clone());
let schedule = WarmupWarmdown::new(500, 12_000, 0.65, 0.1);

// Training loop
for step in 0..12_000 {
    opt.set_lr(5e-4 * schedule.lr_multiplier(step));
    let (inputs, targets) = dataset.sample_batch(4, Device::Mps);
    let logits = model.forward(&inputs).unwrap();
    let logits_flat = logits.reshape(vec![4 * 256, tokenizer.vocab_size()]);
    let targets_flat = targets.reshape(vec![4 * 256]);
    let batch_loss = loss::cross_entropy(&logits_flat, &targets_flat);

    let grads = batch_loss.backward().unwrap();
    clip_grad_norm(&params, &mut grads, 1.0).unwrap();
    opt.step_with_grads(&grads).unwrap();
}
```

Run it with:

```
cargo run --release --example tinystories_train -- --device mps
```

The trainer auto-downloads and tokenizes the dataset on first run, then periodically samples text so you can watch the model go from random tokens to coherent stories:

```
step    50/12000 | train_loss 9.2451 | lr 5.00e-05 | 1.204s | 4267 tok/s
  sample: Once upon a time the the of a...
step   500/12000 | train_loss 4.8320 | lr 5.00e-04 | 1.156s | 4441 tok/s
  sample: Once upon a time there was a little girl named Lily. She liked to play in the park...
```

See [`examples/mnist_train.rs`](examples/mnist_train.rs) for a simpler MNIST classifier starting point.

## What's implemented

**Devices** — CPU and MPS (Metal-backed on macOS)

**DTypes** — `f16`, `f32`, and `i64` tensors (`i64` is used for integer targets / indices)

**Tensor ops** — neg, add, sub, mul, div, powf, log, exp, sqrt, sin, cos, relu, sigmoid, tanh, matmul, gather, index_select, cat

**Reductions** — sum, max, mean, logsumexp, log_softmax, softmax

**Shape ops** — permute, broadcast, reshape, transpose, compact, narrow

**Autograd** — reverse-mode differentiation with gradient accumulation

**Modules** — `Linear`, `Embedding`, `RMSNorm`, `ReLU`, `Sequential`, `CausalSelfAttention`, `MLP`, `GPTBlock`, `GPT`, and the `Module` trait with `to_device`

**Optimizers** — SGD, AdamW (with decoupled weight decay and bias correction)

**LR Schedules** — warmup/constant/warmdown scheduler

**Losses** — `cross_entropy`, `nll_loss`

**Tokenizer** — tiktoken-based wrapper for BPE tokenization

**Data** — MNIST loader, text dataset, token-bin dataset with auto-download

**Checkpoints** — safetensors-based model and optimizer state serialization

**Notable conventions**:

- Gradients are enabled by calling `.attach()` on tensors or using `Var`, which wraps a trainable tensor.
- `Tensor::sum`, `Tensor::mean`, and reductions are explicit (e.g. `sum(vec![0, 1], true)`), and there are no hidden defaults.
- Device movement is explicit and value-oriented through `to_device`.
- CUDA is not implemented — CPU and MPS only for now.

## Design

Each operation implements the `TensorOp` trait with `forward()`, `backward()`, and `dependencies()`. Adding a new op means implementing this trait; the autograd engine handles the reverse traversal and gradient accumulation.

Views (permute, broadcast, reshape) only change the layout metadata without copying data. A stride of 0 encodes broadcast dimensions. `compact()` materializes a view into contiguous storage when needed, but short-circuits to a no-op if the tensor is already contiguous.

`Var` wraps a `Tensor` as a trainable parameter. It implements `Deref<Target = Tensor>` so it can be used anywhere a tensor is expected. The optimizer updates `Var` storage in-place, keeping tensor IDs stable across training steps.

`Tensor::to_device(...)` moves tensor data between backends. At the module level, `Module::to_device(...)` moves all trainable parameters, which keeps the user experience close to `model.to(device)` in PyTorch.

MPS support is intentionally small and explicit. The backend accelerates the common `f32` training path on Apple devices and also supports `u32` index buffers for ops like `gather`.

## Building

```
cargo build
cargo test
```

## Development

Use the repo-standard formatting and linting commands:

```
cargo fmt
cargo fmt-check
cargo lint
```

Dependencies: `thiserror`, `rand`, `gemm`, `half`, `tiktoken-rs`, `safetensors`, and `metal` on macOS.
