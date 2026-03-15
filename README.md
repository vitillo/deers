# Deers

A minimal deep learning framework in Rust. Built for understanding, not production — think "PyTorch from scratch" in ~2k lines of code.

Deers implements reverse-mode automatic differentiation over a define-by-run computation graph: operations build the graph during the forward pass, and `.backward()` traverses it in reverse to compute gradients.

## Quick example

```rust
use deers::{Device, DType, Tensor, Var};

// Create a trainable variable
let x = Var::new(Tensor::randn((2, 3), DType::F32, Device::Cpu));

// Operations build the computation graph
let y = x.matmul(&x.transpose(None)); // (2, 3) @ (3, 2) -> (2, 2)
let loss = y.sum(vec![0, 1], false);

// Backward pass computes gradients
let grads = loss.backward().unwrap();
let grad_x = grads.get(x.id()).unwrap();
```

## Training on MNIST

```rust
use deers::dataset::MNISTDataset;
use deers::nn::{self, Module};
use deers::optim::SGD;
use deers::{loss, Device, Tensor};

// Load data
let dataset = MNISTDataset::load().unwrap();
let images = dataset.train_images.reshape((60000, 784));

// Define model
let model = nn::seq()
    .add(nn::Linear::new(784, 128))
    .add(nn::ReLU)
    .add(nn::Linear::new(128, 10));

// Train
let mut sgd = SGD::new(model.vars(), 0.1);

for batch in images.chunks(256) {
    let logits = model.forward(&batch).unwrap();
    let batch_loss = loss::cross_entropy(&logits, &labels);
    sgd.backward_step(&batch_loss).unwrap();
}
```

A full working example is in [`examples/mnist_train.rs`](examples/mnist_train.rs). It reaches ~91% accuracy in 3 epochs with a two-layer MLP.

```
cargo run --release --example mnist_train
```

## What's implemented

**Tensor ops** — neg, add, sub, mul, div, powf, log, exp, relu, matmul, broadcast_add

**Reductions** — sum, max, logsumexp, log_softmax

**Shape ops** — permute, broadcast, reshape, transpose, compact (no-copy views via strides)

**Autograd** — full reverse-mode differentiation with gradient accumulation

**Modules** — `Linear`, `ReLU`, `Sequential`, and the `Module` trait

**Optimizers** — SGD

**Losses** — cross_entropy, nll_loss

**Data** — MNIST loader

## Design

Each operation implements the `TensorOp` trait with `forward()`, `backward()`, and `dependencies()`. Adding a new op means implementing this trait — the autograd engine handles the rest.

Views (permute, broadcast, reshape) only change the layout metadata without copying data. A stride of 0 encodes broadcast dimensions. `compact()` materializes a view into contiguous storage when needed, but short-circuits to a no-op if the tensor is already contiguous.

`Var` wraps a `Tensor` as a trainable parameter. It implements `Deref<Target=Tensor>` so it can be used anywhere a tensor is expected. The optimizer updates `Var` storage in-place, keeping tensor IDs stable across training steps.

## Building

```
cargo build
cargo test
```

Dependencies: `thiserror`, `rand`, `gemm`.
