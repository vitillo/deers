# Deers

A minimal deep learning framework in Rust. Built for understanding, not production: think "PyTorch from scratch" in a small codebase you can actually read.

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
use deers::{loss, Device};

// Load data
let dataset = MNISTDataset::load().unwrap();
let train_images = dataset.train_images.reshape((60000, 784)).to_device(Device::Mps).unwrap();
let train_labels = dataset.train_labels;

// Define model
let model = nn::seq()
    .add(nn::Linear::new(784, 128))
    .add(nn::ReLU)
    .add(nn::Linear::new(128, 10));
model.to_device(Device::Mps).unwrap();

// Train
let mut sgd = SGD::new(model.vars(), 0.1);
let batch_size = 256;
let num_batches = 60000 / batch_size;

for batch_idx in 0..num_batches {
    let start = batch_idx * batch_size;
    let images = train_images.narrow(0, start, batch_size);
    let labels = train_labels.narrow(0, start, batch_size);
    let logits = model.forward(&images).unwrap();
    let batch_loss = loss::cross_entropy(&logits, &labels);
    sgd.backward_step(&batch_loss).unwrap();
}
```

A full working example is in [`examples/mnist_train.rs`](/Users/roberto/projects/deers/examples/mnist_train.rs). It reaches about 92% test accuracy in 3 epochs with a two-layer MLP.

```
cargo run --release --example mnist_train
cargo run --release --example mnist_train -- --device mps
```

## What's implemented

**Devices** — CPU and MPS (Metal-backed on macOS)

**DTypes** — `f32`, `f64`, and `u32` tensors (`u32` is used for integer targets / indices)

**Tensor ops** — neg, add, sub, mul, div, powf, log, exp, relu, matmul, gather

**Reductions** — sum, max, logsumexp, log_softmax

**Shape ops** — permute, broadcast, reshape, transpose, compact, narrow

**Autograd** — reverse-mode differentiation with gradient accumulation

**Modules** — `Linear`, `ReLU`, `Sequential`, and the `Module` trait with `to_device`

**Optimizers** — SGD

**Losses** — `cross_entropy`, `nll_loss`

**Data** — MNIST loader

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

Dependencies: `thiserror`, `rand`, `gemm`, and `metal` on macOS.
