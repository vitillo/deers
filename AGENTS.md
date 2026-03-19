# DEERS — Minimal PyTorch Clone in Rust

## What is this?

A didactic deep learning framework in Rust. The goal is **understanding over performance** — code should be minimal, readable, and easy to modify. Think "PyTorch from scratch for learning", not production framework.

## Project structure

```
src/
  lib.rs          — Module declarations
  tensor.rs       — Tensor struct (Arc<RwLock<Storage>> + Layout + op history)
  ops.rs          — All operators (forward + backward), implement TensorOp trait
  backprop.rs     — Reverse-mode autodiff, topological sort, gradient store
  storage.rs      — Storage backend abstraction (trait + enum dispatch)
  storage/cpu.rs  — CPU storage with strided iterator
  layout.rs       — Shape, strides, offset (supports views without copies)
  dtype.rs        — F16/F32/F64 type enum
  device.rs       — Device enum (CPU, CUDA stub)
  error.rs        — Error types (thiserror)
  test_utils.rs   — Approx trait for float comparison in tests
tests/
  tensor.rs       — 80+ integration tests covering ops and gradients
data/mnist/       — MNIST dataset files
```

## Architecture

- **Define-by-run**: computation graph built during forward pass (like PyTorch)
- **TensorOp trait**: each op implements `forward()`, `backward()`, `dependencies()`
- **No-copy views**: permute, broadcast, reshape only change Layout, not data
- **Strided iteration**: handles arbitrary memory layouts via strides + offset
- **Gradient tracking**: `requires_grad` propagates through ops; backward does reverse topo sort

## What's implemented

- **Element-wise**: neg, add, sub, mul, div, powf, log, exp (+ scalar variants)
- **Reductions**: sum (with keep_dims), max (backward TODO), logsumexp
- **Shape ops**: permute, broadcast, reshape, transpose, compact
- **Linear algebra**: matmul (naive 2D)
- **Autograd**: full backward pass for most ops
- **Data**: MNIST loading

## What's NOT implemented yet

- Max backward pass
- CUDA backend (enum exists, no implementation)
- F16 support (enum exists, not wired up)
- Parameter/Module system (nn.Module equivalent)
- Optimizers (SGD, Adam, etc.)
- Loss functions
- Higher-level layers (Linear, Conv, etc.)
- Automatic broadcasting (must be explicit)

## Dependencies

Only `thiserror`. Everything else is pure Rust.

## Building & testing

```bash
cargo build
cargo test
cargo fmt-check
cargo lint
```

Run `cargo fmt-check` and `cargo lint` after code changes before marking work done.

## Design principles

- **Minimal code**: fewer lines > more features. No abstractions until needed.
- **Compose when possible**: implement things like sigmoid, softmax, and mean from existing primitives instead of adding custom kernels. Only add storage-level ops when composition would create real performance problems, such as `cat` on MPS needing memcpy to avoid a GPU→CPU→GPU roundtrip.
- **Follow PyTorch/candle conventions**: before implementing a new feature (op, loss, module, etc.), always check how PyTorch and candle structure it. Use their design decisions to guide ours — where to put the API, whether to use a custom op or compose from primitives, naming, etc.
- **Work backwards from nanochat**: only implement what the target model actually uses.
- **Incremental milestones**: implement one piece at a time, review it, then move on.
- **Readable**: a newcomer should be able to follow the code. Prefer explicit over clever.
- **Easy to extend**: adding a new op = implement TensorOp trait with forward/backward.
- **Correctness first**: every op should have gradient tests.

## Commit messages

- Focus on **why**, not what — the diff already shows the what.
- First line: short summary of the change.
- Body: explain the motivation, trade-offs, or reasoning behind the decision.
