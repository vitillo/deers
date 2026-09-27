//! Built-in reference models.

/// GPT-style decoder-only reference implementation.
pub mod gpt2;
/// Backward-compatible re-exports of [`gpt2`] and [`qwen3`].
pub mod gpt;
/// Small MNIST MLP reference model.
pub mod mnist;
/// Qwen3 decoder-only reference implementation.
pub mod qwen3;
