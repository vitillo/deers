//! Backward-compatible re-exports of the split architecture modules.
//!
//! New code should import from [`crate::models::gpt2`] or [`crate::models::qwen3`]
//! directly.
//! This module keeps every previous `models::gpt::` path working.

pub use super::gpt2::{
    Block, CausalSelfAttention, GPT, GPTConfig, GptMlpKind, GptNormKind, KvCache, MLP,
    RopeScaling, apply_rotary_emb, precompute_rotary_embeddings,
    precompute_rotary_embeddings_scaled,
};
pub use super::qwen3::{Qwen3, Qwen3Block, Qwen3Config};
