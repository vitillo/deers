//! Backward-compatible re-exports of the split architecture modules.
//!
//! New code should import from [`crate::models::gpt2`] or [`crate::models::qwen3`]
//! directly.
//! The shared `RopeScaling`, `KvCache`, `CausalSelfAttention`, and rotary helper
//! names here are the GPT-2 copies; Qwen3 code must import those names from
//! [`crate::models::qwen3`].

pub use super::gpt2::{
    Block, CausalSelfAttention, GPT, GPTConfig, GptMlpKind, GptNormKind, KvCache, MLP,
    RopeScaling, apply_rotary_emb, precompute_rotary_embeddings,
    precompute_rotary_embeddings_scaled,
};
pub use super::qwen3::{Qwen3, Qwen3Block, Qwen3Config};
