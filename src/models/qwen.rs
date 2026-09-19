//! One Qwen3 decoder block: pre-norm residual attention plus feed-forward.
//!
//! The block runs two residual sub-blocks in sequence. Each one normalizes
//! its input first, transforms it, and adds the result back to the
//! unnormalized signal:
//!
//! ```text
//! h = x + attention(rms_norm(x))
//! y = h + swiglu(rms_norm(h))
//! ```
//!
//! Normalizing before the transform keeps each sub-block's input at unit
//! scale no matter how deep the stack grows. Adding the output back (instead
//! of replacing the signal) keeps a direct path from the block input to its
//! output, so gradients reach early layers and a silent sub-block still
//! forwards its input unchanged.
//!
//! Attention contributes token mixing: grouped queries share key/value heads
//! to cut KV memory, per-head QK-Norm bounds the logits before softmax, and
//! RoPE rotates queries and keys by absolute position. The feed-forward
//! contributes channel mixing: a SwiGLU gate selects features per token and
//! the down projection returns them to model width.

use crate::error::Result;
use crate::nn::{Module, ParamBuilder, Parameter, RMSNorm, SwiGLU};
use crate::tensor::Tensor;
use crate::{DType, Device};

use super::gpt::{CausalSelfAttention, precompute_rotary_embeddings};

/// Qwen3 decoder-block dimensions.
#[derive(Clone, Debug)]
pub struct Qwen3Config {
    /// Model width each residual stream carries.
    pub hidden: usize,
    /// Query head count.
    pub n_q_heads: usize,
    /// Key/value head count shared across query groups.
    pub n_kv_heads: usize,
    /// Per-head channel width, independent of `hidden / n_q_heads`.
    pub head_dim: usize,
    /// Inner width of the SwiGLU feed-forward.
    pub mlp_dim: usize,
    /// Epsilon for both block RMSNorms.
    pub rms_norm_eps: f64,
    /// RoPE base frequency.
    pub rope_theta: f32,
}

impl Qwen3Config {
    /// Qwen3-0.6B block dimensions.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden: 1024,
            n_q_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            mlp_dim: 3072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    /// Builds the RoPE cos/sin caches for `seq_len` positions.
    pub fn rotary_cache(&self, seq_len: usize, dtype: DType, device: Device) -> (Tensor, Tensor) {
        precompute_rotary_embeddings(seq_len, self.head_dim, self.rope_theta, dtype, device)
    }
}

/// One pre-norm residual Qwen3 decoder block.
///
/// Owns affine input and post-attention norms, grouped-query attention with
/// QK-Norm and RoPE, and a SwiGLU feed-forward. Parameter names follow the
/// deers block convention (`attn.*`, `mlp.*`) with Qwen's layernorm names, so
/// the sharded-checkpoint map extends to this block by adding three entries.
#[derive(Debug)]
pub struct QwenBlock {
    input_norm: RMSNorm,
    attn: CausalSelfAttention,
    post_attn_norm: RMSNorm,
    mlp: SwiGLU,
}

impl QwenBlock {
    /// Creates a block whose weights are registered under `builder`.
    pub fn new(builder: ParamBuilder, config: &Qwen3Config) -> Self {
        Self {
            input_norm: RMSNorm::new_affine(
                builder.pp("input_layernorm"),
                config.hidden,
                config.rms_norm_eps,
            ),
            attn: CausalSelfAttention::new_gqa_with_head_dim(
                builder.pp("attn"),
                config.hidden,
                config.n_q_heads,
                config.n_kv_heads,
                config.head_dim,
            ),
            post_attn_norm: RMSNorm::new_affine(
                builder.pp("post_attention_layernorm"),
                config.hidden,
                config.rms_norm_eps,
            ),
            mlp: SwiGLU::new(builder.pp("mlp"), config.hidden, config.mlp_dim, config.hidden),
        }
    }

    /// Runs the pre-norm attention and SwiGLU residual block on `[B, T, C]`.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = x + &self.attn.forward(&self.input_norm.forward(x)?, cos, sin)?;
        Ok(&h + &self.mlp.forward(&self.post_attn_norm.forward(&h)?)?)
    }

    /// Returns the trainable parameters owned by the block.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.input_norm.parameters();
        parameters.extend(self.attn.parameters());
        parameters.extend(self.post_attn_norm.parameters());
        parameters.extend(self.mlp.parameters());
        parameters
    }

    /// Moves the block parameters to `device`.
    pub fn to_device(&self, device: Device) -> Result<()> {
        for parameter in self.parameters() {
            parameter.to_device(device)?;
        }
        Ok(())
    }
}
