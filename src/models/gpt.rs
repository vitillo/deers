//! Decoder-only GPT language model with multi-head self-attention, RoPE
//! positional embeddings, RMSNorm, and a feed-forward MLP block.
//!
//! Attention uses grouped-query attention: key and value heads are fewer
//! than query heads, and each key/value head is shared by a group of query
//! heads. Sharing cuts the key/value projection weights and the runtime
//! key/value memory by the group size while keeping one distinct query
//! per head.

use half::{bf16, f16};

use crate::error::Result;
use crate::nn::{Embedding, Linear, Module, ParamBuilder, Parameter, RMSNorm, functional};
use crate::tensor::Tensor;
use crate::{DType, Device};

/// RoPE scaling applied when precomputing the rotary cache.
///
/// Rotary embeddings store one angle per position and channel, so a cache built for short
/// training sequences runs out of angular room on longer inputs. Scaling stretches the cache:
/// use `None` for training length, `Linear` or `Yarn` to serve longer contexts. Qwen3 ships
/// with a large theta (1_000_000) and extends context with YaRN-style scaling.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RopeScaling {
    /// No scaling. Reproduces the original RoPE caches exactly.
    #[default]
    None,
    /// Positional interpolation. Divides every inverse frequency by `factor`.
    Linear {
        /// Ratio of the extended context length to the original one.
        factor: f32,
    },
    /// YaRN scaling. Interpolates low frequencies, extrapolates high ones, and
    /// multiplies cos/sin by `0.1 * ln(factor) + 1`.
    Yarn {
        /// Ratio of the extended context length to the original one.
        factor: f32,
        /// Training context length the base frequencies were built for.
        original_max_position_embeddings: usize,
        /// Band edge below which frequencies interpolate. Defaults to 32.
        beta_fast: f32,
        /// Band edge above which frequencies extrapolate. Defaults to 1.
        beta_slow: f32,
    },
}

impl RopeScaling {
    /// YaRN scaling with the reference band defaults (`beta_fast` 32, `beta_slow` 1).
    pub fn yarn(factor: f32, original_max_position_embeddings: usize) -> Self {
        Self::Yarn {
            factor,
            original_max_position_embeddings,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// Derives RoPE inverse frequencies plus the cos/sin multiplier for a scaling choice.
fn rope_inv_freq(head_dim: usize, base: f32, scaling: RopeScaling) -> (Vec<f32>, f32) {
    let half_dim = head_dim / 2;
    let default: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    match scaling {
        RopeScaling::None => (default, 1.0),
        RopeScaling::Linear { factor } => {
            assert!(factor.is_finite() && factor > 0.0, "RoPE linear factor must be positive");
            (default.iter().map(|freq| freq / factor).collect(), 1.0)
        }
        RopeScaling::Yarn { factor, original_max_position_embeddings, beta_fast, beta_slow } => {
            assert!(factor.is_finite() && factor > 0.0, "RoPE YaRN factor must be positive");
            assert!(
                original_max_position_embeddings > 0,
                "RoPE YaRN original_max_position_embeddings must be positive"
            );

            let dim = head_dim as f32;
            let low = find_correction_dim(beta_fast, dim, base, original_max_position_embeddings)
                .floor()
                .max(0.0);
            let mut high = find_correction_dim(beta_slow, dim, base, original_max_position_embeddings)
                .ceil()
                .min(dim - 1.0);
            if low == high {
                // A zero-width band would divide by zero in the ramp below.
                high += 0.001;
            }
            let inv_freq = default
                .iter()
                .enumerate()
                .map(|(i, freq)| {
                    let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    freq / factor * ramp + freq * (1.0 - ramp)
                })
                .collect();
            let attention_factor =
                if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };
            (inv_freq, attention_factor)
        }
    }
}

/// Inverse-dimension formula locating the band edge for `rotations` full turns.
fn find_correction_dim(rotations: f32, dim: f32, base: f32, max_position_embeddings: usize) -> f32 {
    const TAU: f32 = 2.0 * std::f32::consts::PI;
    dim * (max_position_embeddings as f32 / (rotations * TAU)).ln() / (2.0 * base.ln())
}

/// Precomputes RoPE cos/sin caches with shape `[1, seq_len, 1, head_dim / 2]`.
pub fn precompute_rotary_embeddings(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    dtype: DType,
    device: Device,
) -> (Tensor, Tensor) {
    precompute_rotary_embeddings_scaled(seq_len, head_dim, base, RopeScaling::None, dtype, device)
}

/// Precomputes RoPE cos/sin caches with shape `[1, seq_len, 1, head_dim / 2]` under `scaling`.
pub fn precompute_rotary_embeddings_scaled(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    scaling: RopeScaling,
    dtype: DType,
    device: Device,
) -> (Tensor, Tensor) {
    assert!(head_dim.is_multiple_of(2), "RoPE requires an even head dimension");

    let (inv_freq, attention_factor) = rope_inv_freq(head_dim, base, scaling);

    let freqs: Vec<f32> =
        (0..seq_len).flat_map(|t| inv_freq.iter().map(move |&freq| t as f32 * freq)).collect();

    let half_dim = head_dim / 2;
    let shape = vec![1, seq_len, 1, half_dim];
    match dtype {
        DType::F16 => {
            let cos: Vec<f16> =
                freqs.iter().map(|&x| f16::from_f32(x.cos() * attention_factor)).collect();
            let sin: Vec<f16> =
                freqs.iter().map(|&x| f16::from_f32(x.sin() * attention_factor)).collect();
            (Tensor::from_vec(cos, shape.clone(), device), Tensor::from_vec(sin, shape, device))
        }
        DType::F32 => {
            let cos: Vec<f32> = freqs.iter().map(|&x| x.cos() * attention_factor).collect();
            let sin: Vec<f32> = freqs.iter().map(|&x| x.sin() * attention_factor).collect();
            (Tensor::from_vec(cos, shape.clone(), device), Tensor::from_vec(sin, shape, device))
        }
        DType::BF16 => {
            let cos: Vec<bf16> =
                freqs.iter().map(|&x| bf16::from_f32(x.cos() * attention_factor)).collect();
            let sin: Vec<bf16> =
                freqs.iter().map(|&x| bf16::from_f32(x.sin() * attention_factor)).collect();
            (Tensor::from_vec(cos, shape.clone(), device), Tensor::from_vec(sin, shape, device))
        }
        DType::I64 => panic!("RoPE requires a floating-point dtype"),
    }
}

/// Applies rotary embeddings to a multi-head attention tensor with shape `[B, T, H, D]`.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    assert_eq!(x.layout().ndim(), 4, "RoPE expects a 4D attention tensor");
    assert_eq!(cos.layout().shape(), sin.layout().shape(), "RoPE cos/sin shapes must match");

    let head_dim = x.layout().shape()[3];
    assert!(head_dim.is_multiple_of(2), "RoPE requires an even head dimension");

    let half_dim = head_dim / 2;
    assert_eq!(
        cos.layout().shape()[3],
        half_dim,
        "RoPE cache last dimension must equal head_dim / 2"
    );
    assert_eq!(
        cos.layout().shape()[1],
        x.layout().shape()[1],
        "RoPE cache must match sequence length"
    );

    let x1 = x.narrow(3, 0, half_dim);
    let x2 = x.narrow(3, half_dim, half_dim);
    let y1 = &x1 * cos + &x2 * sin;
    let y2 = &x1 * &(sin * -1.0) + &x2 * cos;
    Tensor::cat(&[y1, y2], 3)
}

/// Epsilon for QK-Norm. Matches the Qwen3 default `rms_norm_eps`, which Qwen3
/// reuses for its query/key norms.
const QK_NORM_EPS: f64 = 1e-6;

/// Key/value cache for incremental decoding.
///
/// Prefill scores the whole prompt once and stores its keys and values.
/// Decode then appends one token's keys and values and attends over the
/// cache instead of recomputing history. Without the cache, generating
/// token `t` rescores all `t` past tokens, so a long run costs quadratic
/// work in the prompt length. The cache keeps every past key and value,
/// so each step only projects, norms, and rotates the new token and reads
/// the stored history back. That is the whole game for long generation:
/// per-step work stops paying recomputation and pays only the attention
/// read over the cache.
///
/// The cache holds keys and values after QK-Norm, RoPE, and the
/// grouped-query repeat, shaped `[B, H_q, T_cached, D]`. Norms are
/// token-local, so caching their output is exact. RoPE runs at each
/// token's absolute position before the append, so stored keys stay
/// rotated correctly with no re-rotation. Repeating up front keeps decode
/// a plain append plus attention, at the price of a cache `group_size`
/// wider than the key/value heads.
#[derive(Debug, Default)]
pub struct KvCache {
    kv: Option<(Tensor, Tensor)>,
}

impl KvCache {
    /// Creates an empty cache. Fill it with `CausalSelfAttention::prefill`.
    pub fn new() -> Self {
        Self { kv: None }
    }

    /// Returns true while no tokens are cached yet.
    pub fn is_empty(&self) -> bool {
        self.kv.is_none()
    }

    /// Returns the number of cached tokens. Zero while empty.
    pub fn len(&self) -> usize {
        self.kv.as_ref().map(|(k, _)| k.layout().shape()[2]).unwrap_or(0)
    }

    /// Returns the cached keys shaped `[B, H_q, T_cached, D]`. Panics while empty.
    fn keys(&self) -> &Tensor {
        &self.kv.as_ref().expect("cache is empty; prefill the prompt first").0
    }

    /// Returns the cached values shaped `[B, H_q, T_cached, D]`. Panics while empty.
    fn values(&self) -> &Tensor {
        &self.kv.as_ref().expect("cache is empty; prefill the prompt first").1
    }

    /// Appends ready-to-attend keys and values shaped `[B, H_q, T_new, D]`.
    ///
    /// Batch, heads, head width, dtype, and device must match the stored
    /// cache, so a stray tensor fails loudly instead of scoring garbage.
    fn append(&mut self, k: Tensor, v: Tensor) {
        assert_eq!(k.layout().shape(), v.layout().shape(), "cache keys and values must match");
        let shape = k.layout().shape();
        assert_eq!(shape.ndim(), 4, "cache entries must be shaped [B, H, T, D]");
        match &self.kv {
            None => {
                self.kv = Some((k, v));
            }
            Some((prev_k, prev_v)) => {
                assert_eq!(shape[0], prev_k.layout().shape()[0], "cache batch mismatch");
                assert_eq!(shape[1], prev_k.layout().shape()[1], "cache head mismatch");
                assert_eq!(shape[3], prev_k.layout().shape()[3], "cache head width mismatch");
                assert_eq!(k.dtype(), prev_k.dtype(), "cache dtype mismatch");
                assert_eq!(k.device(), prev_k.device(), "cache device mismatch");
                assert_eq!(v.dtype(), prev_v.dtype(), "cache dtype mismatch");
                assert_eq!(v.device(), prev_v.device(), "cache device mismatch");
                let full_k = Tensor::cat(&[prev_k.clone(), k], 2);
                let full_v = Tensor::cat(&[prev_v.clone(), v], 2);
                self.kv = Some((full_k, full_v));
            }
        }
    }
}

/// Minimal causal self-attention with bias-free projections and RoPE on queries/keys.
///
/// `n_kv_heads` key/value heads are shared across `n_q_heads` query heads.
/// Equal counts is plain multi-head attention.
///
/// Queries and keys pass through a per-head RMSNorm before RoPE. Large models let
/// attention logits grow until softmax saturates and gradients vanish. QK-Norm caps
/// each head at unit RMS, so scores stay bounded and training stays stable at scale.
#[derive(Debug)]
pub struct CausalSelfAttention {
    n_embd: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group_size: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
}

impl CausalSelfAttention {
    /// Creates a causal self-attention module whose projections are registered under `builder`.
    pub fn new(builder: ParamBuilder, n_embd: usize, n_q_heads: usize) -> Self {
        Self::new_gqa(builder, n_embd, n_q_heads, n_q_heads)
    }

    /// Creates causal self-attention with `n_kv_heads` key/value heads shared
    /// across `n_q_heads` query heads. Each key/value head serves `n_q_heads /
    /// n_kv_heads` query heads. Pass `n_kv_heads == n_q_heads` for plain MHA.
    pub fn new_gqa(
        builder: ParamBuilder,
        n_embd: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
    ) -> Self {
        assert!(n_embd.is_multiple_of(n_q_heads), "n_embd must be divisible by n_q_heads");
        Self::new_gqa_with_head_dim(builder, n_embd, n_q_heads, n_kv_heads, n_embd / n_q_heads)
    }

    /// Creates causal self-attention with an explicit per-head width.
    ///
    /// Same grouping as [`new_gqa`](Self::new_gqa), but the head width is
    /// independent of the model width: queries project to `n_q_heads *
    /// head_dim` and keys/values to `n_kv_heads * head_dim`. Qwen3 needs
    /// this because its head width (128) exceeds `hidden / n_q_heads`.
    pub fn new_gqa_with_head_dim(
        builder: ParamBuilder,
        n_embd: usize,
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert!(n_q_heads.is_multiple_of(n_kv_heads), "n_q_heads must be divisible by n_kv_heads");
        assert!(head_dim > 0, "head_dim must be positive");
        Self {
            n_embd,
            n_q_heads,
            n_kv_heads,
            group_size: n_q_heads / n_kv_heads,
            head_dim,
            q_proj: Linear::no_bias(builder.pp("q_proj"), n_embd, n_q_heads * head_dim),
            k_proj: Linear::no_bias(builder.pp("k_proj"), n_embd, n_kv_heads * head_dim),
            v_proj: Linear::no_bias(builder.pp("v_proj"), n_embd, n_kv_heads * head_dim),
            out_proj: Linear::no_bias(builder.pp("out_proj"), n_q_heads * head_dim, n_embd),
            q_norm: RMSNorm::new_affine(builder.pp("q_norm"), head_dim, QK_NORM_EPS),
            k_norm: RMSNorm::new_affine(builder.pp("k_norm"), head_dim, QK_NORM_EPS),
        }
    }

    /// Projects queries, keys, and values shaped `[B, H_q, T, D]`.
    ///
    /// QK-Norm, RoPE, and the grouped-query repeat all apply here, so cached
    /// keys and values already carry them and decode never recomputes them.
    /// Also returns the batch size and sequence length of `x`.
    fn project_qkv(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, usize, usize)> {
        let shape = x.layout().shape();
        assert_eq!(shape.ndim(), 3, "attention expects input shape [B, T, C]");

        let batch_size = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        assert_eq!(channels, self.n_embd, "input channel size must match model width");

        let x_flat = x.reshape(vec![batch_size * seq_len, channels]); // [B*T, C]
        let q = self.q_proj.forward(&x_flat)?.rearrange(
            "(b t) (h d) -> b t h d",
            &[("b", batch_size), ("t", seq_len), ("h", self.n_q_heads)],
        );
        let k = self.k_proj.forward(&x_flat)?.rearrange(
            "(b t) (h d) -> b t h d",
            &[("b", batch_size), ("t", seq_len), ("h", self.n_kv_heads)],
        );
        let v = self.v_proj.forward(&x_flat)?.rearrange(
            "(b t) (h d) -> b t h d",
            &[("b", batch_size), ("t", seq_len), ("h", self.n_kv_heads)],
        );

        // QK-Norm runs before RoPE: rotation preserves each head norm, so unit RMS
        // vectors enter both the rotation and the dot product scores.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // RoPE rotates each head independently, so one rotation serves the whole query group.
        let q = apply_rotary_emb(&q, cos, sin).rearrange("b t h d -> b h t d", &[]);
        let k = apply_rotary_emb(&k, cos, sin);
        let k = if self.group_size == 1 {
            k
        } else {
            k.repeat("b t kv d -> b t (kv g) d", &[("g", self.group_size)])
        };
        let k = k.rearrange("b t h d -> b h t d", &[]);
        let v = if self.group_size == 1 {
            v
        } else {
            v.repeat("b t kv d -> b t (kv g) d", &[("g", self.group_size)])
        };
        let v = v.rearrange("b t h d -> b h t d", &[]);
        Ok((q, k, v, batch_size, seq_len))
    }

    /// Attends queries over keys and values under `mask`, returning `[B, T, C]`.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = Tensor::einsum("b h t d, b h s d -> b h t s", q, k) * scale;
        let attn = (&scores + &mask).softmax(3);
        let y_flat = attn.matmul(v).rearrange("b h t d -> (b t) (h d)", &[]);

        let out = self.out_proj.forward(&y_flat)?;
        Ok(out.rearrange("(b t) c -> b t c", &[("b", batch_size), ("t", seq_len)]))
    }

    /// Runs self-attention on `[B, T, C]` inputs using the provided RoPE caches.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (q, k, v, batch_size, seq_len) = self.project_qkv(x, cos, sin)?;
        let mask = functional::causal_mask(batch_size, seq_len, 0, x.dtype(), x.device());
        self.attend(&q, &k, &v, mask, batch_size, seq_len)
    }

    /// Scores the whole prompt at once and stores its keys and values in `cache`.
    ///
    /// `cos` and `sin` cover the prompt positions starting at zero. The cache
    /// must be empty; each following token arrives through `decode`.
    pub fn prefill(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        assert!(cache.is_empty(), "prefill expects an empty cache; decode appends to it");
        let (q, k, v, batch_size, seq_len) = self.project_qkv(x, cos, sin)?;
        let mask = functional::causal_mask(batch_size, seq_len, 0, x.dtype(), x.device());
        let out = self.attend(&q, &k, &v, mask, batch_size, seq_len)?;
        cache.append(k, v);
        Ok(out)
    }

    /// Appends one token's keys and values, then attends over the whole cache.
    ///
    /// `x` holds exactly one token. `cos` and `sin` hold that token's absolute
    /// position: narrow them from the full rotary cache at the cached length,
    /// so the stored keys stay rotated correctly without re-rotation.
    pub fn decode(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        assert!(!cache.is_empty(), "decode expects a non-empty cache; prefill the prompt first");
        let (q, k, v, batch_size, seq_len) = self.project_qkv(x, cos, sin)?;
        assert_eq!(seq_len, 1, "decode expects a single token, got {seq_len}");
        let offset = cache.len();
        cache.append(k, v);
        let mask = functional::causal_mask(batch_size, 1, offset, x.dtype(), x.device());
        self.attend(&q, cache.keys(), cache.values(), mask, batch_size, 1)
    }

    /// Returns the trainable parameters owned by the attention module.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.q_proj.parameters();
        parameters.extend(self.k_proj.parameters());
        parameters.extend(self.v_proj.parameters());
        parameters.extend(self.out_proj.parameters());
        parameters.extend(self.q_norm.parameters());
        parameters.extend(self.k_norm.parameters());
        parameters
    }

    /// Moves the attention parameters to `device`.
    pub fn to_device(&self, device: Device) -> Result<()> {
        for parameter in self.parameters() {
            parameter.to_device(device)?;
        }
        Ok(())
    }
}

/// Minimal transformer MLP with bias-free projections and a `relu^2` activation.
#[derive(Debug)]
pub struct MLP {
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    /// Creates an MLP whose projections are registered under `builder`.
    pub fn new(builder: ParamBuilder, n_embd: usize, hidden_dim: usize) -> Self {
        Self {
            up_proj: Linear::no_bias(builder.pp("up_proj"), n_embd, hidden_dim),
            down_proj: Linear::no_bias(builder.pp("down_proj"), hidden_dim, n_embd),
        }
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.layout().shape();
        assert_eq!(shape.ndim(), 3, "MLP expects input shape [B, T, C]");

        let batch_size = shape[0];
        let seq_len = shape[1];

        let x_flat = x.rearrange("b t c -> (b t) c", &[]);
        let y = self.up_proj.forward(&x_flat)?; // [B*T, H]
        let y = y.relu(); // [B*T, H]
        let y = &y * &y; // [B*T, H]
        let y = self.down_proj.forward(&y)?; // [B*T, C]
        Ok(y.rearrange("(b t) c -> b t c", &[("b", batch_size), ("t", seq_len)]))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.up_proj.parameters();
        parameters.extend(self.down_proj.parameters());
        parameters
    }
}

/// Pre-norm residual GPT block.
#[derive(Debug)]
pub struct Block {
    norm1: RMSNorm,
    attn: CausalSelfAttention,
    norm2: RMSNorm,
    mlp: MLP,
}

impl Block {
    /// Creates a transformer block whose trainable weights are registered under `builder`.
    pub fn new(
        builder: ParamBuilder,
        n_embd: usize,
        n_head: usize,
        hidden_dim: usize,
        eps: f64,
    ) -> Self {
        Self {
            norm1: RMSNorm::new(eps),
            attn: CausalSelfAttention::new(builder.pp("attn"), n_embd, n_head),
            norm2: RMSNorm::new(eps),
            mlp: MLP::new(builder.pp("mlp"), n_embd, hidden_dim),
        }
    }

    /// Runs the pre-norm attention and MLP residual block.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x + &self.attn.forward(&self.norm1.forward(x)?, cos, sin)?;
        let y = self.mlp.forward(&self.norm2.forward(&x)?)?;
        Ok(&x + &y)
    }

    /// Returns the trainable parameters owned by the block.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.attn.parameters();
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

/// Minimal GPT configuration for the nanochat-style decoder stack.
#[derive(Clone, Debug)]
pub struct GPTConfig {
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length supported by the rotary cache.
    pub sequence_len: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads per block.
    pub n_head: usize,
    /// Hidden width of the model.
    pub n_embd: usize,
    /// Inner width of the MLP projection.
    pub mlp_hidden_dim: usize,
    /// Epsilon used by RMSNorm.
    pub rms_norm_eps: f64,
    /// Base frequency used by RoPE.
    pub rope_base: f32,
    /// RoPE scaling applied to the precomputed rotary cache.
    pub rope_scaling: RopeScaling,
}

impl GPTConfig {
    /// Returns the per-head channel width.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// Minimal GPT model: token embedding, decoder blocks, final norm, and LM head.
#[derive(Debug)]
pub struct GPT {
    vocab_size: usize,
    wte: Embedding,
    blocks: Vec<Block>,
    norm: RMSNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
}

impl GPT {
    /// Creates a GPT model whose trainable weights are registered under `builder`.
    pub fn new(config: GPTConfig, builder: ParamBuilder) -> Self {
        assert!(config.n_embd.is_multiple_of(config.n_head), "n_embd must be divisible by n_head");

        let wte = Embedding::new(builder.pp("wte"), config.vocab_size, config.n_embd);
        let blocks = (0..config.n_layer)
            .map(|index| {
                Block::new(
                    builder.pp("blocks").pp(index.to_string()),
                    config.n_embd,
                    config.n_head,
                    config.mlp_hidden_dim,
                    config.rms_norm_eps,
                )
            })
            .collect();
        let norm = RMSNorm::new(config.rms_norm_eps);
        let lm_head = Linear::no_bias(builder.pp("lm_head"), config.n_embd, config.vocab_size);
        let (cos, sin) = precompute_rotary_embeddings_scaled(
            config.sequence_len,
            config.head_dim(),
            config.rope_base,
            config.rope_scaling,
            DType::F32,
            Device::Cpu,
        );

        Self { vocab_size: config.vocab_size, wte, blocks, norm, lm_head, cos, sin }
    }

    /// Runs the decoder on token ids shaped `[B, T]`.
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let shape = idx.layout().shape();
        assert_eq!(shape.ndim(), 2, "GPT expects token ids with shape [B, T]");

        let batch_size = shape[0];
        let seq_len = shape[1];
        assert!(seq_len <= self.cos.layout().shape()[1], "sequence length exceeds rotary cache");
        assert_eq!(
            idx.device(),
            self.cos.device(),
            "token ids and rotary cache must be on the same device"
        );

        let cos = self.cos.narrow(1, 0, seq_len); // [1, T, 1, D/2]
        let sin = self.sin.narrow(1, 0, seq_len); // [1, T, 1, D/2]

        let mut x = self.wte.forward(idx)?; // [B, T, C]
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?; // [B, T, C]
        }
        x = self.norm.forward(&x)?; // [B, T, C]

        let x_flat = x.rearrange("b t c -> (b t) c", &[]);
        let logits = self.lm_head.forward(&x_flat)?; // [B*T, V]
        Ok(logits.rearrange(
            "(b t) v -> b t v",
            &[("b", batch_size), ("t", seq_len), ("v", self.vocab_size)],
        ))
    }

    /// Returns the trainable parameters owned by the model.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.wte.parameters();
        for block in &self.blocks {
            parameters.extend(block.parameters());
        }
        parameters.extend(self.lm_head.parameters());
        parameters
    }

    /// Moves the model parameters and rotary caches to `device`.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        self.wte.to_device(device)?;
        for block in &self.blocks {
            block.to_device(device)?;
        }
        self.lm_head.to_device(device)?;
        self.cos = self.cos.to_device(device)?;
        self.sin = self.sin.to_device(device)?;
        Ok(())
    }
}
