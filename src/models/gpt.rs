//! Decoder-only GPT language model with multi-head self-attention, RoPE
//! positional embeddings, RMSNorm, and a feed-forward MLP block.

use half::f16;

use crate::error::Result;
use crate::nn::{Embedding, Linear, Module, ParamBuilder, Parameter, RMSNorm, functional};
use crate::tensor::Tensor;
use crate::{DType, Device};

/// Precomputes RoPE cos/sin caches with shape `[1, seq_len, 1, head_dim / 2]`.
pub fn precompute_rotary_embeddings(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    dtype: DType,
    device: Device,
) -> (Tensor, Tensor) {
    assert!(head_dim.is_multiple_of(2), "RoPE requires an even head dimension");

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let channel = (2 * i) as f32;
            1.0 / base.powf(channel / head_dim as f32)
        })
        .collect();

    let freqs: Vec<f32> =
        (0..seq_len).flat_map(|t| inv_freq.iter().map(move |&freq| t as f32 * freq)).collect();

    let shape = vec![1, seq_len, 1, half_dim];
    match dtype {
        DType::F16 => {
            let cos: Vec<f16> = freqs.iter().map(|&x| f16::from_f32(x.cos())).collect();
            let sin: Vec<f16> = freqs.iter().map(|&x| f16::from_f32(x.sin())).collect();
            (Tensor::from_vec(cos, shape.clone(), device), Tensor::from_vec(sin, shape, device))
        }
        DType::F32 => {
            let cos: Vec<f32> = freqs.iter().map(|&x| x.cos()).collect();
            let sin: Vec<f32> = freqs.iter().map(|&x| x.sin()).collect();
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

/// Minimal causal self-attention with bias-free projections and RoPE on queries/keys.
pub struct CausalSelfAttention {
    n_head: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl CausalSelfAttention {
    /// Creates a causal self-attention module whose projections are registered under `builder`.
    pub fn new(builder: ParamBuilder, n_embd: usize, n_head: usize) -> Self {
        assert!(n_embd.is_multiple_of(n_head), "n_embd must be divisible by n_head");
        let head_dim = n_embd / n_head;
        Self {
            n_head,
            head_dim,
            q_proj: Linear::no_bias(builder.pp("q_proj"), n_embd, n_embd),
            k_proj: Linear::no_bias(builder.pp("k_proj"), n_embd, n_embd),
            v_proj: Linear::no_bias(builder.pp("v_proj"), n_embd, n_embd),
            out_proj: Linear::no_bias(builder.pp("out_proj"), n_embd, n_embd),
        }
    }

    /// Runs self-attention on `[B, T, C]` inputs using the provided RoPE caches.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let shape = x.layout().shape();
        assert_eq!(shape.ndim(), 3, "attention expects input shape [B, T, C]");

        let batch_size = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        assert_eq!(
            channels,
            self.n_head * self.head_dim,
            "input channel size must match attention width"
        );

        let x_flat = x.reshape(vec![batch_size * seq_len, channels]); // [B*T, C]
        let q = self.q_proj.forward(&x_flat)?.reshape(vec![
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ]); // [B, T, H, D]
        let k = self.k_proj.forward(&x_flat)?.reshape(vec![
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ]); // [B, T, H, D]
        let v = self.v_proj.forward(&x_flat)?.reshape(vec![
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ]); // [B, T, H, D]

        let q = apply_rotary_emb(&q, cos, sin).permute(vec![0, 2, 1, 3]); // [B, H, T, D]
        let k = apply_rotary_emb(&k, cos, sin).permute(vec![0, 2, 1, 3]); // [B, H, T, D]
        let v = v.permute(vec![0, 2, 1, 3]); // [B, H, T, D]

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(Some((2, 3)))) * scale; // [B, H, T, T]
        let mask = functional::causal_mask(batch_size, seq_len, 0, x.dtype(), x.device()); // [B, 1, T, T]
        let attn = (&scores + &mask).softmax(3); // [B, H, T, T]
        let y =
            attn.matmul(&v).permute(vec![0, 2, 1, 3]).reshape(vec![batch_size, seq_len, channels]); // [B, T, C]

        let y_flat = y.reshape(vec![batch_size * seq_len, channels]); // [B*T, C]
        self.out_proj.forward(&y_flat).map(|out| out.reshape(vec![batch_size, seq_len, channels]))
    }

    /// Returns the trainable parameters owned by the attention module.
    pub fn parameters(&self) -> Vec<Parameter> {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj]
            .iter()
            .flat_map(|proj| proj.parameters())
            .collect()
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
        let channels = shape[2];

        let x_flat = x.reshape(vec![batch_size * seq_len, channels]); // [B*T, C]
        let y = self.up_proj.forward(&x_flat)?; // [B*T, H]
        let y = y.relu(); // [B*T, H]
        let y = &y * &y; // [B*T, H]
        let y = self.down_proj.forward(&y)?; // [B*T, C]
        Ok(y.reshape(vec![batch_size, seq_len, channels])) // [B, T, C]
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.up_proj.parameters().into_iter().chain(self.down_proj.parameters()).collect()
    }
}

/// Pre-norm residual GPT block.
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
        self.attn.parameters().into_iter().chain(self.mlp.parameters()).collect()
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
#[derive(Clone)]
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
}

impl GPTConfig {
    /// Returns the per-head channel width.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// Minimal GPT model: token embedding, decoder blocks, final norm, and LM head.
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
        let (cos, sin) = precompute_rotary_embeddings(
            config.sequence_len,
            config.head_dim(),
            config.rope_base,
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

        let channels = x.layout().shape()[2];
        let x_flat = x.reshape(vec![batch_size * seq_len, channels]); // [B*T, C]
        let logits = self.lm_head.forward(&x_flat)?; // [B*T, V]
        Ok(logits.reshape(vec![batch_size, seq_len, self.vocab_size])) // [B, T, V]
    }

    /// Returns the trainable parameters owned by the model.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.wte
            .parameters()
            .into_iter()
            .chain(self.blocks.iter().flat_map(|b| b.parameters()))
            .chain(self.lm_head.parameters())
            .collect()
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
