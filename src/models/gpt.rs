use half::f16;

use crate::error::Result;
use crate::nn::{Linear, Module, Parameter, functional};
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
    pub fn new(n_embd: usize, n_head: usize) -> Self {
        assert!(n_embd.is_multiple_of(n_head), "n_embd must be divisible by n_head");
        let head_dim = n_embd / n_head;
        Self {
            n_head,
            head_dim,
            q_proj: Linear::no_bias(n_embd, n_embd),
            k_proj: Linear::no_bias(n_embd, n_embd),
            v_proj: Linear::no_bias(n_embd, n_embd),
            out_proj: Linear::no_bias(n_embd, n_embd),
        }
    }

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

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.q_proj.parameters();
        parameters.extend(self.k_proj.parameters());
        parameters.extend(self.v_proj.parameters());
        parameters.extend(self.out_proj.parameters());
        parameters
    }

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
    pub fn new(n_embd: usize, hidden_dim: usize) -> Self {
        Self {
            up_proj: Linear::no_bias(n_embd, hidden_dim),
            down_proj: Linear::no_bias(hidden_dim, n_embd),
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
        let mut parameters = self.up_proj.parameters();
        parameters.extend(self.down_proj.parameters());
        parameters
    }
}
