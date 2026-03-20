//! Neural network modules: traits, layers, and composition.

pub mod functional;
pub mod parameter;

use crate::error::Result;
use crate::tensor::Tensor;
use crate::{DType, Device};
pub use parameter::Parameter;

/// A neural network layer or model.
pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Returns all trainable parameters in this module.
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }

    fn to_device(&self, device: Device) -> Result<()> {
        for parameter in self.parameters() {
            parameter.to_device(device)?;
        }
        Ok(())
    }
}

/// Fully connected layer: `y = x @ weight` (+ optional bias).
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
}

impl Linear {
    /// Creates a Linear layer with uniform [-k, k] initialization (k = 1/sqrt(in)).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::new_inner(in_features, out_features, true)
    }

    /// Creates a Linear layer without bias.
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        Self::new_inner(in_features, out_features, false)
    }

    fn new_inner(in_features: usize, out_features: usize, bias: bool) -> Self {
        let k = 1.0 / (in_features as f64).sqrt();
        let weight = Parameter::new(
            Tensor::rand((in_features, out_features), DType::F32, Device::Cpu) * 2.0 * k - k,
        );
        let bias = if bias {
            Some(Parameter::new(
                Tensor::rand((out_features,), DType::F32, Device::Cpu) * 2.0 * k - k,
            ))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight);
        match &self.bias {
            Some(bias) => Ok(&out + &**bias),
            None => Ok(out),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            parameters.push(bias.clone());
        }
        parameters
    }
}

/// Embedding lookup table: maps integer indices to dense vectors.
pub struct Embedding {
    weight: Parameter,
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let weight =
            Parameter::new(Tensor::randn((vocab_size, hidden_size), DType::F32, Device::Cpu));
        Self { weight }
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let hidden_size = self.weight.layout().shape()[1];
        let mut final_dims: Vec<usize> = indices.layout().shape().iter().copied().collect();
        final_dims.push(hidden_size);
        let flat = indices.reshape(vec![indices.layout().size()]);
        let selected = self.weight.index_select(0, &flat);
        Ok(selected.reshape(final_dims))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}

/// RMSNorm: `x / sqrt(mean(x²) + eps)`.
pub struct RMSNorm {
    eps: f64,
}

impl RMSNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_axis = x.layout().ndim() - 1;
        let mean_sq = (x * x).mean(vec![last_axis], true);
        let inv_norm = (mean_sq + self.eps).scalar_powf(-0.5);
        Ok(x * &inv_norm)
    }
}

/// Element-wise ReLU activation.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu())
    }
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

        let q = functional::apply_rotary_emb(&q, cos, sin).permute(vec![0, 2, 1, 3]); // [B, H, T, D]
        let k = functional::apply_rotary_emb(&k, cos, sin).permute(vec![0, 2, 1, 3]); // [B, H, T, D]
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

/// A sequence of modules applied in order.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

/// Creates an empty Sequential to build with `.add()`.
pub fn seq() -> Sequential {
    Sequential { layers: vec![] }
}
