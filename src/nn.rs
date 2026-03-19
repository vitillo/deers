//! Neural network modules: traits, layers, and composition.

pub mod functional;

use crate::error::Result;
use crate::tensor::Tensor;
use crate::var::Var;
use crate::{DType, Device};

/// A neural network layer or model.
pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Returns all trainable variables in this module.
    fn vars(&self) -> Vec<Var> {
        vec![]
    }

    fn to_device(&self, device: Device) -> Result<()> {
        for var in self.vars() {
            var.to_device(device)?;
        }
        Ok(())
    }
}

/// Fully connected layer: `y = x @ weight` (+ optional bias).
pub struct Linear {
    weight: Var,
    bias: Option<Var>,
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
        let weight = Var::new(
            Tensor::rand((in_features, out_features), DType::F32, Device::Cpu) * 2.0 * k - k,
        );
        let bias = if bias {
            Some(Var::new(Tensor::rand((out_features,), DType::F32, Device::Cpu) * 2.0 * k - k))
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

    fn vars(&self) -> Vec<Var> {
        let mut vars = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            vars.push(bias.clone());
        }
        vars
    }
}

/// Embedding lookup table: maps integer indices to dense vectors.
pub struct Embedding {
    weight: Var,
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let weight = Var::new(Tensor::randn((vocab_size, hidden_size), DType::F32, Device::Cpu));
        Self { weight }
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let hidden_size = self.weight.layout().shape()[1];
        let mut final_dims: Vec<usize> = indices.layout().shape().iter().copied().collect();
        final_dims.push(hidden_size);
        let flat = indices.reshape(vec![indices.layout().size()]);
        let selected = self.weight.index_select(&flat);
        Ok(selected.reshape(final_dims))
    }

    fn vars(&self) -> Vec<Var> {
        vec![self.weight.clone()]
    }
}

/// RMSNorm: `x / sqrt(mean(x²) + eps) * weight`.
pub struct RMSNorm {
    weight: Var,
    eps: f64,
}

impl RMSNorm {
    pub fn new(size: usize, eps: f64) -> Self {
        let weight = Var::new(Tensor::ones(vec![size], DType::F32, Device::Cpu));
        Self { weight, eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_axis = x.layout().ndim() - 1;
        let mean_sq = (x * x).mean(vec![last_axis], true);
        let inv_norm = (mean_sq + self.eps).scalar_powf(-0.5);
        let normalized = x * &inv_norm;
        Ok(&normalized * &*self.weight)
    }

    fn vars(&self) -> Vec<Var> {
        vec![self.weight.clone()]
    }
}

/// Element-wise ReLU activation.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu())
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

    fn vars(&self) -> Vec<Var> {
        self.layers.iter().flat_map(|l| l.vars()).collect()
    }
}

/// Creates an empty Sequential to build with `.add()`.
pub fn seq() -> Sequential {
    Sequential { layers: vec![] }
}
