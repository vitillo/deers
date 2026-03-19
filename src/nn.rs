//! Neural network modules: traits, layers, and composition.

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
        let weight =
            Var::new(Tensor::rand((in_features, out_features), DType::F32, Device::Cpu) * 2.0 * k - k);
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
