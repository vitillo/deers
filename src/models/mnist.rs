use crate::Tensor;
use crate::error::Result;
use crate::nn::{Linear, Module, ParamBuilder, Parameter, ReLU};

/// A two-layer MLP for MNIST classification.
pub struct MnistMLP {
    fc1: Linear,
    relu: ReLU,
    fc2: Linear,
}

impl MnistMLP {
    /// Creates an MNIST MLP whose trainable weights are registered under `builder`.
    pub fn new(builder: ParamBuilder, hidden_size: usize) -> Self {
        Self {
            fc1: Linear::new(builder.pp("fc1"), 28 * 28, hidden_size),
            relu: ReLU,
            fc2: Linear::new(builder.pp("fc2"), hidden_size, 10),
        }
    }
}

impl Module for MnistMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.relu.forward(&x)?;
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut parameters = self.fc1.parameters();
        parameters.extend(self.fc2.parameters());
        parameters
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::{Module, ParamStore};
    use crate::{Device, Tensor};

    use super::MnistMLP;

    #[test]
    fn test_mnist_mlp_forward_shape() {
        // Arrange
        let model = MnistMLP::new(crate::nn::ParamStore::new().root(), 128);
        let x = Tensor::from_vec(vec![1.0f32; 2 * 28 * 28], (2, 28 * 28), Device::Cpu);

        // Act
        let out = model.forward(&x).unwrap();

        // Assert
        assert_eq!(out.layout().shape, (2, 10).into());
    }

    #[test]
    fn test_mnist_mlp_builder_registers_named_parameters() {
        // Arrange
        let store = ParamStore::new();

        // Act
        let _model = MnistMLP::new(store.root(), 64);
        let names = store.named_parameters().into_iter().map(|(name, _)| name).collect::<Vec<_>>();

        // Assert
        assert_eq!(names, vec!["fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight"]);
    }
}
