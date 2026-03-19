use std::ops::Deref;

use crate::error::Result;
use crate::tensor::Tensor;

/// A mutable handle to a tensor whose value the optimizer can update.
///
/// Derefs to `Tensor`, so all tensor methods work directly on a `Var`.
/// Clone is cheap (shares the underlying Arc storage).
#[derive(Clone)]
pub struct Var(Tensor);

impl Var {
    /// Wraps a tensor as a trainable variable (automatically enables grad tracking).
    pub fn new(tensor: Tensor) -> Self {
        Self(tensor.attach())
    }

    /// Copies `src` data into the variable's existing storage, keeping the same tensor id.
    pub(crate) fn set(&self, src: &Tensor) -> Result<()> {
        let src_guard = src.storage();
        let mut dst_guard = self.0.storage_mut();
        *dst_guard = src_guard.clone();
        Ok(())
    }

    pub fn to_device(&self, device: crate::Device) -> Result<()> {
        if self.device() == device {
            return Ok(());
        }
        let tensor = self.0.to_device(device)?;
        self.set(&tensor)
    }
}

impl Deref for Var {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Var;
    use crate::{Device, Tensor};

    #[test]
    fn test_new() {
        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

        // Act
        let var = Var::new(tensor);

        // Assert
        assert!(var.requires_grad());
        assert_eq!(var.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
