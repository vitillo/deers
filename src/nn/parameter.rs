//! Trainable parameters with gradient tracking, in-place update, and
//! hierarchical naming via [`ParamStore`] and [`ParamBuilder`].

use std::ops::Deref;

use crate::Device;
use crate::error::Result;
use crate::tensor::Tensor;

/// A mutable handle to a tensor whose value the optimizer can update.
///
/// Derefs to `Tensor`, so all tensor methods work directly on a `Parameter`.
/// Clone is cheap (shares the underlying Arc storage).
#[derive(Clone, Debug)]
pub struct Parameter(Tensor);

impl Parameter {
    /// Wraps a tensor as a trainable parameter (automatically enables grad tracking).
    pub fn new(tensor: Tensor) -> Self {
        Self(tensor.attach())
    }

    /// Copies `src` data into the variable's existing storage, keeping the same tensor id.
    ///
    /// The source storage is cloned before taking the write lock, so `src` may
    /// share storage with this parameter (as when re-applying an idempotent
    /// conversion) without deadlocking on the storage lock.
    pub fn set(&self, src: &Tensor) -> Result<()> {
        let storage = src.storage().clone();
        *self.0.storage_mut() = storage;
        Ok(())
    }

    /// Moves the parameter to `device` while preserving its tensor identity.
    pub fn to_device(&self, device: Device) -> Result<()> {
        if self.device() != device {
            let tensor = self.0.to_device(device)?;
            self.set(&tensor)?;
        }
        Ok(())
    }
}

impl Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Parameter;
    use crate::{Device, Tensor};

    #[test]
    fn test_new() {
        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

        // Act
        let var = Parameter::new(tensor);

        // Assert
        assert!(var.requires_grad());
        assert_eq!(var.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_set() {
        // Arrange
        let var = Parameter::new(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu));
        let replacement = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], (3,), Device::Cpu);

        // Act
        var.set(&replacement).unwrap();

        // Assert
        assert!(var.requires_grad());
        assert_eq!(var.to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_set_with_shared_storage_does_not_deadlock() {
        // Arrange: `detach` shares the parameter's storage, as when an
        // idempotent dtype conversion sets the converted tensor back.
        let var = Parameter::new(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu));

        // Act
        var.set(&var.detach()).unwrap();

        // Assert
        assert_eq!(var.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_device() {
        // Arrange
        let Some(device) = [Device::Cuda, Device::Mps].into_iter().find(|d| d.is_available())
        else {
            return;
        };
        let var = Parameter::new(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu));
        let id = var.id();

        // Act
        var.to_device(device).unwrap();

        // Assert
        assert_eq!(var.id(), id);
        assert_eq!(var.device(), device);
        assert!(var.requires_grad());
        assert_eq!(var.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
