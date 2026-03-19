//! Optimizers for updating trainable variables.

use crate::error::Result;
use crate::tensor::Tensor;
use crate::var::Var;

/// Stochastic gradient descent optimizer.
pub struct SGD {
    vars: Vec<Var>,
    lr: f64,
}

impl SGD {
    pub fn new(vars: Vec<Var>, lr: f64) -> Self {
        Self { vars, lr }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Runs backward on `loss`, then updates each variable: w = w - lr * grad.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        for var in &self.vars {
            if let Some(grad) = grads.get(var.id()) {
                let updated = (&**var - &(grad * self.lr)).attach();
                var.set(&updated)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SGD;
    use crate::{Device, Tensor, Var};

    #[test]
    fn test_sgd_step() {
        // Arrange
        let x = Var::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
        let target = Tensor::from_vec(vec![3.0f32], (1,), Device::Cpu);
        let mut sgd = SGD::new(vec![x.clone()], 0.1);

        // Act
        let diff = &*x - &target;
        let loss = (&diff * &diff).sum(vec![0], true);
        sgd.backward_step(&loss).unwrap();
        let val: Vec<f32> = x.to_vec().unwrap();

        // Assert
        assert!((val[0] - 0.6).abs() < 1e-5, "x after step = {}", val[0]);
    }

    #[test]
    fn test_sgd_loss_decreases() {
        // Arrange
        let x = Var::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
        let target = Tensor::from_vec(vec![3.0f32], (1,), Device::Cpu);
        let mut sgd = SGD::new(vec![x.clone()], 0.1);
        let mut losses = Vec::new();

        // Act
        for _ in 0..5 {
            let diff = &*x - &target;
            let loss = (&diff * &diff).sum(vec![0], true);
            let loss_val: Vec<f32> = loss.to_vec().unwrap();
            losses.push(loss_val[0]);
            sgd.backward_step(&loss).unwrap();
        }

        // Assert
        assert!(losses.windows(2).all(|window| window[1] < window[0]), "{losses:?}");
    }

    #[test]
    fn test_sgd_preserves_grad_tracking() {
        // Arrange
        let x = Var::new(Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
        let mut sgd = SGD::new(vec![x.clone()], 0.01);

        // Act
        let loss = (&*x * &*x).sum(vec![0], true);
        sgd.backward_step(&loss).unwrap();

        // Assert
        assert!(x.requires_grad());
    }
}
