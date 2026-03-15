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
