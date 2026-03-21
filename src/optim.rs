//! Optimizers for updating trainable parameters.

use std::collections::HashMap;

use crate::error::Result;
use crate::nn::Parameter;
use crate::tensor::{Tensor, TensorId};

/// Stochastic gradient descent optimizer.
pub struct SGD {
    parameters: Vec<Parameter>,
    lr: f64,
}

impl SGD {
    pub fn new(parameters: Vec<Parameter>, lr: f64) -> Self {
        Self { parameters, lr }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Runs backward on `loss`, then updates each parameter: w = w - lr * grad.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        for parameter in &self.parameters {
            if let Some(grad) = grads.get(parameter.id()) {
                let step = grad * self.lr;
                let updated = (&**parameter - &step).attach();
                parameter.set(&updated)?;
            }
        }
        Ok(())
    }
}

/// Configuration for the AdamW optimizer, separate from its runtime state.
pub struct AdamWConfig {
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
}

impl AdamWConfig {
    pub fn new(lr: f64) -> Self {
        Self { lr, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0 }
    }

    pub fn betas(mut self, betas: (f64, f64)) -> Self {
        self.betas = betas;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn build(self, parameters: Vec<Parameter>) -> AdamW {
        AdamW {
            parameters,
            lr: self.lr,
            betas: self.betas,
            eps: self.eps,
            weight_decay: self.weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }
}

/// AdamW optimizer with decoupled weight decay.
///
/// Implements the standard AdamW update rule:
///
/// ```text
/// m = β₁ * m + (1 - β₁) * grad
/// v = β₂ * v + (1 - β₂) * grad²
/// m̂ = m / (1 - β₁^t)
/// v̂ = v / (1 - β₂^t)
/// w = w * (1 - lr * λ) - lr * m̂ / (√v̂ + ε)
/// ```
///
/// Build with [`AdamWConfig`]:
/// ```ignore
/// let opt = AdamWConfig::new(1e-3)
///     .weight_decay(0.01)
///     .build(model.parameters());
/// ```
pub struct AdamW {
    parameters: Vec<Parameter>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    m: HashMap<TensorId, Tensor>,
    v: HashMap<TensorId, Tensor>,
    step: usize,
}

impl AdamW {
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step += 1;

        let (beta1, beta2) = self.betas;
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        for param in &self.parameters {
            let grad = match grads.get(param.id()) {
                Some(g) => g,
                None => continue,
            };

            // First moment: m = β₁ * m + (1 - β₁) * grad
            let m = self.m.entry(param.id()).or_insert_with(|| param.zeros_like());
            *m = &*m * beta1 + &grad * (1.0 - beta1);

            // Second moment: v = β₂ * v + (1 - β₂) * grad²
            let v = self.v.entry(param.id()).or_insert_with(|| param.zeros_like());
            *v = &*v * beta2 + &(&grad * &grad) * (1.0 - beta2);

            // Bias-corrected estimates
            let m_hat = &*m * (1.0 / bias_correction1);
            let v_hat = &*v * (1.0 / bias_correction2);

            // Decoupled weight decay: w = w * (1 - lr * λ)
            let decayed = if self.weight_decay > 0.0 {
                &**param * (1.0 - self.lr * self.weight_decay)
            } else {
                (**param).clone()
            };

            // Parameter update: w = w_decayed - lr * m̂ / (√v̂ + ε)
            let update = &m_hat / &(&v_hat.sqrt() + self.eps);
            let updated = (&decayed - &(&update * self.lr)).attach();
            param.set(&updated)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    #[test]
    fn test_sgd_step() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
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
        let x = Parameter::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
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
        let x = Parameter::new(Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
        let mut sgd = SGD::new(vec![x.clone()], 0.01);

        // Act
        let loss = (&*x * &*x).sum(vec![0], true);
        sgd.backward_step(&loss).unwrap();

        // Assert
        assert!(x.requires_grad());
    }

    #[test]
    fn test_adamw_step() {
        // Arrange — minimize (x - 3)^2 starting from x = 0
        let x = Parameter::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
        let target = Tensor::from_vec(vec![3.0f32], (1,), Device::Cpu);
        let mut opt = AdamWConfig::new(0.1).build(vec![x.clone()]);

        // Act
        let diff = &*x - &target;
        let loss = (&diff * &diff).sum(vec![0], true);
        opt.backward_step(&loss).unwrap();

        // Assert — x should move toward 3
        let val: Vec<f32> = x.to_vec().unwrap();
        assert!(val[0] > 0.0, "x should increase toward target, got {}", val[0]);
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn test_adamw_loss_decreases() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(vec![0.0f32], (1,), Device::Cpu));
        let target = Tensor::from_vec(vec![3.0f32], (1,), Device::Cpu);
        let mut opt = AdamWConfig::new(0.1).build(vec![x.clone()]);
        let mut losses = Vec::new();

        // Act
        for _ in 0..20 {
            let diff = &*x - &target;
            let loss = (&diff * &diff).sum(vec![0], true);
            let loss_val: Vec<f32> = loss.to_vec().unwrap();
            losses.push(loss_val[0]);
            opt.backward_step(&loss).unwrap();
        }

        // Assert
        assert!(
            *losses.last().unwrap() < losses[0] * 0.2,
            "loss should decrease significantly: {losses:?}"
        );
    }

    #[test]
    fn test_adamw_weight_decay() {
        // Arrange — only weight decay should pull x toward 0
        let x = Parameter::new(Tensor::from_vec(vec![5.0f32], (1,), Device::Cpu));
        let mut opt = AdamWConfig::new(0.1).weight_decay(0.1).build(vec![x.clone()]);

        // Act — loss = x * 0 means grad ≈ 0, only weight decay acts
        let loss = (&*x * 0.0).sum(vec![0], true);
        opt.backward_step(&loss).unwrap();
        let val: Vec<f32> = x.to_vec().unwrap();

        // Assert
        assert!(val[0] < 5.0, "weight decay should shrink x, got {}", val[0]);
    }

    #[test]
    fn test_adamw_preserves_grad_tracking() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
        let mut opt = AdamWConfig::new(0.01).build(vec![x.clone()]);

        // Act
        let loss = (&*x * &*x).sum(vec![0], true);
        opt.backward_step(&loss).unwrap();

        // Assert
        assert!(x.requires_grad());
    }

    #[test]
    fn test_adamw_converges_quadratic() {
        // Arrange — minimize f(x,y) = x^2 + y^2
        let x = Parameter::new(Tensor::from_vec(vec![5.0f32], (1,), Device::Cpu));
        let y = Parameter::new(Tensor::from_vec(vec![-3.0f32], (1,), Device::Cpu));
        let mut opt = AdamWConfig::new(0.1).build(vec![x.clone(), y.clone()]);

        // Act
        for _ in 0..100 {
            let loss = (&(&*x * &*x) + &(&*y * &*y)).sum(vec![0], true);
            opt.backward_step(&loss).unwrap();
        }

        // Assert
        let x_val: Vec<f32> = x.to_vec().unwrap();
        let y_val: Vec<f32> = y.to_vec().unwrap();
        assert!(x_val[0].abs() < 0.1, "x should be near 0, got {}", x_val[0]);
        assert!(y_val[0].abs() < 0.1, "y should be near 0, got {}", y_val[0]);
    }
}
