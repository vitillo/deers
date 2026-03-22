//! Optimizers for updating trainable parameters.

use std::collections::HashMap;

use crate::GradientStore;
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
        self.step_with_grads(&grads)
    }

    pub fn step_with_grads(&mut self, grads: &GradientStore) -> Result<()> {
        for parameter in &self.parameters {
            if let Some(grad) = grads.get(parameter.id()) {
                let grad = grad.detach();
                let w = parameter.detach();
                let step = &grad * self.lr;
                let updated = (&w - &step).attach();
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
        self.step_with_grads(&grads)
    }

    pub fn step_with_grads(&mut self, grads: &GradientStore) -> Result<()> {
        self.step += 1;

        let (beta1, beta2) = self.betas;
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        for param in &self.parameters {
            let grad = match grads.get(param.id()) {
                Some(g) => g.detach(),
                None => continue,
            };
            let w = param.detach();

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
            let decayed =
                if self.weight_decay > 0.0 { &w * (1.0 - self.lr * self.weight_decay) } else { w };

            // Parameter update: w = w_decayed - lr * m̂ / (√v̂ + ε)
            let update = &m_hat / &(&v_hat.sqrt() + self.eps);
            let updated = (&decayed - &(&update * self.lr)).attach();
            param.set(&updated)?;
        }

        Ok(())
    }
}

/// Clips the global gradient norm across `parameters` to `max_norm`.
///
/// Returns the norm before clipping so callers can log it.
pub fn clip_grad_norm(
    parameters: &[Parameter],
    grads: &mut GradientStore,
    max_norm: f64,
) -> Result<f32> {
    assert!(max_norm > 0.0, "max_norm must be positive");
    if parameters.is_empty() {
        return Ok(0.0);
    }

    let device = parameters[0].device();
    let dtype = parameters[0].dtype();
    let mut total = Tensor::zeros((1,), dtype, device);

    for parameter in parameters {
        let Some(grad) = grads.get(parameter.id()) else {
            continue;
        };
        let axes = (0..grad.layout().ndim()).collect::<Vec<_>>();
        total = &total + &(&grad * &grad).sum(axes, true);
    }

    let total_norm = total.sqrt();
    let total_norm_value = total_norm.to_vec::<f32>()?[0];
    if total_norm_value <= max_norm as f32 {
        return Ok(total_norm_value);
    }

    let scale = max_norm / (f64::from(total_norm_value) + 1e-6);
    for parameter in parameters {
        let Some(grad) = grads.get(parameter.id()) else {
            continue;
        };
        grads.insert(parameter.id(), (&grad * scale).detach());
    }

    Ok(total_norm_value)
}

/// A learning rate schedule maps a step number to a multiplier in [0, 1].
///
/// Usage in a training loop:
/// ```ignore
/// let lr = base_lr * schedule.lr_multiplier(step);
/// opt.set_lr(lr);
/// ```
pub trait LrSchedule {
    /// Returns the learning rate multiplier for the given step.
    fn lr_multiplier(&self, step: usize) -> f64;
}

/// nanochat-style schedule: linear warmup → constant → linear warmdown.
///
/// ```text
/// 1.0 |    /‾‾‾‾‾‾‾‾‾\
///     |   /             \
/// f   |  /               \___
///     | /
/// 0.0 +------------------------
///     0  warmup        total
/// ```
pub struct WarmupWarmdown {
    warmup_steps: usize,
    total_steps: usize,
    warmdown_steps: usize,
    final_lr_frac: f64,
}

impl WarmupWarmdown {
    pub fn new(
        warmup_steps: usize,
        total_steps: usize,
        warmdown_ratio: f64,
        final_lr_frac: f64,
    ) -> Self {
        let warmdown_steps = (warmdown_ratio * total_steps as f64).round() as usize;
        Self { warmup_steps, total_steps, warmdown_steps, final_lr_frac }
    }
}

impl LrSchedule for WarmupWarmdown {
    fn lr_multiplier(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup: 0 → 1
            (step + 1) as f64 / self.warmup_steps as f64
        } else if step + self.warmdown_steps <= self.total_steps {
            // Constant phase
            1.0
        } else {
            // Linear warmdown: 1 → final_lr_frac
            let remaining = self.total_steps.saturating_sub(step) as f64;
            let progress = remaining / self.warmdown_steps as f64;
            progress + (1.0 - progress) * self.final_lr_frac
        }
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
    fn test_warmup_warmdown_warmup_phase() {
        // Arrange — 10 warmup steps, 100 total, 50% warmdown, final_frac=0.05
        let sched = WarmupWarmdown::new(10, 100, 0.5, 0.05);

        // Act / Assert — linear ramp from 0.1 to 1.0
        assert!((sched.lr_multiplier(0) - 0.1).abs() < 1e-10);
        assert!((sched.lr_multiplier(4) - 0.5).abs() < 1e-10);
        assert!((sched.lr_multiplier(9) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_warmup_warmdown_constant_phase() {
        // Arrange
        let sched = WarmupWarmdown::new(10, 100, 0.5, 0.05);

        // Act / Assert — constant at 1.0 between warmup and warmdown
        assert!((sched.lr_multiplier(10) - 1.0).abs() < 1e-10);
        assert!((sched.lr_multiplier(30) - 1.0).abs() < 1e-10);
        assert!((sched.lr_multiplier(50) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_warmup_warmdown_warmdown_phase() {
        // Arrange
        let sched = WarmupWarmdown::new(10, 100, 0.5, 0.05);

        // Act / Assert — linear decay toward final_lr_frac
        let mid_warmdown = sched.lr_multiplier(75);
        assert!(mid_warmdown < 1.0 && mid_warmdown > 0.05);

        let at_end = sched.lr_multiplier(100);
        assert!((at_end - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_warmup_warmdown_monotonic() {
        // Arrange
        let sched = WarmupWarmdown::new(10, 100, 0.5, 0.05);

        // Act — collect multipliers for all steps
        let multipliers: Vec<f64> = (0..=100).map(|s| sched.lr_multiplier(s)).collect();

        // Assert — warmup is increasing, warmdown is decreasing
        assert!(multipliers[..10].windows(2).all(|w| w[1] > w[0]));
        assert!(multipliers[51..].windows(2).all(|w| w[1] <= w[0]));
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

    #[test]
    fn test_clip_grad_norm_scales_large_gradients() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(vec![0.0f32, 0.0], (2,), Device::Cpu));
        let mut grads = GradientStore::new();
        grads.insert(x.id(), Tensor::from_vec(vec![3.0f32, 4.0], (2,), Device::Cpu));

        // Act
        let norm = clip_grad_norm(std::slice::from_ref(&x), &mut grads, 1.0).unwrap();
        let clipped = grads.get(x.id()).unwrap().to_vec::<f32>().unwrap();

        // Assert
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((clipped[0] - 0.6).abs() < 1e-4);
        assert!((clipped[1] - 0.8).abs() < 1e-4);
    }
}
