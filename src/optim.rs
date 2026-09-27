//! Optimizers for updating trainable parameters.

use std::collections::{BTreeMap, HashMap, HashSet};

use half::{bf16, f16};

use crate::DType;
use crate::GradientStore;
use crate::error::{Error, Result};
use crate::nn::Parameter;
use crate::tensor::{Tensor, TensorId};

/// Stochastic gradient descent optimizer.
#[derive(Debug)]
pub struct SGD {
    parameters: Vec<Parameter>,
    lr: f64,
}

impl SGD {
    /// Creates an SGD optimizer over `parameters` with learning rate `lr`.
    pub fn new(parameters: Vec<Parameter>, lr: f64) -> Self {
        Self { parameters, lr }
    }

    /// Returns the current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Sets the learning rate used on subsequent steps.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Runs backward on `loss`, then updates each parameter: w = w - lr * grad.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step_with_grads(&grads)
    }

    /// Applies one SGD step using a precomputed gradient store.
    pub fn step_with_grads(&mut self, grads: &GradientStore) -> Result<()> {
        let mut seen = HashSet::new();
        for parameter in &self.parameters {
            if !seen.insert(parameter.id()) {
                continue;
            }
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

/// One AdamW parameter group with its own decoupled weight decay.
///
/// Group parameters so norms and biases can opt out of decay following the
/// standard recipe (decay nD weights, `weight_decay = 0.0` for biases and norms):
/// ```ignore
/// let opt = AdamWConfig::new(1e-3)
///     .build_with_groups(vec![
///         AdamWParamGroup::new(decay_params, 0.01),
///         AdamWParamGroup::new(no_decay_params, 0.0),
///     ]);
/// ```
#[derive(Debug, Clone)]
pub struct AdamWParamGroup {
    parameters: Vec<Parameter>,
    weight_decay: f64,
}

impl AdamWParamGroup {
    /// Creates a group over `parameters` with decoupled weight decay `weight_decay`.
    pub fn new(parameters: Vec<Parameter>, weight_decay: f64) -> Self {
        Self { parameters, weight_decay }
    }
}

/// Configuration for the AdamW optimizer, separate from its runtime state.
#[derive(Debug)]
pub struct AdamWConfig {
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
}

impl AdamWConfig {
    /// Creates a default AdamW config with learning rate `lr`.
    pub fn new(lr: f64) -> Self {
        Self { lr, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.0 }
    }

    /// Sets the `(beta1, beta2)` momentum coefficients.
    pub fn betas(mut self, betas: (f64, f64)) -> Self {
        self.betas = betas;
        self
    }

    /// Sets the epsilon added to the denominator for numerical stability.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the decoupled weight decay coefficient.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builds an AdamW optimizer over `parameters`.
    ///
    /// This is the single-group default: every parameter shares the config's
    /// `weight_decay`. Use [`build_with_groups`](Self::build_with_groups) to
    /// exclude norms and biases per group.
    pub fn build(self, parameters: Vec<Parameter>) -> AdamW {
        let weight_decay = self.weight_decay;
        self.build_with_groups(vec![AdamWParamGroup::new(parameters, weight_decay)])
    }

    /// Builds an AdamW optimizer over per-group parameters.
    ///
    /// Each group carries its own `weight_decay`; the config's `weight_decay`
    /// applies only to [`build`](Self::build). Shared hyperparameters
    /// (`lr`, `betas`, `eps`) come from the config.
    pub fn build_with_groups(self, groups: Vec<AdamWParamGroup>) -> AdamW {
        AdamW {
            groups,
            lr: self.lr,
            betas: self.betas,
            eps: self.eps,
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
///
/// To exclude norms and biases, build with per-group decay:
/// ```ignore
/// let opt = AdamWConfig::new(1e-3)
///     .build_with_groups(vec![
///         AdamWParamGroup::new(decay_params, 0.01),
///         AdamWParamGroup::new(no_decay_params, 0.0),
///     ]);
/// ```
#[derive(Debug)]
pub struct AdamW {
    groups: Vec<AdamWParamGroup>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    m: HashMap<TensorId, Tensor>,
    v: HashMap<TensorId, Tensor>,
    step: usize,
}

impl AdamW {
    /// Sets the learning rate used on subsequent steps.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Returns the current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Returns the number of optimizer steps already applied.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Overrides the internal optimizer step counter.
    pub fn set_step_count(&mut self, step: usize) {
        self.step = step;
    }

    /// Returns the Adam moments tracked for `parameter`, if any.
    pub fn state_for(&self, parameter: &Parameter) -> Option<(Tensor, Tensor)> {
        let m = self.m.get(&parameter.id())?.clone();
        let v = self.v.get(&parameter.id())?.clone();
        Some((m, v))
    }

    /// Loads Adam moments for `parameter`.
    pub fn load_state_for(&mut self, parameter: &Parameter, m: Tensor, v: Tensor) {
        self.m.insert(parameter.id(), m.detach());
        self.v.insert(parameter.id(), v.detach());
    }

    /// Exports optimizer state using the same names as the model parameters.
    pub fn state_dict(&self, named_parameters: &[(String, Parameter)]) -> BTreeMap<String, Tensor> {
        let mut state = BTreeMap::new();
        for (name, parameter) in named_parameters {
            if let Some((m, v)) = self.state_for(parameter) {
                state.insert(format!("{name}.exp_avg"), m.detach());
                state.insert(format!("{name}.exp_avg_sq"), v.detach());
            }
        }
        state
    }

    /// Loads optimizer state from a named tensor map and restores the step counter.
    pub fn load_state_dict(
        &mut self,
        named_parameters: &[(String, Parameter)],
        state: &BTreeMap<String, Tensor>,
        step: usize,
    ) -> Result<()> {
        if state.len() != named_parameters.len() * 2 {
            return Err(Error::Checkpoint(format!(
                "optimizer state count mismatch: expected {}, found {}",
                named_parameters.len() * 2,
                state.len()
            )));
        }

        self.step = step;
        self.m.clear();
        self.v.clear();

        for (name, parameter) in named_parameters {
            let m_key = format!("{name}.exp_avg");
            let v_key = format!("{name}.exp_avg_sq");
            let m = state
                .get(&m_key)
                .ok_or_else(|| Error::Checkpoint(format!("missing optimizer state: {m_key}")))?;
            let v = state
                .get(&v_key)
                .ok_or_else(|| Error::Checkpoint(format!("missing optimizer state: {v_key}")))?;
            self.load_state_for(parameter, m.clone(), v.clone());
        }

        Ok(())
    }

    /// Runs backward on `loss`, then applies one AdamW step.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step_with_grads(&grads)
    }

    /// Applies one AdamW step using a precomputed gradient store.
    pub fn step_with_grads(&mut self, grads: &GradientStore) -> Result<()> {
        self.step += 1;

        let (beta1, beta2) = self.betas;
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        let mut seen = HashSet::new();
        for group in &self.groups {
            let weight_decay = group.weight_decay;
            for param in &group.parameters {
                if !seen.insert(param.id()) {
                    continue;
                }
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
                    if weight_decay > 0.0 { &w * (1.0 - self.lr * weight_decay) } else { w };

                // Parameter update: w = w_decayed - lr * m̂ / (√v̂ + ε)
                let update = &m_hat / &(&v_hat.sqrt() + self.eps);
                let updated = (&decayed - &(&update * self.lr)).attach();
                param.set(&updated)?;
            }
        }

        Ok(())
    }
}

/// Clips the global gradient norm across `parameters` to `max_norm`.
///
/// Returns the norm before clipping so callers can log it. F16 norms are
/// accumulated in f32 on the host so they cannot overflow the F16 range.
/// Panics if `max_norm` is not positive or the parameters are not floats.
pub fn clip_grad_norm(
    parameters: &[Parameter],
    grads: &mut GradientStore,
    max_norm: f64,
) -> Result<f32> {
    assert!(max_norm > 0.0, "max_norm must be positive");
    if parameters.is_empty() {
        return Ok(0.0);
    }

    let total_norm_value = match parameters[0].dtype() {
        DType::F16 => {
            let mut seen = HashSet::new();
            let mut total = 0.0f32;
            for parameter in parameters {
                if !seen.insert(parameter.id()) {
                    continue;
                }
                let Some(grad) = grads.get(parameter.id()) else {
                    continue;
                };
                total += grad.to_vec::<f16>()?.iter().map(|v| v.to_f32().powi(2)).sum::<f32>();
            }
            total.sqrt()
        }
        DType::BF16 => grad_norm(parameters, grads).to_vec::<bf16>()?[0].to_f32(),
        DType::F32 => grad_norm(parameters, grads).to_vec::<f32>()?[0],
        DType::I64 => panic!(
            "clip_grad_norm: {:?} is not supported, use float parameters",
            parameters[0].dtype()
        ),
    };
    if !total_norm_value.is_finite() {
        return Ok(total_norm_value);
    }
    if total_norm_value <= max_norm as f32 {
        return Ok(total_norm_value);
    }

    let scale = max_norm / (f64::from(total_norm_value) + 1e-6);
    let mut seen = HashSet::new();
    for parameter in parameters {
        if !seen.insert(parameter.id()) {
            continue;
        }
        let Some(grad) = grads.get(parameter.id()) else {
            continue;
        };
        grads.insert(parameter.id(), (&grad * scale).detach());
    }

    Ok(total_norm_value)
}

/// Computes the global gradient L2 norm on device in the parameter dtype.
fn grad_norm(parameters: &[Parameter], grads: &GradientStore) -> Tensor {
    let mut seen = HashSet::new();
    let mut total = Tensor::zeros((1,), parameters[0].dtype(), parameters[0].device());
    for parameter in parameters {
        if !seen.insert(parameter.id()) {
            continue;
        }
        let Some(grad) = grads.get(parameter.id()) else {
            continue;
        };
        let axes = (0..grad.layout().ndim()).collect::<Vec<_>>();
        total = &total + &(&grad * &grad).sum(axes, true);
    }
    total.sqrt()
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
#[derive(Debug)]
pub struct WarmupWarmdown {
    warmup_steps: usize,
    total_steps: usize,
    warmdown_steps: usize,
    final_lr_frac: f64,
}

impl WarmupWarmdown {
    /// Creates a warmup/hold/warmdown learning-rate schedule.
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

/// Cosine decay schedule: linear warmup, then cosine annealing to a floor.
///
/// Formula with `decay = total_steps - warmup_steps` and `t = step - warmup_steps`:
///
/// ```text
/// mult(step) = (step + 1) / warmup_steps                                    if step < warmup_steps
/// mult(step) = min + (1 - min) * (1 + cos(π * t / decay)) / 2               if warmup_steps <= step < total_steps
/// mult(step) = min                                                          if step >= total_steps
/// ```
///
/// Boundary behavior: the multiplier is exactly 1.0 at the end of warmup,
/// exactly `min_lr_frac` at `step == total_steps`, and stays at `min_lr_frac`
/// beyond `total_steps`. Panics if `total_steps == 0`, `warmup_steps >
/// `total_steps`, or `min_lr_frac` is outside [0, 1].
///
/// Usage matches the [`LrSchedule`] seam:
/// ```ignore
/// let lr = base_lr * schedule.lr_multiplier(step);
/// opt.set_lr(lr);
/// ```
#[derive(Debug)]
pub struct CosineDecay {
    warmup_steps: usize,
    total_steps: usize,
    min_lr_frac: f64,
}

impl CosineDecay {
    /// Creates a cosine decay schedule over `total_steps` with a linear
    /// warmup of `warmup_steps` (0 disables warmup) and a floor of
    /// `min_lr_frac` times the base learning rate.
    pub fn new(warmup_steps: usize, total_steps: usize, min_lr_frac: f64) -> Self {
        assert!(total_steps > 0, "total_steps must be positive");
        assert!(
            warmup_steps <= total_steps,
            "warmup_steps ({warmup_steps}) must not exceed total_steps ({total_steps})"
        );
        assert!(
            (0.0..=1.0).contains(&min_lr_frac),
            "min_lr_frac must be in [0, 1], got {min_lr_frac}"
        );
        Self { warmup_steps, total_steps, min_lr_frac }
    }
}

impl LrSchedule for CosineDecay {
    fn lr_multiplier(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            // Linear warmup: 1/warmup → 1, matching WarmupWarmdown.
            return (step + 1) as f64 / self.warmup_steps as f64;
        }
        if step >= self.total_steps {
            return self.min_lr_frac;
        }
        // warmup_steps <= step < total_steps implies decay_steps > 0.
        let decay_steps = self.total_steps - self.warmup_steps;
        let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
        self.min_lr_frac
            + (1.0 - self.min_lr_frac) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
    }
}

/// Step decay schedule: hold the rate constant, then multiply by
/// `decay_rate` every `decay_steps` steps, floored at `min_lr_frac`.
///
/// Formula with `drops = step / decay_steps` (integer division):
///
/// ```text
/// mult(step) = max(min_lr_frac, decay_rate^drops)
/// ```
///
/// Boundary behavior: the multiplier is exactly 1.0 for `step < decay_steps`,
/// drops to `decay_rate` at `step == decay_steps`, and never falls below
/// `min_lr_frac`. Panics if `decay_steps == 0`, `decay_rate` is outside
/// (0, 1), or `min_lr_frac` is outside [0, 1].
#[derive(Debug)]
pub struct StepDecay {
    decay_steps: usize,
    decay_rate: f64,
    min_lr_frac: f64,
}

impl StepDecay {
    /// Creates a step decay schedule that multiplies the rate by
    /// `decay_rate` every `decay_steps` steps, floored at `min_lr_frac`
    /// times the base learning rate.
    pub fn new(decay_steps: usize, decay_rate: f64, min_lr_frac: f64) -> Self {
        assert!(decay_steps > 0, "decay_steps must be positive");
        assert!(
            decay_rate > 0.0 && decay_rate < 1.0,
            "decay_rate must be in (0, 1), got {decay_rate}"
        );
        assert!(
            (0.0..=1.0).contains(&min_lr_frac),
            "min_lr_frac must be in [0, 1], got {min_lr_frac}"
        );
        Self { decay_steps, decay_rate, min_lr_frac }
    }
}

impl LrSchedule for StepDecay {
    fn lr_multiplier(&self, step: usize) -> f64 {
        let drops = step / self.decay_steps;
        self.decay_rate.powi(drops as i32).max(self.min_lr_frac)
    }
}

/// Which direction of a monitored metric counts as improvement for
/// [`ReduceOnPlateau`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// Lower is better (e.g. validation loss).
    Min,
    /// Higher is better (e.g. validation accuracy).
    Max,
}

/// Plateau reduction schedule: shrink the rate by `factor` when a monitored
/// metric stops improving.
///
/// Unlike step-indexed schedules this one is metric-driven. Feed each new
/// observation to [`step_metric`](Self::step_metric); it updates the internal
/// state and returns the current multiplier, which is also what the
/// [`LrSchedule`] impl reports (its `step` argument is ignored). Apply it to
/// an optimizer the usual way:
/// ```ignore
/// let lr = base_lr * schedule.step_metric(val_loss);
/// opt.set_lr(lr);
/// ```
///
/// Improvement rule with absolute `threshold` (default 1e-4):
///
/// ```text
/// Min mode: improved ⟺ metric < best - threshold
/// Max mode: improved ⟺ metric > best + threshold
/// ```
///
/// State transitions per observation: the first observation sets `best` with
/// no reduction. An improvement sets `best` and resets the bad-epoch count.
/// Otherwise the bad-epoch count grows; while a cooldown from a recent
/// reduction is still running it just ticks down, and otherwise once bad
/// epochs exceed `patience` the multiplier drops to
/// `max(min_lr_frac, current * factor)` and a new cooldown of `cooldown`
/// observations starts. The multiplier never rises again; call
/// [`reset`](Self::reset) to restart the search (e.g. after a regime change).
/// Panics if `factor` is outside (0, 1) or `min_lr_frac` is outside [0, 1].
#[derive(Debug)]
pub struct ReduceOnPlateau {
    factor: f64,
    patience: usize,
    min_lr_frac: f64,
    mode: PlateauMode,
    threshold: f64,
    cooldown: usize,
    best: Option<f64>,
    bad_epochs: usize,
    cooldown_remaining: usize,
    current: f64,
}

impl ReduceOnPlateau {
    /// Creates a plateau schedule that multiplies the rate by `factor` after
    /// `patience` consecutive observations without improvement, floored at
    /// `min_lr_frac`. Defaults to [`PlateauMode::Min`], threshold 1e-4, and
    /// no cooldown; use the builder methods to change them.
    pub fn new(factor: f64, patience: usize, min_lr_frac: f64) -> Self {
        assert!(factor > 0.0 && factor < 1.0, "factor must be in (0, 1), got {factor}");
        assert!(
            (0.0..=1.0).contains(&min_lr_frac),
            "min_lr_frac must be in [0, 1], got {min_lr_frac}"
        );
        Self {
            factor,
            patience,
            min_lr_frac,
            mode: PlateauMode::Min,
            threshold: 1e-4,
            cooldown: 0,
            best: None,
            bad_epochs: 0,
            cooldown_remaining: 0,
            current: 1.0,
        }
    }

    /// Sets whether lower ([`PlateauMode::Min`]) or higher
    /// ([`PlateauMode::Max`]) metrics count as improvement.
    pub fn mode(mut self, mode: PlateauMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the absolute improvement threshold (default 1e-4). Must be
    /// non-negative; changes smaller than this do not reset the bad-epoch
    /// count.
    pub fn threshold(mut self, threshold: f64) -> Self {
        assert!(threshold >= 0.0, "threshold must be non-negative, got {threshold}");
        self.threshold = threshold;
        self
    }

    /// Sets how many observations after a reduction pass before another
    /// reduction can happen.
    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Feeds one metric observation, advances the schedule state, and returns
    /// the current multiplier. Non-finite metrics are ignored: they leave the
    /// state untouched and return the current multiplier.
    pub fn step_metric(&mut self, metric: f64) -> f64 {
        if !metric.is_finite() {
            return self.current;
        }
        let Some(best) = self.best else {
            self.best = Some(metric);
            return self.current;
        };
        let improved = match self.mode {
            PlateauMode::Min => metric < best - self.threshold,
            PlateauMode::Max => metric > best + self.threshold,
        };
        if improved {
            self.best = Some(metric);
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.cooldown_remaining > 0 {
                self.cooldown_remaining -= 1;
            } else if self.bad_epochs > self.patience {
                self.current = (self.current * self.factor).max(self.min_lr_frac);
                self.bad_epochs = 0;
                self.cooldown_remaining = self.cooldown;
            }
        }
        self.current
    }

    /// Restarts the plateau search: clears the best metric, counters, and
    /// restores the multiplier to 1.0.
    pub fn reset(&mut self) {
        self.best = None;
        self.bad_epochs = 0;
        self.cooldown_remaining = 0;
        self.current = 1.0;
    }
}

impl LrSchedule for ReduceOnPlateau {
    fn lr_multiplier(&self, _step: usize) -> f64 {
        self.current
    }
}

/// Annealing shape for [`OneCycle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneCycleAnneal {
    /// Straight-line interpolation between phase endpoints.
    Linear,
    /// Cosine interpolation between phase endpoints.
    Cosine,
}

/// One-cycle schedule: ramp from a low rate up to the base rate, then anneal
/// down to a floor, over `total_steps`.
///
/// With `div = div_factor`, `final_div = final_div_factor`,
/// `up = round(pct_start * total_steps)` clamped to `[1, total_steps]`,
/// `start = 1 / div`, and `end = 1 / (div * final_div)`:
///
/// ```text
/// mult(step) = anneal(start → 1, step / up)              if step < up
/// mult(step) = anneal(1 → end, (step - up) / (total - up))  if up <= step < total
/// mult(step) = end                                       if step >= total
/// ```
///
/// where `anneal(a → b, p)` is `a + (b - a) * p` for
/// [`OneCycleAnneal::Linear`] and `b + (a - b) * (1 + cos(π * p)) / 2` for
/// [`OneCycleAnneal::Cosine`].
///
/// Boundary behavior: the multiplier is exactly `1 / div_factor` at step 0,
/// exactly 1.0 at the peak (`step == up`), exactly `end` at
/// `step == total_steps`, and stays at `end` beyond `total_steps`. Panics if
/// `total_steps == 0`, `pct_start` is outside (0, 1), `div_factor < 1.0`, or
/// `final_div_factor < 1.0`.
#[derive(Debug)]
pub struct OneCycle {
    total_steps: usize,
    up_steps: usize,
    start: f64,
    end: f64,
    anneal: OneCycleAnneal,
}

impl OneCycle {
    /// Creates a one-cycle schedule peaking at the base rate after
    /// `pct_start` of `total_steps`, starting at `1 / div_factor` of base and
    /// ending at `1 / (div_factor * final_div_factor)` of base, with linear
    /// annealing. Use [`anneal`](Self::anneal) for cosine annealing.
    pub fn new(
        total_steps: usize,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    ) -> Self {
        assert!(total_steps > 0, "total_steps must be positive");
        assert!(
            pct_start > 0.0 && pct_start < 1.0,
            "pct_start must be in (0, 1), got {pct_start}"
        );
        assert!(div_factor >= 1.0, "div_factor must be >= 1.0, got {div_factor}");
        assert!(
            final_div_factor >= 1.0,
            "final_div_factor must be >= 1.0, got {final_div_factor}"
        );
        let up_steps =
            ((pct_start * total_steps as f64).round() as usize).clamp(1, total_steps);
        Self {
            total_steps,
            up_steps,
            start: 1.0 / div_factor,
            end: 1.0 / (div_factor * final_div_factor),
            anneal: OneCycleAnneal::Linear,
        }
    }

    /// Sets the annealing shape used in both phases.
    pub fn anneal(mut self, anneal: OneCycleAnneal) -> Self {
        self.anneal = anneal;
        self
    }
}

impl LrSchedule for OneCycle {
    fn lr_multiplier(&self, step: usize) -> f64 {
        if step < self.up_steps {
            let progress = step as f64 / self.up_steps as f64;
            return match self.anneal {
                OneCycleAnneal::Linear => self.start + (1.0 - self.start) * progress,
                OneCycleAnneal::Cosine => {
                    1.0 + (self.start - 1.0) * (1.0
                        + (std::f64::consts::PI * progress).cos())
                        / 2.0
                }
            };
        }
        if step >= self.total_steps {
            return self.end;
        }
        // up_steps <= step < total_steps implies down_steps > 0.
        let down_steps = self.total_steps - self.up_steps;
        let progress = (step - self.up_steps) as f64 / down_steps as f64;
        match self.anneal {
            OneCycleAnneal::Linear => 1.0 - (1.0 - self.end) * progress,
            OneCycleAnneal::Cosine => {
                self.end
                    + (1.0 - self.end) * (1.0 + (std::f64::consts::PI * progress).cos())
                        / 2.0
            }
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
    fn test_adamw_param_groups_default_matches_single_group() {
        // Arrange — same params, lr, and decay built both ways
        let make_params = || {
            vec![
                Parameter::new(Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu)),
                Parameter::new(Tensor::from_vec(vec![-1.0f32], (1,), Device::Cpu)),
            ]
        };
        let single_params = make_params();
        let grouped_params = make_params();
        let mut single = AdamWConfig::new(0.05).weight_decay(0.1).build(single_params.clone());
        let mut grouped = AdamWConfig::new(0.05).build_with_groups(vec![
            AdamWParamGroup::new(vec![grouped_params[0].clone()], 0.1),
            AdamWParamGroup::new(vec![grouped_params[1].clone()], 0.1),
        ]);

        // Act — one step each with identical grads (grad = param)
        let mut grads = GradientStore::new();
        for param in single_params.iter().chain(grouped_params.iter()) {
            grads.insert(param.id(), (**param).clone().detach());
        }
        single.step_with_grads(&grads).unwrap();
        grouped.step_with_grads(&grads).unwrap();

        // Assert — identical results: single-group default is unchanged
        for (a, b) in single_params.iter().zip(grouped_params.iter()) {
            let a_val: Vec<f32> = a.to_vec().unwrap();
            let b_val: Vec<f32> = b.to_vec().unwrap();
            assert_eq!(a_val.len(), b_val.len());
            for (x, y) in a_val.iter().zip(b_val.iter()) {
                assert!((x - y).abs() < 1e-6, "{a_val:?} vs {b_val:?}");
            }
        }
    }

    #[test]
    fn test_adamw_param_groups_selective_decay() {
        // Arrange — decay group vs no-decay group, zero grads isolate decay
        let decayed = Parameter::new(Tensor::from_vec(vec![5.0f32], (1,), Device::Cpu));
        let frozen = Parameter::new(Tensor::from_vec(vec![5.0f32], (1,), Device::Cpu));
        let mut opt = AdamWConfig::new(0.1).build_with_groups(vec![
            AdamWParamGroup::new(vec![decayed.clone()], 0.1),
            AdamWParamGroup::new(vec![frozen.clone()], 0.0),
        ]);

        // Act — loss = 0 * w means grad ≈ 0, only decay acts
        let loss = (&(&*decayed * 0.0) + &(&*frozen * 0.0)).sum(vec![0], true);
        opt.backward_step(&loss).unwrap();

        // Assert — decay applies per group
        let decayed_val: Vec<f32> = decayed.to_vec().unwrap();
        let frozen_val: Vec<f32> = frozen.to_vec().unwrap();
        assert!(
            (decayed_val[0] - 5.0 * (1.0 - 0.1 * 0.1)).abs() < 1e-5,
            "decayed param = {}",
            decayed_val[0]
        );
        assert!((frozen_val[0] - 5.0).abs() < 1e-5, "frozen param = {}", frozen_val[0]);
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

    #[test]
    fn test_clip_grad_norm_scales_f16_gradients() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(
            vec![f16::from_f32(0.0), f16::from_f32(0.0)],
            (2,),
            Device::Cpu,
        ));
        let mut grads = GradientStore::new();
        grads.insert(
            x.id(),
            Tensor::from_vec(vec![f16::from_f32(3.0), f16::from_f32(4.0)], (2,), Device::Cpu),
        );

        // Act
        let norm = clip_grad_norm(std::slice::from_ref(&x), &mut grads, 1.0).unwrap();
        let clipped: Vec<f32> = grads
            .get(x.id())
            .unwrap()
            .to_vec::<f16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();

        // Assert
        assert!((norm - 5.0).abs() < 1e-2);
        assert!((clipped[0] - 0.6).abs() < 1e-2);
        assert!((clipped[1] - 0.8).abs() < 1e-2);
    }

    #[test]
    fn test_clip_grad_norm_scales_f16_gradients_with_norm_above_f16_range() {
        // Arrange
        let n = 65_536;
        let x = Parameter::new(Tensor::from_vec(vec![f16::from_f32(0.0); n], (n,), Device::Cpu));
        let mut grads = GradientStore::new();
        grads.insert(x.id(), Tensor::from_vec(vec![f16::from_f32(1.0); n], (n,), Device::Cpu));

        // Act
        let norm = clip_grad_norm(std::slice::from_ref(&x), &mut grads, 1.0).unwrap();
        let clipped = grads.get(x.id()).unwrap().to_vec::<f16>().unwrap();

        // Assert
        assert!((norm - 256.0).abs() < 1e-2);
        assert!(clipped.iter().all(|v| (v.to_f32() - 1.0 / 256.0).abs() < 1e-5));
    }

    #[test]
    fn test_clip_grad_norm_scales_bf16_gradients() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(
            vec![bf16::from_f32(0.0), bf16::from_f32(0.0)],
            (2,),
            Device::Cpu,
        ));
        let mut grads = GradientStore::new();
        grads.insert(
            x.id(),
            Tensor::from_vec(vec![bf16::from_f32(3.0), bf16::from_f32(4.0)], (2,), Device::Cpu),
        );

        // Act
        let norm = clip_grad_norm(std::slice::from_ref(&x), &mut grads, 1.0).unwrap();
        let clipped: Vec<f32> = grads
            .get(x.id())
            .unwrap()
            .to_vec::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();

        // Assert
        assert!((norm - 5.0).abs() < 1e-2);
        assert!((clipped[0] - 0.6).abs() < 1e-2);
        assert!((clipped[1] - 0.8).abs() < 1e-2);
    }

    #[test]
    fn test_cosine_decay_warmup_ramp() {
        // Arrange
        let sched = CosineDecay::new(10, 100, 0.0);

        // Act / Assert — linear ramp matching WarmupWarmdown
        assert!((sched.lr_multiplier(0) - 0.1).abs() < 1e-12);
        assert!((sched.lr_multiplier(4) - 0.5).abs() < 1e-12);
        assert!((sched.lr_multiplier(9) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_decay_midpoint_and_end() {
        // Arrange — no warmup, 100 steps, floor 0.1
        let sched = CosineDecay::new(0, 100, 0.1);

        // Act / Assert — start at 1, midpoint at (1 + min) / 2, end at min
        assert!((sched.lr_multiplier(0) - 1.0).abs() < 1e-12);
        assert!((sched.lr_multiplier(50) - 0.55).abs() < 1e-12);
        assert!((sched.lr_multiplier(100) - 0.1).abs() < 1e-12);
        assert!((sched.lr_multiplier(150) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_decay_monotonic_after_warmup() {
        // Arrange
        let sched = CosineDecay::new(10, 100, 0.05);

        // Act
        let multipliers: Vec<f64> = (0..=120).map(|s| sched.lr_multiplier(s)).collect();

        // Assert — ramp up, then non-increasing decay clamped at the floor
        assert!(multipliers[..10].windows(2).all(|w| w[1] > w[0]));
        assert!(multipliers[10..].windows(2).all(|w| w[1] <= w[0]));
        assert!(multipliers[100..].iter().all(|&m| m == 0.05));
    }

    #[test]
    #[should_panic(expected = "total_steps must be positive")]
    fn test_cosine_decay_rejects_zero_total_steps() {
        // Arrange / Act / Assert — panics
        CosineDecay::new(0, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "must not exceed total_steps")]
    fn test_cosine_decay_rejects_warmup_beyond_total() {
        // Arrange / Act / Assert — panics
        CosineDecay::new(11, 10, 0.0);
    }

    #[test]
    #[should_panic(expected = "min_lr_frac must be in [0, 1]")]
    fn test_cosine_decay_rejects_floor_above_one() {
        // Arrange / Act / Assert — panics
        CosineDecay::new(0, 10, 1.5);
    }

    #[test]
    fn test_step_decay_drops_at_boundaries() {
        // Arrange — halve every 10 steps, floor 0.01
        let sched = StepDecay::new(10, 0.5, 0.01);

        // Act / Assert
        assert!((sched.lr_multiplier(0) - 1.0).abs() < 1e-12);
        assert!((sched.lr_multiplier(9) - 1.0).abs() < 1e-12);
        assert!((sched.lr_multiplier(10) - 0.5).abs() < 1e-12);
        assert!((sched.lr_multiplier(25) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_step_decay_floors_at_minimum() {
        // Arrange
        let sched = StepDecay::new(10, 0.5, 0.01);

        // Act / Assert — 0.5^7 < 0.01, so the floor wins far out
        assert!((sched.lr_multiplier(70) - 0.01).abs() < 1e-12);
        assert!((sched.lr_multiplier(1000) - 0.01).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "decay_steps must be positive")]
    fn test_step_decay_rejects_zero_decay_steps() {
        // Arrange / Act / Assert — panics
        StepDecay::new(0, 0.5, 0.0);
    }

    #[test]
    #[should_panic(expected = "decay_rate must be in (0, 1)")]
    fn test_step_decay_rejects_growth_rate() {
        // Arrange / Act / Assert — panics
        StepDecay::new(10, 1.5, 0.0);
    }

    #[test]
    fn test_plateau_holds_while_improving() {
        // Arrange
        let mut sched = ReduceOnPlateau::new(0.5, 2, 0.0);

        // Act
        let multipliers: Vec<f64> = [1.0, 0.9, 0.8, 0.7].iter().map(|&m| sched.step_metric(m)).collect();

        // Assert — steady improvement never reduces the rate
        assert!(multipliers.iter().all(|&m| m == 1.0));
        assert_eq!(sched.lr_multiplier(999), 1.0);
    }

    #[test]
    fn test_plateau_reduces_after_patience_exhausted() {
        // Arrange — patience 2 means the third bad epoch triggers reduction
        let mut sched = ReduceOnPlateau::new(0.5, 2, 0.0);
        sched.step_metric(1.0);

        // Act
        let first_bad = sched.step_metric(1.0);
        let second_bad = sched.step_metric(1.0);
        let third_bad = sched.step_metric(1.0);

        // Assert
        assert_eq!(first_bad, 1.0);
        assert_eq!(second_bad, 1.0);
        assert_eq!(third_bad, 0.5);
    }

    #[test]
    fn test_plateau_improvement_resets_bad_count() {
        // Arrange
        let mut sched = ReduceOnPlateau::new(0.5, 2, 0.0);
        sched.step_metric(1.0);
        sched.step_metric(1.0);
        sched.step_metric(1.0);

        // Act — a new best restarts the patience window
        sched.step_metric(0.5);
        let first_bad = sched.step_metric(0.5);
        let second_bad = sched.step_metric(0.5);

        // Assert
        assert_eq!(first_bad, 1.0);
        assert_eq!(second_bad, 1.0);
    }

    #[test]
    fn test_plateau_floors_and_ignores_nan() {
        // Arrange
        let mut sched = ReduceOnPlateau::new(0.1, 0, 0.05);
        sched.step_metric(1.0);

        // Act
        let after_nan = sched.step_metric(f64::NAN);
        let reduced = sched.step_metric(1.0);
        let floored = sched.step_metric(1.0);

        // Assert — NaN leaves state untouched, then 1*0.1 floors at 0.05
        assert_eq!(after_nan, 1.0);
        assert!((reduced - 0.1).abs() < 1e-12);
        assert!((floored - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_plateau_max_mode_and_cooldown() {
        // Arrange — accuracy-like metric with a 1-epoch cooldown
        let mut sched =
            ReduceOnPlateau::new(0.5, 0, 0.0).mode(PlateauMode::Max).cooldown(1);
        sched.step_metric(0.5);

        // Act
        let reduced = sched.step_metric(0.5);
        let cooling = sched.step_metric(0.5);
        let reduced_again = sched.step_metric(0.5);

        // Assert
        assert_eq!(reduced, 0.5);
        assert_eq!(cooling, 0.5);
        assert_eq!(reduced_again, 0.25);
    }

    #[test]
    #[should_panic(expected = "factor must be in (0, 1)")]
    fn test_plateau_rejects_growth_factor() {
        // Arrange / Act / Assert — panics
        ReduceOnPlateau::new(1.0, 2, 0.0);
    }

    #[test]
    fn test_one_cycle_linear_endpoints() {
        // Arrange — 100 steps, peak at 30, start 1/25, end 1/25000
        let sched = OneCycle::new(100, 0.3, 25.0, 1000.0);

        // Act / Assert
        assert!((sched.lr_multiplier(0) - 0.04).abs() < 1e-12);
        assert!((sched.lr_multiplier(30) - 1.0).abs() < 1e-12);
        assert!((sched.lr_multiplier(100) - 0.00004).abs() < 1e-12);
        assert!((sched.lr_multiplier(150) - 0.00004).abs() < 1e-12);
    }

    #[test]
    fn test_one_cycle_linear_midpoints() {
        // Arrange
        let sched = OneCycle::new(100, 0.3, 25.0, 1000.0);

        // Act / Assert — halfway up and halfway down interpolate linearly
        assert!((sched.lr_multiplier(15) - 0.52).abs() < 1e-12);
        let mid_down = sched.lr_multiplier(65);
        assert!((mid_down - (1.0 + 0.00004) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_one_cycle_cosine_stays_in_bounds_and_peaks() {
        // Arrange
        let sched = OneCycle::new(100, 0.3, 25.0, 1000.0).anneal(OneCycleAnneal::Cosine);

        // Act
        let multipliers: Vec<f64> = (0..=100).map(|s| sched.lr_multiplier(s)).collect();

        // Assert — rises to the peak, falls to the floor, never leaves [end, 1]
        assert!((multipliers[0] - 0.04).abs() < 1e-12);
        assert!((multipliers[30] - 1.0).abs() < 1e-12);
        assert!((multipliers[100] - 0.00004).abs() < 1e-12);
        assert!(multipliers[..30].windows(2).all(|w| w[1] >= w[0]));
        assert!(multipliers[30..].windows(2).all(|w| w[1] <= w[0]));
    }

    #[test]
    #[should_panic(expected = "pct_start must be in (0, 1)")]
    fn test_one_cycle_rejects_degenerate_split() {
        // Arrange / Act / Assert — panics
        OneCycle::new(100, 1.0, 25.0, 1000.0);
    }

    #[test]
    #[should_panic(expected = "div_factor must be >= 1.0")]
    fn test_one_cycle_rejects_div_below_one() {
        // Arrange / Act / Assert — panics
        OneCycle::new(100, 0.3, 0.5, 1000.0);
    }

    #[test]
    fn test_schedule_composes_with_optimizer_lr() {
        // Arrange
        let x = Parameter::new(Tensor::from_vec(vec![1.0f32], (1,), Device::Cpu));
        let mut opt = SGD::new(vec![x], 0.1);
        let sched = StepDecay::new(10, 0.5, 0.0);
        let base_lr = 0.1;

        // Act
        opt.set_lr(base_lr * sched.lr_multiplier(10));

        // Assert
        assert!((opt.lr() - 0.05).abs() < 1e-12);
    }
}
