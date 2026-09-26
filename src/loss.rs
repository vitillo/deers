//! Loss functions composed from primitive tensor operations.

use half::{bf16, f16};

use crate::tensor::Tensor;

/// Reduction applied to the per-sample losses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reduction {
    /// Average over the included samples.
    #[default]
    Mean,
    /// Sum over the included samples.
    Sum,
    /// Return the per-sample losses with shape `(batch,)`.
    None,
}

/// Negative log-likelihood loss.
///
/// Takes log-probabilities of shape `(batch, classes)` and integer targets
/// with shape `(batch,)`. Returns a scalar loss tensor.
///
/// Equivalent to PyTorch's `F.nll_loss` or candle's `loss::nll`.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    nll_loss_with_options(log_probs, targets, Reduction::Mean, None)
}

/// Negative log-likelihood loss with reduction and ignore-index support.
///
/// `ignore_index` drops matching target positions before reducing: they
/// contribute zero loss, zero gradient, and are excluded from the `Mean`
/// divisor. A batch where every position is ignored returns zero (with zero
/// gradient) instead of NaN. Ignored targets are clamped to class `0` before
/// gathering so out-of-range sentinels such as `-100` never index out of
/// bounds.
pub fn nll_loss_with_options(
    log_probs: &Tensor,
    targets: &Tensor,
    reduction: Reduction,
    ignore_index: Option<i64>,
) -> Tensor {
    let batch = log_probs.layout().shape()[0];
    if reduction == Reduction::Mean && ignore_index.is_none() {
        // Legacy path: keep default behavior (including output shape) unchanged.
        let batch_size = batch as f64;
        let picked = log_probs.gather(1, &targets.reshape((targets.layout().shape()[0], 1)));
        return picked.sum(vec![0, 1], true) * (-1.0 / batch_size);
    }
    let (safe_targets, keep, kept) = ignore_mask(targets, log_probs.dtype(), batch, ignore_index);
    let picked = log_probs.gather(1, &safe_targets.reshape((batch, 1)));
    let mut per_sample = -&picked.reshape((batch,));
    if let Some(keep) = keep {
        per_sample = &per_sample * &keep;
    }
    match reduction {
        Reduction::None => per_sample,
        Reduction::Sum => per_sample.sum(vec![0], true),
        Reduction::Mean => {
            if kept == 0 {
                per_sample.sum(vec![0], true) * 0.0
            } else {
                per_sample.sum(vec![0], true) * (1.0 / kept as f64)
            }
        }
    }
}

/// Cross-entropy loss (log-softmax + NLL combined).
///
/// Takes raw logits of shape `(batch, classes)` and integer targets
/// with shape `(batch,)`. Returns a scalar loss tensor.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    cross_entropy_with_options(logits, targets, Reduction::Mean, None)
}

/// Cross-entropy loss with reduction and ignore-index support.
///
/// See [`nll_loss_with_options`] for the `ignore_index` semantics.
pub fn cross_entropy_with_options(
    logits: &Tensor,
    targets: &Tensor,
    reduction: Reduction,
    ignore_index: Option<i64>,
) -> Tensor {
    let log_probs = logits.log_softmax(1);
    nll_loss_with_options(&log_probs, targets, reduction, ignore_index)
}

/// Builds gather-safe targets, an optional keep mask, and the kept count.
///
/// Returns `(safe_targets, keep_mask, kept)`. Without `ignore_index` the
/// targets pass through untouched with no mask.
fn ignore_mask(
    targets: &Tensor,
    loss_dtype: crate::DType,
    batch: usize,
    ignore_index: Option<i64>,
) -> (Tensor, Option<Tensor>, usize) {
    let Some(ignored) = ignore_index else {
        return (targets.clone(), None, batch);
    };
    let raw: Vec<i64> = targets.to_vec().unwrap();
    debug_assert_eq!(raw.len(), batch);
    let mut kept = 0;
    let mut safe = Vec::with_capacity(batch);
    let mut kept_flags = Vec::with_capacity(batch);
    for t in raw {
        if t == ignored {
            safe.push(0);
            kept_flags.push(false);
        } else {
            safe.push(t);
            kept_flags.push(true);
            kept += 1;
        }
    }
    let device = targets.device();
    let safe_targets = Tensor::from_vec(safe, (batch,), device);
    let keep = match loss_dtype {
        crate::DType::F32 => Tensor::from_vec(
            kept_flags.iter().map(|k| if *k { 1.0f32 } else { 0.0 }).collect::<Vec<f32>>(),
            (batch,),
            device,
        ),
        crate::DType::F16 => Tensor::from_vec(
            kept_flags
                .iter()
                .map(|k| if *k { f16::from_f32(1.0) } else { f16::from_f32(0.0) })
                .collect::<Vec<f16>>(),
            (batch,),
            device,
        ),
        crate::DType::BF16 => Tensor::from_vec(
            kept_flags
                .iter()
                .map(|k| if *k { bf16::from_f32(1.0) } else { bf16::ZERO })
                .collect::<Vec<bf16>>(),
            (batch,),
            device,
        ),
        // Integer log-probs carry no gradient; the mask dtype is irrelevant.
        crate::DType::I64 => Tensor::from_vec(
            kept_flags.iter().map(|k| if *k { 1i64 } else { 0 }).collect::<Vec<i64>>(),
            (batch,),
            device,
        ),
    };
    // The mask must match the loss dtype for the elementwise multiply.
    (safe_targets, Some(keep), kept)
}

#[cfg(test)]
mod tests {
    use super::{
        Reduction, cross_entropy, cross_entropy_with_options, nll_loss, nll_loss_with_options,
    };
    use crate::{Device, Tensor};

    #[test]
    fn test_nll_loss_forward() {
        // Arrange
        let log_probs = Tensor::from_vec(
            vec![-0.9076f32, -1.2076, -2.4076, -0.4076, -1.9076, -1.5076],
            (2, 3),
            Device::Cpu,
        );
        let targets = Tensor::from_vec(vec![0i64, 2], (2,), Device::Cpu);

        // Act
        let loss = nll_loss(&log_probs, &targets);
        let loss_val: Vec<f32> = loss.to_vec().unwrap();

        // Assert
        assert!((loss_val[0] - 1.2076).abs() < 1e-4, "loss={}", loss_val[0]);
    }

    #[test]
    fn test_nll_loss_backward() {
        // Arrange
        let log_probs =
            Tensor::from_vec(vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5], (2, 3), Device::Cpu)
                .attach();
        let targets = Tensor::from_vec(vec![1i64, 0], (2,), Device::Cpu);

        // Act
        let loss = nll_loss(&log_probs, &targets);
        let grads = loss.backward().unwrap();
        let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();

        // Assert
        assert_eq!(grad, vec![0.0, -0.5, 0.0, -0.5, 0.0, 0.0]);
    }

    #[test]
    fn test_cross_entropy_end_to_end() {
        // Arrange
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, -1.0, 0.0], (2, 3), Device::Cpu).attach();
        let targets = Tensor::from_vec(vec![2i64, 0], (2,), Device::Cpu);

        // Act
        let loss = cross_entropy(&logits, &targets);
        let grads = loss.backward().unwrap();
        let grad: Vec<f32> = grads.get(logits.id()).unwrap().to_vec().unwrap();
        let row1_sum: f32 = grad[0..3].iter().sum();
        let row2_sum: f32 = grad[3..6].iter().sum();

        // Assert
        assert_eq!(grad.len(), 6);
        assert!(row1_sum.abs() < 1e-4, "row1_sum={row1_sum}");
        assert!(row2_sum.abs() < 1e-4, "row2_sum={row2_sum}");
    }

    #[test]
    fn test_nll_loss_sum_reduction() {
        // Arrange
        let log_probs =
            Tensor::from_vec(vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5], (2, 3), Device::Cpu)
                .attach();
        let targets = Tensor::from_vec(vec![1i64, 0], (2,), Device::Cpu);

        // Act
        let loss = nll_loss_with_options(&log_probs, &targets, Reduction::Sum, None);
        let loss_val: Vec<f32> = loss.to_vec().unwrap();
        let grads = loss.backward().unwrap();
        let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();

        // Assert
        assert!((loss_val[0] - 1.6).abs() < 1e-4, "loss={}", loss_val[0]);
        assert_eq!(grad, vec![0.0, -1.0, 0.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_nll_loss_none_reduction() {
        // Arrange
        let log_probs =
            Tensor::from_vec(vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5], (2, 3), Device::Cpu)
                .attach();
        let targets = Tensor::from_vec(vec![1i64, 0], (2,), Device::Cpu);

        // Act
        let loss = nll_loss_with_options(&log_probs, &targets, Reduction::None, None);
        let loss_val: Vec<f32> = loss.to_vec().unwrap();
        let grads = loss.backward().unwrap();
        let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();

        // Assert
        assert_eq!(loss.layout().shape().as_slice(), &[2]);
        assert!((loss_val[0] - 1.2).abs() < 1e-4, "loss0={}", loss_val[0]);
        assert!((loss_val[1] - 0.4).abs() < 1e-4, "loss1={}", loss_val[1]);
        assert_eq!(grad, vec![0.0, -1.0, 0.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_nll_loss_ignore_index_mean() {
        // Arrange
        let log_probs =
            Tensor::from_vec(vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5], (2, 3), Device::Cpu)
                .attach();
        let targets = Tensor::from_vec(vec![1i64, -100], (2,), Device::Cpu);

        // Act
        let loss = nll_loss_with_options(&log_probs, &targets, Reduction::Mean, Some(-100));
        let loss_val: Vec<f32> = loss.to_vec().unwrap();
        let grads = loss.backward().unwrap();
        let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();

        // Assert
        assert!((loss_val[0] - 1.2).abs() < 1e-4, "loss={}", loss_val[0]);
        assert_eq!(grad, vec![0.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_nll_loss_ignore_index_none() {
        // Arrange
        let log_probs =
            Tensor::from_vec(vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5], (2, 3), Device::Cpu)
                .attach();
        let targets = Tensor::from_vec(vec![1i64, -100], (2,), Device::Cpu);

        // Act
        let loss = nll_loss_with_options(&log_probs, &targets, Reduction::None, Some(-100));
        let loss_val: Vec<f32> = loss.to_vec().unwrap();
        let grads = loss.backward().unwrap();
        let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();

        // Assert
        assert!((loss_val[0] - 1.2).abs() < 1e-4, "loss0={}", loss_val[0]);
        assert_eq!(loss_val[1], 0.0);
        assert_eq!(grad, vec![0.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cross_entropy_ignore_index_sum() {
        // Arrange
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 1.0, -1.0, 0.0], (2, 3), Device::Cpu).attach();
        let targets = Tensor::from_vec(vec![2i64, -100], (2,), Device::Cpu);

        // Act
        let ignored = cross_entropy_with_options(&logits, &targets, Reduction::Sum, Some(-100))
            .to_vec::<f32>()
            .unwrap();
        let first_logits = logits.narrow(0, 0, 1);
        let first_target = targets.narrow(0, 0, 1);
        let kept = cross_entropy_with_options(&first_logits, &first_target, Reduction::Sum, None)
            .to_vec::<f32>()
            .unwrap();
        let grads = cross_entropy_with_options(&logits, &targets, Reduction::Sum, Some(-100))
            .backward()
            .unwrap();
        let grad: Vec<f32> = grads.get(logits.id()).unwrap().to_vec().unwrap();

        // Assert
        assert!((ignored[0] - kept[0]).abs() < 1e-5, "ignored={} kept={}", ignored[0], kept[0]);
        assert_eq!(grad[3..6], vec![0.0, 0.0, 0.0]);
        let row1_sum: f32 = grad[0..3].iter().sum();
        assert!(row1_sum.abs() < 1e-4, "row1_sum={row1_sum}");
    }
}
