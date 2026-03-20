//! Loss functions composed from primitive tensor operations.

use crate::tensor::Tensor;

/// Negative log-likelihood loss.
///
/// Takes log-probabilities of shape `(batch, classes)` and integer targets
/// with shape `(batch,)`. Returns a scalar loss tensor.
///
/// Equivalent to PyTorch's `F.nll_loss` or candle's `loss::nll`.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    let batch_size = log_probs.layout().shape()[0] as f64;
    let picked = log_probs.gather(1, &targets.reshape((targets.layout().shape()[0], 1)));
    picked.sum(vec![0, 1], true) * (-1.0 / batch_size)
}

/// Cross-entropy loss (log-softmax + NLL combined).
///
/// Takes raw logits of shape `(batch, classes)` and integer targets
/// with shape `(batch,)`. Returns a scalar loss tensor.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    let log_probs = logits.log_softmax(1);
    nll_loss(&log_probs, targets)
}

#[cfg(test)]
mod tests {
    use super::{cross_entropy, nll_loss};
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
}
