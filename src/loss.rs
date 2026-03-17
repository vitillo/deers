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
    let picked = log_probs.gather(1, targets); // (batch, 1)
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
