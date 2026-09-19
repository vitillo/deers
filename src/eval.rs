//! Evaluation harness: perplexity scoring over a text sample plus reference parity checks.
//!
//! Perplexity scores a language model by the average number of next-token choices it hesitates
//! between: it exponentiates the mean negative log-probability over a corpus, so a model that
//! assigns probability 1 to every true next token scores 1 while a model guessing uniformly over
//! V tokens scores V. Lower is better because it means the model wastes less probability on
//! tokens that never occur.

use crate::models::gpt::GPT;
use crate::tokenizer::Tokenizer;
use crate::{DType, Device, Tensor, no_grad};

/// Tiny bundled sample for the default eval. Larger corpora stay opt-in: tokenize them yourself
/// and pass the ids to [`score_model`].
pub const SAMPLE_TEXT: &str =
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";

/// Perplexity aggregate over a corpus. `token_log_probs` holds `log p(true token)` per scored
/// position, `mean_nll` is its negated mean, and `perplexity` is `exp(mean_nll)`.
#[derive(Clone, Debug)]
pub struct PerplexityReport {
    /// `log p(true token)` at each scored position, in order.
    pub token_log_probs: Vec<f32>,
    /// Number of scored next-token predictions.
    pub n_tokens: usize,
    /// Mean negative log-likelihood over the scored positions.
    pub mean_nll: f64,
    /// `exp(mean_nll)`. Always finite and at least 1 for a real model output.
    pub perplexity: f64,
}

/// Outcome of one reference comparison. `passed` is true below `tolerance`.
#[derive(Clone, Copy, Debug)]
pub struct ParityCheck {
    /// Block under test, e.g. `"attention"`.
    pub name: &'static str,
    /// Largest absolute element difference seen.
    pub max_abs_diff: f32,
    /// Allowed difference. Use 1e-4 for f32 on CPU.
    pub tolerance: f32,
}

impl ParityCheck {
    /// Returns true when the observed difference fits the tolerance.
    pub fn passed(&self) -> bool {
        self.max_abs_diff <= self.tolerance
    }
}

/// Tokenizes [`SAMPLE_TEXT`] with an existing tokenizer. Panics when the text yields no tokens.
pub fn sample_token_ids(tokenizer: &impl Tokenizer) -> Vec<i64> {
    let ids: Vec<i64> = tokenizer.encode(SAMPLE_TEXT).into_iter().map(|id| id as i64).collect();
    assert!(!ids.is_empty(), "sample text must tokenize to at least one token");
    ids
}

/// Scores raw logits against targets. `logits` is `[N, V]`, `targets` is `[N]` with I64 ids.
/// Panics when there is nothing to score or the shapes disagree.
pub fn perplexity_from_logits(logits: &Tensor, targets: &Tensor) -> PerplexityReport {
    let shape = logits.layout().shape();
    assert_eq!(shape.ndim(), 2, "logits must have shape [N, V]");
    assert_eq!(targets.layout().shape().as_slice(), &[shape[0]], "targets must have shape [N]");
    assert_eq!(targets.dtype(), DType::I64, "targets must hold I64 token ids");
    let n_tokens = shape[0];
    assert!(n_tokens > 0, "need at least one scored token");

    let log_probs = logits.log_softmax(1);
    let picked = log_probs.gather(1, &targets.reshape(vec![n_tokens, 1]));
    let token_log_probs: Vec<f32> =
        picked.to_vec().expect("log-probabilities must be readable as f32");
    let mean_nll = -token_log_probs.iter().sum::<f32>() as f64 / n_tokens as f64;
    PerplexityReport { token_log_probs, n_tokens, mean_nll, perplexity: mean_nll.exp() }
}

/// Scores a model on token ids without tracking gradients. Position `i` predicts `ids[i + 1]`,
/// so the report holds `len - 1` predictions. Panics on fewer than two ids.
pub fn score_model(model: &GPT, token_ids: &[i64]) -> PerplexityReport {
    assert!(token_ids.len() >= 2, "need at least two tokens to score one prediction");
    let seq_len = token_ids.len();
    no_grad(|| {
        let idx = Tensor::from_vec(token_ids.to_vec(), (1, seq_len), Device::Cpu);
        let logits = model.forward(&idx).expect("eval forward must succeed");
        let vocab_size = logits.layout().shape()[2];
        let shifted = logits.narrow(1, 0, seq_len - 1).reshape(vec![seq_len - 1, vocab_size]);
        let targets = Tensor::from_vec(token_ids[1..].to_vec(), (seq_len - 1,), Device::Cpu);
        perplexity_from_logits(&shifted, &targets)
    })
}

/// Compares two flat outputs elementwise. Panics on empty or mismatched inputs.
pub fn check_close(
    name: &'static str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> ParityCheck {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    assert!(!actual.is_empty(), "{name}: nothing to compare");
    let max_abs_diff =
        actual.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).fold(0.0, f32::max);
    ParityCheck { name, max_abs_diff, tolerance }
}

#[cfg(test)]
mod tests {
    use super::{check_close, perplexity_from_logits, sample_token_ids, score_model};
    use crate::models::gpt::{GPTConfig, RopeScaling};
    use crate::nn::ParamStore;
    use crate::tokenizer::{Gpt2Tokenizer, Tokenizer};
    use crate::{Device, Tensor};

    fn tiny_config(vocab_size: usize) -> GPTConfig {
        GPTConfig {
            vocab_size,
            sequence_len: 8,
            n_layer: 1,
            n_head: 2,
            n_embd: 4,
            mlp_hidden_dim: 8,
            rms_norm_eps: 1e-5,
            rope_base: 10_000.0,
            rope_scaling: RopeScaling::None,
        }
    }

    #[test]
    fn uniform_logits_score_vocab_size() {
        // Arrange
        let logits = Tensor::zeros((3, 4), crate::DType::F32, Device::Cpu);
        let targets = Tensor::from_vec(vec![0i64, 1, 3], (3,), Device::Cpu);

        // Act
        let report = perplexity_from_logits(&logits, &targets);

        // Assert
        assert_eq!(report.n_tokens, 3);
        assert!((report.mean_nll - 1.3862944).abs() < 1e-5, "mean_nll={}", report.mean_nll);
        assert!((report.perplexity - 4.0).abs() < 1e-5, "perplexity={}", report.perplexity);
        assert_eq!(report.token_log_probs.len(), 3);
    }

    #[test]
    fn skewed_logits_match_hand_computation() {
        // Arrange: logits [2, 0] on target 0 give ppl = 1 + e^-2.
        let logits = Tensor::from_vec(vec![2.0f32, 0.0], (1, 2), Device::Cpu);
        let targets = Tensor::from_vec(vec![0i64], (1,), Device::Cpu);

        // Act
        let report = perplexity_from_logits(&logits, &targets);

        // Assert
        assert!((report.mean_nll - 0.1269280).abs() < 1e-5, "mean_nll={}", report.mean_nll);
        assert!((report.perplexity - 1.1353353).abs() < 1e-5, "perplexity={}", report.perplexity);
    }

    #[test]
    #[should_panic(expected = "at least one scored token")]
    fn empty_targets_panic() {
        // Arrange
        let logits = Tensor::zeros((0, 4), crate::DType::F32, Device::Cpu);
        let targets = Tensor::from_vec(Vec::<i64>::new(), (0,), Device::Cpu);

        // Act
        perplexity_from_logits(&logits, &targets);
    }

    #[test]
    #[should_panic(expected = "at least two tokens")]
    fn single_token_model_input_panics() {
        // Arrange
        let model = crate::models::gpt::GPT::new(tiny_config(8), ParamStore::new().root());

        // Act
        score_model(&model, &[1]);
    }

    #[test]
    fn model_scores_one_fewer_token_than_input() {
        // Arrange
        let model = crate::models::gpt::GPT::new(tiny_config(8), ParamStore::new().root());

        // Act
        let report = score_model(&model, &[1, 2, 3, 4]);

        // Assert
        assert_eq!(report.n_tokens, 3);
        assert_eq!(report.token_log_probs.len(), 3);
        assert!(report.perplexity.is_finite(), "perplexity={}", report.perplexity);
        assert!(report.perplexity >= 1.0, "perplexity={}", report.perplexity);
        assert!(
            (report.perplexity.ln() - report.mean_nll).abs() < 1e-9,
            "mean_nll={} perplexity={}",
            report.mean_nll,
            report.perplexity
        );
    }

    #[test]
    fn check_close_reports_max_difference() {
        // Arrange
        let actual = vec![1.0f32, 2.0, 4.0];
        let expected = vec![1.0f32, 2.5, 3.0];

        // Act
        let strict = check_close("block", &actual, &expected, 1e-4);
        let loose = check_close("block", &actual, &expected, 2.0);

        // Assert
        assert_eq!(strict.max_abs_diff, 1.0);
        assert!(!strict.passed());
        assert!(loose.passed());
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn check_close_rejects_mismatched_lengths() {
        // Arrange
        let actual = vec![1.0f32];
        let expected = vec![1.0f32, 2.0];

        // Act
        check_close("block", &actual, &expected, 1e-4);
    }

    #[test]
    fn sample_text_tokenizes() {
        // Arrange
        let tokenizer = Gpt2Tokenizer::new();

        // Act
        let ids = sample_token_ids(&tokenizer);

        // Assert
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| (id as usize) < tokenizer.vocab_size()));
    }
}
