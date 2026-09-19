//! Token sampling: turn next-token logits into a token choice.
//!
//! Temperature controls sharpness: lower values sharpen the distribution toward the most likely token, and 0 selects it outright.
//! Top-k keeps only the k most likely tokens and discards the rest.
//! Top-p keeps the smallest set of most likely tokens whose probabilities add up past p.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// How to turn a row of next-token logits into one token id.
///
/// `temperature` of 0 selects the highest logit directly. `top_k` and
/// `top_p` narrow the candidates before sampling; `seed` makes the draw
/// reproducible.
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    /// Distribution sharpness; 0 means greedy. Defaults to 1.0.
    pub temperature: f32,
    /// Keep only the k most likely tokens. Defaults to no filtering.
    pub top_k: Option<usize>,
    /// Keep the smallest likely set summing past p. Defaults to no filtering.
    pub top_p: Option<f32>,
    /// Seed for the sampling draw. Defaults to 0.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    /// A config with defaults: temperature 1.0, no filtering, seed 0.
    pub fn new() -> Self {
        Self::default()
    }

    fn validate(&self) {
        assert!(
            !self.temperature.is_nan() && self.temperature >= 0.0,
            "sampling temperature must be >= 0, got {}",
            self.temperature
        );
        if let Some(k) = self.top_k {
            assert!(k > 0, "top_k must be >= 1, got {k}");
        }
        if let Some(p) = self.top_p {
            assert!(
                (0.0..=1.0).contains(&p),
                "top_p must be within [0, 1], got {p}"
            );
        }
    }
}

/// Draw one token id from `logits` under `config`.
///
/// Applies temperature scaling, then top-k, then top-p masking, then a
/// seeded draw from the resulting distribution. Temperature 0 skips the
/// draw and returns the argmax.
///
/// Panics on empty logits or an invalid config.
pub fn sample_token(logits: &[f32], config: &SamplingConfig) -> u32 {
    assert!(!logits.is_empty(), "sample_token needs non-empty logits");
    config.validate();
    if config.temperature == 0.0 {
        return argmax(logits);
    }
    let scaled = apply_temperature(logits, config.temperature);
    let mut filtered = scaled;
    if let Some(k) = config.top_k {
        filtered = apply_top_k(&filtered, k);
    }
    if let Some(p) = config.top_p {
        filtered = apply_top_p(&filtered, p);
    }
    sample_from_probs(&softmax(&filtered), config.seed)
}

/// Scale logits by temperature: higher flattens, lower sharpens.
pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    assert!(
        !temperature.is_nan() && temperature > 0.0,
        "temperature must be > 0, got {temperature}"
    );
    logits.iter().map(|&x| x / temperature).collect()
}

/// Mask every logit outside the top k with negative infinity.
///
/// A k at or above the vocabulary size leaves the logits untouched.
/// Ties break toward the lower index so exactly k entries survive.
pub fn apply_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    assert!(!logits.is_empty(), "apply_top_k needs non-empty logits");
    assert!(k > 0, "top_k must be >= 1, got {k}");
    if k >= logits.len() {
        return logits.to_vec();
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let mut kept = vec![false; logits.len()];
    for &i in order.iter().take(k) {
        kept[i] = true;
    }
    logits
        .iter()
        .enumerate()
        .map(|(i, &x)| if kept[i] { x } else { f32::NEG_INFINITY })
        .collect()
}

/// Mask every logit outside the nucleus with negative infinity.
///
/// Sorts by probability, then keeps the smallest set whose probabilities
/// add up past p. Always keeps at least the most likely token, so p of 0
/// keeps exactly one and p of 1 keeps everything.
pub fn apply_top_p(logits: &[f32], p: f32) -> Vec<f32> {
    assert!(!logits.is_empty(), "apply_top_p needs non-empty logits");
    assert!(
        (0.0..=1.0).contains(&p),
        "top_p must be within [0, 1], got {p}"
    );
    if p >= 1.0 {
        return logits.to_vec();
    }
    let probs = softmax(logits);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    let mut kept = vec![false; logits.len()];
    let mut cumulative = 0.0f32;
    for &i in &order {
        kept[i] = true;
        cumulative += probs[i];
        if cumulative >= p {
            break;
        }
    }
    logits
        .iter()
        .enumerate()
        .map(|(i, &x)| if kept[i] { x } else { f32::NEG_INFINITY })
        .collect()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in logits.iter().enumerate().skip(1) {
        if x > logits[best] {
            best = i;
        }
    }
    best as u32
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn sample_from_probs(probs: &[f32], seed: u64) -> u32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let draw: f32 = rng.random_range(0.0..1.0);
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if draw < cumulative {
            return i as u32;
        }
    }
    // Rounding can leave cumulative just under 1.0; fall back to the last id.
    probs.len() as u32 - 1
}
