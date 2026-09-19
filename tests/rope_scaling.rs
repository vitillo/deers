use deers::models::gpt::{
    GPT, GPTConfig, RopeScaling, precompute_rotary_embeddings,
    precompute_rotary_embeddings_scaled,
};
use deers::nn::ParamStore;
use deers::{DType, Device, Tensor};

fn tiny_config(scaling: RopeScaling) -> GPTConfig {
    GPTConfig {
        vocab_size: 8,
        sequence_len: 4,
        n_layer: 1,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
        rope_scaling: scaling,
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{label}[{index}]: got {a}, expected {e}");
    }
}

#[test]
fn test_unscaled_cache_matches_legacy_values() {
    // Arrange
    let seq_len = 2;
    let head_dim = 4;

    // Act
    let (legacy_cos, legacy_sin) =
        precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, Device::Cpu);
    let (cos, sin) = precompute_rotary_embeddings_scaled(
        seq_len,
        head_dim,
        10_000.0,
        RopeScaling::None,
        DType::F32,
        Device::Cpu,
    );

    // Assert
    assert_eq!(cos.to_vec::<f32>().unwrap(), legacy_cos.to_vec::<f32>().unwrap());
    assert_eq!(sin.to_vec::<f32>().unwrap(), legacy_sin.to_vec::<f32>().unwrap());
    assert_close(
        &cos.to_vec::<f32>().unwrap(),
        &[1.0, 1.0, 0.54030231, 0.99995],
        1e-6,
        "unscaled cos",
    );
    assert_close(
        &sin.to_vec::<f32>().unwrap(),
        &[0.0, 0.0, 0.84147098, 0.00999983],
        1e-6,
        "unscaled sin",
    );
}

#[test]
fn test_linear_scaling_matches_direct_formula() {
    // Arrange
    let scaling = RopeScaling::Linear { factor: 2.0 };

    // Act
    let (cos, sin) = precompute_rotary_embeddings_scaled(
        4,
        4,
        10_000.0,
        scaling,
        DType::F32,
        Device::Cpu,
    );

    // Assert
    let cos = cos.to_vec::<f32>().unwrap();
    let sin = sin.to_vec::<f32>().unwrap();
    assert_close(&cos[0..2], &[1.0, 1.0], 1e-6, "linear cos at t=0");
    assert_close(&sin[0..2], &[0.0, 0.0], 1e-6, "linear sin at t=0");
    assert_close(&cos[6..8], &[0.0707372, 0.999_887_5], 1e-6, "linear cos at t=3");
    assert_close(&sin[6..8], &[0.997_495, 0.01499944], 1e-6, "linear sin at t=3");
}

fn yarn_reference(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    factor: f32,
    original_max: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let tau = 2.0 * std::f32::consts::PI;
    let dim = head_dim as f32;
    let correction = |rotations: f32| {
        dim * (original_max as f32 / (rotations * tau)).ln() / (2.0 * base.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let mut high = correction(1.0).ceil().min(dim - 1.0);
    if low == high {
        high += 0.001;
    }
    let attention = 0.1 * factor.ln() + 1.0;
    let mut cos = Vec::with_capacity(seq_len * half_dim);
    let mut sin = Vec::with_capacity(seq_len * half_dim);
    for t in 0..seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / base.powf((2 * i) as f32 / dim);
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let scaled = freq / factor * ramp + freq * (1.0 - ramp);
            let angle = t as f32 * scaled;
            cos.push(angle.cos() * attention);
            sin.push(angle.sin() * attention);
        }
    }
    (cos, sin)
}

#[test]
fn test_yarn_cache_matches_direct_formula() {
    // Arrange
    let scaling = RopeScaling::yarn(4.0, 32768);
    let (expected_cos, expected_sin) = yarn_reference(2, 8, 1_000_000.0, 4.0, 32768);

    // Act
    let (cos, sin) = precompute_rotary_embeddings_scaled(
        2,
        8,
        1_000_000.0,
        scaling,
        DType::F32,
        Device::Cpu,
    );

    // Assert
    let cos = cos.to_vec::<f32>().unwrap();
    let sin = sin.to_vec::<f32>().unwrap();
    assert_close(&cos, &expected_cos, 1e-6, "yarn cos");
    assert_close(&sin, &expected_sin, 1e-6, "yarn sin");
    assert_close(&cos[0..4], &[1.138_629_4; 4], 1e-6, "yarn attention factor at t=0");
    assert_close(&sin[0..4], &[0.0; 4], 1e-6, "yarn sin at t=0");
}

#[test]
fn test_yarn_factor_one_matches_unscaled() {
    // Arrange
    let scaling = RopeScaling::yarn(1.0, 32768);

    // Act
    let (cos, sin) =
        precompute_rotary_embeddings_scaled(4, 8, 10_000.0, scaling, DType::F32, Device::Cpu);
    let (expected_cos, expected_sin) =
        precompute_rotary_embeddings(4, 8, 10_000.0, DType::F32, Device::Cpu);

    // Assert
    assert_close(
        &cos.to_vec::<f32>().unwrap(),
        &expected_cos.to_vec::<f32>().unwrap(),
        1e-6,
        "yarn factor=1 cos",
    );
    assert_close(
        &sin.to_vec::<f32>().unwrap(),
        &expected_sin.to_vec::<f32>().unwrap(),
        1e-6,
        "yarn factor=1 sin",
    );
}

#[test]
#[should_panic(expected = "RoPE linear factor must be positive")]
fn test_linear_zero_factor_panics() {
    // Arrange
    let scaling = RopeScaling::Linear { factor: 0.0 };

    // Act
    precompute_rotary_embeddings_scaled(4, 8, 10_000.0, scaling, DType::F32, Device::Cpu);
}

#[test]
#[should_panic(expected = "RoPE linear factor must be positive")]
fn test_linear_negative_factor_panics() {
    // Arrange
    let scaling = RopeScaling::Linear { factor: -2.0 };

    // Act
    precompute_rotary_embeddings_scaled(4, 8, 10_000.0, scaling, DType::F32, Device::Cpu);
}

#[test]
#[should_panic(expected = "RoPE YaRN factor must be positive")]
fn test_yarn_zero_factor_panics() {
    // Arrange
    let scaling = RopeScaling::yarn(0.0, 32768);

    // Act
    precompute_rotary_embeddings_scaled(4, 8, 10_000.0, scaling, DType::F32, Device::Cpu);
}

#[test]
#[should_panic(expected = "RoPE YaRN original_max_position_embeddings must be positive")]
fn test_yarn_zero_original_max_panics() {
    // Arrange
    let scaling = RopeScaling::yarn(4.0, 0);

    // Act
    precompute_rotary_embeddings_scaled(4, 8, 10_000.0, scaling, DType::F32, Device::Cpu);
}

#[test]
fn test_gpt_uses_configured_scaling() {
    // Arrange
    let plain = GPT::new(tiny_config(RopeScaling::None), ParamStore::new().root());
    let scaled =
        GPT::new(tiny_config(RopeScaling::Linear { factor: 2.0 }), ParamStore::new().root());
    for (dst, src) in scaled.parameters().iter().zip(plain.parameters().iter()) {
        let shape: Vec<usize> = src.layout().shape().iter().copied().collect();
        dst.set(&Tensor::from_vec(src.to_vec::<f32>().unwrap(), shape, Device::Cpu)).unwrap();
    }
    let idx = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), Device::Cpu);

    // Act
    let plain_logits = plain.forward(&idx).unwrap().to_vec::<f32>().unwrap();
    let scaled_logits = scaled.forward(&idx).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(plain_logits.len(), 3 * 8);
    assert_eq!(scaled_logits.len(), 3 * 8);
    assert!(plain_logits.iter().all(|x| x.is_finite()));
    assert!(scaled_logits.iter().all(|x| x.is_finite()));
    let max_diff = plain_logits
        .iter()
        .zip(scaled_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "scaled model output matches unscaled: {max_diff}");
}
