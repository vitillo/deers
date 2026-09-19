use candle_core::{D, Device as CDevice, Tensor as CTensor};
use candle_nn::ops as candle_ops;
use deers::eval::{ParityCheck, check_close, sample_token_ids, score_model};
use deers::models::gpt::{self, GPTConfig, RopeScaling};
use deers::nn::{Module, ParamStore, Parameter, RMSNorm};
use deers::tokenizer::{Gpt2Tokenizer, Tokenizer};
use deers::{DType, Device, Tensor};

const TOL: f32 = 1e-4;

fn assert_passed(check: &ParityCheck) {
    assert!(
        check.passed(),
        "{} differs by {} over tolerance {}",
        check.name,
        check.max_abs_diff,
        check.tolerance
    );
}

fn candle_tensor(data: Vec<f32>, shape: &[usize]) -> CTensor {
    CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap()
}

fn candle_rms_norm(x: &CTensor, eps: f64) -> CTensor {
    let mean_sq = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
    let inv_norm = (mean_sq + eps).unwrap().powf(-0.5).unwrap();
    x.broadcast_mul(&inv_norm).unwrap()
}

fn candle_rotary(seq_len: usize, head_dim: usize, base: f32) -> (CTensor, CTensor) {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> =
        (0..half_dim).map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32)).collect();
    let freqs: Vec<f32> =
        (0..seq_len).flat_map(|t| inv_freq.iter().map(move |&freq| t as f32 * freq)).collect();
    let shape = [1, seq_len, 1, half_dim];
    let cos = freqs.iter().map(|&x| x.cos()).collect();
    let sin = freqs.iter().map(|&x| x.sin()).collect();
    (candle_tensor(cos, &shape), candle_tensor(sin, &shape))
}

fn candle_rotate(x: &CTensor, cos: &CTensor, sin: &CTensor) -> CTensor {
    let half_dim = x.dims4().unwrap().3 / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim).unwrap();
    let x2 = x.narrow(D::Minus1, half_dim, half_dim).unwrap();
    let y1 = x1.broadcast_mul(cos).unwrap().broadcast_add(&x2.broadcast_mul(sin).unwrap()).unwrap();
    let y2 = x1
        .broadcast_mul(&sin.neg().unwrap())
        .unwrap()
        .broadcast_add(&x2.broadcast_mul(cos).unwrap())
        .unwrap();
    CTensor::cat(&[&y1, &y2], D::Minus1).unwrap()
}

/// Epsilon for the QK-Norm step. Mirrors the fixed `QK_NORM_EPS` in `gpt.rs`.
const CANDLE_QK_NORM_EPS: f64 = 1e-6;

fn candle_attention(
    x: &CTensor,
    weights: &[CTensor],
    n_head: usize,
    cos: &CTensor,
    sin: &CTensor,
) -> CTensor {
    let (batch_size, seq_len, channels) = x.dims3().unwrap();
    let head_dim = channels / n_head;
    let flat = x.reshape((batch_size * seq_len, channels)).unwrap();
    let project = |w: &CTensor| {
        flat.matmul(w).unwrap().reshape((batch_size, seq_len, n_head, head_dim)).unwrap()
    };
    let q = project(&weights[0]);
    let q = candle_rms_norm(&q, CANDLE_QK_NORM_EPS).broadcast_mul(&weights[4]).unwrap();
    let q = candle_rotate(&q, cos, sin)
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let k = project(&weights[1]);
    let k = candle_rms_norm(&k, CANDLE_QK_NORM_EPS).broadcast_mul(&weights[5]).unwrap();
    let k = candle_rotate(&k, cos, sin)
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let v = project(&weights[2]).transpose(1, 2).unwrap().contiguous().unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * scale).unwrap();
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let scores = scores.broadcast_add(&candle_tensor(mask, &[1, 1, seq_len, seq_len])).unwrap();
    let attn = candle_ops::softmax(&scores, D::Minus1).unwrap();
    attn.matmul(&v)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((batch_size, seq_len, channels))
        .unwrap()
        .reshape((batch_size * seq_len, channels))
        .unwrap()
        .matmul(&weights[3])
        .unwrap()
        .reshape((batch_size, seq_len, channels))
        .unwrap()
}

fn deers_weights_to_candle(params: &[Parameter]) -> Vec<CTensor> {
    params
        .iter()
        .map(|p| {
            candle_tensor(
                p.to_vec::<f32>().unwrap(),
                &p.layout().shape().iter().copied().collect::<Vec<_>>(),
            )
        })
        .collect()
}

#[test]
fn rmsnorm_block_matches_candle() {
    // Arrange
    let norm = RMSNorm::new(1e-5);
    let values: Vec<f32> = vec![0.5, -1.0, 2.0, 0.25, 1.5, 0.0, -0.5, 1.0];
    let x = Tensor::from_vec(values.clone(), (1, 2, 4), Device::Cpu);

    // Act
    let actual = norm.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let expected = candle_rms_norm(&candle_tensor(values, &[1, 2, 4]), 1e-5)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Assert
    assert_passed(&check_close("rmsnorm", &actual, &expected, TOL));
}

#[test]
fn attention_block_matches_candle() {
    // Arrange
    let store = ParamStore::new();
    let attn = gpt::CausalSelfAttention::new(store.root(), 8, 2);
    let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 1.0).collect();
    let x = Tensor::from_vec(values.clone(), (1, 3, 8), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(3, 4, 10_000.0, DType::F32, Device::Cpu);
    let weights = deers_weights_to_candle(&attn.parameters());
    let (cos_ref, sin_ref) = candle_rotary(3, 4, 10_000.0);

    // Act
    let actual = attn.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();
    let expected =
        candle_attention(&candle_tensor(values, &[1, 3, 8]), &weights, 2, &cos_ref, &sin_ref)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

    // Assert
    assert_passed(&check_close("attention", &actual, &expected, TOL));
}

#[test]
fn mlp_block_matches_candle() {
    // Arrange
    let store = ParamStore::new();
    let mlp = gpt::MLP::new(store.root(), 8, 16);
    let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 1.0).collect();
    let x = Tensor::from_vec(values.clone(), (1, 3, 8), Device::Cpu);
    let weights = deers_weights_to_candle(&mlp.parameters());

    // Act
    let actual = mlp.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let hidden =
        candle_tensor(values, &[3, 8]).matmul(&weights[0]).unwrap().relu().unwrap().sqr().unwrap();
    let expected =
        hidden.matmul(&weights[1]).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Assert
    assert_passed(&check_close("mlp", &actual, &expected, TOL));
}

#[test]
fn bundled_sample_scores_end_to_end_on_cpu() {
    // Arrange
    let tokenizer = Gpt2Tokenizer::new();
    let ids = sample_token_ids(&tokenizer);
    let config = GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        sequence_len: 128,
        n_layer: 1,
        n_head: 2,
        n_embd: 8,
        mlp_hidden_dim: 16,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
        rope_scaling: RopeScaling::None,
    };
    let model = gpt::GPT::new(config, ParamStore::new().root());

    // Act
    let report = score_model(&model, &ids);

    // Assert
    assert_eq!(report.n_tokens, ids.len() - 1);
    assert!(report.perplexity.is_finite(), "perplexity={}", report.perplexity);
    assert!(report.perplexity >= 1.0, "perplexity={}", report.perplexity);
    assert!(
        (report.perplexity.ln() - report.mean_nll).abs() < 1e-9,
        "mean_nll={} perplexity={}",
        report.mean_nll,
        report.perplexity
    );
}
