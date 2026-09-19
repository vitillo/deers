use std::collections::BTreeMap;

use candle_core::{D, Device as CDevice, Tensor as CTensor};
use candle_nn::ops as candle_ops;
use deers::models::gpt;
use deers::nn::{ParamStore, Parameter};
use deers::{Device, Tensor};

const TOL: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < TOL, "{label}[{index}]: got {a}, expected {e}");
    }
}

fn det_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.05 - 0.5).collect()
}

fn candle_tensor(data: Vec<f32>, shape: &[usize]) -> CTensor {
    CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap()
}

fn candle_rms_norm(x: &CTensor, eps: f64) -> CTensor {
    let mean_sq = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
    let inv_norm = (mean_sq + eps).unwrap().powf(-0.5).unwrap();
    x.broadcast_mul(&inv_norm).unwrap()
}

fn candle_rotate(x: &CTensor, cos: &CTensor, sin: &CTensor) -> CTensor {
    let head_dim = x.dims4().unwrap().3;
    let half_dim = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim).unwrap();
    let x2 = x.narrow(D::Minus1, half_dim, half_dim).unwrap();
    let y1 = x1.broadcast_mul(cos).unwrap().broadcast_sub(&x2.broadcast_mul(sin).unwrap()).unwrap();
    let y2 = x1
        .broadcast_mul(sin)
        .unwrap()
        .broadcast_add(&x2.broadcast_mul(cos).unwrap())
        .unwrap();
    CTensor::cat(&[&y1, &y2], D::Minus1).unwrap()
}

fn candle_rope_cache(seq_len: usize, head_dim: usize, base: f32) -> (CTensor, CTensor) {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> =
        (0..half_dim).map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32)).collect();
    let freqs: Vec<f32> =
        (0..seq_len).flat_map(|t| inv_freq.iter().map(move |&freq| t as f32 * freq)).collect();
    let shape = [1, seq_len, 1, half_dim];
    let cos = candle_tensor(freqs.iter().map(|&x| x.cos()).collect(), &shape);
    let sin = candle_tensor(freqs.iter().map(|&x| x.sin()).collect(), &shape);
    (cos, sin)
}

fn candle_causal_mask(seq_len: usize) -> CTensor {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    candle_tensor(mask, &[1, 1, seq_len, seq_len])
}

/// Independent candle mirror of `Qwen3Block::forward`: affine RMSNorms, GQA
/// attention with QK-Norm and RoPE, causal mask, and SwiGLU residuals.
#[allow(clippy::too_many_arguments)]
fn candle_qwen3_block(
    x: &CTensor,
    weights: &BTreeMap<String, CTensor>,
    cos: &CTensor,
    sin: &CTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> CTensor {
    let (batch_size, seq_len, channels) = x.dims3().unwrap();
    let group_size = n_q_heads / n_kv_heads;
    let get = |name: &str| weights[name].clone();

    let normed =
        candle_rms_norm(x, 1e-6).broadcast_mul(&get("input_layernorm.weight")).unwrap();
    let flat = normed.reshape((batch_size * seq_len, channels)).unwrap();
    let project = |w: CTensor, heads: usize| {
        flat.matmul(&w).unwrap().reshape((batch_size, seq_len, heads, head_dim)).unwrap()
    };
    let q = candle_rotate(
        &candle_rms_norm(&project(get("attn.q_proj.weight"), n_q_heads), 1e-6)
            .broadcast_mul(&get("attn.q_norm.weight"))
            .unwrap(),
        cos,
        sin,
    )
    .transpose(1, 2)
    .unwrap()
    .contiguous()
    .unwrap();
    let k = candle_rotate(
        &candle_rms_norm(&project(get("attn.k_proj.weight"), n_kv_heads), 1e-6)
            .broadcast_mul(&get("attn.k_norm.weight"))
            .unwrap(),
        cos,
        sin,
    );
    let repeat_heads = |t: CTensor| {
        let mut heads = Vec::new();
        for kv in 0..n_kv_heads {
            let head = t.narrow(2, kv, 1).unwrap();
            for _ in 0..group_size {
                heads.push(head.clone());
            }
        }
        CTensor::cat(&heads.iter().collect::<Vec<_>>(), 2)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let k = repeat_heads(k);
    let v = repeat_heads(project(get("attn.v_proj.weight"), n_kv_heads));

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * scale).unwrap();
    let scores = scores.broadcast_add(&candle_causal_mask(seq_len)).unwrap();
    let attn = candle_ops::softmax(&scores, D::Minus1).unwrap();
    let attended = attn
        .matmul(&v)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((batch_size, seq_len, n_q_heads * head_dim))
        .unwrap()
        .reshape((batch_size * seq_len, n_q_heads * head_dim))
        .unwrap()
        .matmul(&get("attn.out_proj.weight"))
        .unwrap()
        .reshape((batch_size, seq_len, channels))
        .unwrap();
    let h = x.broadcast_add(&attended).unwrap();

    let normed2 =
        candle_rms_norm(&h, 1e-6).broadcast_mul(&get("post_attention_layernorm.weight")).unwrap();
    let flat2 = normed2.reshape((batch_size * seq_len, channels)).unwrap();
    let gate = flat2.matmul(&get("mlp.gate_proj.weight")).unwrap();
    let silu = gate.broadcast_mul(&candle_ops::sigmoid(&gate).unwrap()).unwrap();
    let up = flat2.matmul(&get("mlp.up_proj.weight")).unwrap();
    let mlp = silu
        .broadcast_mul(&up)
        .unwrap()
        .matmul(&get("mlp.down_proj.weight"))
        .unwrap()
        .reshape((batch_size, seq_len, channels))
        .unwrap();
    h.broadcast_add(&mlp).unwrap()
}

fn named_candle_weights(params: &[(String, Parameter)]) -> BTreeMap<String, CTensor> {
    params
        .iter()
        .map(|(name, parameter)| {
            let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
            (name.clone(), candle_tensor(parameter.to_vec::<f32>().unwrap(), &shape))
        })
        .collect()
}

fn small_block() -> (gpt::Qwen3Block, Vec<(String, Parameter)>) {
    let store = ParamStore::new();
    let block = gpt::Qwen3Block::new(store.root(), 8, 4, 2, 4, 16, 1e-6);
    let named = store.named_parameters();
    (block, named)
}

#[test]
fn qwen3_block_matches_candle_reference() {
    // Arrange: grouped heads (4 query over 2 key/value) at small dims.
    let (block, named) = small_block();
    let (batch_size, seq_len) = (2, 3);
    let values = det_vec(batch_size * seq_len * 8);
    let x = Tensor::from_vec(values.clone(), (batch_size, seq_len, 8), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(3, 4, 10_000.0, deers::DType::F32, Device::Cpu);
    let (ccos, csin) = candle_rope_cache(3, 4, 10_000.0);
    let weights = named_candle_weights(&named);
    let expected = candle_qwen3_block(&candle_tensor(values, &[2, 3, 8]), &weights, &ccos, &csin, 4, 2, 4)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let actual = block.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_close(&actual, &expected, "qwen3 block forward");
}

#[test]
fn qwen3_block_full_dims_shape_and_param_names() {
    // Arrange: Qwen3-0.6B block dims with a tiny two-token input.
    let store = ParamStore::new();
    let block = gpt::Qwen3Block::new(store.root(), 1024, 16, 8, 128, 3072, 1e-6);
    let x = Tensor::from_vec(det_vec(2 * 1024), (1, 2, 1024), Device::Cpu);
    let (cos, sin) =
        gpt::precompute_rotary_embeddings(2, 128, 1_000_000.0, deers::DType::F32, Device::Cpu);

    // Act
    let out = block.forward(&x, &cos, &sin).unwrap();
    let names: Vec<String> =
        store.named_parameters().into_iter().map(|(name, _)| name).collect();
    let shapes: BTreeMap<String, Vec<usize>> = store
        .named_parameters()
        .into_iter()
        .map(|(name, parameter)| {
            (name, parameter.layout().shape().iter().copied().collect())
        })
        .collect();

    // Assert: the residual width holds while the query width decouples to 16 * 128.
    assert_eq!(out.layout().shape().iter().copied().collect::<Vec<_>>(), vec![1, 2, 1024]);
    assert!(out.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert_eq!(
        names,
        vec![
            "attn.k_norm.weight",
            "attn.k_proj.weight",
            "attn.out_proj.weight",
            "attn.q_norm.weight",
            "attn.q_proj.weight",
            "attn.v_proj.weight",
            "input_layernorm.weight",
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "post_attention_layernorm.weight",
        ]
    );
    assert_eq!(shapes["attn.q_proj.weight"], vec![1024, 2048]);
    assert_eq!(shapes["attn.k_proj.weight"], vec![1024, 1024]);
    assert_eq!(shapes["attn.out_proj.weight"], vec![2048, 1024]);
    assert_eq!(shapes["mlp.gate_proj.weight"], vec![1024, 3072]);
}

#[test]
fn qwen3_block_residual_carry_through() {
    // Arrange: zero every sublayer projection so both residuals carry the input.
    let (block, named) = small_block();
    let params: BTreeMap<String, Parameter> = named.into_iter().collect();
    for name in [
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.out_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        let shape: Vec<usize> = params[name].layout().shape().iter().copied().collect();
        let zeros = Tensor::from_vec(vec![0.0f32; shape.iter().product()], shape, Device::Cpu);
        params[name].set(&zeros).unwrap();
    }
    let values = det_vec(2 * 8);
    let x = Tensor::from_vec(values.clone(), (1, 2, 8), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(2, 4, 10_000.0, deers::DType::F32, Device::Cpu);

    // Act
    let actual = block.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(actual, values);
}

#[test]
fn qwen3_block_backward_reaches_all_parameters() {
    // Arrange
    let (block, named) = small_block();
    let x = Tensor::from_vec(det_vec(2 * 8), (1, 2, 8), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(2, 4, 10_000.0, deers::DType::F32, Device::Cpu);

    // Act
    let loss = block.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let grads = loss.backward().unwrap();
    let norms: Vec<(String, f32)> = named
        .iter()
        .map(|(name, parameter)| {
            let norm = grads
                .get(parameter.id())
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();
            (name.clone(), norm)
        })
        .collect();

    // Assert
    assert_eq!(norms.len(), 11);
    for (name, norm) in &norms {
        assert!(norm.is_finite() && *norm > 0.0, "{name} has no gradient");
    }
}

#[test]
fn qwen3_block_plain_head_count_runs() {
    // Arrange: query heads equal key/value heads, so no repeat applies.
    let store = ParamStore::new();
    let block = gpt::Qwen3Block::new(store.root(), 8, 4, 4, 4, 8, 1e-6);
    let x = Tensor::from_vec(det_vec(2 * 8), (1, 2, 8), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(2, 4, 10_000.0, deers::DType::F32, Device::Cpu);

    // Act
    let out = block.forward(&x, &cos, &sin).unwrap();
    let values = out.to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(out.layout().shape().iter().copied().collect::<Vec<_>>(), vec![1, 2, 8]);
    assert!(values.iter().all(|v| v.is_finite()));
}
