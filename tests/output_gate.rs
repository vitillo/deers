use std::collections::BTreeMap;

use candle_core::{D, Device as CDevice, Tensor as CTensor};
use candle_nn::ops as candle_ops;
use deers::models::gpt::{CausalSelfAttention, KvCache, precompute_rotary_embeddings};
use deers::nn::{ParamStore, Parameter};
use deers::{DType, Device, Tensor};

fn devices() -> Vec<Device> {
    [Device::Cpu, Device::Cuda, Device::Mps]
        .into_iter()
        .filter(|device| device.is_available())
        .collect()
}

fn det_vec(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 13) as f32 * 0.05 - 0.3).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{label}[{index}]: got {a}, expected {e}");
    }
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
    let y2 = x1.broadcast_mul(sin).unwrap().broadcast_add(&x2.broadcast_mul(cos).unwrap()).unwrap();
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

/// Independent candle mirror of gated `CausalSelfAttention::forward`, following
/// the Hugging Face Qwen3.5 full-attention formula: the doubled query
/// projection chunks per head into queries and a gate, and `sigmoid(gate)`
/// scales the attended output before the output projection.
#[allow(clippy::too_many_arguments)]
fn candle_gated_attention(
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

    let flat = x.reshape((batch_size * seq_len, channels)).unwrap();
    let qg = flat
        .matmul(&get("q_proj.weight"))
        .unwrap()
        .reshape((batch_size, seq_len, n_q_heads, 2 * head_dim))
        .unwrap();
    let q = qg.narrow(D::Minus1, 0, head_dim).unwrap();
    let gate = qg
        .narrow(D::Minus1, head_dim, head_dim)
        .unwrap()
        .reshape((batch_size, seq_len, n_q_heads * head_dim))
        .unwrap();

    let q = candle_rms_norm(&q, 1e-6).broadcast_mul(&get("q_norm.weight")).unwrap();
    let q = candle_rotate(&q, cos, sin).transpose(1, 2).unwrap().contiguous().unwrap();
    let k = flat
        .matmul(&get("k_proj.weight"))
        .unwrap()
        .reshape((batch_size, seq_len, n_kv_heads, head_dim))
        .unwrap();
    let k = candle_rms_norm(&k, 1e-6).broadcast_mul(&get("k_norm.weight")).unwrap();
    let k = candle_rotate(&k, cos, sin);
    let v = flat
        .matmul(&get("v_proj.weight"))
        .unwrap()
        .reshape((batch_size, seq_len, n_kv_heads, head_dim))
        .unwrap();
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
    let v = repeat_heads(v);

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
        .unwrap();
    let gated = attended
        .broadcast_mul(&candle_ops::sigmoid(&gate).unwrap())
        .unwrap()
        .reshape((batch_size * seq_len, n_q_heads * head_dim))
        .unwrap()
        .matmul(&get("out_proj.weight"))
        .unwrap();
    gated.reshape((batch_size, seq_len, channels)).unwrap()
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

/// Builds output-gated attention with deterministic weights on every parameter.
fn gated(
    n_embd: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
) -> (CausalSelfAttention, Vec<(String, Parameter)>) {
    let store = ParamStore::new();
    let attn = CausalSelfAttention::new_gqa_with_head_dim_and_output_gate(
        store.root(),
        n_embd,
        n_q_heads,
        n_kv_heads,
        head_dim,
        true,
    );
    let named = store.named_parameters();
    let params: BTreeMap<String, Parameter> = named.iter().cloned().collect();
    let widths = [
        ("q_proj.weight", 2 * n_q_heads * head_dim),
        ("k_proj.weight", n_kv_heads * head_dim),
        ("v_proj.weight", n_kv_heads * head_dim),
    ];
    for (name, width) in widths {
        params[name]
            .set(&Tensor::from_vec(det_vec(n_embd * width), vec![n_embd, width], device))
            .unwrap();
    }
    params["out_proj.weight"]
        .set(&Tensor::from_vec(
            det_vec(n_q_heads * head_dim * n_embd),
            vec![n_q_heads * head_dim, n_embd],
            device,
        ))
        .unwrap();
    for name in ["q_norm.weight", "k_norm.weight"] {
        params[name].set(&Tensor::from_vec(det_vec(head_dim), vec![head_dim], device)).unwrap();
    }
    (attn, named)
}

/// Splits a doubled query-projection matrix into its per-head query and gate
/// halves. Each head owns a `[query | gate]` slice, so the halves interleave
/// across the width rather than splitting it down the middle.
fn split_qg(qg: &[f32], n_embd: usize, n_q_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut queries = vec![0.0; n_embd * n_q_heads * head_dim];
    let mut gates = vec![0.0; n_embd * n_q_heads * head_dim];
    for row in 0..n_embd {
        for head in 0..n_q_heads {
            for dim in 0..head_dim {
                queries[(row * n_q_heads + head) * head_dim + dim] =
                    qg[(row * 2 * n_q_heads + 2 * head) * head_dim + dim];
                gates[(row * n_q_heads + head) * head_dim + dim] =
                    qg[(row * 2 * n_q_heads + 2 * head + 1) * head_dim + dim];
            }
        }
    }
    (queries, gates)
}

fn rope(seq_len: usize, head_dim: usize, device: Device) -> (Tensor, Tensor) {
    precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device)
}

#[test]
fn output_gate_matches_candle_reference() {
    // Arrange: grouped heads (4 query over 2 key/value) with a decoupled width.
    let (n_embd, n_q_heads, n_kv_heads, head_dim) = (8, 4, 2, 4);
    let (batch_size, seq_len) = (2, 3);
    let device = Device::Cpu;
    let (attn, named) = gated(n_embd, n_q_heads, n_kv_heads, head_dim, device);
    let input = det_vec(batch_size * seq_len * n_embd);
    let x = Tensor::from_vec(input.clone(), (batch_size, seq_len, n_embd), device);
    let (cos, sin) = rope(seq_len, head_dim, device);
    let (ccos, csin) = candle_rope_cache(seq_len, head_dim, 10_000.0);
    let weights = named_candle_weights(&named);
    let expected = candle_gated_attention(
        &candle_tensor(input, &[batch_size, seq_len, n_embd]),
        &weights,
        &ccos,
        &csin,
        n_q_heads,
        n_kv_heads,
        head_dim,
    )
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();

    // Act
    let actual = attn.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_close(&actual, &expected, 1e-4, "gated attention forward");
}

#[test]
fn gated_q_proj_doubles_width() {
    // Arrange
    let (n_embd, n_q_heads, n_kv_heads, head_dim) = (8, 4, 2, 4);
    let store = ParamStore::new();
    let gated_attn = CausalSelfAttention::new_gqa_with_head_dim_and_output_gate(
        store.root(),
        n_embd,
        n_q_heads,
        n_kv_heads,
        head_dim,
        true,
    );
    let plain = CausalSelfAttention::new_gqa_with_head_dim(
        ParamStore::new().root(),
        n_embd,
        n_q_heads,
        n_kv_heads,
        head_dim,
    );

    // Act
    let shapes = |attn: &CausalSelfAttention| {
        attn.parameters()
            .iter()
            .map(|parameter| parameter.layout().shape().iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };

    // Assert: only the query projection doubles; every other shape holds.
    assert_eq!(
        shapes(&gated_attn),
        vec![vec![8, 32], vec![8, 8], vec![8, 8], vec![16, 8], vec![4], vec![4],]
    );
    assert_eq!(shapes(&plain)[0], vec![8, 16]);
    assert_eq!(shapes(&gated_attn)[1..], shapes(&plain)[1..]);
}

#[test]
fn zero_gate_halves_ungated_output() {
    // Arrange: a gated module whose gate half is all zeros, next to an
    // ungated module carrying the same query/key/value/output weights.
    let (n_embd, n_q_heads, n_kv_heads, head_dim) = (8, 4, 2, 2);
    for device in devices() {
        let (gated_attn, named) = gated(n_embd, n_q_heads, n_kv_heads, head_dim, device);
        let params: BTreeMap<String, Parameter> = named.into_iter().collect();
        let qg = params["q_proj.weight"].to_vec::<f32>().unwrap();
        let (queries, _) = split_qg(&qg, n_embd, n_q_heads, head_dim);
        let plain = CausalSelfAttention::new_gqa_with_head_dim(
            ParamStore::new().root(),
            n_embd,
            n_q_heads,
            n_kv_heads,
            head_dim,
        );
        let plain_params = plain.parameters();
        plain_params[0]
            .set(&Tensor::from_vec(queries, vec![n_embd, n_q_heads * head_dim], device))
            .unwrap();
        for (gated_name, plain_param) in [
            ("k_proj.weight", &plain_params[1]),
            ("v_proj.weight", &plain_params[2]),
            ("out_proj.weight", &plain_params[3]),
        ] {
            plain_param.set(&params[gated_name].detach()).unwrap();
        }
        for (gated_name, plain_param) in
            [("q_norm.weight", &plain_params[4]), ("k_norm.weight", &plain_params[5])]
        {
            plain_param.set(&params[gated_name].detach()).unwrap();
        }
        // Zero the gate half in place: sigmoid(0) is exactly one half.
        let mut gated_qg = qg;
        for row in 0..n_embd {
            for head in 0..n_q_heads {
                for dim in 0..head_dim {
                    gated_qg[(row * 2 * n_q_heads + 2 * head + 1) * head_dim + dim] = 0.0;
                }
            }
        }
        params["q_proj.weight"]
            .set(&Tensor::from_vec(gated_qg, vec![n_embd, 2 * n_q_heads * head_dim], device))
            .unwrap();
        let x = Tensor::from_vec(det_vec(2 * n_embd), (1, 2, n_embd), device);
        let (cos, sin) = rope(2, head_dim, device);

        // Act
        let gated_out = gated_attn.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();
        let plain_out = plain.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();

        // Assert: a zero gate scales every output channel by exactly one half.
        let expected: Vec<f32> = plain_out.iter().map(|v| v * 0.5).collect();
        assert_close(&gated_out, &expected, 1e-6, "zero gate halves output");
        if device == Device::Cpu {
            assert_close(
                &gated_out[..8],
                &[
                    0.009437499,
                    0.019562502,
                    -0.0028125025,
                    0.007312501,
                    0.0011874977,
                    -0.021187503,
                    -0.011062501,
                    -0.017187505,
                ],
                1e-6,
                "zero gate pinned output",
            );
        }
    }
}

#[test]
fn cached_decode_matches_full_forward_with_gate() {
    // Arrange
    let (batch, seq, prompt_len, n_embd, n_q_heads, n_kv_heads, head_dim) = (1, 5, 3, 8, 4, 2, 2);
    for device in devices() {
        let (attn, _) = gated(n_embd, n_q_heads, n_kv_heads, head_dim, device);
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);
        let mut cache = KvCache::new();

        // Act: prefill the prompt, decode the rest, stitch back into order.
        let full = attn.forward(&x, &cos, &sin).unwrap().to_vec::<f32>().unwrap();
        let prompt = x.narrow(1, 0, prompt_len);
        let mut pieces = vec![
            attn.prefill(
                &prompt,
                &cos.narrow(1, 0, prompt_len),
                &sin.narrow(1, 0, prompt_len),
                &mut cache,
            )
            .unwrap(),
        ];
        for pos in prompt_len..seq {
            pieces.push(
                attn.decode(
                    &x.narrow(1, pos, 1),
                    &cos.narrow(1, pos, 1),
                    &sin.narrow(1, pos, 1),
                    &mut cache,
                )
                .unwrap(),
            );
        }
        let stitched = Tensor::cat(&pieces, 1).to_vec::<f32>().unwrap();

        // Assert: the gate reads the current token only, so caching stays exact.
        assert_close(&stitched, &full, 1e-5, "gated prefill/decode parity");
        assert_eq!(cache.len(), seq);
    }
}

#[test]
fn gradients_reach_gate_half() {
    // Arrange
    let (n_embd, n_q_heads, n_kv_heads, head_dim) = (8, 4, 2, 2);
    let device = Device::Cpu;
    let (attn, named) = gated(n_embd, n_q_heads, n_kv_heads, head_dim, device);
    let x = Tensor::from_vec(det_vec(2 * n_embd), (1, 2, n_embd), device);
    let (cos, sin) = rope(2, head_dim, device);

    // Act
    let loss = attn.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let grads = loss.backward().unwrap();

    // Assert: every parameter learns, and the gate half of the doubled
    // query projection learns separately from the query half.
    for (name, parameter) in &named {
        let grad = grads.get(parameter.id()).unwrap().to_vec::<f32>().unwrap();
        assert!(
            grad.iter().all(|g| g.is_finite()) && grad.iter().any(|&g| g != 0.0),
            "{name} has no gradient"
        );
    }
    let q_proj_grad =
        named.iter().find(|(name, _)| name == "q_proj.weight").expect("q_proj.weight").1.id();
    let qg_grad = grads.get(q_proj_grad).unwrap().to_vec::<f32>().unwrap();
    let (query_grad, gate_grad) = split_qg(&qg_grad, n_embd, n_q_heads, head_dim);
    assert!(query_grad.iter().any(|&g| g != 0.0), "query half has no gradient");
    assert!(gate_grad.iter().any(|&g| g != 0.0), "gate half has no gradient");
    assert_ne!(query_grad, gate_grad, "gate half shadows the query half");
}
