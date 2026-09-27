use deers::models::gpt::{CausalSelfAttention, apply_rotary_emb, precompute_rotary_embeddings};
use deers::nn::{ParamStore, Parameter, functional};
use deers::{DType, Device, Tensor};

const QK_EPS: f64 = 1e-6;

fn devices() -> Vec<Device> {
    [Device::Cpu, Device::Cuda, Device::Mps]
        .into_iter()
        .filter(|device| device.is_available())
        .collect()
}

fn det_vec(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 13) as f32 * 0.05 - 0.3).collect()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "[{index}]: got {a}, expected {e}");
    }
}

/// RMS normalization over the last axis with an explicit scale, mirroring
/// `RMSNorm` without calling it.
fn manual_rms_norm(x: &Tensor, weight: &Tensor) -> Tensor {
    let last = x.layout().ndim() - 1;
    let mean_sq = (x * x).mean(vec![last], true);
    let inv_norm = (mean_sq + QK_EPS).scalar_powf(-0.5);
    let normed = x * &inv_norm;
    &normed * weight
}

/// Plain multi-head attention forward with hand-rolled QK-Norm.
fn manual_mha_forward(
    x: &Tensor,
    q_w: &Tensor,
    k_w: &Tensor,
    v_w: &Tensor,
    out_w: &Tensor,
    qn_w: &Tensor,
    kn_w: &Tensor,
    n_q_heads: usize,
    cos: &Tensor,
    sin: &Tensor,
) -> Tensor {
    let shape = x.layout().shape().clone();
    let batch = shape[0];
    let seq = shape[1];
    let channels = shape[2];
    let head_dim = channels / n_q_heads;

    let x_flat = x.reshape(vec![batch * seq, channels]);
    let q = x_flat.matmul(q_w).reshape(vec![batch, seq, n_q_heads, head_dim]);
    let q = manual_rms_norm(&q, qn_w);
    let k = x_flat.matmul(k_w).reshape(vec![batch, seq, n_q_heads, head_dim]);
    let k = manual_rms_norm(&k, kn_w);
    let v = x_flat.matmul(v_w).reshape(vec![batch, seq, n_q_heads, head_dim]);

    let q = apply_rotary_emb(&q, cos, sin).rearrange("b t h d -> b h t d", &[]);
    let k = apply_rotary_emb(&k, cos, sin).rearrange("b t h d -> b h t d", &[]);
    let v = v.rearrange("b t h d -> b h t d", &[]);

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(Some((2, 3)))) * scale;
    let mask = functional::causal_mask(batch, seq, 0, x.dtype(), x.device());
    let attn = (&scores + &mask).softmax(3);
    let y_flat = attn.matmul(&v).rearrange("b h t d -> (b t) (h d)", &[]);

    y_flat.matmul(out_w).rearrange("(b t) c -> b t c", &[("b", batch), ("t", seq)])
}

fn rope(seq_len: usize, head_dim: usize, device: Device) -> (Tensor, Tensor) {
    precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device)
}

fn grad_of(grads: &deers::GradientStore, param: &Parameter) -> Vec<f32> {
    values(&grads.get(param.id()).unwrap())
}

#[test]
fn qk_norm_matches_manual_reference() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads) = (1, 3, 8, 4);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        let store = ParamStore::new();
        let attn = CausalSelfAttention::new(store.root(), n_embd, n_q_heads);
        let q_w = Tensor::from_vec(det_vec(n_embd * n_embd), vec![n_embd, n_embd], device);
        let k_w = Tensor::from_vec(det_vec(n_embd * n_embd), vec![n_embd, n_embd], device);
        let v_w = Tensor::from_vec(det_vec(n_embd * n_embd), vec![n_embd, n_embd], device);
        let out_w = Tensor::from_vec(det_vec(n_embd * n_embd), vec![n_embd, n_embd], device);
        let qn_w = Tensor::from_vec(vec![1.5f32, 0.5], vec![head_dim], device);
        let kn_w = Tensor::from_vec(vec![2.0f32, 0.25], vec![head_dim], device);
        let params = attn.parameters();
        for (param, weight) in params.iter().zip([&q_w, &k_w, &v_w, &out_w, &qn_w, &kn_w]) {
            param.set(weight).unwrap();
        }
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);

        // Act
        let actual = values(&attn.forward(&x, &cos, &sin).unwrap());
        let expected =
            values(&manual_mha_forward(&x, &q_w, &k_w, &v_w, &out_w, &qn_w, &kn_w, n_q_heads, &cos, &sin));

        // Assert
        assert_close(&actual, &expected, 1e-4);
        assert_close(
            &actual[..8],
            &[
                -0.043250006,
                -0.030375006,
                -0.025625005,
                -0.012750003,
                -0.016124997,
                -0.0129999975,
                -0.00012499644,
                0.037125003
            ],
            1e-4,
        );
    }
}

#[test]
fn grouped_qk_norm_matches_plain_reference() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads, n_kv_heads) = (1, 3, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let group_size = n_q_heads / n_kv_heads;
    for device in devices() {
        let grouped =
            CausalSelfAttention::new_gqa(ParamStore::new().root(), n_embd, n_q_heads, n_kv_heads);
        let grouped_params = grouped.parameters();
        let widths = [n_q_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim, n_embd];
        for (param, width) in grouped_params[..4].iter().zip(widths) {
            let data = det_vec(n_embd * width);
            param.set(&Tensor::from_vec(data, vec![n_embd, width], device)).unwrap();
        }
        grouped_params[4]
            .set(&Tensor::from_vec(vec![1.5f32, 0.5], vec![head_dim], device))
            .unwrap();
        grouped_params[5]
            .set(&Tensor::from_vec(vec![2.0f32, 0.25], vec![head_dim], device))
            .unwrap();
        let plain = CausalSelfAttention::new(ParamStore::new().root(), n_embd, n_q_heads);
        let plain_params = plain.parameters();
        plain_params[0].set(&grouped_params[0]).unwrap();
        for (src_param, dst_param) in
            [(&grouped_params[1], &plain_params[1]), (&grouped_params[2], &plain_params[2])]
        {
            let narrow: Vec<Tensor> = (0..n_q_heads)
                .map(|head| {
                    src_param.reshape(vec![n_embd, n_kv_heads, head_dim]).narrow(
                        1,
                        head / group_size,
                        1,
                    )
                })
                .collect();
            let tiled = Tensor::cat(&narrow, 1).reshape(vec![n_embd, n_q_heads * head_dim]);
            dst_param.set(&tiled).unwrap();
        }
        plain_params[3].set(&grouped_params[3]).unwrap();
        plain_params[4].set(&grouped_params[4]).unwrap();
        plain_params[5].set(&grouped_params[5]).unwrap();
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);

        // Act
        let grouped_out = values(&grouped.forward(&x, &cos, &sin).unwrap());
        let plain_out = values(&plain.forward(&x, &cos, &sin).unwrap());

        // Assert
        assert_close(&grouped_out, &plain_out, 1e-4);
        assert_close(
            &grouped_out[..8],
            &[
                0.018874997,
                0.039125003,
                -0.005625005,
                0.014625002,
                0.0023749953,
                -0.042375006,
                -0.022125002,
                -0.03437501
            ],
            1e-4,
        );
    }
}

#[test]
fn qk_scale_leaves_output_unchanged() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads) = (1, 3, 8, 4);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        let build = |scale: f32| {
            let attn = CausalSelfAttention::new(ParamStore::new().root(), n_embd, n_q_heads);
            for (index, param) in attn.parameters().iter().enumerate() {
                let shape: Vec<usize> = param.layout().shape().iter().copied().collect();
                let len: usize = shape.iter().product();
                let mut data = det_vec(len);
                if index < 2 {
                    for entry in &mut data {
                        *entry *= scale;
                    }
                }
                param.set(&Tensor::from_vec(data, shape, device)).unwrap();
            }
            attn
        };
        let baseline = build(1.0);
        let scaled = build(4.0);
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);

        // Act
        let baseline_out = values(&baseline.forward(&x, &cos, &sin).unwrap());
        let scaled_out = values(&scaled.forward(&x, &cos, &sin).unwrap());

        // Assert
        assert_close(&scaled_out, &baseline_out, 1e-4);
        assert_close(
            &baseline_out[..8],
            &[
                -0.043250006,
                -0.030375006,
                -0.025625005,
                -0.012750003,
                -0.016124997,
                -0.0129999975,
                -0.00012499644,
                0.037125003
            ],
            1e-4,
        );
    }
}

#[test]
fn gradients_reach_qk_norm_weights() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads) = (1, 3, 8, 4);
    let head_dim = n_embd / n_q_heads;
    let device = Device::Cpu;
    let store = ParamStore::new();
    let attn = CausalSelfAttention::new(store.root().pp("attn"), n_embd, n_q_heads);
    for param in attn.parameters() {
        let shape: Vec<usize> = param.layout().shape().iter().copied().collect();
        let len: usize = shape.iter().product();
        param.set(&Tensor::from_vec(det_vec(len), shape, device)).unwrap();
    }
    let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
    let (cos, sin) = rope(seq, head_dim, device);

    // Act
    let loss = attn.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let grads = loss.backward().unwrap();
    let named = store.named_parameters();
    let names: Vec<String> = named.iter().map(|(name, _)| name.clone()).collect();

    // Assert
    assert!(names.contains(&"attn.q_norm.weight".to_string()));
    assert!(names.contains(&"attn.k_norm.weight".to_string()));
    for (name, param) in &named {
        if name.ends_with("norm.weight") {
            let grad = grad_of(&grads, param);
            assert!(grad.iter().any(|&entry| entry != 0.0), "{name} got no gradient");
        }
    }
}
