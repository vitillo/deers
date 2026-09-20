use deers::models::gpt::{CausalSelfAttention, precompute_rotary_embeddings};
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

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

const CPU_TOL: f32 = 1e-4;
const MPS_TOL: f32 = 2e-3;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "[{index}]: got {a}, expected {e}");
    }
}

/// Accelerator kernels reorder floating-point summation, so cross-backend
/// goldens compare within tolerance: exact on CPU, looser on accelerators.
fn tol_for(device: Device) -> f32 {
    match device {
        Device::Cpu => CPU_TOL,
        _ => MPS_TOL,
    }
}

/// Builds grouped-query attention with deterministic weights on every parameter.
fn gqa(n_embd: usize, n_q_heads: usize, n_kv_heads: usize, device: Device) -> CausalSelfAttention {
    let head_dim = n_embd / n_q_heads;
    let attn =
        CausalSelfAttention::new_gqa(ParamStore::new().root(), n_embd, n_q_heads, n_kv_heads);
    let params = attn.parameters();
    let widths = [n_q_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim, n_embd];
    for (param, width) in params[..4].iter().zip(widths) {
        let data = det_vec(n_embd * width);
        param.set(&Tensor::from_vec(data, vec![n_embd, width], device)).unwrap();
    }
    for param in &params[4..] {
        param.set(&Tensor::from_vec(det_vec(head_dim), vec![head_dim], device)).unwrap();
    }
    attn
}

/// Builds plain MHA whose key/value heads repeat the grouped module's heads.
fn naive_mha_from(
    grouped: &CausalSelfAttention,
    n_embd: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
) -> CausalSelfAttention {
    let head_dim = n_embd / n_q_heads;
    let group_size = n_q_heads / n_kv_heads;
    let naive = CausalSelfAttention::new(ParamStore::new().root(), n_embd, n_q_heads);
    let src = grouped.parameters();
    let dst = naive.parameters();
    dst[0].set(&src[0]).unwrap();
    for (src_param, dst_param) in [(&src[1], &dst[1]), (&src[2], &dst[2])] {
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
    dst[3].set(&src[3]).unwrap();
    dst[4].set(&src[4]).unwrap();
    dst[5].set(&src[5]).unwrap();
    naive
}

fn rope(seq_len: usize, head_dim: usize, device: Device) -> (Tensor, Tensor) {
    precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device)
}

fn grad_of(grads: &deers::GradientStore, param: &Parameter) -> Vec<f32> {
    values(&grads.get(param.id()).unwrap())
}

#[test]
fn gqa_matches_naive_repeated_heads() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads, n_kv_heads) = (1, 3, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        let grouped = gqa(n_embd, n_q_heads, n_kv_heads, device);
        let naive = naive_mha_from(&grouped, n_embd, n_q_heads, n_kv_heads);
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);

        // Act
        let grouped_out = values(&grouped.forward(&x, &cos, &sin).unwrap());
        let naive_out = values(&naive.forward(&x, &cos, &sin).unwrap());

        // Assert
        assert_eq!(grouped_out, naive_out);
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
                -0.03437501,
            ],
            tol_for(device),
        );
    }
}

#[test]
fn mha_constructor_matches_gqa_with_equal_heads() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads) = (1, 2, 4, 2);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        let plain = CausalSelfAttention::new(ParamStore::new().root(), n_embd, n_q_heads);
        let grouped =
            CausalSelfAttention::new_gqa(ParamStore::new().root(), n_embd, n_q_heads, n_q_heads);
        for (plain_param, grouped_param) in
            plain.parameters().iter().zip(grouped.parameters().iter())
        {
            let shape: Vec<usize> = plain_param.layout().shape().iter().copied().collect();
            let data = det_vec(shape.iter().product());
            let tensor = Tensor::from_vec(data, shape, device);
            plain_param.set(&tensor).unwrap();
            grouped_param.set(&tensor).unwrap();
        }
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);

        // Act
        let plain_out = values(&plain.forward(&x, &cos, &sin).unwrap());
        let grouped_out = values(&grouped.forward(&x, &cos, &sin).unwrap());

        // Assert
        assert_eq!(grouped_out, plain_out);
        assert_close(
            &plain_out,
            &[
                -0.01575,
                -0.0127500035,
                -0.0016250028,
                0.009499998,
                -0.015338032,
                -0.012801903,
                -0.0054932944,
                0.0018153149,
            ],
            tol_for(device),
        );
    }
}

#[test]
fn gradients_reach_every_projection() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads, n_kv_heads) = (1, 3, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let group_size = n_q_heads / n_kv_heads;
    let device = Device::Cpu;
    let grouped = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let naive = naive_mha_from(&grouped, n_embd, n_q_heads, n_kv_heads);
    let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
    let (cos, sin) = rope(seq, head_dim, device);

    // Act
    let grouped_loss = grouped.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let naive_loss = naive.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let grouped_grads = grouped_loss.backward().unwrap();
    let naive_grads = naive_loss.backward().unwrap();

    // Assert
    let grouped_params = grouped.parameters();
    let naive_params = naive.parameters();
    for (index, param) in grouped_params.iter().enumerate() {
        let grad = grad_of(&grouped_grads, param);
        assert!(grad.iter().any(|&entry| entry != 0.0), "projection {index} got no gradient");
    }
    let grouped_k = grad_of(&grouped_grads, &grouped_params[1]);
    let naive_k = grad_of(&naive_grads, &naive_params[1]);
    for kv in 0..n_kv_heads {
        for row in 0..n_embd {
            for dim in 0..head_dim {
                let shared = grouped_k[(row * n_kv_heads + kv) * head_dim + dim];
                let mut summed = 0.0;
                for rep in 0..group_size {
                    let head = kv * group_size + rep;
                    summed += naive_k[(row * n_q_heads + head) * head_dim + dim];
                }
                assert!(
                    (shared - summed).abs() < 1e-4,
                    "kv head {kv} grad differs: {shared} vs {summed}"
                );
            }
        }
    }
    assert_eq!(
        &grad_of(&grouped_grads, &grouped_params[0])[..4],
        &[0.0012603741, -0.00081668067, 0.005150421, -0.003959463]
    );
}

#[test]
fn shared_rope_heads_feed_each_query_identically() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads, n_kv_heads) = (1, 2, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let group_size = n_q_heads / n_kv_heads;
    let device = Device::Cpu;
    let grouped = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let params = grouped.parameters();
    let head0 = params[0].reshape(vec![n_embd, n_q_heads, head_dim]).narrow(1, 0, 1);
    let tiled =
        Tensor::cat(&vec![head0.clone(); n_q_heads], 1).reshape(vec![n_embd, n_q_heads * head_dim]);
    params[0].set(&tiled).unwrap();
    // Identity output projection so each output channel exposes one attention head directly.
    let mut identity = vec![0.0; n_embd * n_embd];
    for index in 0..n_embd {
        identity[index * n_embd + index] = 1.0;
    }
    params[3].set(&Tensor::from_vec(identity, vec![n_embd, n_embd], device)).unwrap();
    let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
    let (cos, sin) = rope(seq, head_dim, device);

    // Act
    let out = grouped.forward(&x, &cos, &sin).unwrap().reshape(vec![seq, n_q_heads, head_dim]);
    let out = values(&out);

    // Assert
    for pos in 0..seq {
        for rep in 1..group_size {
            for dim in 0..head_dim {
                for kv in 0..n_kv_heads {
                    let first = out[(pos * n_q_heads + kv * group_size) * head_dim + dim];
                    let other = out[(pos * n_q_heads + kv * group_size + rep) * head_dim + dim];
                    assert_eq!(first, other);
                }
            }
        }
    }
    assert_eq!(
        out,
        vec![
            0.05250001,
            0.10000001,
            0.05250001,
            0.10000001,
            0.05000001,
            4.200265e-9,
            0.05000001,
            4.200265e-9,
            0.013253556,
            -0.030436728,
            0.013253556,
            -0.030436728,
            0.02136946,
            0.0024896108,
            0.02136946,
            0.0024896108,
        ]
    );
}
