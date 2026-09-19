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

/// Builds grouped-query attention with deterministic projection weights.
fn gqa(n_embd: usize, n_head: usize, n_kv_head: usize, device: Device) -> CausalSelfAttention {
    let head_dim = n_embd / n_head;
    let attn = CausalSelfAttention::new_gqa(ParamStore::new().root(), n_embd, n_head, n_kv_head);
    let widths = [n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim, n_embd];
    for (param, width) in attn.parameters().iter().zip(widths) {
        let data = det_vec(n_embd * width);
        param.set(&Tensor::from_vec(data, vec![n_embd, width], device)).unwrap();
    }
    attn
}

/// Builds plain MHA whose key/value heads repeat the grouped module's heads.
fn naive_mha_from(
    grouped: &CausalSelfAttention,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
) -> CausalSelfAttention {
    let head_dim = n_embd / n_head;
    let group = n_head / n_kv_head;
    let naive = CausalSelfAttention::new(ParamStore::new().root(), n_embd, n_head);
    let src = grouped.parameters();
    let dst = naive.parameters();
    dst[0].set(&src[0]).unwrap();
    for (src_param, dst_param) in [(&src[1], &dst[1]), (&src[2], &dst[2])] {
        let narrow: Vec<Tensor> = (0..n_head)
            .map(|head| {
                src_param.reshape(vec![n_embd, n_kv_head, head_dim]).narrow(1, head / group, 1)
            })
            .collect();
        let tiled = Tensor::cat(&narrow, 1).reshape(vec![n_embd, n_head * head_dim]);
        dst_param.set(&tiled).unwrap();
    }
    dst[3].set(&src[3]).unwrap();
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
    let (batch, seq, n_embd, n_head, n_kv_head) = (1, 3, 8, 4, 2);
    let head_dim = n_embd / n_head;
    for device in devices() {
        let grouped = gqa(n_embd, n_head, n_kv_head, device);
        let naive = naive_mha_from(&grouped, n_embd, n_head, n_kv_head);
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);

        // Act
        let grouped_out = values(&grouped.forward(&x, &cos, &sin).unwrap());
        let naive_out = values(&naive.forward(&x, &cos, &sin).unwrap());

        // Assert
        assert_eq!(grouped_out, naive_out);
        assert_eq!(
            grouped_out[..8].to_vec(),
            vec![
                0.018874997,
                0.039125003,
                -0.005625005,
                0.014625002,
                0.0023749953,
                -0.042375006,
                -0.022125002,
                -0.03437501
            ]
        );
    }
}

#[test]
fn mha_constructor_matches_gqa_with_equal_heads() {
    // Arrange
    let (batch, seq, n_embd, n_head) = (1, 2, 4, 2);
    let head_dim = n_embd / n_head;
    for device in devices() {
        let plain = CausalSelfAttention::new(ParamStore::new().root(), n_embd, n_head);
        let grouped =
            CausalSelfAttention::new_gqa(ParamStore::new().root(), n_embd, n_head, n_head);
        for (plain_param, grouped_param) in
            plain.parameters().iter().zip(grouped.parameters().iter())
        {
            let width = plain_param.layout().shape()[1];
            let data = det_vec(n_embd * width);
            let tensor = Tensor::from_vec(data, vec![n_embd, width], device);
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
        assert_eq!(
            plain_out,
            vec![
                -0.01575,
                -0.0127500035,
                -0.0016250028,
                0.009499998,
                -0.015242673,
                -0.012746319,
                -0.0053746966,
                0.0019969258
            ]
        );
    }
}

#[test]
fn gradients_reach_every_projection() {
    // Arrange
    let (batch, seq, n_embd, n_head, n_kv_head) = (1, 3, 8, 4, 2);
    let head_dim = n_embd / n_head;
    let group = n_head / n_kv_head;
    let device = Device::Cpu;
    let grouped = gqa(n_embd, n_head, n_kv_head, device);
    let naive = naive_mha_from(&grouped, n_embd, n_head, n_kv_head);
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
    for kv in 0..n_kv_head {
        for row in 0..n_embd {
            for dim in 0..head_dim {
                let shared = grouped_k[(row * n_kv_head + kv) * head_dim + dim];
                let mut summed = 0.0;
                for rep in 0..group {
                    let head = kv * group + rep;
                    summed += naive_k[(row * n_head + head) * head_dim + dim];
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
        &[-4.863824e-5, -8.130807e-6, -0.00046206897, -0.00021508266]
    );
}

#[test]
fn shared_rope_heads_feed_each_query_identically() {
    // Arrange
    let (batch, seq, n_embd, n_head, n_kv_head) = (1, 2, 8, 4, 2);
    let head_dim = n_embd / n_head;
    let group = n_head / n_kv_head;
    let device = Device::Cpu;
    let grouped = gqa(n_embd, n_head, n_kv_head, device);
    let params = grouped.parameters();
    let head0 = params[0].reshape(vec![n_embd, n_head, head_dim]).narrow(1, 0, 1);
    let tiled =
        Tensor::cat(&vec![head0.clone(); n_head], 1).reshape(vec![n_embd, n_head * head_dim]);
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
    let out = grouped.forward(&x, &cos, &sin).unwrap().reshape(vec![seq, n_head, head_dim]);
    let out = values(&out);

    // Assert
    for pos in 0..seq {
        for rep in 1..group {
            for dim in 0..head_dim {
                for kv in 0..n_kv_head {
                    let first = out[(pos * n_head + kv * group) * head_dim + dim];
                    let other = out[(pos * n_head + kv * group + rep) * head_dim + dim];
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
            0.010342829,
            -0.040110618,
            0.010342829,
            -0.040110618,
            0.021309288,
            0.0024948437,
            0.021309288,
            0.0024948437,
        ]
    );
}
