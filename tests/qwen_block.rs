use deers::models::gpt::{apply_rotary_emb, precompute_rotary_embeddings};
use deers::models::qwen::{Qwen3Config, QwenBlock};
use deers::nn::{ParamStore, Parameter, functional};
use deers::{DType, Device, Tensor};

fn small_config(n_kv_heads: usize) -> Qwen3Config {
    Qwen3Config {
        hidden: 32,
        n_q_heads: 8,
        n_kv_heads,
        head_dim: 16,
        mlp_dim: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    }
}

fn det_vec(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 13) as f32 * 0.05 - 0.3).collect()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

/// Overwrites every block parameter with deterministic values.
fn deterministic(block: &QwenBlock, device: Device) {
    for param in block.parameters() {
        let shape: Vec<usize> = param.layout().shape().iter().copied().collect();
        let len = shape.iter().product();
        param.set(&Tensor::from_vec(det_vec(len), shape, device)).unwrap();
    }
}

fn rope(seq_len: usize, head_dim: usize, theta: f32, device: Device) -> (Tensor, Tensor) {
    precompute_rotary_embeddings(seq_len, head_dim, theta, DType::F32, device)
}

/// Affine RMS normalization from raw tensor ops, mirroring `RMSNorm`.
fn manual_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let last = x.layout().ndim() - 1;
    let mean_sq = (x * x).mean(vec![last], true);
    &(x * &(mean_sq + eps).scalar_powf(-0.5)) * weight
}

/// Full block forward wired by hand from tensor ops and the block's weights.
#[allow(clippy::too_many_arguments)]
fn manual_block_forward(
    x: &Tensor,
    weights: &[Tensor],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    cos: &Tensor,
    sin: &Tensor,
) -> Tensor {
    let shape = x.layout().shape().clone();
    let (batch, seq, hidden) = (shape[0], shape[1], shape[2]);
    let group_size = n_q_heads / n_kv_heads;

    // Attention sub-block over the input norm.
    let flat = manual_rms_norm(x, &weights[0], eps).reshape(vec![batch * seq, hidden]);
    let q = manual_rms_norm(
        &flat.matmul(&weights[1]).reshape(vec![batch, seq, n_q_heads, head_dim]),
        &weights[5],
        eps,
    );
    let k = manual_rms_norm(
        &flat.matmul(&weights[2]).reshape(vec![batch, seq, n_kv_heads, head_dim]),
        &weights[6],
        eps,
    );
    let v = flat.matmul(&weights[3]).reshape(vec![batch, seq, n_kv_heads, head_dim]);
    let q = apply_rotary_emb(&q, cos, sin).rearrange("b t h d -> b h t d", &[]);
    let k = apply_rotary_emb(&k, cos, sin);
    let k = if group_size == 1 {
        k
    } else {
        k.repeat("b t kv d -> b t (kv g) d", &[("g", group_size)])
    };
    let k = k.rearrange("b t h d -> b h t d", &[]);
    let v = if group_size == 1 {
        v
    } else {
        v.repeat("b t kv d -> b t (kv g) d", &[("g", group_size)])
    };
    let v = v.rearrange("b t h d -> b h t d", &[]);
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = Tensor::einsum("b h t d, b h s d -> b h t s", &q, &k) * scale;
    let mask = functional::causal_mask(batch, seq, 0, x.dtype(), x.device());
    let context = (&scores + &mask).softmax(3).matmul(&v).rearrange("b h t d -> (b t) (h d)", &[]);
    let attended =
        context.matmul(&weights[4]).rearrange("(b t) c -> b t c", &[("b", batch), ("t", seq)]);
    let h = x + &attended;

    // SwiGLU sub-block over the post-attention norm.
    let flat = manual_rms_norm(&h, &weights[7], eps).reshape(vec![batch * seq, hidden]);
    let gated = flat.matmul(&weights[8]).silu() * flat.matmul(&weights[9]);
    let fed = gated.matmul(&weights[10]).rearrange("(b t) c -> b t c", &[("b", batch), ("t", seq)]);
    &h + &fed
}

fn grad_of(grads: &deers::GradientStore, param: &Parameter) -> Vec<f32> {
    values(&grads.get(param.id()).unwrap())
}

#[test]
fn qwen_block_matches_reference_composition() {
    // Arrange
    let device = Device::Cpu;
    let config = small_config(4);
    let block = QwenBlock::new(ParamStore::new().root(), &config);
    deterministic(&block, device);
    let weights: Vec<Tensor> = block.parameters().iter().map(|p| p.detach()).collect();
    let (batch, seq) = (1, 3);
    let x = Tensor::from_vec(
        det_vec(batch * seq * config.hidden),
        vec![batch, seq, config.hidden],
        device,
    );
    let (cos, sin) = rope(seq, config.head_dim, config.rope_theta, device);

    // Act
    let actual = values(&block.forward(&x, &cos, &sin).unwrap());
    let expected = values(&manual_block_forward(
        &x,
        &weights,
        config.n_q_heads,
        config.n_kv_heads,
        config.head_dim,
        config.rms_norm_eps,
        &cos,
        &sin,
    ));

    // Assert
    assert_eq!(actual, expected);
    assert_eq!(
        actual[..8].to_vec(),
        vec![
            -0.47327715,
            -0.3010012,
            0.023309544,
            0.7928997,
            0.96602976,
            -1.0741744,
            -0.5301795,
            -0.5054862
        ]
    );
}

#[test]
fn zero_block_forwards_its_input() {
    // Arrange: every weight zeroed, so both sub-blocks contribute nothing.
    let device = Device::Cpu;
    let config = small_config(4);
    let block = QwenBlock::new(ParamStore::new().root(), &config);
    for param in block.parameters() {
        let shape: Vec<usize> = param.layout().shape().iter().copied().collect();
        param.set(&Tensor::zeros(shape, DType::F32, device)).unwrap();
    }
    let input = det_vec(2 * config.hidden);
    let x = Tensor::from_vec(input.clone(), vec![1, 2, config.hidden], device);
    let (cos, sin) = rope(2, config.head_dim, config.rope_theta, device);

    // Act
    let out = values(&block.forward(&x, &cos, &sin).unwrap());

    // Assert: the residuals carry the signal through untouched.
    assert_eq!(out, input);
}

#[test]
fn gradients_reach_every_projection_and_norm() {
    // Arrange
    let device = Device::Cpu;
    let config = small_config(4);
    let block = QwenBlock::new(ParamStore::new().root(), &config);
    deterministic(&block, device);
    let x = Tensor::from_vec(det_vec(2 * config.hidden), vec![1, 2, config.hidden], device);
    let (cos, sin) = rope(2, config.head_dim, config.rope_theta, device);

    // Act
    let loss = block.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let grads = loss.backward().unwrap();

    // Assert: all 11 parameter tensors (2 norms, 4 projections, 2 QK
    // norms, 3 SwiGLU projections) observe the loss.
    let params = block.parameters();
    assert_eq!(params.len(), 11);
    for (index, param) in params.iter().enumerate() {
        let grad = grad_of(&grads, param);
        assert!(grad.iter().any(|&entry| entry != 0.0), "block parameter {index} got no gradient");
    }
    assert_eq!(
        &grad_of(&grads, &params[4])[..4],
        &[-1.0441353, -0.2878458, -0.71510607, -0.51174974]
    );
}

#[test]
fn grouped_and_plain_head_counts_both_work() {
    // Arrange: the same width with 4 shared KV heads versus 8 plain heads.
    let device = Device::Cpu;
    let grouped = QwenBlock::new(ParamStore::new().root(), &small_config(4));
    let plain = QwenBlock::new(ParamStore::new().root(), &small_config(8));
    deterministic(&grouped, device);
    deterministic(&plain, device);
    assert!(
        grouped.parameters().iter().map(|p| p.layout().size()).sum::<usize>()
            < plain.parameters().iter().map(|p| p.layout().size()).sum::<usize>()
    );
    let x = Tensor::from_vec(det_vec(2 * 32), vec![1, 2, 32], device);
    let (cos, sin) = rope(2, 16, 1_000_000.0, device);

    // Act
    let grouped_out = values(&grouped.forward(&x, &cos, &sin).unwrap());
    let plain_out = values(&plain.forward(&x, &cos, &sin).unwrap());

    // Assert
    assert_eq!(grouped_out.len(), 64);
    assert_eq!(plain_out.len(), 64);
    assert_ne!(grouped_out, plain_out);
    assert_eq!(grouped_out[..4].to_vec(), vec![-0.47327715, -0.3010012, 0.023309544, 0.7928997]);
    assert_eq!(plain_out[..4].to_vec(), vec![-0.33898187, 2.0187504, 3.0758362, 3.3147247]);
}

#[test]
fn qwen3_0_6b_dims_wire_up() {
    // Arrange
    let store = ParamStore::new();
    let block = QwenBlock::new(store.root(), &Qwen3Config::qwen3_0_6b());

    // Act
    let named: Vec<(String, Vec<usize>)> = store
        .named_parameters()
        .into_iter()
        .map(|(name, param)| (name, param.layout().shape().iter().copied().collect()))
        .collect();

    // Assert: query width 2048 and KV width 1024 hang off hidden 1024.
    assert_eq!(
        named,
        vec![
            ("attn.k_norm.weight".to_owned(), vec![128]),
            ("attn.k_proj.weight".to_owned(), vec![1024, 1024]),
            ("attn.out_proj.weight".to_owned(), vec![2048, 1024]),
            ("attn.q_norm.weight".to_owned(), vec![128]),
            ("attn.q_proj.weight".to_owned(), vec![1024, 2048]),
            ("attn.v_proj.weight".to_owned(), vec![1024, 1024]),
            ("input_layernorm.weight".to_owned(), vec![1024]),
            ("mlp.down_proj.weight".to_owned(), vec![3072, 1024]),
            ("mlp.gate_proj.weight".to_owned(), vec![1024, 3072]),
            ("mlp.up_proj.weight".to_owned(), vec![1024, 3072]),
            ("post_attention_layernorm.weight".to_owned(), vec![1024]),
        ]
    );
    let x = Tensor::from_vec(det_vec(2 * 1024), vec![1, 2, 1024], Device::Cpu);
    let config = Qwen3Config::qwen3_0_6b();
    let (cos, sin) = config.rotary_cache(2, DType::F32, Device::Cpu);
    let out = block.forward(&x, &cos, &sin).unwrap();
    assert_eq!(out.layout().shape().as_slice(), &[1, 2, 1024]);
    assert!(values(&out).iter().all(|v| v.is_finite()));
}
