use candle_core::{D, Device as CDevice, Tensor as CTensor, Var};
use candle_nn::{loss as candle_loss, ops as candle_ops};
use deers::loss;
use deers::models::gpt;
use deers::nn::ParamStore;
use deers::{Device, Tensor};

const CPU_TOL: f32 = 1e-4;
const MPS_TOL: f32 = 2e-3;

struct CandleGptRef {
    config: gpt::GPTConfig,
    vars: Vec<Var>,
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");

    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{label}[{index}]: got {a}, expected {e}");
    }
}

fn devices() -> Vec<Device> {
    [Device::Cpu, Device::Mps].into_iter().filter(|device| device.is_available()).collect()
}

fn tol_for(device: Device) -> f32 {
    match device {
        Device::Cpu => CPU_TOL,
        _ => MPS_TOL,
    }
}

fn test_config() -> gpt::GPTConfig {
    gpt::GPTConfig {
        vocab_size: 8,
        sequence_len: 4,
        n_layer: 1,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    }
}

fn token_ids_i64() -> Vec<i64> {
    vec![1, 2, 3, 4, 3, 2]
}

fn token_ids_u32() -> Vec<u32> {
    vec![1, 2, 3, 4, 3, 2]
}

fn targets_i64() -> Vec<i64> {
    vec![2, 3, 4, 3, 2, 1]
}

fn targets_u32() -> Vec<u32> {
    vec![2, 3, 4, 3, 2, 1]
}

fn candle_tensor(data: Vec<f32>, shape: &[usize]) -> CTensor {
    CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap()
}

fn candle_grad(grads: &candle_core::backprop::GradStore, var: &Var) -> Vec<f32> {
    grads.get(var.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

fn candle_rms_norm(x: &CTensor, eps: f64) -> CTensor {
    let mean_sq = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
    let inv_norm = (mean_sq + eps).unwrap().powf(-0.5).unwrap();
    x.broadcast_mul(&inv_norm).unwrap()
}

fn candle_precompute_rotary_embeddings(
    seq_len: usize,
    head_dim: usize,
    base: f32,
) -> (CTensor, CTensor) {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let channel = (2 * i) as f32;
            1.0 / base.powf(channel / head_dim as f32)
        })
        .collect();
    let freqs: Vec<f32> =
        (0..seq_len).flat_map(|t| inv_freq.iter().map(move |&freq| t as f32 * freq)).collect();
    let shape = [1, seq_len, 1, half_dim];
    let cos = freqs.iter().map(|&x| x.cos()).collect();
    let sin = freqs.iter().map(|&x| x.sin()).collect();
    (candle_tensor(cos, &shape), candle_tensor(sin, &shape))
}

fn candle_apply_rotary_emb(x: &CTensor, cos: &CTensor, sin: &CTensor) -> CTensor {
    let head_dim = x.dims4().unwrap().3;
    let half_dim = head_dim / 2;
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

fn candle_causal_mask(batch_size: usize, seq_len: usize) -> CTensor {
    let mask: Vec<f32> = (0..batch_size)
        .flat_map(|_| {
            (0..seq_len).flat_map(move |i| {
                (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
            })
        })
        .collect();
    candle_tensor(mask, &[batch_size, 1, seq_len, seq_len])
}

fn candle_attention_forward(
    x: &CTensor,
    q_proj: &CTensor,
    k_proj: &CTensor,
    v_proj: &CTensor,
    out_proj: &CTensor,
    n_head: usize,
    cos: &CTensor,
    sin: &CTensor,
) -> CTensor {
    let (batch_size, seq_len, channels) = x.dims3().unwrap();
    let head_dim = channels / n_head;
    let x_flat = x.reshape((batch_size * seq_len, channels)).unwrap(); // [B*T, C]
    let q =
        x_flat.matmul(q_proj).unwrap().reshape((batch_size, seq_len, n_head, head_dim)).unwrap(); // [B, T, H, D]
    let k =
        x_flat.matmul(k_proj).unwrap().reshape((batch_size, seq_len, n_head, head_dim)).unwrap(); // [B, T, H, D]
    let v =
        x_flat.matmul(v_proj).unwrap().reshape((batch_size, seq_len, n_head, head_dim)).unwrap(); // [B, T, H, D]

    let q = candle_apply_rotary_emb(&q, cos, sin).transpose(1, 2).unwrap().contiguous().unwrap(); // [B, H, T, D]
    let k = candle_apply_rotary_emb(&k, cos, sin).transpose(1, 2).unwrap().contiguous().unwrap(); // [B, H, T, D]
    let v = v.transpose(1, 2).unwrap().contiguous().unwrap(); // [B, H, T, D]

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * scale).unwrap(); // [B, H, T, T]
    let scores = scores.broadcast_add(&candle_causal_mask(batch_size, seq_len)).unwrap();
    let attn = candle_ops::softmax(&scores, D::Minus1).unwrap(); // [B, H, T, T]
    let y = attn
        .matmul(&v)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((batch_size, seq_len, channels))
        .unwrap(); // [B, T, C]

    y.reshape((batch_size * seq_len, channels))
        .unwrap()
        .matmul(out_proj)
        .unwrap()
        .reshape((batch_size, seq_len, channels))
        .unwrap() // [B, T, C]
}

fn candle_mlp_forward(x: &CTensor, up_proj: &CTensor, down_proj: &CTensor) -> CTensor {
    let (batch_size, seq_len, channels) = x.dims3().unwrap();
    let y = x.reshape((batch_size * seq_len, channels)).unwrap().matmul(up_proj).unwrap(); // [B*T, H]
    let y = y.relu().unwrap().sqr().unwrap(); // [B*T, H]
    y.matmul(down_proj).unwrap().reshape((batch_size, seq_len, channels)).unwrap() // [B, T, C]
}

fn candle_gpt_forward(
    ids: &[u32],
    batch_size: usize,
    seq_len: usize,
    config: &gpt::GPTConfig,
    weights: &[CTensor],
) -> CTensor {
    let wte = &weights[0];
    let q_proj = &weights[1];
    let k_proj = &weights[2];
    let v_proj = &weights[3];
    let out_proj = &weights[4];
    let up_proj = &weights[5];
    let down_proj = &weights[6];
    let lm_head = &weights[7];

    let ids = CTensor::from_vec(ids.to_vec(), &[batch_size * seq_len], &CDevice::Cpu).unwrap();
    let mut x = wte.embedding(&ids).unwrap().reshape((batch_size, seq_len, config.n_embd)).unwrap(); // [B, T, C]
    let (cos, sin) =
        candle_precompute_rotary_embeddings(seq_len, config.head_dim(), config.rope_base);

    let norm1 = candle_rms_norm(&x, config.rms_norm_eps);
    let attn = candle_attention_forward(
        &norm1,
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        config.n_head,
        &cos,
        &sin,
    );
    x = x.broadcast_add(&attn).unwrap(); // [B, T, C]

    let norm2 = candle_rms_norm(&x, config.rms_norm_eps);
    let mlp = candle_mlp_forward(&norm2, up_proj, down_proj);
    x = x.broadcast_add(&mlp).unwrap(); // [B, T, C]

    let x = candle_rms_norm(&x, config.rms_norm_eps); // [B, T, C]
    x.reshape((batch_size * seq_len, config.n_embd))
        .unwrap()
        .matmul(lm_head)
        .unwrap()
        .reshape((batch_size, seq_len, config.vocab_size))
        .unwrap() // [B, T, V]
}

impl CandleGptRef {
    fn from_model(config: gpt::GPTConfig, model: &gpt::GPT) -> Self {
        let vars = model
            .parameters()
            .iter()
            .map(|parameter| {
                let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
                let tensor = candle_tensor(parameter.to_vec::<f32>().unwrap(), &shape);
                Var::from_tensor(&tensor).unwrap()
            })
            .collect();
        Self { config, vars }
    }

    fn weights(&self) -> Vec<CTensor> {
        self.vars.iter().map(|var| var.as_tensor().clone()).collect()
    }

    fn forward(&self, ids: &[u32], batch_size: usize, seq_len: usize) -> Vec<f32> {
        candle_gpt_forward(ids, batch_size, seq_len, &self.config, &self.weights())
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn loss_and_grads(
        &self,
        ids: &[u32],
        targets: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> (f32, Vec<Vec<f32>>) {
        let logits = candle_gpt_forward(ids, batch_size, seq_len, &self.config, &self.weights());
        let loss = candle_loss::cross_entropy(
            &logits.reshape((batch_size * seq_len, self.config.vocab_size)).unwrap(),
            &CTensor::from_vec(targets.to_vec(), &[batch_size * seq_len], &CDevice::Cpu).unwrap(),
        )
        .unwrap();
        let loss_value = loss.to_vec0::<f32>().unwrap();
        let grads = loss.backward().unwrap();
        let grad_values = self.vars.iter().map(|var| candle_grad(&grads, var)).collect();
        (loss_value, grad_values)
    }
}

#[test]
fn test_gpt_forward_conforms_with_candle_on_cpu_and_mps() {
    // Arrange
    let mut model = gpt::GPT::new(test_config(), ParamStore::new().root());
    let candle = CandleGptRef::from_model(test_config(), &model);
    let batch_size = 2;
    let seq_len = 3;
    let expected = candle.forward(&token_ids_u32(), batch_size, seq_len);

    // Act
    for device in devices() {
        if device == Device::Mps {
            model.to_device(device).unwrap();
        }

        let idx = Tensor::from_vec(token_ids_i64(), (batch_size, seq_len), device);
        let actual = model.forward(&idx).unwrap().to_vec::<f32>().unwrap();

        // Assert
        assert_close(&actual, &expected, tol_for(device), &format!("gpt forward on {:?}", device));
    }
}

#[test]
fn test_gpt_backward_conforms_with_candle_on_cpu_and_mps() {
    // Arrange
    let config = test_config();
    let mut model = gpt::GPT::new(test_config(), ParamStore::new().root());
    let candle = CandleGptRef::from_model(test_config(), &model);
    let batch_size = 2;
    let seq_len = 3;
    let (expected_loss, expected_grads) =
        candle.loss_and_grads(&token_ids_u32(), &targets_u32(), batch_size, seq_len);

    // Act
    for device in devices() {
        if device == Device::Mps {
            model.to_device(device).unwrap();
        }

        let idx = Tensor::from_vec(token_ids_i64(), (batch_size, seq_len), device);
        let targets = Tensor::from_vec(targets_i64(), (batch_size * seq_len,), device);
        let logits = model.forward(&idx).unwrap();
        let loss = loss::cross_entropy(
            &logits.reshape((batch_size * seq_len, config.vocab_size)),
            &targets,
        );
        let actual_loss = loss.to_vec::<f32>().unwrap()[0];
        let grads = loss.backward().unwrap();
        let parameters = model.parameters();

        // Assert
        assert_close(
            &[actual_loss],
            &[expected_loss],
            tol_for(device),
            &format!("gpt loss on {:?}", device),
        );

        for (index, (parameter, expected_grad)) in
            parameters.iter().zip(expected_grads.iter()).enumerate()
        {
            let actual_grad = grads.get(parameter.id()).unwrap().to_vec::<f32>().unwrap();
            assert_close(
                &actual_grad,
                expected_grad,
                tol_for(device),
                &format!("gpt parameter grad {index} on {:?}", device),
            );
        }
    }
}
