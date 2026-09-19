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

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "[{index}]: got {a}, expected {e}");
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

fn rope(seq_len: usize, head_dim: usize, device: Device) -> (Tensor, Tensor) {
    precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device)
}

fn grad_of(grads: &deers::GradientStore, param: &Parameter) -> Vec<f32> {
    values(&grads.get(param.id()).unwrap())
}

/// Scores `prompt_len` tokens at once, then decodes the rest one token at a
/// time, stitching every output back into full-sequence order.
fn decode_rest(
    attn: &CausalSelfAttention,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    prompt_len: usize,
    cache: &mut KvCache,
) -> Tensor {
    let seq_len = x.layout().shape()[1];
    let prompt = x.narrow(1, 0, prompt_len);
    let prompt_cos = cos.narrow(1, 0, prompt_len);
    let prompt_sin = sin.narrow(1, 0, prompt_len);
    let mut pieces = vec![attn.prefill(&prompt, &prompt_cos, &prompt_sin, cache).unwrap()];
    for pos in prompt_len..seq_len {
        let token = x.narrow(1, pos, 1);
        let token_cos = cos.narrow(1, pos, 1);
        let token_sin = sin.narrow(1, pos, 1);
        pieces.push(attn.decode(&token, &token_cos, &token_sin, cache).unwrap());
    }
    Tensor::cat(&pieces, 1)
}

#[test]
fn cached_decode_matches_full_recomputation() {
    // Arrange
    let (batch, seq, prompt_len, n_embd, n_q_heads, n_kv_heads) = (1, 5, 3, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);
        let mut cache = KvCache::new();

        // Act
        let full = values(&attn.forward(&x, &cos, &sin).unwrap());
        let stitched = values(&decode_rest(&attn, &x, &cos, &sin, prompt_len, &mut cache));

        // Assert
        assert_close(&stitched, &full, 1e-5);
        assert_eq!(cache.len(), seq);
        if device == Device::Cpu {
            assert_eq!(
                stitched[..8].to_vec(),
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
}

#[test]
fn single_token_prompt_decodes() {
    // Arrange
    let (batch, seq, n_embd, n_q_heads, n_kv_heads) = (1, 4, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
        let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
        let (cos, sin) = rope(seq, head_dim, device);
        let mut cache = KvCache::new();

        // Act
        let full = values(&attn.forward(&x, &cos, &sin).unwrap());
        let stitched = values(&decode_rest(&attn, &x, &cos, &sin, 1, &mut cache));

        // Assert
        assert_close(&stitched, &full, 1e-5);
        assert_eq!(cache.len(), seq);
    }
}

#[test]
fn cache_grows_across_many_steps() {
    // Arrange
    let (batch, seq, prompt_len, n_embd, n_q_heads, n_kv_heads) = (1, 8, 2, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let device = Device::Cpu;
    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
    let (cos, sin) = rope(seq, head_dim, device);
    let mut cache = KvCache::new();

    // Act
    let prompt = x.narrow(1, 0, prompt_len);
    let first = attn
        .prefill(&prompt, &cos.narrow(1, 0, prompt_len), &sin.narrow(1, 0, prompt_len), &mut cache)
        .unwrap();
    assert_eq!(cache.len(), prompt_len);
    let mut pieces = vec![first];
    for pos in prompt_len..seq {
        let token = x.narrow(1, pos, 1);
        pieces.push(
            attn.decode(&token, &cos.narrow(1, pos, 1), &sin.narrow(1, pos, 1), &mut cache)
                .unwrap(),
        );
        assert_eq!(cache.len(), pos + 1);
    }
    let stitched = values(&Tensor::cat(&pieces, 1));
    let full = values(&attn.forward(&x, &cos, &sin).unwrap());

    // Assert
    assert_close(&stitched, &full, 1e-5);
    assert_eq!(cache.len(), seq);
}

#[test]
fn cached_gradients_match_full_recomputation() {
    // Arrange
    let (batch, seq, prompt_len, n_embd, n_q_heads, n_kv_heads) = (1, 5, 3, 8, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let device = Device::Cpu;
    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x = Tensor::from_vec(det_vec(batch * seq * n_embd), vec![batch, seq, n_embd], device);
    let (cos, sin) = rope(seq, head_dim, device);

    // Act
    let full_loss = attn.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let full_grads = full_loss.backward().unwrap();
    let mut cache = KvCache::new();
    let stitched = decode_rest(&attn, &x, &cos, &sin, prompt_len, &mut cache);
    let stitched_loss = stitched.sum(vec![0, 1, 2], false);
    let stitched_grads = stitched_loss.backward().unwrap();

    // Assert
    let params = attn.parameters();
    for param in &params {
        let full = grad_of(&full_grads, param);
        let stitched = grad_of(&stitched_grads, param);
        assert_close(&stitched, &full, 1e-5);
        assert!(stitched.iter().any(|&entry| entry != 0.0), "a projection got no gradient");
    }
    assert_eq!(
        &grad_of(&stitched_grads, &params[0])[..4],
        &[-0.010818105, 0.0063312226, -0.0007816993, -0.0025316612]
    );
}

#[test]
#[should_panic(expected = "decode expects a non-empty cache")]
fn decode_rejects_empty_cache() {
    // Arrange
    let (n_embd, n_q_heads, n_kv_heads) = (8, 4, 2);
    let device = Device::Cpu;
    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x = Tensor::from_vec(det_vec(n_embd), vec![1, 1, n_embd], device);
    let (cos, sin) = rope(1, n_embd / n_q_heads, device);
    let mut cache = KvCache::new();

    // Act
    let _ = attn.decode(&x, &cos, &sin, &mut cache);

    // Assert: panics before returning.
}

#[test]
#[should_panic(expected = "prefill expects an empty cache")]
fn prefill_rejects_nonempty_cache() {
    // Arrange
    let (n_embd, n_q_heads, n_kv_heads) = (8, 4, 2);
    let device = Device::Cpu;
    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x = Tensor::from_vec(det_vec(2 * n_embd), vec![1, 2, n_embd], device);
    let (cos, sin) = rope(2, n_embd / n_q_heads, device);
    let mut cache = KvCache::new();
    attn.prefill(&x, &cos, &sin, &mut cache).unwrap();

    // Act
    let _ = attn.prefill(&x, &cos, &sin, &mut cache);

    // Assert: panics before returning.
}
