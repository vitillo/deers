use deers::models::gpt::{CausalSelfAttention, KvCache, precompute_rotary_embeddings};
use deers::nn::functional::{FlashMask, causal_mask, flash_attention};
use deers::nn::ParamStore;
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

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    actual.iter().zip(expected).map(|(a, e)| (a - e).abs()).fold(0.0, f32::max)
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

/// Prefills the whole prompt, then decodes nothing: pure prefill-vs-reference parity.
///
/// The reference is always the CPU materialized forward with identical
/// deterministic weights. Same-device comparison on CUDA is avoided on purpose:
/// the materialized CUDA softmax path drifts intermittently against the CPU
/// reference (up to ~1e-2, data-dependent), while the tiled prefill matches the
/// reference to ~1e-7, so the CPU reference is the stricter check of exactness.
fn check_prefill_parity(seq_len: usize, device: Device, tol: f32) {
    // Arrange
    let (batch, n_embd, n_q_heads, n_kv_heads) = (1, 32, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let data = det_vec(batch * seq_len * n_embd);
    let reference = gqa(n_embd, n_q_heads, n_kv_heads, Device::Cpu);
    let ref_x = Tensor::from_vec(data.clone(), vec![batch, seq_len, n_embd], Device::Cpu);
    let (ref_cos, ref_sin) =
        precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, Device::Cpu);
    let expected: Vec<f32> =
        reference.forward(&ref_x, &ref_cos, &ref_sin).unwrap().to_vec().unwrap();

    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x = Tensor::from_vec(data, vec![batch, seq_len, n_embd], device);
    let (cos, sin) = precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device);
    let mut cache = KvCache::new();

    // Act
    let prefilled: Vec<f32> = attn.prefill(&x, &cos, &sin, &mut cache).unwrap().to_vec().unwrap();

    // Assert
    assert_eq!(expected.len(), prefilled.len(), "length mismatch at T={seq_len}");
    let worst = max_abs_diff(&prefilled, &expected);
    assert!(worst < tol, "prefill drifted at T={seq_len} on {device:?}: worst {worst} >= {tol}");
    assert_eq!(cache.len(), seq_len);
}

#[test]
fn flash_prefill_matches_materialized_at_128() {
    // Arrange: 128 prompt tokens, one query block of tiled attention.
    for device in devices() {
        if device == Device::Mps {
            continue;
        }
        // Act + Assert
        check_prefill_parity(128, device, 1e-5);
    }
}

#[test]
fn flash_prefill_matches_materialized_at_512() {
    // Arrange: 512 prompt tokens, four query blocks of tiled attention.
    for device in devices() {
        if device == Device::Mps {
            continue;
        }
        // Act + Assert
        check_prefill_parity(512, device, 1e-5);
    }
}

#[test]
fn flash_prefill_matches_materialized_at_2048() {
    // Arrange: 2048 prompt tokens, sixteen query blocks; the materialized
    // baseline builds a 2048x2048 score matrix per head while flash never does.
    for device in devices() {
        if device == Device::Mps {
            continue;
        }
        // Act + Assert
        check_prefill_parity(2048, device, 1e-5);
    }
}

#[test]
fn flash_prefill_covers_ragged_tile_edges() {
    // Arrange: 100 tokens is not a multiple of the 128-wide blocks, so the
    // trailing query and key tiles are partial and the diagonal mask applies.
    for device in devices() {
        if device == Device::Mps {
            continue;
        }
        // Act + Assert
        check_prefill_parity(100, device, 1e-5);
    }
}

#[test]
fn flash_prefill_stitched_decode_matches_full() {
    // Arrange: prefill 512 tokens tiled, then decode 64 more one at a time.
    let (batch, seq_len, prompt_len, n_embd, n_q_heads, n_kv_heads) = (1, 576, 512, 32, 4, 2);
    let head_dim = n_embd / n_q_heads;
    for device in devices() {
        if device == Device::Mps {
            continue;
        }
        let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
        let data = det_vec(batch * seq_len * n_embd);
        let x = Tensor::from_vec(data.clone(), vec![batch, seq_len, n_embd], device);
        let (cos, sin) =
            precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device);
        let mut cache = KvCache::new();

        // Act
        let reference = gqa(n_embd, n_q_heads, n_kv_heads, Device::Cpu);
        let ref_x = Tensor::from_vec(data, vec![batch, seq_len, n_embd], Device::Cpu);
        let (ref_cos, ref_sin) =
            precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, Device::Cpu);
        let full: Vec<f32> =
            reference.forward(&ref_x, &ref_cos, &ref_sin).unwrap().to_vec().unwrap();
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
        for pos in prompt_len..seq_len {
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
        let stitched: Vec<f32> = Tensor::cat(&pieces, 1).to_vec().unwrap();

        // Assert
        let worst = max_abs_diff(&stitched, &full);
        assert!(worst < 1e-5, "stitched run drifted on {device:?}: worst {worst}");
        assert_eq!(cache.len(), seq_len);
    }
}

#[test]
fn flash_prefill_gradients_match_materialized() {
    // Arrange: small prompt so the debug-CPU backward stays fast.
    let (batch, seq_len, prompt_len, n_embd, n_q_heads, n_kv_heads) = (1, 96, 64, 16, 4, 2);
    let head_dim = n_embd / n_q_heads;
    let device = Device::Cpu;
    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x =
        Tensor::from_vec(det_vec(batch * seq_len * n_embd), vec![batch, seq_len, n_embd], device);
    let (cos, sin) = precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device);

    // Act
    let full_loss = attn.forward(&x, &cos, &sin).unwrap().sum(vec![0, 1, 2], false);
    let full_grads = full_loss.backward().unwrap();
    let mut cache = KvCache::new();
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
    for pos in prompt_len..seq_len {
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
    let stitched_loss = Tensor::cat(&pieces, 1).sum(vec![0, 1, 2], false);
    let stitched_grads = stitched_loss.backward().unwrap();

    // Assert
    for param in attn.parameters() {
        let full: Vec<f32> = full_grads.get(param.id()).unwrap().to_vec().unwrap();
        let stitched: Vec<f32> = stitched_grads.get(param.id()).unwrap().to_vec().unwrap();
        let worst = max_abs_diff(&stitched, &full);
        assert!(worst < 1e-5, "gradient drifted: worst {worst}");
    }
}

/// Builds an additive bias with `-inf` where `allowed(i, j)` is false, else 0.
fn manual_bias(
    tq: usize,
    sk: usize,
    allowed: &impl Fn(usize, usize) -> bool,
    device: Device,
) -> Tensor {
    let data: Vec<f32> = (0..tq)
        .flat_map(|i| (0..sk).map(move |j| if allowed(i, j) { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(data, vec![1, 1, tq, sk], device)
}

/// Scores one mask mode through the fused kernel and the materialized path.
fn check_mask_mode(mask: FlashMask, bias: Tensor, tq: usize, sk: usize, device: Device) {
    // Arrange
    let (batch, heads, head_dim) = (1, 2, 8);
    let q_data = det_vec(batch * heads * tq * head_dim);
    let kv_data = det_vec(batch * heads * sk * head_dim);
    let q = Tensor::from_vec(q_data, vec![batch, heads, tq, head_dim], device);
    let k = Tensor::from_vec(kv_data.clone(), vec![batch, heads, sk, head_dim], device);
    let v = Tensor::from_vec(kv_data, vec![batch, heads, sk, head_dim], device);
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Act
    let fused: Vec<f32> = flash_attention(&q, &k, &v, scale, mask).unwrap().to_vec().unwrap();
    let scores = q.matmul(&k.transpose(None)) * scale + &bias;
    let expected: Vec<f32> = scores.softmax(3).matmul(&v).to_vec().unwrap();

    // Assert
    let worst = max_abs_diff(&fused, &expected);
    assert!(worst < 1e-5, "mask mode drifted on {device:?}: worst {worst}");
}

#[test]
fn flash_mask_modes_match_materialized() {
    // Arrange: every mask shape the fused kernel accepts, each against the
    // same-bias materialized reference.
    for device in devices() {
        if device == Device::Mps {
            continue;
        }
        // Act + Assert
        let zeros = Tensor::zeros(vec![1, 1, 64, 64], DType::F32, device);
        check_mask_mode(FlashMask::None, zeros, 64, 64, device);
        let causal = causal_mask(1, 64, 0, DType::F32, device);
        check_mask_mode(FlashMask::Causal, causal, 64, 64, device);
        let offset = manual_bias(64, 64, &|i, j| j <= i + 16, device);
        check_mask_mode(FlashMask::CausalWithOffset(16), offset, 64, 64, device);
        let band = manual_bias(48, 64, &|i, j| j <= i + 16 && j + 16 >= i, device);
        check_mask_mode(FlashMask::Mask(band.clone()), band, 48, 64, device);
    }
}

#[test]
fn flash_prefill_on_mps_fails_loudly() {
    // Arrange: MPS has no tuned tiling path, so prefill must refuse loudly
    // instead of silently staging through the CPU.
    if !Device::Mps.is_available() {
        return;
    }

    // Act
    let (n_embd, n_q_heads, n_kv_heads) = (8, 4, 2);
    let device = Device::Mps;
    let attn = gqa(n_embd, n_q_heads, n_kv_heads, device);
    let x = Tensor::from_vec(det_vec(2 * n_embd), vec![1, 2, n_embd], device);
    let (cos, sin) =
        precompute_rotary_embeddings(2, n_embd / n_q_heads, 10_000.0, DType::F32, device);
    let mut cache = KvCache::new();
    let result = attn.prefill(&x, &cos, &sin, &mut cache);

    // Assert
    assert!(result.is_err(), "MPS prefill must fail loudly until a native kernel lands");
}
