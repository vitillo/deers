use half::f16;

use crate::{DType, Device, Tensor};

/// Precomputes RoPE cos/sin caches with shape `[1, seq_len, 1, head_dim / 2]`.
pub fn precompute_rotary_embeddings(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    dtype: DType,
    device: Device,
) -> (Tensor, Tensor) {
    assert!(head_dim.is_multiple_of(2), "RoPE requires an even head dimension");

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let channel = (2 * i) as f32;
            1.0 / base.powf(channel / head_dim as f32)
        })
        .collect();

    let freqs: Vec<f32> =
        (0..seq_len).flat_map(|t| inv_freq.iter().map(move |&freq| t as f32 * freq)).collect();

    let shape = vec![1, seq_len, 1, half_dim];
    match dtype {
        DType::F16 => {
            let cos: Vec<f16> = freqs.iter().map(|&x| f16::from_f32(x.cos())).collect();
            let sin: Vec<f16> = freqs.iter().map(|&x| f16::from_f32(x.sin())).collect();
            (Tensor::from_vec(cos, shape.clone(), device), Tensor::from_vec(sin, shape, device))
        }
        DType::F32 => {
            let cos: Vec<f32> = freqs.iter().map(|&x| x.cos()).collect();
            let sin: Vec<f32> = freqs.iter().map(|&x| x.sin()).collect();
            (Tensor::from_vec(cos, shape.clone(), device), Tensor::from_vec(sin, shape, device))
        }
        DType::I64 => panic!("RoPE requires a floating-point dtype"),
    }
}

/// Applies rotary embeddings to a multi-head attention tensor with shape `[B, T, H, D]`.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    assert_eq!(x.layout().ndim(), 4, "RoPE expects a 4D attention tensor");
    assert_eq!(cos.layout().shape(), sin.layout().shape(), "RoPE cos/sin shapes must match");

    let head_dim = x.layout().shape()[3];
    assert!(head_dim.is_multiple_of(2), "RoPE requires an even head dimension");

    let half_dim = head_dim / 2;
    assert_eq!(
        cos.layout().shape()[3],
        half_dim,
        "RoPE cache last dimension must equal head_dim / 2"
    );
    assert_eq!(
        cos.layout().shape()[1],
        x.layout().shape()[1],
        "RoPE cache must match sequence length"
    );

    let x1 = x.narrow(3, 0, half_dim);
    let x2 = x.narrow(3, half_dim, half_dim);
    let y1 = &x1 * cos + &x2 * sin;
    let y2 = &x1 * &(sin * -1.0) + &x2 * cos;
    Tensor::cat(&[y1, y2], 3)
}

/// Builds an additive causal attention mask with shape `[batch, 1, tgt_len, tgt_len + seqlen_offset]`.
///
/// Allowed positions contain `0`, masked positions contain `-inf`, so the result can be added
/// directly to attention logits before softmax.
pub fn causal_mask(
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: Device,
) -> Tensor {
    let total_len = tgt_len + seqlen_offset;
    match dtype {
        DType::F16 => {
            let mask: Vec<f16> = (0..batch_size)
                .flat_map(|_| {
                    (0..tgt_len).flat_map(|i| {
                        (0..total_len).map(move |j| {
                            if j >= seqlen_offset && j - seqlen_offset > i {
                                f16::NEG_INFINITY
                            } else {
                                f16::from_f32(0.0)
                            }
                        })
                    })
                })
                .collect();
            Tensor::from_vec(mask, vec![batch_size, 1, tgt_len, total_len], device)
        }
        DType::F32 => {
            let mask: Vec<f32> = (0..batch_size)
                .flat_map(|_| {
                    (0..tgt_len).flat_map(|i| {
                        (0..total_len).map(move |j| {
                            if j >= seqlen_offset && j - seqlen_offset > i {
                                f32::NEG_INFINITY
                            } else {
                                0.0
                            }
                        })
                    })
                })
                .collect();
            Tensor::from_vec(mask, vec![batch_size, 1, tgt_len, total_len], device)
        }
        DType::I64 => panic!("causal_mask requires a floating-point dtype"),
    }
}
