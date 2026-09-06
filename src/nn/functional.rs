//! Stateless helper functions for building neural networks (causal masks,
//! composable primitives).

use half::f16;

use crate::{DType, Device, Tensor};

/// Applies inverted dropout: each element is zeroed with probability `p` and
/// survivors are scaled by `1 / (1 - p)`.
///
/// Returns the input unchanged when `training` is false or `p` is zero, so
/// evaluation is an exact identity that still carries gradients.
pub fn dropout(x: &Tensor, p: f64, training: bool) -> Tensor {
    assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1), got {p}");
    if !training || p == 0.0 {
        return x.clone();
    }
    let keep = 1.0 - p;
    let scale = 1.0 / keep;
    let shape: Vec<usize> = x.layout().shape().iter().copied().collect();
    let device = x.device();
    match x.dtype() {
        DType::F16 => {
            let uniform: Vec<f16> =
                Tensor::rand(shape.clone(), DType::F16, device).to_vec().unwrap();
            let mask: Vec<f16> = uniform
                .iter()
                .map(|v| {
                    if v.to_f32() < keep as f32 {
                        f16::from_f32(scale as f32)
                    } else {
                        f16::from_f32(0.0)
                    }
                })
                .collect();
            x * &Tensor::from_vec(mask, shape, device)
        }
        DType::F32 => {
            let uniform: Vec<f32> =
                Tensor::rand(shape.clone(), DType::F32, device).to_vec().unwrap();
            let mask: Vec<f32> =
                uniform.iter().map(|&v| if v < keep as f32 { scale as f32 } else { 0.0 }).collect();
            x * &Tensor::from_vec(mask, shape, device)
        }
        DType::I64 => panic!("dropout requires a floating-point dtype"),
    }
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
