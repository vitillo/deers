//! Stateless helper functions for building neural networks (causal masks,
//! composable primitives).

use half::f16;

use crate::{DType, Device, Tensor};

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
