//! Helper functions for building neural networks (causal masks,
//! composable primitives).

use std::sync::Mutex;

use half::{bf16, f16};

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
        DType::BF16 => {
            let uniform: Vec<bf16> =
                Tensor::rand(shape.clone(), DType::BF16, device).to_vec().unwrap();
            let mask: Vec<bf16> = uniform
                .iter()
                .map(|v| {
                    if v.to_f32() < keep as f32 { bf16::from_f32(scale as f32) } else { bf16::ZERO }
                })
                .collect();
            x * &Tensor::from_vec(mask, shape, device)
        }
        DType::I64 => panic!("dropout requires a floating-point dtype"),
    }
}

/// Maximum entries held by the [`causal_mask`] cache.
///
/// Forward and prefill reuse one entry per shape while every decode step
/// shares the single-query entry, so a handful of slots covers generation.
/// The bound keeps a shape-churned workload from pinning device memory.
pub const CAUSAL_MASK_CACHE_CAP: usize = 16;

/// Identifies one cached causal mask: the shape inputs plus dtype and device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CausalMaskKey {
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: Device,
}

/// Causal masks by key, oldest first. Small enough for a linear scan, and
/// `Tensor` clones share storage, so hits cost no rebuild and no upload.
static CAUSAL_MASK_CACHE: Mutex<Vec<(CausalMaskKey, Tensor)>> = Mutex::new(Vec::new());

/// Returns the number of masks currently held by the [`causal_mask`] cache.
///
/// Diagnostics for tests: the count never exceeds [`CAUSAL_MASK_CACHE_CAP`]
/// no matter how many distinct shapes flow through the cache.
pub fn causal_mask_cache_len() -> usize {
    CAUSAL_MASK_CACHE.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// Builds an additive causal attention mask with shape `[batch, 1, tgt_len, tgt_len + seqlen_offset]`.
///
/// Allowed positions contain `0`, masked positions contain `-inf`, so the result can be added
/// directly to attention logits before softmax.
///
/// Masks are cached by shape inputs, dtype, and device under a bounded
/// first-in-first-out cache, so repeated forwards, prefills, and decode
/// steps with the same key reuse one device tensor instead of rebuilding on
/// the host and uploading again. A single query token (`tgt_len == 1`)
/// masks nothing, so decode steps share one tiny `[batch, 1, 1, 1]` zeros
/// tensor that broadcasts over heads and keys, whatever the offset.
/// Reused masks equal freshly built ones element for element.
pub fn causal_mask(
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: Device,
) -> Tensor {
    assert!(dtype != DType::I64, "causal_mask requires a floating-point dtype");
    let key = CausalMaskKey {
        batch_size,
        tgt_len,
        // Single-query masks hold only zeros (see `build_causal_mask`), so
        // every decode offset shares one entry instead of one wide row each.
        seqlen_offset: if tgt_len == 1 { 0 } else { seqlen_offset },
        dtype,
        device,
    };
    let mut cache = CAUSAL_MASK_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, mask)) = cache.iter().find(|(k, _)| *k == key) {
        return mask.clone();
    }
    let mask = build_causal_mask(batch_size, tgt_len, seqlen_offset, dtype, device);
    if cache.len() >= CAUSAL_MASK_CACHE_CAP {
        cache.remove(0);
    }
    cache.push((key, mask.clone()));
    mask
}

/// Builds the mask behind [`causal_mask`] without consulting the cache.
fn build_causal_mask(
    batch_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: Device,
) -> Tensor {
    if tgt_len == 1 {
        // With one query row (`i == 0`), `j - seqlen_offset > i` never holds,
        // so no position is masked. One zero per batch broadcasts over the
        // heads and keys of `[B, H, 1, S]` logits exactly like a wide row.
        return Tensor::zeros(vec![batch_size, 1, 1, 1], dtype, device);
    }
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
        DType::BF16 => {
            let mask: Vec<bf16> = (0..batch_size)
                .flat_map(|_| {
                    (0..tgt_len).flat_map(|i| {
                        (0..total_len).map(move |j| {
                            if j >= seqlen_offset && j - seqlen_offset > i {
                                bf16::NEG_INFINITY
                            } else {
                                bf16::ZERO
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
