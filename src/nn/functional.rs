//! Stateless helper functions for building neural networks (causal masks,
//! composable primitives).

use half::{bf16, f16};

use crate::{
    DType, Device, Tensor,
    error::{Error, Result},
};

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

/// Query rows per online-softmax block. One `[B, H, Br, D]` output tile stays
/// live at a time, so peak traffic is weights plus one tile, never `T x T` scores.
const FLASH_Q_BLOCK: usize = 128;
/// Key/value columns per online-softmax block. Score tiles are `Br x Bc`.
const FLASH_KV_BLOCK: usize = 128;

/// Block-tiled causal attention with running softmax statistics (flash form).
///
/// Computes the same result as materialized `softmax(scale * Q @ K^T + mask) @ V`
/// over the full prompt, but never builds the `T x T` score matrix. Each query
/// block streams over the key/value blocks left to right, keeping three running
/// statistics per query row: the row maximum `m`, the rescaled exp-sum `l`, and
/// the unnormalized output `o`. Every key block rescales the past accumulators
/// down to the new maximum before folding its own `exp(S - m_new) @ V` block in,
/// so `o / l` at the end equals the exact full-row softmax output.
///
/// Causal masking is positional, not materialized: key blocks fully past the
/// query block are skipped outright (which also keeps all-`-inf` rows, whose
/// maximum would poison the rescale, out of the loop), and the diagonal block
/// adds a small additive mask built from global `(row, col)` positions.
///
/// Every tile's arithmetic runs in native per-backend kernels (`matmul` hits
/// cuBLAS on CUDA, gemm on CPU), so both covered backends compute on-device;
/// only the block loop itself is host orchestration. MPS has no tuned path and
/// fails loudly instead of silently staging through the CPU.
///
/// Inputs are post-RoPE, post-repeat `[B, H, T, D]` tensors sharing one float
/// dtype and device, with equal sequence lengths (prefill self-attention over
/// the whole prompt). Returns `[B, H, T, D]`. Built entirely from
/// differentiable primitives, so gradients flow exactly like the materialized
/// path. Shape mismatches panic: they are caller bugs, not runtime failures.
///
/// This mirrors the fused, tiled architecture candle uses for the same
/// operation, reimplemented here from the online-softmax equations.
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    if q.device() == Device::Mps || k.device() == Device::Mps || v.device() == Device::Mps {
        return Err(Error::NotImplemented("mps flash_attention is not implemented"));
    }
    let shape: Vec<usize> = q.layout().shape().iter().copied().collect();
    assert_eq!(shape.len(), 4, "flash_attention expects 4D [B, H, T, D] inputs");
    let (batch, heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(seq_len > 0, "flash_attention needs a non-empty sequence");
    for (name, t) in [("k", k), ("v", v)] {
        let other: Vec<usize> = t.layout().shape().iter().copied().collect();
        assert_eq!(other, shape, "flash_attention {name} shape {other:?} must match q {shape:?}");
    }
    assert_eq!(q.dtype(), k.dtype(), "flash_attention q/k dtype mismatch");
    assert_eq!(q.dtype(), v.dtype(), "flash_attention q/v dtype mismatch");
    assert_eq!(q.device(), k.device(), "flash_attention q/k device mismatch");
    assert_eq!(q.device(), v.device(), "flash_attention q/v device mismatch");
    assert_ne!(q.dtype(), DType::I64, "flash_attention requires a floating-point dtype");
    let dtype = q.dtype();
    let device = q.device();
    // Projected q/k/v arrive as permuted views, and `narrow` needs contiguous
    // inputs, so compact once up front (a no-op when already compact). Each tile
    // then narrows a compact base on every backend.
    let (q, k, v) = (q.compact(), k.compact(), v.compact());

    // One output tile per query block, concatenated along the sequence axis.
    let mut outputs = Vec::new();
    let mut query_start = 0;
    while query_start < seq_len {
        let query_len = FLASH_Q_BLOCK.min(seq_len - query_start);
        let query = q.narrow(2, query_start, query_len); // [B, H, Br, D]

        // Seed the running statistics from the first key block: block zero always
        // overlaps the query block, so every row has a valid maximum to start.
        let key_len = FLASH_KV_BLOCK.min(seq_len);
        let (mut running_max, mut running_sum, mut running_out) =
            flash_block(&query, &k, &v, 0, key_len, query_start, scale, dtype, device);

        // Fold later key blocks in left to right, skipping blocks fully past
        // the query rows (causal future), which carry no valid entries.
        let mut key_start = key_len;
        while key_start < seq_len {
            if key_start >= query_start + query_len {
                break;
            }
            let len = FLASH_KV_BLOCK.min(seq_len - key_start);
            let (block_max, block_sum, block_out) =
                flash_block(&query, &k, &v, key_start, len, query_start, scale, dtype, device);
            // Rescale past and present to their joint maximum, then add.
            // `m + relu(diff)` is the elementwise maximum of two equal shapes.
            let new_max = &running_max + &(&block_max - &running_max).relu();
            let past = (&running_max - &new_max).exp();
            let present = (&block_max - &new_max).exp();
            running_sum = &(&running_sum * &past) + &(&block_sum * &present);
            running_out = &(&running_out * &past) + &(&block_out * &present);
            running_max = new_max;
            key_start += len;
        }

        let norm = running_sum.broadcast(vec![batch, heads, query_len, head_dim]);
        outputs.push(&running_out / &norm);
        query_start += query_len;
    }
    Ok(Tensor::cat(&outputs, 2))
}

/// Scores one `(query block, key block)` pair into unnormalized statistics.
///
/// Returns the tile's row maximum `m` (`[B, H, Br, 1]`), its rescaled exp-sum
/// `l` (`[B, H, Br, 1]`), and its `exp(S - m) @ V` contribution (`[B, H, Br, D]`).
/// The caller rescales these against its own running statistics. `key_start`
/// positions the causal mask in global coordinates; fully allowed tiles skip
/// the mask build entirely and every surviving row keeps at least one valid
/// entry, so the row maximum is always finite.
fn flash_block(
    query: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_start: usize,
    key_len: usize,
    query_start: usize,
    scale: f64,
    dtype: DType,
    device: Device,
) -> (Tensor, Tensor, Tensor) {
    let query_len = query.layout().shape()[2];
    let key = k.narrow(2, key_start, key_len).transpose(None); // [B, H, D, Bc]
    let value = v.narrow(2, key_start, key_len); // [B, H, Bc, D]
    let mut scores = query.matmul(&key) * scale; // [B, H, Br, Bc]
    if key_start + key_len > query_start + 1 {
        // Diagonal tile: mask positions whose global column outruns their row.
        let mask = causal_tile_mask(query_len, key_len, query_start, key_start, dtype, device);
        scores = &scores + &mask;
    }
    let block_max = scores.max(vec![3], true); // [B, H, Br, 1]
    let weights = (&scores - &block_max).exp();
    let block_sum = weights.sum(vec![3], true); // [B, H, Br, 1]
    let block_out = weights.matmul(&value); // [B, H, Br, D]
    (block_max, block_sum, block_out)
}

/// Additive causal mask for one score tile in global coordinates.
///
/// Allowed positions (global column `<=` global row) hold `0`, masked ones
/// hold `-inf`, so the tile adds it straight onto its scores like the
/// materialized path adds its full causal mask. Shaped `[1, 1, Br, Bc]` so it
/// broadcasts over batch and heads without repeating storage per head.
fn causal_tile_mask(
    query_len: usize,
    key_len: usize,
    query_start: usize,
    key_start: usize,
    dtype: DType,
    device: Device,
) -> Tensor {
    let shape = vec![1, 1, query_len, key_len];
    match dtype {
        DType::F32 => {
            let mask: Vec<f32> = (0..query_len)
                .flat_map(|i| {
                    (0..key_len).map(move |j| {
                        if key_start + j > query_start + i { f32::NEG_INFINITY } else { 0.0 }
                    })
                })
                .collect();
            Tensor::from_vec(mask, shape, device)
        }
        DType::F16 => {
            let mask: Vec<f16> = (0..query_len)
                .flat_map(|i| {
                    (0..key_len).map(move |j| {
                        if key_start + j > query_start + i {
                            f16::NEG_INFINITY
                        } else {
                            f16::from_f32(0.0)
                        }
                    })
                })
                .collect();
            Tensor::from_vec(mask, shape, device)
        }
        DType::BF16 => {
            let mask: Vec<bf16> = (0..query_len)
                .flat_map(|i| {
                    (0..key_len).map(move |j| {
                        if key_start + j > query_start + i {
                            bf16::NEG_INFINITY
                        } else {
                            bf16::ZERO
                        }
                    })
                })
                .collect();
            Tensor::from_vec(mask, shape, device)
        }
        DType::I64 => panic!("flash_attention requires a floating-point dtype"),
    }
}
