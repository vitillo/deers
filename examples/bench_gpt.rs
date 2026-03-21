//! Benchmark: deers vs candle GPT forward + backward.
//!
//! Builds the same small GPT architecture in both libraries and compares
//! forward-only and forward+backward timings on CPU.
//!
//! Run:
//!   cargo run --release --example bench_gpt

use std::time::Instant;

// --- Config shared by both implementations ---

const VOCAB_SIZE: usize = 512; // small for benchmarking
const SEQ_LEN: usize = 64;
const N_LAYER: usize = 2;
const N_HEAD: usize = 4;
const N_EMBD: usize = 64;
const MLP_HIDDEN: usize = N_EMBD * 4;
const BATCH_SIZE: usize = 4;
const WARMUP: usize = 2;
const ITERATIONS: usize = 10;

fn main() {
    println!(
        "GPT benchmark: vocab={VOCAB_SIZE}, seq={SEQ_LEN}, layers={N_LAYER}, heads={N_HEAD}, d={N_EMBD}"
    );
    println!("batch={BATCH_SIZE}, warmup={WARMUP}, iterations={ITERATIONS}\n");

    bench_deers();
    bench_candle();
}

// ---------------------------------------------------------------------------
// deers
// ---------------------------------------------------------------------------

fn bench_deers() {
    use deers::models::gpt::{GPT, GPTConfig};
    use deers::{Device, Tensor, loss};

    let config = GPTConfig {
        vocab_size: VOCAB_SIZE,
        sequence_len: SEQ_LEN,
        n_layer: N_LAYER,
        n_head: N_HEAD,
        n_embd: N_EMBD,
        mlp_hidden_dim: MLP_HIDDEN,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    };
    let model = GPT::new(config);

    let num_params: usize = model.parameters().iter().map(|p| p.layout().size()).sum();

    // Random-ish token ids
    let ids: Vec<i64> = (0..BATCH_SIZE * SEQ_LEN).map(|i| (i % VOCAB_SIZE) as i64).collect();
    let input = Tensor::from_vec(ids.clone(), (BATCH_SIZE, SEQ_LEN), Device::Cpu);
    let target_ids: Vec<i64> =
        (0..BATCH_SIZE * SEQ_LEN).map(|i| ((i + 1) % VOCAB_SIZE) as i64).collect();
    let targets = Tensor::from_vec(target_ids, (BATCH_SIZE * SEQ_LEN,), Device::Cpu);

    // Warmup
    for _ in 0..WARMUP {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape(vec![BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]);
        let l = loss::cross_entropy(&logits_flat, &targets);
        let _ = l.backward();
    }

    // Forward only
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape(vec![BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]);
        let _l = loss::cross_entropy(&logits_flat, &targets);
    }
    let fwd_us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

    // Forward + backward
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape(vec![BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]);
        let l = loss::cross_entropy(&logits_flat, &targets);
        let _ = l.backward();
    }
    let fwd_bwd_us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

    println!("=== deers cpu ({num_params} params) ===");
    println!("  forward:          {fwd_us:>10.0} µs");
    println!("  forward+backward: {fwd_bwd_us:>10.0} µs");
    println!("  backward only:    {:>10.0} µs\n", fwd_bwd_us - fwd_us);
}

// ---------------------------------------------------------------------------
// candle — same architecture, built from primitives
// ---------------------------------------------------------------------------

fn bench_candle() {
    use candle_core::{DType, Device as CDevice, Tensor as CTensor};
    use candle_nn::{VarBuilder, VarMap};

    let device = CDevice::Cpu;
    let dtype = DType::F32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let model = CandleGPT::new(vb).unwrap();
    let num_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();

    // Same token ids
    let ids: Vec<u32> = (0..BATCH_SIZE * SEQ_LEN).map(|i| (i % VOCAB_SIZE) as u32).collect();
    let input = CTensor::from_vec(ids, (BATCH_SIZE, SEQ_LEN), &device).unwrap();
    let target_ids: Vec<u32> =
        (0..BATCH_SIZE * SEQ_LEN).map(|i| ((i + 1) % VOCAB_SIZE) as u32).collect();
    let targets = CTensor::from_vec(target_ids, BATCH_SIZE * SEQ_LEN, &device).unwrap();

    // Warmup
    for _ in 0..WARMUP {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape((BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)).unwrap();
        let l = candle_nn::loss::cross_entropy(&logits_flat, &targets).unwrap();
        let _ = l.backward();
    }

    // Forward only
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape((BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)).unwrap();
        let _l = candle_nn::loss::cross_entropy(&logits_flat, &targets).unwrap();
    }
    let fwd_us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

    // Forward + backward
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape((BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)).unwrap();
        let l = candle_nn::loss::cross_entropy(&logits_flat, &targets).unwrap();
        let _ = l.backward();
    }
    let fwd_bwd_us = t0.elapsed().as_micros() as f64 / ITERATIONS as f64;

    println!("=== candle cpu ({num_params} params) ===");
    println!("  forward:          {fwd_us:>10.0} µs");
    println!("  forward+backward: {fwd_bwd_us:>10.0} µs");
    println!("  backward only:    {:>10.0} µs\n", fwd_bwd_us - fwd_us);
}

// ---------------------------------------------------------------------------
// Candle GPT implementation (matches deers architecture)
// ---------------------------------------------------------------------------

use candle_core::{Device as CDevice, Result as CResult, Tensor as CTensor};
use candle_nn::{Module, VarBuilder};

struct CandleGPT {
    wte: candle_nn::Embedding,
    blocks: Vec<CandleBlock>,
    norm: candle_nn::RmsNorm,
    lm_head: candle_nn::Linear,
    cos: CTensor,
    sin: CTensor,
}

impl CandleGPT {
    fn new(vb: VarBuilder) -> CResult<Self> {
        let wte = candle_nn::embedding(VOCAB_SIZE, N_EMBD, vb.pp("wte"))?;
        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(CandleBlock::new(vb.pp(format!("block_{i}")))?);
        }
        let norm = candle_nn::rms_norm(N_EMBD, 1e-5, vb.pp("norm"))?;
        let lm_head = candle_nn::linear_no_bias(N_EMBD, VOCAB_SIZE, vb.pp("lm_head"))?;
        let (cos, sin) = candle_rotary_cache(vb.device())?;
        Ok(Self { wte, blocks, norm, lm_head, cos, sin })
    }

    fn forward(&self, idx: &CTensor) -> CResult<CTensor> {
        let (_b, t) = idx.dims2()?;
        let cos = self.cos.narrow(1, 0, t)?;
        let sin = self.sin.narrow(1, 0, t)?;

        let mut x = self.wte.forward(idx)?; // [B, T, C]
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }
        x = self.norm.forward(&x)?;
        // Flatten to [B*T, C] for lm_head
        let (b, t, c) = x.dims3()?;
        let x_flat = x.reshape((b * t, c))?;
        let logits = self.lm_head.forward(&x_flat)?; // [B*T, V]
        logits.reshape((b, t, VOCAB_SIZE))
    }
}

struct CandleBlock {
    norm1: candle_nn::RmsNorm,
    attn: CandleAttention,
    norm2: candle_nn::RmsNorm,
    mlp: CandleMLP,
}

impl CandleBlock {
    fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            norm1: candle_nn::rms_norm(N_EMBD, 1e-5, vb.pp("norm1"))?,
            attn: CandleAttention::new(vb.pp("attn"))?,
            norm2: candle_nn::rms_norm(N_EMBD, 1e-5, vb.pp("norm2"))?,
            mlp: CandleMLP::new(vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &CTensor, cos: &CTensor, sin: &CTensor) -> CResult<CTensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?, cos, sin)?)?;
        let y = self.mlp.forward(&self.norm2.forward(&x)?)?;
        &x + y
    }
}

struct CandleAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
}

impl CandleAttention {
    fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(N_EMBD, N_EMBD, vb.pp("q"))?,
            k_proj: candle_nn::linear_no_bias(N_EMBD, N_EMBD, vb.pp("k"))?,
            v_proj: candle_nn::linear_no_bias(N_EMBD, N_EMBD, vb.pp("v"))?,
            out_proj: candle_nn::linear_no_bias(N_EMBD, N_EMBD, vb.pp("out"))?,
        })
    }

    fn forward(&self, x: &CTensor, cos: &CTensor, sin: &CTensor) -> CResult<CTensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = N_EMBD / N_HEAD;
        let x_flat = x.reshape((b * t, c))?;

        let q = self.q_proj.forward(&x_flat)?.reshape((b, t, N_HEAD, head_dim))?;
        let k = self.k_proj.forward(&x_flat)?.reshape((b, t, N_HEAD, head_dim))?;
        let v = self.v_proj.forward(&x_flat)?.reshape((b, t, N_HEAD, head_dim))?;

        let q = candle_apply_rotary(&q, cos, sin)?.permute((0, 2, 1, 3))?.contiguous()?; // [B, H, T, D]
        let k = candle_apply_rotary(&k, cos, sin)?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(scale, 0.0)?; // [B, H, T, T]

        // Causal mask: upper triangle filled with -inf
        let mask = candle_causal_mask(t, x.device())?.broadcast_left((b, N_HEAD))?;

        let scores_masked = scores.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax(&scores_masked, 3)?;
        let y = attn.matmul(&v)?.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, t, c))?;

        let y_flat = y.reshape((b * t, c))?;
        self.out_proj.forward(&y_flat)?.reshape((b, t, c))
    }
}

struct CandleMLP {
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl CandleMLP {
    fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            up_proj: candle_nn::linear_no_bias(N_EMBD, MLP_HIDDEN, vb.pp("up"))?,
            down_proj: candle_nn::linear_no_bias(MLP_HIDDEN, N_EMBD, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &CTensor) -> CResult<CTensor> {
        let (b, t, c) = x.dims3()?;
        let x_flat = x.reshape((b * t, c))?;
        let y = self.up_proj.forward(&x_flat)?.relu()?;
        let y = (&y * &y)?; // relu²
        let y = self.down_proj.forward(&y)?;
        y.reshape((b, t, N_EMBD))
    }
}

/// Precompute RoPE cos/sin cache: [1, seq_len, 1, head_dim/2]
fn candle_rotary_cache(device: &CDevice) -> CResult<(CTensor, CTensor)> {
    let head_dim = N_EMBD / N_HEAD;
    let half_dim = head_dim / 2;

    let mut cos_data = Vec::with_capacity(SEQ_LEN * half_dim);
    let mut sin_data = Vec::with_capacity(SEQ_LEN * half_dim);

    for t in 0..SEQ_LEN {
        for i in 0..half_dim {
            let freq = 1.0 / 10_000_f32.powf((2 * i) as f32 / head_dim as f32);
            let angle = t as f32 * freq;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }

    let cos = CTensor::from_vec(cos_data, (1, SEQ_LEN, 1, half_dim), device)?;
    let sin = CTensor::from_vec(sin_data, (1, SEQ_LEN, 1, half_dim), device)?;
    Ok((cos, sin))
}

/// Causal mask: [T, T] with 0 on/below diagonal, -inf above.
fn candle_causal_mask(t: usize, device: &CDevice) -> CResult<CTensor> {
    let mut data = vec![0.0_f32; t * t];
    for row in 0..t {
        for col in (row + 1)..t {
            data[row * t + col] = -1e9;
        }
    }
    CTensor::from_vec(data, (t, t), device)
}

/// Apply rotary embeddings to [B, T, H, D] tensor.
fn candle_apply_rotary(x: &CTensor, cos: &CTensor, sin: &CTensor) -> CResult<CTensor> {
    let (_b, _t, _h, d) = x.dims4()?;
    let half = d / 2;

    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;

    let rotated_x1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let rotated_x2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;

    CTensor::cat(&[rotated_x1, rotated_x2], 3)
}
