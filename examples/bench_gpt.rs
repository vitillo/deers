//! Benchmark: deers vs candle GPT forward + backward.
//!
//! Builds the same small GPT architecture in both libraries and compares
//! forward-only and forward+backward timings on CPU.
//!
//! Run:
//!   cargo run --release --example bench_gpt
//!   cargo run --release --example bench_gpt -- --profile
//!   cargo run --release --example bench_gpt -- --device mps --profile

use std::env;
use std::process;
use std::time::Instant;

// --- Config shared by both implementations ---

// Matches the TinyStories training model shape.
const VOCAB_SIZE: usize = 50_257;
const SEQ_LEN: usize = 256;
const N_LAYER: usize = 4;
const N_HEAD: usize = 4;
const N_EMBD: usize = 128;
const MLP_HIDDEN: usize = N_EMBD * 4;
const BATCH_SIZE: usize = 4;
const WARMUP: usize = 1;
const ITERATIONS: usize = 3;

fn average_us(iterations: usize, mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iterations {
        f();
    }
    t0.elapsed().as_micros() as f64 / iterations as f64
}

fn average_sample_us(iterations: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut total_us = 0.0;
    for _ in 0..iterations {
        total_us += f();
    }
    total_us / iterations as f64
}

fn main() {
    let options = parse_args();

    println!(
        "GPT benchmark: vocab={VOCAB_SIZE}, seq={SEQ_LEN}, layers={N_LAYER}, heads={N_HEAD}, d={N_EMBD}"
    );
    println!("batch={BATCH_SIZE}, warmup={WARMUP}, iterations={ITERATIONS}\n");
    if cfg!(debug_assertions) {
        println!("note: run with --release for meaningful benchmark timings\n");
    }

    bench_deers(options.device, options.profile);
    bench_candle(options.device);
}

// ---------------------------------------------------------------------------
// deers
// ---------------------------------------------------------------------------

fn bench_deers(device: deers::Device, profile_enabled: bool) {
    use deers::models::gpt::{GPT, GPTConfig};

    if let Err(err) = device.check_available() {
        eprintln!("error: device {device:?} is not available: {err}");
        process::exit(1);
    }
    use deers::{ProfilerConfig, Tensor, loss, profile};

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
    let store = deers::nn::ParamStore::new();
    let mut model = GPT::new(config, store.root());
    model.to_device(device).unwrap();

    let num_params: usize = model.parameters().iter().map(|p| p.layout().size()).sum();

    // Random-ish token ids
    let ids: Vec<i64> = (0..BATCH_SIZE * SEQ_LEN).map(|i| (i % VOCAB_SIZE) as i64).collect();
    let input = Tensor::from_vec(ids.clone(), (BATCH_SIZE, SEQ_LEN), device);
    let target_ids: Vec<i64> =
        (0..BATCH_SIZE * SEQ_LEN).map(|i| ((i + 1) % VOCAB_SIZE) as i64).collect();
    let targets = Tensor::from_vec(target_ids, (BATCH_SIZE * SEQ_LEN,), device);
    let step_loss = || {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape(vec![BATCH_SIZE * SEQ_LEN, VOCAB_SIZE]);
        loss::cross_entropy(&logits_flat, &targets)
    };

    // Warmup
    for _ in 0..WARMUP {
        let l = step_loss();
        let _ = l.backward();
        device.synchronize();
    }

    // Forward only
    let fwd_us = average_us(ITERATIONS, || {
        let _ = step_loss();
        device.synchronize();
    });

    // Forward + backward
    let fwd_bwd_us = average_us(ITERATIONS, || {
        let l = step_loss();
        let _ = l.backward();
        device.synchronize();
    });

    // Backward only, measured directly after building a fresh graph.
    let bwd_us = average_sample_us(ITERATIONS, || {
        let l = step_loss();
        device.synchronize();
        let t0 = Instant::now();
        let _ = l.backward();
        device.synchronize();
        t0.elapsed().as_micros() as f64
    });

    println!("=== deers {device:?} ({num_params} params) ===");
    println!("  forward:          {fwd_us:>10.0} µs");
    println!("  forward+backward: {fwd_bwd_us:>10.0} µs");
    println!("  backward only:    {bwd_us:>10.0} µs\n");

    if !profile_enabled {
        return;
    }

    let profile_config = ProfilerConfig::default().record_shapes(true).profile_memory(true);
    let (_, forward_prof) = profile(profile_config, || {
        let _ = step_loss();
    });

    let loss = step_loss();
    let (_, backward_prof) = profile(profile_config, || {
        let _ = loss.backward();
    });

    println!("--- deers profile ({device:?}, forward, one step) ---");
    println!("{}", forward_prof.table());
    println!();
    println!("--- deers profile ({device:?}, backward, one step) ---");
    println!("{}", backward_prof.table());
    println!();
}

#[derive(Clone, Copy)]
struct Options {
    device: deers::Device,
    profile: bool,
}

fn parse_args() -> Options {
    let mut args = env::args().skip(1);
    let mut device = deers::Device::Cpu;
    let mut profile = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                let value = args.next().unwrap_or_else(|| usage("missing value for --device"));
                device = match value.as_str() {
                    "cpu" => deers::Device::Cpu,
                    "cuda" => deers::Device::Cuda,
                    "mps" => deers::Device::Mps,
                    other => usage(&format!("unsupported device: {other}")),
                };
            }
            "--profile" => profile = true,
            "--help" | "-h" => usage(""),
            other => usage(&format!("unexpected argument: {other}")),
        }
    }

    Options { device, profile }
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
        eprintln!();
    }
    eprintln!(
        "Usage: cargo run --release --example bench_gpt -- [--device cpu|cuda|mps] [--profile]"
    );
    process::exit(if message.is_empty() { 0 } else { 1 });
}

// ---------------------------------------------------------------------------
// candle — same architecture, built from primitives
// ---------------------------------------------------------------------------

fn bench_candle(device: deers::Device) {
    use candle_core::{DType, Device as CDevice, Tensor as CTensor};
    use candle_nn::{VarBuilder, VarMap};

    let (device, device_name) = match device {
        deers::Device::Cpu => (CDevice::Cpu, "cpu"),
        deers::Device::Mps => {
            // bench_deers already exited if MPS is unavailable, so this is
            // only reached on macOS.
            #[cfg(not(target_os = "macos"))]
            unreachable!("MPS device checked before bench_candle is called");
            #[cfg(target_os = "macos")]
            match CDevice::new_metal(0) {
                Ok(d) => (d, "metal"),
                Err(err) => {
                    println!("=== candle skipped ===");
                    println!("  failed to initialize candle metal device: {err}\n");
                    return;
                }
            }
        }
        deers::Device::Cuda => {
            #[cfg(not(target_os = "linux"))]
            {
                println!("=== candle skipped ===");
                println!("  candle cuda benchmark is only wired on Linux in this example\n");
                return;
            }
            #[cfg(target_os = "linux")]
            match CDevice::new_cuda(0) {
                Ok(d) => (d, "cuda"),
                Err(err) => {
                    println!("=== candle skipped ===");
                    println!("  failed to initialize candle cuda device: {err}\n");
                    return;
                }
            }
        }
    };
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
    let step_loss = || {
        let logits = model.forward(&input).unwrap();
        let logits_flat = logits.reshape((BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)).unwrap();
        candle_nn::loss::cross_entropy(&logits_flat, &targets).unwrap()
    };

    // Warmup
    for _ in 0..WARMUP {
        let l = step_loss();
        let _ = l.backward();
        device.synchronize().unwrap();
    }

    // Forward only
    let fwd_us = average_us(ITERATIONS, || {
        let _ = step_loss();
        device.synchronize().unwrap();
    });

    // Forward + backward
    let fwd_bwd_us = average_us(ITERATIONS, || {
        let l = step_loss();
        let _ = l.backward();
        device.synchronize().unwrap();
    });

    // Backward only, measured directly after building a fresh graph.
    let bwd_us = average_sample_us(ITERATIONS, || {
        let l = step_loss();
        device.synchronize().unwrap();
        let t0 = Instant::now();
        let _ = l.backward();
        device.synchronize().unwrap();
        t0.elapsed().as_micros() as f64
    });

    println!("=== candle {device_name} ({num_params} params) ===");
    println!("  forward:          {fwd_us:>10.0} µs");
    println!("  forward+backward: {fwd_bwd_us:>10.0} µs");
    println!("  backward only:    {bwd_us:>10.0} µs\n");
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
