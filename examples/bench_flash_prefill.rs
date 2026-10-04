//! Benchmark: materialized vs flash (tiled online-softmax) prefill.
//!
//! Scores prompts of 128, 512, and 2048 tokens through `CausalSelfAttention`
//! twice: once via `forward` (the materialized `T x T` score path) and once via
//! `prefill` (the flash tiled path), reporting wall time and the worst absolute
//! drift between the two outputs. Flash must match within tolerance while never
//! building the score matrix.
//!
//! Run:
//!   cargo run --release --example bench_flash_prefill

use std::time::Instant;

use deers::models::gpt::{CausalSelfAttention, KvCache, precompute_rotary_embeddings};
use deers::nn::ParamStore;
use deers::{DType, Device, Tensor};

const N_EMBD: usize = 64;
const N_Q_HEADS: usize = 8;
const N_KV_HEADS: usize = 2;
const WARMUP: usize = 1;
const ITERATIONS: usize = 5;

fn det_vec(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 13) as f32 * 0.05 - 0.3).collect()
}

fn bench_device(device: Device) {
    let head_dim = N_EMBD / N_Q_HEADS;
    for seq_len in [128, 512, 2048] {
        let attn =
            CausalSelfAttention::new_gqa(ParamStore::new().root(), N_EMBD, N_Q_HEADS, N_KV_HEADS);
        attn.to_device(device).unwrap();
        let x = Tensor::from_vec(det_vec(seq_len * N_EMBD), vec![1, seq_len, N_EMBD], device);
        let (cos, sin) =
            precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, device);

        for _ in 0..WARMUP {
            let _ = attn.forward(&x, &cos, &sin).unwrap();
            let mut cache = KvCache::new();
            let _ = attn.prefill(&x, &cos, &sin, &mut cache).unwrap();
        }
        device.synchronize();

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = attn.forward(&x, &cos, &sin).unwrap();
        }
        device.synchronize();
        let materialized_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            let mut cache = KvCache::new();
            let _ = attn.prefill(&x, &cos, &sin, &mut cache).unwrap();
        }
        device.synchronize();
        let flash_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;

        let full: Vec<f32> = attn.forward(&x, &cos, &sin).unwrap().to_vec().unwrap();
        let mut cache = KvCache::new();
        let prefilled: Vec<f32> =
            attn.prefill(&x, &cos, &sin, &mut cache).unwrap().to_vec().unwrap();
        let worst = full.iter().zip(&prefilled).map(|(a, e)| (a - e).abs()).fold(0.0, f32::max);

        println!(
            "device={device:<4} T={seq_len:<5} materialized={materialized_ms:>9.2}ms flash={flash_ms:>9.2}ms worst-drift={worst:.2e}"
        );
    }
}

fn main() {
    if cfg!(debug_assertions) {
        println!("note: run with --release for meaningful benchmark timings\n");
    }
    bench_device(Device::Cpu);
    if Device::Cuda.is_available() {
        bench_device(Device::Cuda);
    } else {
        println!("cuda unavailable, skipping");
    }
}
