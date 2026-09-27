//! Benchmark: full-recompute generation versus prefill plus cached decode.
//!
//! Times one attention block generating tokens two ways on CPU. The
//! full-recompute path rescores the whole prefix for every new token. The
//! cached path prefills the prompt once into a `KvCache` and then decodes
//! one token at a time. Results print as wall times and per-token times
//! at several prompt lengths, with the speedup and the scaling trend.
//!
//! Run:
//!   cargo run --release --example bench_kvcache

use std::time::Instant;

// Matches the TinyStories benchmark model width in `bench_gpt.rs`.
const N_EMBD: usize = 128;
const N_HEAD: usize = 4;
const PROMPT_LENS: [usize; 3] = [64, 256, 1024];
const GEN_TOKENS: usize = 32;
const WARMUP: usize = 1;
const ITERATIONS: usize = 3;

fn main() {
    use deers::Device;

    println!(
        "KV-cache benchmark: d={N_EMBD}, heads={N_HEAD}, generate={GEN_TOKENS} tokens, \
         warmup={WARMUP}, iterations={ITERATIONS}"
    );
    println!("device={:?}\n", Device::Cpu);
    if cfg!(debug_assertions) {
        println!("note: run with --release for meaningful benchmark timings\n");
    }

    let mut speedups = Vec::new();
    for &prompt_len in &PROMPT_LENS {
        speedups.push(bench_prompt(prompt_len));
    }

    println!("--- scaling trend (cached total vs full recompute) ---");
    for (&prompt_len, &speedup) in PROMPT_LENS.iter().zip(speedups.iter()) {
        println!("  prompt {prompt_len:>5}: {speedup:.2}x");
    }
    match PROMPT_LENS.iter().zip(speedups.iter()).find(|&(_, &s)| s > 1.0) {
        Some((&prompt_len, _)) => {
            println!("cached generation wins from prompt {prompt_len} upward");
        }
        None => {
            println!("cached generation did not beat full recomputation at these prompt lengths");
        }
    }
}

/// Times both generation paths for one prompt length. Returns the speedup
/// of the cached total over the full-recompute total.
fn bench_prompt(prompt_len: usize) -> f64 {
    use deers::models::gpt::{CausalSelfAttention, KvCache, precompute_rotary_embeddings};
    use deers::{DType, Device, Tensor};

    let device = Device::Cpu;
    let total_len = prompt_len + GEN_TOKENS;
    let head_dim = N_EMBD / N_HEAD;

    let store = deers::nn::ParamStore::new();
    let attn = CausalSelfAttention::new(store.root(), N_EMBD, N_HEAD);
    let x = Tensor::randn((1, total_len, N_EMBD), DType::F32, device);
    let (cos, sin) =
        precompute_rotary_embeddings(total_len, head_dim, 10_000.0, DType::F32, device);

    // Rescoring the growing prefix mirrors naive generation, where token
    // `t` pays attention over all `prompt_len + t` past tokens.
    let mut full_step = || {
        for step in 0..GEN_TOKENS {
            let len = prompt_len + step + 1;
            let prefix = x.narrow(1, 0, len);
            let _ = attn.forward(&prefix, &cos.narrow(1, 0, len), &sin.narrow(1, 0, len)).unwrap();
            device.synchronize();
        }
    };
    // Prefill stays inside the cached total so the comparison is honest.
    let mut cached_step = || {
        let mut cache = KvCache::new();
        let prompt = x.narrow(1, 0, prompt_len);
        let _ = attn
            .prefill(
                &prompt,
                &cos.narrow(1, 0, prompt_len),
                &sin.narrow(1, 0, prompt_len),
                &mut cache,
            )
            .unwrap();
        device.synchronize();
        for step in 0..GEN_TOKENS {
            let pos = prompt_len + step;
            let token = x.narrow(1, pos, 1);
            let _ = attn
                .decode(&token, &cos.narrow(1, pos, 1), &sin.narrow(1, pos, 1), &mut cache)
                .unwrap();
            device.synchronize();
        }
    };

    for _ in 0..WARMUP {
        full_step();
        cached_step();
    }
    let full_ms = average_ms(ITERATIONS, &mut full_step);
    let cached_ms = average_ms(ITERATIONS, &mut cached_step);
    let speedup = full_ms / cached_ms;

    println!("=== prompt={prompt_len}, generate={GEN_TOKENS} ===");
    println!(
        "  full recompute: {full_ms:>10.1} ms total ({:>8.2} ms/token)",
        full_ms / GEN_TOKENS as f64
    );
    println!(
        "  cached total:   {cached_ms:>10.1} ms total ({:>8.2} ms/token, speedup {speedup:.2}x)",
        cached_ms / GEN_TOKENS as f64
    );
    println!();
    speedup
}

fn average_ms(iterations: usize, f: &mut impl FnMut()) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iterations {
        f();
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}
