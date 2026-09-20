//! Benchmark: deers vs candle Qwen3-0.6B prefill + greedy decode on real weights.
//!
//! Both sides load the same `model.safetensors`, score the same chat prompt,
//! and run the same workload: one full prefill plus `--decode-steps`
//! single-token decodes. Deers runs in BF16 (the checkpoint dtype) on
//! `--device`; candle runs F32 on CPU — candle-core in this repo builds
//! without its `cuda` feature, whose nvcc kernels do not compile against the
//! CUDA 13.3 toolkit here, so CPU is its only available backend.
//!
//! Run (downloads weights once into `~/.cache/deers/qwen3-0.6B/`):
//!   cargo run --release --example bench_qwen3 -- --device cpu
//!   cargo run --release --features cuda --example bench_qwen3 -- --device cuda

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use deers::models::gpt::{KvCache, Qwen3, Qwen3Config};
use deers::nn::ParamStore;
use deers::tokenizer::{ChatMessage, Qwen3Tokenizer, Tokenizer};
use deers::{DType, Device, Tensor, no_grad};
use half::bf16;

const WEIGHT_URL: &str = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors";
const CONFIG_URL: &str = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json";
const WARMUP: usize = 1;
const DEFAULT_DECODE_STEPS: usize = 32;
const DEFAULT_ITERATIONS: usize = 3;

fn main() {
    let options = parse_args();
    let dir = weight_dir();
    download_once(&dir.join("model.safetensors"), WEIGHT_URL);
    download_once(&dir.join("config.json"), CONFIG_URL);

    let tokenizer = Qwen3Tokenizer::new();
    let prompt = Qwen3Tokenizer::apply_chat_template(
        &[
            ChatMessage { role: "system", content: "You are a helpful assistant." },
            ChatMessage { role: "user", content: "What is 2 plus 2?" },
        ],
        true,
    );
    let prompt_ids = tokenizer.encode(&prompt);

    println!(
        "Qwen3-0.6B benchmark: prompt={} tokens, decode={} tokens, iterations={}",
        prompt_ids.len(),
        options.decode_steps,
        options.iterations
    );
    println!("weights: {}", dir.display());
    if cfg!(debug_assertions) {
        println!("note: run with --release for meaningful benchmark timings\n");
    }

    let deers_first = bench_deers(&options, &prompt_ids);
    let candle_first = bench_candle(&dir, &options, &prompt_ids);
    println!(
        "first-token top-1: deers={deers_first} candle={candle_first} ({})",
        if deers_first == candle_first { "agree" } else { "DIVERGE" }
    );
}

// ---------------------------------------------------------------------------
// deers (BF16, selected device)
// ---------------------------------------------------------------------------

fn bench_deers(options: &Options, prompt_ids: &[u32]) -> u32 {
    if let Err(err) = options.device.check_available() {
        eprintln!("error: device {:?} is not available: {err}", options.device);
        process::exit(1);
    }
    eprintln!("loading Qwen3-0.6B into deers on {:?}", options.device);
    let store = ParamStore::new();
    let mut model = Qwen3::new(Qwen3Config::qwen3_06b(), store.root());
    model.to_dtype(DType::BF16).expect("BF16 conversion must succeed");
    store
        .load_sharded(&weight_dir(), options.device)
        .expect("real checkpoint must assign with zero validation errors");
    // Weights load straight onto the device; the rotary cache still needs
    // the same move so prefill sees one device everywhere.
    model.to_device(options.device).expect("model must move to the bench device");

    let ids: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
    let prompt_len = prompt_ids.len();

    // One timed round: prefill the prompt, then decode step by step.
    let round = |first_token: &mut Option<u32>| {
        let mut caches: Vec<KvCache> = (0..model.n_layers()).map(|_| KvCache::new()).collect();
        let idx = Tensor::from_vec(ids.clone(), (1, prompt_len), options.device);
        let t0 = Instant::now();
        let logits = no_grad(|| model.prefill(&idx, &mut caches).expect("prefill must succeed"));
        options.device.synchronize();
        let prefill_us = t0.elapsed().as_micros() as f64;
        if first_token.is_none() {
            *first_token = Some(top1_last(&logits, prompt_len));
        }

        let t0 = Instant::now();
        let mut next = first_token.unwrap();
        for step in 0..options.decode_steps {
            let token = Tensor::from_vec(vec![i64::from(next)], (1, 1), options.device);
            let logits = no_grad(|| {
                model.decode(&token, prompt_len + step, &mut caches).expect("decode must succeed")
            });
            options.device.synchronize();
            next = top1_last(&logits, 1);
        }
        let decode_us = t0.elapsed().as_micros() as f64;
        (prefill_us, decode_us)
    };

    for _ in 0..WARMUP {
        round(&mut None);
    }
    let mut first_token = None;
    let mut prefill_total = 0.0;
    let mut decode_total = 0.0;
    for _ in 0..options.iterations {
        let (prefill_us, decode_us) = round(&mut first_token);
        prefill_total += prefill_us;
        decode_total += decode_us;
    }
    let prefill_ms = prefill_total / options.iterations as f64 / 1000.0;
    let decode_ms = decode_total / options.iterations as f64 / 1000.0;
    let tok_per_s = options.decode_steps as f64 / (decode_ms / 1000.0);

    println!("=== deers {:?} (bf16) ===", options.device);
    println!("  prefill ({prompt_len} tokens): {prefill_ms:>10.1} ms");
    println!(
        "  decode ({} tokens): {decode_ms:>10.1} ms ({tok_per_s:.1} tok/s)\n",
        options.decode_steps
    );
    first_token.expect("timed rounds must record the first token")
}

fn top1_last(logits: &Tensor, seq_len: usize) -> u32 {
    let vocab = logits.layout().shape()[2];
    let row: Vec<f32> = logits
        .narrow(1, seq_len - 1, 1)
        .reshape(vec![vocab])
        .to_vec::<bf16>()
        .expect("logits read")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(id, _)| id as u32)
        .expect("non-empty logits")
}

// ---------------------------------------------------------------------------
// candle (F32, CPU)
// ---------------------------------------------------------------------------

fn bench_candle(dir: &std::path::Path, options: &Options, prompt_ids: &[u32]) -> u32 {
    use candle_core::{DType as CDType, Device as CDevice, Tensor as CTensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::qwen3::{Config as CandleQwen3Config, ModelForCausalLM};

    let device = CDevice::Cpu;
    let raw = std::fs::read(dir.join("config.json")).expect("config.json must download");
    let v: serde_json::Value = serde_json::from_slice(&raw).expect("parse config.json");
    let get_usize = |key: &str| {
        v.get(key).and_then(serde_json::Value::as_u64).unwrap_or_else(|| panic!("{key} missing"))
            as usize
    };
    let get_f64 = |key: &str| {
        v.get(key).and_then(serde_json::Value::as_f64).unwrap_or_else(|| panic!("{key} missing"))
    };
    let cfg = CandleQwen3Config {
        vocab_size: get_usize("vocab_size"),
        hidden_size: get_usize("hidden_size"),
        intermediate_size: get_usize("intermediate_size"),
        num_hidden_layers: get_usize("num_hidden_layers"),
        num_attention_heads: get_usize("num_attention_heads"),
        head_dim: get_usize("head_dim"),
        attention_bias: v
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        num_key_value_heads: get_usize("num_key_value_heads"),
        max_position_embeddings: get_usize("max_position_embeddings"),
        sliding_window: v
            .get("sliding_window")
            .and_then(serde_json::Value::as_u64)
            .map(|w| w as usize),
        max_window_layers: get_usize("max_window_layers"),
        tie_word_embeddings: v
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        rope_theta: get_f64("rope_theta"),
        rms_norm_eps: get_f64("rms_norm_eps"),
        use_sliding_window: v
            .get("use_sliding_window")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        hidden_act: candle_nn::Activation::Silu,
    };
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], CDType::F32, &device)
            .expect("mmap Qwen3 safetensors for candle")
    };
    let mut model = ModelForCausalLM::new(&cfg, vb).expect("build candle Qwen3");
    let prompt_len = prompt_ids.len();

    let top1 = |logits: &CTensor| {
        logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(CDType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as u32)
            .expect("non-empty logits")
    };

    let mut round = |first_token: &mut Option<u32>| {
        model.clear_kv_cache();
        let input = CTensor::new(prompt_ids, &device).unwrap().unsqueeze(0).unwrap();
        let t0 = Instant::now();
        let logits = model.forward(&input, 0).unwrap();
        let prefill_us = t0.elapsed().as_micros() as f64;
        if first_token.is_none() {
            // Candle's Qwen3 forward returns last-position logits directly.
            *first_token = Some(top1(&logits));
        }

        let t0 = Instant::now();
        let mut next = first_token.unwrap();
        for step in 0..options.decode_steps {
            let input = CTensor::new(&[next], &device).unwrap().unsqueeze(0).unwrap();
            let logits = model.forward(&input, prompt_len + step).unwrap();
            next = top1(&logits);
        }
        let decode_us = t0.elapsed().as_micros() as f64;
        (prefill_us, decode_us)
    };

    for _ in 0..WARMUP {
        round(&mut None);
    }
    let mut first_token = None;
    let mut prefill_total = 0.0;
    let mut decode_total = 0.0;
    for _ in 0..options.iterations {
        let (prefill_us, decode_us) = round(&mut first_token);
        prefill_total += prefill_us;
        decode_total += decode_us;
    }
    let prefill_ms = prefill_total / options.iterations as f64 / 1000.0;
    let decode_ms = decode_total / options.iterations as f64 / 1000.0;
    let tok_per_s = options.decode_steps as f64 / (decode_ms / 1000.0);

    println!("=== candle cpu (f32) ===");
    println!("  prefill ({prompt_len} tokens): {prefill_ms:>10.1} ms");
    println!(
        "  decode ({} tokens): {decode_ms:>10.1} ms ({tok_per_s:.1} tok/s)\n",
        options.decode_steps
    );
    first_token.expect("timed rounds must record the first token")
}

// ---------------------------------------------------------------------------
// cli + weights
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Options {
    device: Device,
    decode_steps: usize,
    iterations: usize,
}

fn parse_args() -> Options {
    let mut args = env::args().skip(1);
    let mut device = Device::Cpu;
    let mut decode_steps = DEFAULT_DECODE_STEPS;
    let mut iterations = DEFAULT_ITERATIONS;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                let value = args.next().unwrap_or_else(|| usage("missing value for --device"));
                device = match value.as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" => Device::Cuda,
                    other => usage(&format!("unsupported device: {other}")),
                };
            }
            "--decode-steps" => {
                let value =
                    args.next().unwrap_or_else(|| usage("missing value for --decode-steps"));
                decode_steps =
                    value.parse().unwrap_or_else(|_| usage("decode steps must be a number"));
            }
            "--iterations" => {
                let value = args.next().unwrap_or_else(|| usage("missing value for --iterations"));
                iterations = value.parse().unwrap_or_else(|_| usage("iterations must be a number"));
            }
            "--help" | "-h" => usage(""),
            other => usage(&format!("unexpected argument: {other}")),
        }
    }

    Options { device, decode_steps, iterations }
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
        eprintln!();
    }
    eprintln!(
        "Usage: cargo run --release --example bench_qwen3 -- [--device cpu|cuda] [--decode-steps N] [--iterations N]"
    );
    process::exit(if message.is_empty() { 0 } else { 1 });
}

fn weight_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join(".cache/deers/qwen3-0.6B")
}

fn download_once(path: &PathBuf, url: &str) {
    if path.exists() {
        return;
    }
    std::fs::create_dir_all(path.parent().expect("weight file has a parent"))
        .expect("cache dir must create");
    eprintln!("downloading {url}");
    let part_path = path.with_extension("part");
    let response = ureq::get(url).call().expect("download failed");
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&part_path).expect("download file must create");
    std::io::copy(&mut reader, &mut file).expect("download must stream to disk");
    std::fs::rename(&part_path, path).expect("downloaded file must rename");
}
