//! Benchmark: Qwen3-0.6B BF16 prefill on CUDA.
//!
//! Loads the published weights, prefills a fixed 2048-token prompt on the
//! RTX 5080, and reports wall time plus tokens per second. The prompt tiles
//! the parity-test chat ids, so ids stay in range and the run is
//! deterministic. Timings bracket `Device::synchronize`, so queued GPU work
//! is included.
//!
//! Weight resolution: `QWEN3_06B_DIR`, then `target/fixtures/qwen3-0.6b`,
//! then `~/.cache/deers/qwen3-0.6B`.
//!
//! Run:
//!   cargo run --release --features cuda --example bench_qwen3_prefill
//!   cargo run --release --features cuda --example bench_qwen3_prefill -- --tokens 512 --iters 5
//!   cargo run --release --features cuda --example bench_qwen3_prefill -- --profile

use std::env;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use deers::models::gpt::{KvCache, Qwen3, Qwen3Config};
use deers::nn::ParamStore;
use deers::tokenizer::{ChatMessage, Qwen3Tokenizer, Tokenizer};
use deers::{DType, Device, Tensor, no_grad};

const DEFAULT_TOKENS: usize = 2048;
const WARMUP: usize = 1;
const ITERATIONS: usize = 3;

fn weight_dir() -> PathBuf {
    if let Ok(dir) = env::var("QWEN3_06B_DIR") {
        return PathBuf::from(dir);
    }
    let local = PathBuf::from("target/fixtures/qwen3-0.6b");
    if local.join("model.safetensors").exists() {
        return local;
    }
    let home = env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join(".cache/deers/qwen3-0.6B")
}

fn prompt_ids(tokens: usize) -> Vec<i64> {
    let tokenizer = Qwen3Tokenizer::new();
    let prompt = Qwen3Tokenizer::apply_chat_template(
        &[ChatMessage { role: "user", content: "Explain why the sky is blue in one sentence." }],
        true,
    );
    let ids = tokenizer.encode(&prompt);
    assert!(!ids.is_empty(), "tokenizer returned no ids");
    (0..tokens).map(|i| i64::from(ids[i % ids.len()])).collect()
}

fn main() {
    let options = parse_args();
    let dir = weight_dir();
    let weights = dir.join("model.safetensors");
    if !weights.exists() {
        eprintln!("Qwen3-0.6B weights not found in '{}'", dir.display());
        eprintln!("download once with:");
        eprintln!(
            "  mkdir -p {0} && curl -L -o {0}/model.safetensors https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors",
            dir.display()
        );
        process::exit(1);
    }

    let device = Device::Cuda;
    if let Err(err) = device.check_available() {
        eprintln!("error: CUDA is not available: {err}");
        process::exit(1);
    }
    if cfg!(debug_assertions) {
        eprintln!("note: run with --release for meaningful benchmark timings");
    }

    eprintln!("building Qwen3-0.6B (BF16 weights) on CUDA");
    let store = ParamStore::new();
    let mut model = Qwen3::new(Qwen3Config::qwen3_06b(), store.root());
    model.to_dtype(DType::BF16).expect("BF16 conversion must succeed");
    model.to_device(device).expect("model must move to CUDA");
    store.load_sharded(&dir, device).expect("checkpoint must assign with zero validation errors");
    if options.f32_reference {
        // Compare arithmetic on the same rounded checkpoint weights, not a new model.
        model.to_dtype(DType::F32).expect("checkpoint weights must cast to f32");
    }
    let dtype = if options.f32_reference { DType::F32 } else { DType::BF16 };
    eprintln!("loaded {} tensors", store.named_parameters().len());

    let ids = prompt_ids(options.tokens);
    let idx = Tensor::from_vec(ids, (1, options.tokens), device);
    let n_layers = model.n_layers();

    let one_prefill = || {
        let mut caches: Vec<KvCache> = (0..n_layers).map(|_| KvCache::new()).collect();
        no_grad(|| model.prefill(&idx, &mut caches).expect("prefill must succeed"))
    };

    for _ in 0..options.warmup {
        let _ = one_prefill();
        device.synchronize();
    }

    let mut times_ms = Vec::with_capacity(options.iters);
    for _ in 0..options.iters {
        let t0 = Instant::now();
        let logits = one_prefill();
        device.synchronize();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(logits.layout().shape()[1], options.tokens);
        times_ms.push(ms);
    }
    times_ms.sort_by(|a, b| a.total_cmp(b));
    let median = times_ms[times_ms.len() / 2];
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let tok_per_s = options.tokens as f64 / (median / 1000.0);

    println!("Qwen3-0.6B {dtype} prefill: {} tokens on {:?}", options.tokens, device);
    println!(
        "  iters:   {iterations} (+{warmup} warmup)",
        iterations = options.iters,
        warmup = options.warmup
    );
    println!("  median:  {median:.1} ms ({tok_per_s:.1} tok/s)");
    println!("  mean:    {mean:.1} ms");
    println!("  all:     {times_ms:.1?} ms");

    // Read a few final-token logits outside the timed section to verify that
    // both backends computed the same prefill rather than only its shape.
    let logits = one_prefill();
    let sample = logits.compact().narrow(1, options.tokens - 1, 1).compact().narrow(2, 0, 8);
    let sample: Vec<f32> = match dtype {
        DType::BF16 => sample
            .to_vec::<half::bf16>()
            .expect("sample logits must be readable")
            .iter()
            .map(|v| v.to_f32())
            .collect(),
        DType::F32 => sample.to_vec::<f32>().expect("sample logits must be readable"),
        _ => unreachable!("benchmark uses BF16 or F32"),
    };
    println!("  final-token logits[0..8]: {sample:?}");

    if !options.profile {
        return;
    }
    let (_, forward_prof) = deers::profile(
        deers::ProfilerConfig::default().record_shapes(true).profile_memory(true),
        || {
            let _ = one_prefill();
        },
    );
    println!();
    println!("--- deers profile (CUDA prefill, one step) ---");
    println!("{}", forward_prof.table());
}

#[derive(Clone, Copy)]
struct Options {
    tokens: usize,
    warmup: usize,
    iters: usize,
    profile: bool,
    f32_reference: bool,
}

fn parse_args() -> Options {
    let mut args = env::args().skip(1);
    let mut options = Options {
        tokens: DEFAULT_TOKENS,
        warmup: WARMUP,
        iters: ITERATIONS,
        profile: false,
        f32_reference: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokens" => {
                options.tokens = args
                    .next()
                    .unwrap_or_else(|| usage("missing value for --tokens"))
                    .parse()
                    .unwrap_or_else(|_| usage("tokens must be a number"));
            }
            "--warmup" => {
                options.warmup = args
                    .next()
                    .unwrap_or_else(|| usage("missing value for --warmup"))
                    .parse()
                    .unwrap_or_else(|_| usage("warmup must be a number"));
            }
            "--iters" => {
                options.iters = args
                    .next()
                    .unwrap_or_else(|| usage("missing value for --iters"))
                    .parse()
                    .unwrap_or_else(|_| usage("iters must be a number"));
            }
            "--profile" => options.profile = true,
            "--f32-reference" => options.f32_reference = true,
            "--help" | "-h" => usage(""),
            other => usage(&format!("unexpected argument: {other}")),
        }
    }
    if options.iters == 0 {
        usage("iters must be at least 1");
    }
    options
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
        eprintln!();
    }
    eprintln!(
        "Usage: cargo run --release --features cuda --example bench_qwen3_prefill [--tokens N] [--warmup N] [--iters N] [--profile] [--f32-reference]"
    );
    process::exit(if message.is_empty() { 0 } else { 1 });
}
