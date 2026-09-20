//! Head-to-head CPU benchmark: deers vs candle on the same Qwen3-0.6B weights.
//!
//! Loads `model.safetensors` once per side (F32 on both), scores the same
//! token ids at several prompt lengths, then decodes greedily. Reports
//! prefill wall time, per-token decode time, and deers/candle ratios, plus a
//! deers op profile over one prefill and a short decode run.
//!
//! Run:
//!   QWEN3_06B_DIR=target/fixtures/qwen3-0.6b cargo run --release --example qwen3_head_to_head

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType as CDType, Device as CDevice, Tensor as CTensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as CandleQwen3Config, ModelForCausalLM};
use deers::checkpoint::map_qwen_name;
use deers::models::gpt::{KvCache, Qwen3, Qwen3Config};
use deers::nn::ParamStore;
use deers::tokenizer::{ChatMessage, Qwen3Tokenizer, Tokenizer};
use deers::{Device, ProfilerConfig, Tensor, no_grad, profile};

const PROMPT_TOKENS: [usize; 3] = [32, 128, 512];
const DECODE_TOKENS: usize = 32;
const PREFILL_REPS: usize = 3;
const PROFILE_PROMPT: usize = 128;
const PROFILE_DECODES: usize = 8;

/// Loads the published BF16 checkpoint as F32, mirroring the parity test
/// harness (tests/qwen3_candle_parity.rs): same values candle's F32
/// VarBuilder sees, with projections transposed into deers layout.
fn load_deers_f32(dir: &std::path::Path) -> (Qwen3, ParamStore) {
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let store_safe = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let store = ParamStore::new();
    let model = Qwen3::new(Qwen3Config::qwen3_06b(), store.root());
    let by_name: BTreeMap<String, deers::nn::Parameter> =
        store.named_parameters().into_iter().collect();
    for name in store_safe.names() {
        let deers_name =
            map_qwen_name(name).unwrap_or_else(|| panic!("no deers mapping for {name}"));
        if deers_name == "lm_head.weight" {
            continue;
        }
        let view = store_safe.tensor(name).expect("read tensor view");
        let shape = view.shape().to_vec();
        let values: Vec<f32> = match view.dtype() {
            safetensors::Dtype::BF16 => view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect(),
            other => panic!("tensor '{name}' has unsupported dtype {other:?}"),
        };
        let param =
            by_name.get(&deers_name).unwrap_or_else(|| panic!("no deers param {deers_name}"));
        let expected: Vec<usize> = param.layout().shape().iter().copied().collect();
        let needs_transpose = [
            ".attn.q_proj.weight",
            ".attn.k_proj.weight",
            ".attn.v_proj.weight",
            ".attn.out_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ]
        .iter()
        .any(|suffix| deers_name.ends_with(suffix));
        let flat = if needs_transpose {
            let (rows, cols) = (shape[0], shape[1]);
            let source = &values;
            (0..cols).flat_map(|c| (0..rows).map(move |r| source[r * cols + c])).collect()
        } else {
            assert_eq!(shape, expected, "shape mismatch for '{deers_name}'");
            values
        };
        param.set(&Tensor::from_vec(flat, expected, Device::Cpu)).expect("assign weight");
    }
    (model, store)
}

fn fixture_dir() -> PathBuf {
    std::env::var("QWEN3_06B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/fixtures/qwen3-0.6b"))
}

fn candle_config(dir: &std::path::Path) -> CandleQwen3Config {
    let raw = std::fs::read(dir.join("config.json")).expect("read config.json");
    let v: serde_json::Value = serde_json::from_slice(&raw).expect("parse config.json");
    let get_usize = |key: &str| {
        v.get(key).and_then(serde_json::Value::as_u64).unwrap_or_else(|| panic!("{key} missing"))
            as usize
    };
    let get_f64 = |key: &str| {
        v.get(key).and_then(serde_json::Value::as_f64).unwrap_or_else(|| panic!("{key} missing"))
    };
    let reference = Qwen3Config::qwen3_06b();
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
    assert_eq!(cfg.vocab_size, reference.vocab_size);
    assert_eq!(cfg.hidden_size, reference.hidden);
    assert_eq!(cfg.num_hidden_layers, reference.n_layers);
    assert_eq!(cfg.num_attention_heads, reference.n_q_heads);
    assert_eq!(cfg.num_key_value_heads, reference.n_kv_heads);
    assert_eq!(cfg.head_dim, reference.head_dim);
    assert_eq!(cfg.intermediate_size, reference.mlp_hidden_dim);
    cfg
}

/// Builds exactly `len` token ids by repeating one chat-templated sentence.
fn prompt_ids(len: usize) -> Vec<u32> {
    let tokenizer = Qwen3Tokenizer::new();
    let message = Qwen3Tokenizer::apply_chat_template(
        &[ChatMessage {
            role: "user",
            content: "Explain why the sky is blue in one short sentence. ",
        }],
        true,
    );
    let unit = tokenizer.encode(&message);
    assert!(!unit.is_empty(), "prompt template must encode to tokens");
    (0..len).map(|i| unit[i % unit.len()]).collect()
}

fn top1(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(id, _)| id as u32)
        .expect("non-empty logits")
}

fn deers_last_top1(logits: &Tensor, pos: usize, vocab: usize) -> u32 {
    let row =
        logits.narrow(1, pos, 1).reshape(vec![vocab]).to_vec::<f32>().expect("read deers logits");
    top1(&row)
}

fn candle_last_top1(logits: &CTensor) -> u32 {
    let row = logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_dtype(CDType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    top1(&row)
}

fn deers_prefill(model: &Qwen3, ids: &[u32]) -> (Vec<KvCache>, Tensor) {
    let flat: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
    let idx = Tensor::from_vec(flat, (1, ids.len()), Device::Cpu);
    let mut caches: Vec<KvCache> = (0..model.n_layers()).map(|_| KvCache::new()).collect();
    let logits = model.prefill(&idx, &mut caches).expect("deers prefill");
    (caches, logits)
}

fn deers_decode(model: &Qwen3, caches: &mut [KvCache], token: u32, pos: usize) -> Tensor {
    let input = Tensor::from_vec(vec![i64::from(token)], (1, 1), Device::Cpu);
    model.decode(&input, pos, caches).expect("deers decode")
}

fn candle_prefill(model: &mut ModelForCausalLM, ids: &[u32]) -> CTensor {
    model.clear_kv_cache();
    let input = CTensor::new(ids, &CDevice::Cpu).unwrap().unsqueeze(0).unwrap();
    model.forward(&input, 0).expect("candle prefill")
}

fn candle_decode(model: &mut ModelForCausalLM, token: u32, pos: usize) -> CTensor {
    let input = CTensor::new(&[token], &CDevice::Cpu).unwrap().unsqueeze(0).unwrap();
    model.forward(&input, pos).expect("candle decode")
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn machine_profile() -> String {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':').map(|(_, name)| name.trim().to_string()))
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let mem_gb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find(|line| line.starts_with("MemTotal")).and_then(|line| {
                line.split_whitespace().nth(1)?.parse::<f64>().ok().map(|kb| kb / 1_048_576.0)
            })
        })
        .unwrap_or(0.0);
    let build = if cfg!(debug_assertions) { "debug (TIMING INVALID)" } else { "release" };
    format!("{cpu} / {threads} threads / {mem_gb:.0} GiB RAM / {build}")
}

fn main() {
    println!("machine: {}", machine_profile());
    if cfg!(debug_assertions) {
        eprintln!("run with --release: debug timings mean nothing here");
        std::process::exit(2);
    }
    let dir = fixture_dir();
    assert!(dir.join("model.safetensors").exists(), "missing weights in {}", dir.display());

    eprintln!("loading deers (F32)");
    let (deers_model, _deers_store) = load_deers_f32(&dir);
    eprintln!("loading candle (F32)");
    let cfg = candle_config(&dir);
    let mut candle_model = unsafe {
        let vb = VarBuilder::from_mmaped_safetensors(
            &[dir.join("model.safetensors")],
            CDType::F32,
            &CDevice::Cpu,
        )
        .expect("mmap weights for candle");
        ModelForCausalLM::new(&cfg, vb).expect("build candle Qwen3")
    };
    let vocab = Qwen3Config::qwen3_06b().vocab_size;

    // Warmup so allocator pages, the gemm thread pool, and caches settle before timing.
    eprintln!("warming up");
    let warm = prompt_ids(PROMPT_TOKENS[0]);
    no_grad(|| {
        let (mut caches, logits) = deers_prefill(&deers_model, &warm);
        let mut next = deers_last_top1(&logits, warm.len() - 1, vocab);
        for step in 0..4 {
            let out = deers_decode(&deers_model, &mut caches, next, warm.len() + step);
            next = deers_last_top1(&out, 0, vocab);
        }
    });
    let logits = candle_prefill(&mut candle_model, &warm);
    let mut next = candle_last_top1(&logits);
    for step in 0..4 {
        let out = candle_decode(&mut candle_model, next, warm.len() + step);
        next = candle_last_top1(&out);
    }

    println!(
        "{:>8} {:>16} {:>16} {:>8} {:>16} {:>16} {:>8}",
        "prompt",
        "deers prefill",
        "candle prefill",
        "ratio",
        "deers tok/s",
        "candle tok/s",
        "ratio"
    );
    for &len in &PROMPT_TOKENS {
        let ids = prompt_ids(len);

        // Agreement guard: same weights and same prompt must pick the same next token.
        // Candle returns last-position logits only, so both sides compare there.
        let (_, deers_logits) = no_grad(|| deers_prefill(&deers_model, &ids));
        let candle_logits = candle_prefill(&mut candle_model, &ids);
        let agree =
            deers_last_top1(&deers_logits, len - 1, vocab) == candle_last_top1(&candle_logits);
        if !agree {
            println!("warning: first-position top-1 disagrees at prompt {len}; timing still valid");
        }

        let mut prefill_deers = Vec::with_capacity(PREFILL_REPS);
        let mut prefill_candle = Vec::with_capacity(PREFILL_REPS);
        for _ in 0..PREFILL_REPS {
            let start = Instant::now();
            let (caches, _) = no_grad(|| deers_prefill(&deers_model, &ids));
            drop(caches);
            prefill_deers.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            candle_prefill(&mut candle_model, &ids);
            prefill_candle.push(start.elapsed().as_secs_f64());
        }
        let prefill_deers = median(&mut prefill_deers);
        let prefill_candle = median(&mut prefill_candle);

        let mut decode_deers = Vec::with_capacity(2);
        let mut decode_candle = Vec::with_capacity(2);
        for _ in 0..2 {
            let (mut caches, logits) = no_grad(|| deers_prefill(&deers_model, &ids));
            let mut next = no_grad(|| deers_last_top1(&logits, len - 1, vocab));
            let start = Instant::now();
            no_grad(|| {
                for step in 0..DECODE_TOKENS {
                    let out = deers_decode(&deers_model, &mut caches, next, len + step);
                    next = deers_last_top1(&out, 0, vocab);
                }
            });
            decode_deers.push(start.elapsed().as_secs_f64() / DECODE_TOKENS as f64);

            let logits = candle_prefill(&mut candle_model, &ids);
            let mut next = candle_last_top1(&logits);
            let start = Instant::now();
            for step in 0..DECODE_TOKENS {
                let out = candle_decode(&mut candle_model, next, len + step);
                next = candle_last_top1(&out);
            }
            decode_candle.push(start.elapsed().as_secs_f64() / DECODE_TOKENS as f64);
        }
        let per_tok_deers = median(&mut decode_deers);
        let per_tok_candle = median(&mut decode_candle);
        println!(
            "{len:>8} {prefill_deers:>15.3}s {prefill_candle:>15.3}s {:>7.2}x {:>14.1}/s {:>14.1}/s {:>7.2}x",
            prefill_deers / prefill_candle,
            1.0 / per_tok_deers,
            1.0 / per_tok_candle,
            per_tok_deers / per_tok_candle,
        );
    }

    // Deers op profile: one prefill plus a short greedy decode run.
    let ids = prompt_ids(PROFILE_PROMPT);
    let config = ProfilerConfig::default().record_shapes(true).profile_memory(true);
    let (_, prefill_prof) = profile(config, || {
        no_grad(|| {
            let (mut caches, logits) = deers_prefill(&deers_model, &ids);
            let mut next = deers_last_top1(&logits, ids.len() - 1, vocab);
            let _ = (&mut caches, &mut next);
        })
    });
    println!("--- deers profile (prefill, {PROFILE_PROMPT} tokens) ---");
    println!("{}", prefill_prof.table());
    let (_, decode_prof) = profile(config, || {
        no_grad(|| {
            let (mut caches, logits) = deers_prefill(&deers_model, &ids);
            let mut next = deers_last_top1(&logits, ids.len() - 1, vocab);
            for step in 0..PROFILE_DECODES {
                let out = deers_decode(&deers_model, &mut caches, next, ids.len() + step);
                next = deers_last_top1(&out, 0, vocab);
            }
        })
    });
    println!("--- deers profile (prefill + {PROFILE_DECODES} decodes) ---");
    println!("{}", decode_prof.table());

    // BF16 probe: the shipped qwen3_run path runs BF16, where CPU matmuls
    // funnel through F32 (see src/storage/cpu.rs). Time the same 128-token
    // prefill after an exact F32->BF16 round trip of the weights.
    let mut bf16_model = deers_model;
    bf16_model.to_dtype(deers::DType::BF16).expect("convert to BF16");
    let ids = prompt_ids(PROFILE_PROMPT);
    let mut bf16_times = Vec::with_capacity(2);
    for _ in 0..2 {
        let start = Instant::now();
        let (caches, _) = no_grad(|| deers_prefill(&bf16_model, &ids));
        drop(caches);
        bf16_times.push(start.elapsed().as_secs_f64());
    }
    println!(
        "deers BF16 prefill at {PROFILE_PROMPT} tokens: {:.3}s (F32 median in the table above)",
        median(&mut bf16_times)
    );
}
