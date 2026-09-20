use std::collections::BTreeMap;
use std::path::PathBuf;

use candle_core::{DType as CDType, Device as CDevice, Tensor as CTensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as CandleQwen3Config, ModelForCausalLM};
use deers::checkpoint::map_qwen_name;
use deers::models::gpt::{Qwen3, Qwen3Config};
use deers::nn::ParamStore;
use deers::tokenizer::{ChatMessage, Qwen3Tokenizer, Tokenizer};
use deers::{DType, Device, Tensor, no_grad};
use half::bf16;

const GREEDY_STEPS: usize = 8;
const LOGIT_TOL: f32 = 1e-2;
/// Cross-backend tolerance for same-dtype CPU/CUDA comparisons.
const ACCEL_TOL: f32 = 2e-3;

fn fixture_dir() -> PathBuf {
    std::env::var("QWEN3_06B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/fixtures/qwen3-0.6b"))
}

fn missing_fixture(dir: &std::path::Path) -> String {
    format!(
        "Qwen3-0.6B weights not found in '{}': download once with \
         `mkdir -p {} && curl -L -o {}/model.safetensors https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors && curl -L -o {}/config.json https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json` (never committed)",
        dir.display(),
        dir.display(),
        dir.display(),
        dir.display()
    )
}

fn chat_prompt() -> (String, Vec<u32>) {
    let tokenizer = Qwen3Tokenizer::new();
    let messages = [
        ChatMessage { role: "system", content: "You are a helpful assistant." },
        ChatMessage { role: "user", content: "What is 2 plus 2?" },
    ];
    let prompt = Qwen3Tokenizer::apply_chat_template(&messages, true);
    let ids = tokenizer.encode(&prompt);
    (prompt, ids)
}

fn needs_transpose(deers_name: &str) -> bool {
    [
        ".attn.q_proj.weight",
        ".attn.k_proj.weight",
        ".attn.v_proj.weight",
        ".attn.out_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ]
    .iter()
    .any(|suffix| deers_name.ends_with(suffix))
}

fn load_deers(dir: &std::path::Path) -> (Qwen3, ParamStore) {
    let bytes = std::fs::read(dir.join("model.safetensors"))
        .unwrap_or_else(|_| panic!("{}", missing_fixture(dir)));
    let store_safe =
        safetensors::SafeTensors::deserialize(&bytes).expect("parse model.safetensors");
    let store = ParamStore::new();
    let model = Qwen3::new(Qwen3Config::qwen3_06b(), store.root());
    let by_name: BTreeMap<String, deers::nn::Parameter> =
        store.named_parameters().into_iter().collect();
    assert_eq!(
        by_name["wte.weight"].id(),
        by_name["lm_head.weight"].id(),
        "tied head must share the embedding tensor id"
    );

    for name in store_safe.names() {
        let deers_name = map_qwen_name(name)
            .unwrap_or_else(|| panic!("unexpected tensor '{name}': no deers mapping"));
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
        let param = by_name
            .get(&deers_name)
            .unwrap_or_else(|| panic!("deers model has no parameter '{deers_name}'"));
        let expected: Vec<usize> = param.layout().shape().iter().copied().collect();
        let flat = if needs_transpose(&deers_name) {
            assert_eq!(shape.len(), 2, "tensor '{name}' must be a 2D projection");
            assert_eq!(
                vec![shape[1], shape[0]],
                expected,
                "transposed shape mismatch for '{deers_name}'"
            );
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

fn candle_config(dir: &std::path::Path) -> CandleQwen3Config {
    let raw = std::fs::read(dir.join("config.json"))
        .unwrap_or_else(|_| panic!("{}", missing_fixture(dir)));
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
    let reference = Qwen3Config::qwen3_06b();
    assert_eq!(cfg.vocab_size, reference.vocab_size);
    assert_eq!(cfg.hidden_size, reference.hidden);
    assert_eq!(cfg.num_hidden_layers, reference.n_layers);
    assert_eq!(cfg.num_attention_heads, reference.n_q_heads);
    assert_eq!(cfg.num_key_value_heads, reference.n_kv_heads);
    assert_eq!(cfg.head_dim, reference.head_dim);
    assert_eq!(cfg.intermediate_size, reference.mlp_hidden_dim);
    assert!((cfg.rms_norm_eps - reference.rms_norm_eps).abs() < 1e-12);
    assert!((cfg.rope_theta - f64::from(reference.rope_base)).abs() < 1e-3);
    assert!(cfg.tie_word_embeddings);
    cfg
}

fn load_candle(dir: &std::path::Path, cfg: &CandleQwen3Config) -> ModelForCausalLM {
    let path = dir.join("model.safetensors");
    assert!(path.exists(), "{}", missing_fixture(dir));
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], CDType::F32, &CDevice::Cpu)
            .expect("mmap Qwen3 safetensors for candle")
    };
    ModelForCausalLM::new(cfg, vb).expect("build candle Qwen3")
}

fn deers_prefill_logits(model: &Qwen3, ids: &[u32]) -> Vec<Vec<f32>> {
    deers_prefill_logits_on(model, ids, Device::Cpu)
}

/// F32 prefill logits as rows with index tensors on `device`, so a moved
/// model scores without tripping its rotary-cache device assertion.
fn deers_prefill_logits_on(model: &Qwen3, ids: &[u32], device: Device) -> Vec<Vec<f32>> {
    no_grad(|| {
        let flat: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let idx = Tensor::from_vec(flat, (1, ids.len()), device);
        let logits = model.forward(&idx).expect("deers forward");
        let vocab = logits.layout().shape()[2];
        (0..ids.len())
            .map(|pos| {
                logits
                    .narrow(1, pos, 1)
                    .reshape(vec![vocab])
                    .to_vec::<f32>()
                    .expect("read deers logits")
            })
            .collect()
    })
}

fn candle_prefix_logits(model: &mut ModelForCausalLM, ids: &[u32]) -> Vec<Vec<f32>> {
    let device = CDevice::Cpu;
    (1..=ids.len())
        .map(|len| {
            model.clear_kv_cache();
            let prefix = &ids[..len];
            let input = CTensor::new(prefix, &device).unwrap().unsqueeze(0).unwrap();
            model
                .forward(&input, 0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(CDType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .collect()
}

fn top1(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(id, _)| id as u32)
        .expect("non-empty logits")
}

/// BF16 prefill logits as F32 rows on `device`: the same prompt scores on
/// CPU and CUDA from identical weights, so cross-backend diffs isolate the
/// accelerator kernels from weight conversion.
fn bf16_prefill_logits(model: &Qwen3, ids: &[u32], device: Device) -> Vec<Vec<f32>> {
    no_grad(|| {
        let flat: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let idx = Tensor::from_vec(flat, (1, ids.len()), device);
        let logits = model.forward(&idx).expect("deers bf16 forward");
        assert_eq!(logits.dtype(), DType::BF16);
        let vocab = logits.layout().shape()[2];
        (0..ids.len())
            .map(|pos| {
                logits
                    .narrow(1, pos, 1)
                    .reshape(vec![vocab])
                    .to_vec::<bf16>()
                    .expect("read deers bf16 logits")
                    .iter()
                    .map(|v| v.to_f32())
                    .collect()
            })
            .collect()
    })
}

fn max_abs_diff(rows_a: &[Vec<f32>], rows_b: &[Vec<f32>]) -> f32 {
    rows_a
        .iter()
        .zip(rows_b.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max)
}

fn greedy_deers(model: &Qwen3, prompt: &[u32]) -> Vec<u32> {
    no_grad(|| {
        let mut ids = prompt.to_vec();
        for _ in 0..GREEDY_STEPS {
            let flat: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
            let idx = Tensor::from_vec(flat, (1, ids.len()), Device::Cpu);
            let logits = model.forward(&idx).expect("deers greedy forward");
            let vocab = logits.layout().shape()[2];
            let last = logits
                .narrow(1, ids.len() - 1, 1)
                .reshape(vec![vocab])
                .to_vec::<f32>()
                .expect("read deers last logits");
            ids.push(top1(&last));
        }
        ids[prompt.len()..].to_vec()
    })
}

fn greedy_candle(model: &mut ModelForCausalLM, prompt: &[u32]) -> Vec<u32> {
    let device = CDevice::Cpu;
    model.clear_kv_cache();
    let input = CTensor::new(prompt, &device).unwrap().unsqueeze(0).unwrap();
    let mut next = top1(
        &model
            .forward(&input, 0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(CDType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
    );
    let mut out = vec![next];
    for _ in 1..GREEDY_STEPS {
        let input = CTensor::new(&[next], &device).unwrap().unsqueeze(0).unwrap();
        let logits = model
            .forward(&input, prompt.len() + out.len() - 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(CDType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        next = top1(&logits);
        out.push(next);
    }
    out
}

#[ignore]
#[test]
fn qwen3_06b_candle_parity_prefill_and_greedy() {
    // Arrange: the same published weights and the same chat prompt on both sides.
    let dir = fixture_dir();
    assert!(dir.join("model.safetensors").exists(), "{}", missing_fixture(&dir));
    let (prompt, ids) = chat_prompt();
    let (deers_model, _) = load_deers(&dir);
    let cfg = candle_config(&dir);
    let mut candle_model = load_candle(&dir, &cfg);

    // Act
    let deers_logits = deers_prefill_logits(&deers_model, &ids);
    let candle_logits = candle_prefix_logits(&mut candle_model, &ids);
    let deers_greedy = greedy_deers(&deers_model, &ids);
    let candle_greedy = greedy_candle(&mut candle_model, &ids);

    // Assert: per-position top predictions agree and logits sit within tolerance.
    // NOTE (differential outcome, 2026-09-20): this PASSES. The RoPE sin-sign
    // divergence it used to document was fixed upstream: shared
    // `apply_rotary_emb` now computes the HF/candle direction
    // (y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos), so deers F32/CPU and
    // candle F32/CPU agree to max_abs_diff ~1e-4 with identical greedy
    // continuations. Divergences are never averaged away: the report below
    // prints first, the assertions second.
    assert_eq!(deers_logits.len(), ids.len());
    assert_eq!(candle_logits.len(), ids.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut count = 0u64;
    let mut top1_deers = Vec::with_capacity(ids.len());
    let mut top1_candle = Vec::with_capacity(ids.len());
    for (d, c) in deers_logits.iter().zip(candle_logits.iter()) {
        top1_deers.push(top1(d));
        top1_candle.push(top1(c));
        for (a, b) in d.iter().zip(c.iter()) {
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            sum_abs += f64::from(diff);
            count += 1;
        }
    }
    let mean_abs = sum_abs / count as f64;
    println!("prompt: {prompt:?}");
    println!("prompt ids: {ids:?}");
    println!("per-position top-1 deers : {top1_deers:?}");
    println!("per-position top-1 candle: {top1_candle:?}");
    println!(
        "positions={} max_abs_diff={max_abs:.6} mean_abs_diff={mean_abs:.6} tol={LOGIT_TOL}",
        ids.len()
    );
    println!("deers greedy: {deers_greedy:?}");
    println!("candle greedy: {candle_greedy:?}");
    println!("first-position top-5 deers: {:?}", top_k(&deers_logits[0], 5));
    println!("first-position top-5 candle: {:?}", top_k(&candle_logits[0], 5));
    for (pos, (&d, &c)) in top1_deers.iter().zip(top1_candle.iter()).enumerate() {
        assert_eq!(d, c, "top-1 diverges at prefill position {pos}");
    }
    assert!(max_abs <= LOGIT_TOL, "prefill logits diverge by {max_abs} over tolerance {LOGIT_TOL}");
    assert_eq!(deers_greedy, candle_greedy, "greedy generations diverge");
}

/// Real-weights CUDA proof: the accelerator backend scores the published
/// checkpoint and must match CPU behavior plus agree with candle.
///
/// Same-dtype F32 logits compare within the accelerator tolerance (CUDA
/// kernels reorder summation but F32 carries no quantization noise). BF16
/// compares at decision level: saturating softmax amplifies quantization
/// noise into O(1) mid-logit diffs, so a top-1 flip is allowed only when both
/// sides score it a near-tie within BF16 noise.
///
/// Needs `model.safetensors` + `config.json` (see `missing_fixture`) and a
/// CUDA device; stays ignored so the CPU-only suite needs neither.
#[ignore]
#[test]
fn qwen3_06b_cuda_matches_cpu_and_candle() {
    // Arrange: one model walks every backend/dtype; candle loads F32/CPU.
    if !Device::Cuda.is_available() {
        return;
    }
    let dir = fixture_dir();
    assert!(dir.join("model.safetensors").exists(), "{}", missing_fixture(&dir));
    let (_prompt, ids) = chat_prompt();
    let (mut model, _) = load_deers(&dir);
    let cfg = candle_config(&dir);
    let mut candle_model = load_candle(&dir, &cfg);

    // Act
    let f32_cpu = deers_prefill_logits(&model, &ids);
    model.to_device(Device::Cuda).expect("model must move to CUDA");
    let f32_cuda = deers_prefill_logits_on(&model, &ids, Device::Cuda);
    model.to_dtype(DType::BF16).expect("BF16 conversion must succeed");
    let bf16_cuda = bf16_prefill_logits(&model, &ids, Device::Cuda);
    model.to_device(Device::Cpu).expect("model must move back to CPU");
    let bf16_cpu = bf16_prefill_logits(&model, &ids, Device::Cpu);
    let candle = candle_prefix_logits(&mut candle_model, &ids);

    // Assert: F32 within the accelerator tolerance with agreeing top-1s.
    let f32_diff = max_abs_diff(&f32_cuda, &f32_cpu);
    println!("f32 cpu-vs-cuda max_abs_diff={f32_diff:.6} tol={ACCEL_TOL}");
    assert!(f32_diff <= ACCEL_TOL, "f32 cuda logits diverge by {f32_diff}");
    for (pos, (a, b)) in f32_cuda.iter().zip(f32_cpu.iter()).enumerate() {
        assert_eq!(top1(a), top1(b), "f32 top-1 diverges at position {pos}");
    }

    // Assert: BF16 decisions agree up to near-tie flips; the absolute diff
    // only prints (see the doc comment for why it cannot hold 2e-3).
    let bf16_diff = max_abs_diff(&bf16_cuda, &bf16_cpu);
    println!("bf16 cpu-vs-cuda max_abs_diff={bf16_diff:.6} (informational)");
    for (pos, (a, b)) in bf16_cuda.iter().zip(bf16_cpu.iter()).enumerate() {
        assert!(tops_agree(a, b, BF16_NOISE), "bf16 cpu/cuda decision diverges at position {pos}");
    }

    // Assert: CPU BF16 agrees with candle F32 under the same near-tie rule,
    // isolating quantization from the backend; then CUDA inherits it.
    let quant_diff = max_abs_diff(&bf16_cpu, &candle);
    println!("bf16 cpu-vs-candle-f32 max_abs_diff={quant_diff:.6} (informational)");
    for (pos, (b, c)) in bf16_cpu.iter().zip(candle.iter()).enumerate() {
        assert!(tops_agree(b, c, BF16_NOISE), "bf16/candle decision diverges at position {pos}");
    }
}

/// BF16 noise scale for decision agreement: observed BF16 top-logit diffs
/// stay under 1.0 while genuine top margins run to tens.
const BF16_NOISE: f32 = 2.0;

/// Returns true when both rows pick the same winner, or both score the flip
/// a near-tie within `noise`.
fn tops_agree(a: &[f32], b: &[f32], noise: f32) -> bool {
    let ta = top1(a) as usize;
    let tb = top1(b) as usize;
    if ta == tb {
        return true;
    }
    (a[ta] - a[tb]).abs() < noise && (b[ta] - b[tb]).abs() < noise
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> =
        logits.iter().enumerate().map(|(id, &v)| (id as u32, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    indexed.truncate(k);
    indexed
}

#[test]
fn qwen3_06b_agreed_outputs_are_pinned() {
    // Arrange: the fixed chat prompt both differential sides share.
    // The live `qwen3_06b_candle_parity_prefill_and_greedy` test now passes
    // (the RoPE sin-sign divergence it documented is fixed), so the only
    // outputs pinned here are the prompt rendering and its token ids.
    // When extending coverage, take agreed greedy ids and spot logits from
    // a green live run.
    let (prompt, ids) = chat_prompt();
    let agreed_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2 plus 2?<|im_end|>\n<|im_start|>assistant\n";
    let agreed_ids: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838,
        374, 220, 17, 5519, 220, 17, 30, 151645, 198, 151644, 77091, 198,
    ];

    // Act
    let tokenizer = Qwen3Tokenizer::new();
    let roundtrip = tokenizer.decode(&ids);

    // Assert: the prompt template, the token ids, and their roundtrip.
    assert_eq!(prompt, agreed_prompt);
    assert_eq!(ids, agreed_ids);
    assert_eq!(roundtrip, agreed_prompt);
    assert!(agreed_ids.iter().all(|&id| id < 151936));
}
