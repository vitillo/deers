//! Run the real Qwen3-0.6B weights end to end on CPU.
//!
//! Downloads `model.safetensors` from Hugging Face once into
//! `~/.cache/deers/qwen3-0.6B/`, loads it into the assembled model, scores a
//! chat prompt for perplexity plus per-position top predictions, then
//! generates real text with the sampler over the KV-cache decode loop.
//!
//! Run:
//!   cargo run --release --example qwen3_run

use std::fs::File;
use std::path::PathBuf;

use deers::models::gpt::{KvCache, Qwen3, Qwen3Config};
use deers::nn::ParamStore;
use deers::sample::SamplingConfig;
use deers::tokenizer::{ChatMessage, Qwen3Tokenizer, Tokenizer};
use deers::{DType, Device, Tensor, no_grad};
use half::bf16;

const WEIGHT_URL: &str = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors";
const GENERATE_TOKENS: usize = 100;

fn main() {
    let weight_path = weight_file();
    download_once(&weight_path);

    eprintln!("building Qwen3-0.6B on CPU");
    let store = ParamStore::new();
    let mut model = Qwen3::new(Qwen3Config::qwen3_06b(), store.root());
    model.to_dtype(DType::BF16).expect("BF16 conversion must succeed");
    store
        .load_sharded(weight_path.parent().expect("weight file has a parent"), Device::Cpu)
        .expect("real checkpoint must assign with zero validation errors");
    eprintln!("loaded {} tensors", store.named_parameters().len());

    let tokenizer = Qwen3Tokenizer::new();
    let prompt = Qwen3Tokenizer::apply_chat_template(
        &[ChatMessage { role: "user", content: "Explain why the sky is blue in one sentence." }],
        true,
    );
    let prompt_ids = tokenizer.encode(&prompt);
    println!("prompt tokens: {}", prompt_ids.len());

    eprintln!("prefill over {} prompt tokens", prompt_ids.len());
    let logits = no_grad(|| {
        let ids: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
        let idx = Tensor::from_vec(ids, (1, prompt_ids.len()), Device::Cpu);
        let mut caches: Vec<KvCache> =
            (0..model.n_layers()).map(|_| KvCache::new()).collect();
        model.prefill(&idx, &mut caches).expect("prefill must succeed")
    });
    let rows = to_f32_rows(&logits, prompt_ids.len(), tokenizer.vocab_size());
    proof_of_life(&tokenizer, &prompt_ids, &rows);

    eprintln!("generating {GENERATE_TOKENS} tokens");
    let config = SamplingConfig { temperature: 0.0, ..SamplingConfig::new() };
    let generated = model.generate(&prompt_ids, GENERATE_TOKENS, &config).expect("generate failed");
    let text = tokenizer.decode(&generated);
    println!("--- generated text ---\n{text}\n--- end ---");
}

fn weight_file() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set");
    PathBuf::from(home).join(".cache/deers/qwen3-0.6B/model.safetensors")
}

fn download_once(path: &PathBuf) {
    if path.exists() {
        eprintln!("weights cached at {}", path.display());
        return;
    }
    std::fs::create_dir_all(path.parent().expect("weight file has a parent"))
        .expect("cache dir must create");
    eprintln!("downloading {WEIGHT_URL}");
    let part_path = path.with_extension("part");
    let response = ureq::get(WEIGHT_URL).call().expect("download failed");
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&part_path).expect("download file must create");
    std::io::copy(&mut reader, &mut file).expect("download must stream to disk");
    std::fs::rename(&part_path, path).expect("downloaded file must rename");
    eprintln!("saved to {}", path.display());
}

fn to_f32_rows(logits: &Tensor, seq_len: usize, vocab: usize) -> Vec<Vec<f32>> {
    let flat: Vec<f32> =
        logits.to_vec::<bf16>().expect("logits read").iter().map(|v| v.to_f32()).collect();
    flat.chunks_exact(vocab).take(seq_len).map(<[f32]>::to_vec).collect()
}

fn proof_of_life(tokenizer: &Qwen3Tokenizer, prompt_ids: &[u32], rows: &[Vec<f32>]) {
    let mut nll_sum = 0.0;
    let mut count = 0;
    for (pos, row) in rows.iter().enumerate() {
        let top = row.iter().enumerate().fold(0, |best, (i, &v)| if v > row[best] { i } else { best });
        let piece = tokenizer.decode_lossy(&[top as u32]);
        println!("pos {pos}: top-1 {top} {piece:?}");
        if pos + 1 < prompt_ids.len() {
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
            let log_prob = (row[prompt_ids[pos + 1] as usize] - max) - sum.ln();
            nll_sum -= log_prob as f64;
            count += 1;
        }
    }
    println!("prompt perplexity: {:.3}", (nll_sum / count as f64).exp());
}
