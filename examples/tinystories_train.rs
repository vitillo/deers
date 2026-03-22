//! Train a small GPT on TinyStories.
//!
//! Run:
//!   cargo run --release --example tinystories_train -- [--device cpu|mps]

use std::path::Path;
use std::time::Instant;
use std::{env, process};

use deers::dataset::TextDataset;
use deers::models::gpt::{GPT, GPTConfig};
use deers::optim::{AdamWConfig, LrSchedule, WarmupWarmdown};
use deers::tokenizer::Tokenizer;
use deers::{Device, Tensor, loss};

fn main() {
    let device = parse_device_arg();

    // --- Tokenizer ---
    let t0 = Instant::now();
    let tokenizer = Tokenizer::gpt2();
    println!(
        "Tokenizer: gpt2 (vocab_size={}) [{:.1}s]",
        tokenizer.vocab_size(),
        t0.elapsed().as_secs_f64()
    );

    // --- Dataset ---
    // Use the tiny slice for fast iteration; swap to the full file once perf is sorted.
    let seq_len = 256;
    let path = Path::new("data/TinyStoriesV2-GPT4-tiny.txt");
    println!("Loading {} (seq_len={seq_len})...", path.display());
    let t0 = Instant::now();
    let dataset = TextDataset::load(path, &tokenizer, seq_len).unwrap();
    println!("Loaded {} sequences [{:.1}s]", dataset.len(), t0.elapsed().as_secs_f64());

    // --- Model ---
    let config = GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        sequence_len: seq_len,
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
        mlp_hidden_dim: 128 * 4,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    };
    let t0 = Instant::now();
    let mut model = GPT::new(config);
    model.to_device(device).unwrap();
    let num_params: usize = model.parameters().iter().map(|p| p.layout().size()).sum();
    println!("Model: {num_params} parameters [{:.1}s]", t0.elapsed().as_secs_f64());
    println!("Device: {device:?}\n");

    // --- Training ---
    let base_lr = 3e-4;
    let batch_size = 32;
    let num_batches = dataset.len() / batch_size;
    let epochs = 1;
    let total_steps = epochs * num_batches;

    let schedule = WarmupWarmdown::new(100, total_steps, 0.65, 0.05);
    let mut opt = AdamWConfig::new(base_lr).weight_decay(0.01).build(model.parameters());

    println!("Training: {epochs} epoch(s), batch_size={batch_size}, {num_batches} steps");
    println!("Optimizer: AdamW, lr={base_lr}, wd=0.01\n");

    let train_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches {
            let step = epoch * num_batches + batch_idx;
            let lr = base_lr * schedule.lr_multiplier(step);
            opt.set_lr(lr);

            let step_t0 = Instant::now();

            // Slice batch from dataset and move to device
            let start = batch_idx * batch_size;
            let batch = dataset.data.narrow(0, start, batch_size);
            let inputs = batch.narrow(1, 0, seq_len).to_device(device).unwrap();
            let targets = batch.narrow(1, 1, seq_len).to_device(device).unwrap();

            // Forward
            let logits = model.forward(&inputs).unwrap();
            let vocab_size = tokenizer.vocab_size();
            let logits_flat = logits.reshape(vec![batch_size * seq_len, vocab_size]);
            let targets_flat = targets.reshape(vec![batch_size * seq_len]);
            let batch_loss = loss::cross_entropy(&logits_flat, &targets_flat);
            let loss_val: Vec<f32> = batch_loss.to_vec().unwrap();
            epoch_loss += loss_val[0];

            // Backward + optimizer step
            opt.backward_step(&batch_loss).unwrap();

            let t_step = step_t0.elapsed().as_secs_f64();

            println!(
                "step {:>4}/{total_steps} | loss: {:.4} | lr: {:.2e} | {:.3}s/step",
                step + 1,
                loss_val[0],
                lr,
                t_step,
            );

            if (step + 1).is_multiple_of(10) {
                let sample = generate(&model, &tokenizer, "Once upon a time", 64, device);
                println!("  >> {sample}");
            }
        }

        let avg_loss = epoch_loss / num_batches as f32;
        let elapsed = epoch_start.elapsed().as_secs_f64();
        println!("\nepoch {}/{epochs} | avg_loss: {avg_loss:.4} | {elapsed:.1}s\n", epoch + 1);
    }

    println!("Done in {:.1}s.", train_start.elapsed().as_secs_f64());
}

/// Greedy autoregressive generation: feed the full sequence each step, take argmax.
fn generate(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    device: Device,
) -> String {
    let mut tokens: Vec<i64> = tokenizer.encode(prompt).iter().map(|&t| t as i64).collect();

    for _ in 0..max_tokens {
        let seq_len = tokens.len();
        let input = Tensor::from_vec(tokens.clone(), (1, seq_len), device);
        let logits = model.forward(&input).unwrap(); // [1, T, V]

        // Get logits for the last position
        let last_logits =
            logits.narrow(1, seq_len - 1, 1).reshape(vec![logits.layout().shape()[2]]);
        let probs: Vec<f32> = last_logits.to_vec().unwrap();

        // Argmax
        let next_token =
            probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;

        tokens.push(next_token as i64);
    }

    let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    tokenizer.decode(&u32_tokens)
}

fn parse_device_arg() -> Device {
    let mut args = env::args().skip(1);
    let mut device = Device::Cpu;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                let value = args.next().unwrap_or_else(|| usage("missing value for --device"));
                device = match value.as_str() {
                    "cpu" => Device::Cpu,
                    "mps" => Device::Mps,
                    other => usage(&format!("unsupported device: {other}")),
                };
            }
            "--help" | "-h" => usage(""),
            other => usage(&format!("unexpected argument: {other}")),
        }
    }

    device
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
        eprintln!();
    }
    eprintln!("Usage: cargo run --release --example tinystories_train -- [--device cpu|mps]");
    process::exit(if message.is_empty() { 0 } else { 1 });
}
