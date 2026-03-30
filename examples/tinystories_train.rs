//! Train a TinyStories GPT on CPU, CUDA, or MPS.
//!
//! Run:
//!   cargo run --release --example tinystories_train -- --device mps
//!   cargo run --release --example tinystories_train -- --device mps --prepare-only
//!   cargo run --release --features cuda --example tinystories_train -- --device cuda

use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use deers::checkpoint;
use deers::dataset::TokenBinDataset;
use deers::models::gpt::{GPT, GPTConfig};
use deers::nn::{ParamStore, Parameter};
use deers::optim::{AdamW, AdamWConfig, LrSchedule, WarmupWarmdown, clip_grad_norm};
use deers::tokenizer::Tokenizer;
use deers::tokenizer::{TokenBinPaths, prepare_text_token_bins};
use deers::{Device, GradientStore, Tensor, loss};

const DEFAULT_TEXT_PATH: &str = "data/TinyStoriesV2-GPT4-train.txt";
const DEFAULT_BIN_DIR: &str = "data/tinystories_gpt2";
const DEFAULT_OUT_DIR: &str = "out/tinystories";
const DEFAULT_PROMPT: &str = "Once upon a time";
const DEFAULT_VAL_RATIO: f32 = 0.01;
const DEFAULT_SAMPLE_TEMPERATURE: f32 = 0.8;
const DEFAULT_SAMPLE_TOP_K: usize = 50;
const MODEL_FILE: &str = "model.safetensors";
const OPTIMIZER_FILE: &str = "optimizer.safetensors";

fn main() {
    let options = parse_args();
    if !options.device.is_available() {
        match options.device {
            Device::Cuda => {
                panic!(
                    "device Cuda is not available in this process. Re-run with `--features cuda`: cargo run --release --features cuda --example tinystories_train -- --device cuda"
                );
            }
            _ => panic!("device {:?} is not available in this process", options.device),
        }
    }

    let tokenizer = Tokenizer::gpt2();
    println!("Tokenizer: gpt2 (vocab_size={})", tokenizer.vocab_size());

    let token_paths = resolve_token_bins(&options, &tokenizer);
    println!("Token bins: train={} val={}", token_paths.train.display(), token_paths.val.display());

    if options.prepare_only {
        return;
    }

    let train_dataset = TokenBinDataset::load(&token_paths.train, options.seq_len).unwrap();
    let val_dataset = TokenBinDataset::load(&token_paths.val, options.seq_len).unwrap();
    println!(
        "Loaded token bins: train={} tokens, val={} tokens, seq_len={}",
        train_dataset.num_tokens(),
        val_dataset.num_tokens(),
        options.seq_len
    );

    let config = options.gpt_config(tokenizer.vocab_size());
    let store = ParamStore::new();
    let mut model = GPT::new(config.clone(), store.root());
    model.to_device(options.device).unwrap();

    let parameters = store.parameters();
    let num_params: usize = parameters.iter().map(|parameter| parameter.layout().size()).sum();
    println!("Model: {num_params} parameters on {:?}", options.device);

    fs::create_dir_all(&options.out_dir).unwrap();
    write_run_config(&options, &config, tokenizer.vocab_size(), num_params);

    let schedule = WarmupWarmdown::new(
        options.warmup_steps,
        options.max_steps,
        options.warmdown_ratio,
        options.final_lr_frac,
    );
    let mut opt = AdamWConfig::new(options.lr)
        .betas(options.betas)
        .weight_decay(options.weight_decay)
        .build(parameters.clone());
    let start_step = if let Some(resume) = &options.resume {
        let step = load_checkpoint(&store, &mut opt, &config, options.device, resume);
        println!("Resumed from {} at step {}", resume.display(), step);
        step
    } else {
        0
    };

    println!(
        "Training: start_step={}, target_steps={}, micro_batch={}, grad_accum={}, eval_every={}, save_every={}, sample_every={}",
        start_step,
        options.max_steps,
        options.micro_batch_size,
        options.grad_accum_steps,
        options.eval_every,
        options.save_every,
        options.sample_every
    );

    if options.eval_every > 0 {
        let val_loss = evaluate(&model, &val_dataset, tokenizer.vocab_size(), &options);
        println!(
            "step {:>5}/{:>5} | initial val_loss {:.4}",
            start_step, options.max_steps, val_loss
        );
    }

    if options.sample_every > 0 {
        let sample = generate(
            &model,
            &tokenizer,
            &options.prompt,
            64,
            options.device,
            options.seq_len,
            options.sample_temperature,
            options.sample_top_k,
        );
        println!("step {:>5}/{:>5} | initial sample: {sample}", start_step, options.max_steps);
    }

    let train_start = Instant::now();
    for step in start_step..options.max_steps {
        let step_start = Instant::now();
        let lr = options.lr * schedule.lr_multiplier(step);
        opt.set_lr(lr);

        let (train_loss, grad_norm) = train_step(
            &model,
            &parameters,
            &mut opt,
            &train_dataset,
            tokenizer.vocab_size(),
            &options,
        );
        options.device.synchronize();
        let step_secs = step_start.elapsed().as_secs_f64();

        let step_num = step + 1;
        if step_num.is_multiple_of(options.log_every) || step_num == 1 {
            let tokens_per_step =
                options.micro_batch_size * options.seq_len * options.grad_accum_steps;
            let tokens_per_sec = tokens_per_step as f64 / step_secs.max(1e-9);
            println!(
                "step {:>5}/{:>5} | train_loss {:.4} | grad_norm {:.3} | lr {:.2e} | {:.3}s | {:.0} tok/s",
                step_num, options.max_steps, train_loss, grad_norm, lr, step_secs, tokens_per_sec,
            );
        }

        let val_loss = if options.eval_every > 0 && step_num.is_multiple_of(options.eval_every) {
            let val_loss = evaluate(&model, &val_dataset, tokenizer.vocab_size(), &options);
            println!("  val_loss {:.4}", val_loss);
            Some(val_loss)
        } else {
            None
        };

        if should_save_checkpoint(step_num, start_step, options.save_every) {
            save_checkpoint(&store, &config, &options, &opt, step_num, train_loss, val_loss);
        }

        if options.sample_every > 0 && step_num.is_multiple_of(options.sample_every) {
            let sample = generate(
                &model,
                &tokenizer,
                &options.prompt,
                64,
                options.device,
                options.seq_len,
                options.sample_temperature,
                options.sample_top_k,
            );
            println!("  sample: {sample}");
        }
    }

    println!("Done in {:.1}s", train_start.elapsed().as_secs_f64());
}

fn train_step(
    model: &GPT,
    parameters: &[Parameter],
    opt: &mut deers::optim::AdamW,
    dataset: &TokenBinDataset,
    vocab_size: usize,
    options: &TrainOptions,
) -> (f32, f32) {
    let mut accumulated = GradientStore::new();
    let mut total_loss = 0.0f32;

    for _ in 0..options.grad_accum_steps {
        let (inputs, targets) = dataset.sample_batch(options.micro_batch_size, options.device);
        let logits = model.forward(&inputs).unwrap();
        let logits_flat =
            logits.reshape(vec![options.micro_batch_size * options.seq_len, vocab_size]);
        let targets_flat = targets.reshape(vec![options.micro_batch_size * options.seq_len]);
        let batch_loss = loss::cross_entropy(&logits_flat, &targets_flat);
        total_loss += batch_loss.to_vec::<f32>().unwrap()[0];

        let scaled_loss = &batch_loss * (1.0 / options.grad_accum_steps as f64);
        let grads = scaled_loss.backward().unwrap();
        accumulate_grads(&mut accumulated, parameters, &grads);
    }

    let grad_norm = clip_grad_norm(parameters, &mut accumulated, options.max_grad_norm).unwrap();
    if grad_norm.is_finite() {
        opt.step_with_grads(&accumulated).unwrap();
    }
    (total_loss / options.grad_accum_steps as f32, grad_norm)
}

fn accumulate_grads(dst: &mut GradientStore, parameters: &[Parameter], src: &GradientStore) {
    for parameter in parameters {
        let Some(grad) = src.get(parameter.id()) else {
            continue;
        };
        dst.accumulate(parameter, grad.detach());
    }
}

fn evaluate(
    model: &GPT,
    dataset: &TokenBinDataset,
    vocab_size: usize,
    options: &TrainOptions,
) -> f32 {
    let mut total = 0.0f32;

    for _ in 0..options.eval_batches {
        let (inputs, targets) = dataset.sample_batch(options.micro_batch_size, options.device);
        let logits = model.forward(&inputs).unwrap();
        let logits_flat =
            logits.reshape(vec![options.micro_batch_size * options.seq_len, vocab_size]);
        let targets_flat = targets.reshape(vec![options.micro_batch_size * options.seq_len]);
        let batch_loss = loss::cross_entropy(&logits_flat, &targets_flat);
        total += batch_loss.to_vec::<f32>().unwrap()[0];
    }

    total / options.eval_batches as f32
}

fn generate(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    device: Device,
    sequence_len: usize,
    temperature: f32,
    top_k: usize,
) -> String {
    let mut tokens: Vec<i64> = tokenizer.encode(prompt).into_iter().map(i64::from).collect();

    for _ in 0..max_new_tokens {
        let start = tokens.len().saturating_sub(sequence_len);
        let input_tokens = tokens[start..].to_vec();
        let input = Tensor::from_vec(input_tokens, (1, tokens.len() - start), device);
        let logits = model.forward(&input).unwrap();
        let last_logits = logits.narrow(1, input.layout().shape()[1] - 1, 1);
        let last_logits = last_logits.reshape(vec![logits.layout().shape()[2]]);
        let logits: Vec<f32> = last_logits.to_vec().unwrap();
        let next_token = sample_token(&logits, temperature, top_k);
        tokens.push(next_token as i64);
    }

    let decoded = tokens.into_iter().map(|token| token as u32).collect::<Vec<_>>();
    tokenizer.decode_lossy(&decoded)
}

fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    assert!(temperature > 0.0, "sample_temperature must be positive");
    assert!(top_k > 0, "sample_top_k must be positive");

    let mut candidates = logits.iter().copied().enumerate().collect::<Vec<_>>();
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    candidates.truncate(top_k.min(candidates.len()));

    let scaled_max = candidates[0].1 / temperature;
    let weights = candidates
        .iter()
        .map(|(_, logit)| ((*logit / temperature) - scaled_max).exp())
        .collect::<Vec<_>>();
    let total_weight = weights.iter().sum::<f32>();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return candidates[0].0;
    }

    let mut draw = rand::rng().random::<f32>() * total_weight;
    for ((token, _), weight) in candidates.iter().zip(weights.iter()) {
        draw -= *weight;
        if draw <= 0.0 {
            return *token;
        }
    }

    candidates.last().unwrap().0
}

fn resolve_token_bins(options: &TrainOptions, tokenizer: &Tokenizer) -> TokenBinPaths {
    let train = options.train_bin.clone().unwrap_or_else(|| options.bin_dir.join("train.bin"));
    let val = options.val_bin.clone().unwrap_or_else(|| options.bin_dir.join("val.bin"));

    if options.train_bin.is_none()
        && options.val_bin.is_none()
        && (options.prepare || !train.exists() || !val.exists())
    {
        if !options.text_path.exists() {
            let url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt";
            println!("Dataset not found at {}. Downloading from Hugging Face...", options.text_path.display());
            if let Some(parent) = options.text_path.parent() {
                fs::create_dir_all(parent).unwrap_or_else(|e| {
                    eprintln!("error: failed to create directory {}: {e}", parent.display());
                    process::exit(1);
                });
            }
            let response = ureq::get(url).call().unwrap_or_else(|e| {
                eprintln!("error: failed to download dataset: {e}");
                process::exit(1);
            });
            let mut dest = fs::File::create(&options.text_path).unwrap_or_else(|e| {
                eprintln!("error: failed to create {}: {e}", options.text_path.display());
                process::exit(1);
            });
            std::io::copy(&mut response.into_body().into_reader(), &mut dest).unwrap_or_else(|e| {
                eprintln!("error: failed to write dataset: {e}");
                process::exit(1);
            });
            println!("Downloaded to {}", options.text_path.display());
        }
        println!(
            "Preparing token bins from {} into {}...",
            options.text_path.display(),
            options.bin_dir.display()
        );
        return prepare_text_token_bins(
            &options.text_path,
            tokenizer,
            &options.bin_dir,
            options.val_ratio,
        )
        .unwrap_or_else(|e| {
            eprintln!("error: failed to prepare token bins: {e}");
            process::exit(1);
        });
    }

    assert!(train.exists(), "missing train bin: {}", train.display());
    assert!(val.exists(), "missing val bin: {}", val.display());
    TokenBinPaths { train, val }
}

fn should_save_checkpoint(step_num: usize, start_step: usize, save_every: usize) -> bool {
    if save_every == 0 {
        return false;
    }

    if step_num < save_every {
        return false;
    }

    if start_step == 0 {
        return step_num.is_multiple_of(save_every);
    }

    step_num >= start_step + save_every && (step_num - start_step).is_multiple_of(save_every)
}

fn write_run_config(
    options: &TrainOptions,
    config: &GPTConfig,
    vocab_size: usize,
    num_params: usize,
) {
    let metadata = RunMetadata {
        preset: options.preset.name().to_owned(),
        device: format!("{:?}", options.device),
        vocab_size,
        num_params,
        seq_len: config.sequence_len,
        n_layer: config.n_layer,
        n_head: config.n_head,
        n_embd: config.n_embd,
        mlp_hidden_dim: config.mlp_hidden_dim,
        micro_batch_size: options.micro_batch_size,
        grad_accum_steps: options.grad_accum_steps,
        max_steps: options.max_steps,
        eval_every: options.eval_every,
        save_every: options.save_every,
        sample_every: options.sample_every,
        eval_batches: options.eval_batches,
        log_every: options.log_every,
        lr: options.lr,
        warmup_steps: options.warmup_steps,
        warmdown_ratio: options.warmdown_ratio,
        final_lr_frac: options.final_lr_frac,
        weight_decay: options.weight_decay,
        betas: options.betas,
        max_grad_norm: options.max_grad_norm,
        sample_temperature: options.sample_temperature,
        sample_top_k: options.sample_top_k,
        text_path: options.text_path.display().to_string(),
        bin_dir: options.bin_dir.display().to_string(),
        out_dir: options.out_dir.display().to_string(),
        resume: options.resume.as_ref().map(|path| path.display().to_string()),
        prompt: options.prompt.clone(),
    };

    let path = options.out_dir.join("config.json");
    let json = serde_json::to_string_pretty(&metadata).unwrap();
    std::fs::write(path, json).unwrap();
}

fn save_checkpoint(
    store: &ParamStore,
    config: &GPTConfig,
    options: &TrainOptions,
    opt: &AdamW,
    step: usize,
    train_loss: f32,
    val_loss: Option<f32>,
) {
    let dir = options.out_dir.join(format!("step-{step:05}"));
    fs::create_dir_all(&dir).unwrap();

    let named_parameters = store.named_parameters();
    let optimizer_state = opt.state_dict(&named_parameters);
    store.save(&dir.join(MODEL_FILE)).unwrap();
    checkpoint::save_tensors(&dir.join(OPTIMIZER_FILE), &optimizer_state).unwrap();

    let checkpoint = CheckpointMetadata {
        step,
        train_loss,
        val_loss,
        lr: opt.lr(),
        sequence_len: config.sequence_len,
        n_layer: config.n_layer,
        n_head: config.n_head,
        n_embd: config.n_embd,
        mlp_hidden_dim: config.mlp_hidden_dim,
        model_file: MODEL_FILE.to_owned(),
        optimizer: OptimizerCheckpointMetadata {
            step: opt.step_count(),
            file: OPTIMIZER_FILE.to_owned(),
            tensor_count: optimizer_state.len(),
        },
        tensor_count: named_parameters.len(),
    };
    let json = serde_json::to_string_pretty(&checkpoint).unwrap();
    std::fs::write(dir.join("metadata.json"), json).unwrap();
    std::fs::write(options.out_dir.join("latest-checkpoint.txt"), format!("step-{step:05}\n"))
        .unwrap();
}

fn load_checkpoint(
    store: &ParamStore,
    opt: &mut AdamW,
    config: &GPTConfig,
    device: Device,
    checkpoint_dir: &Path,
) -> usize {
    let checkpoint_dir = resolve_resume_dir(checkpoint_dir);
    let metadata: CheckpointMetadata = serde_json::from_str(
        &std::fs::read_to_string(checkpoint_dir.join("metadata.json")).unwrap(),
    )
    .unwrap();

    assert_eq!(metadata.sequence_len, config.sequence_len, "checkpoint seq_len mismatch");
    assert_eq!(metadata.n_layer, config.n_layer, "checkpoint n_layer mismatch");
    assert_eq!(metadata.n_head, config.n_head, "checkpoint n_head mismatch");
    assert_eq!(metadata.n_embd, config.n_embd, "checkpoint n_embd mismatch");
    assert_eq!(
        metadata.mlp_hidden_dim, config.mlp_hidden_dim,
        "checkpoint mlp_hidden_dim mismatch"
    );
    let named_parameters = store.named_parameters();
    assert_eq!(
        metadata.tensor_count,
        named_parameters.len(),
        "checkpoint parameter count mismatch"
    );

    store.load(&checkpoint_dir.join(&metadata.model_file), device).unwrap();

    assert_eq!(
        metadata.optimizer.tensor_count,
        named_parameters.len() * 2,
        "checkpoint optimizer state count mismatch"
    );
    let optimizer_state =
        checkpoint::load_tensors(&checkpoint_dir.join(&metadata.optimizer.file), device).unwrap();
    opt.load_state_dict(&named_parameters, &optimizer_state, metadata.optimizer.step).unwrap();
    metadata.step
}

fn resolve_resume_dir(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.to_path_buf();
    }

    if path.is_file() {
        let target = std::fs::read_to_string(path).unwrap();
        let checkpoint = target.trim();
        let checkpoint = path.parent().unwrap_or_else(|| Path::new(".")).join(checkpoint);
        assert!(checkpoint.is_dir(), "resume pointer does not target a checkpoint dir");
        return checkpoint;
    }

    panic!("resume checkpoint must be a directory: {}", path.display());
}

#[derive(Clone, Copy)]
enum Preset {
    Air16Fast,
    Air16Overnight,
}

impl Preset {
    fn from_str(value: &str) -> Self {
        match value {
            "air16-fast" => Self::Air16Fast,
            "air16-overnight" => Self::Air16Overnight,
            other => usage(&format!("unsupported preset: {other}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Air16Fast => "air16-fast",
            Self::Air16Overnight => "air16-overnight",
        }
    }
}

#[derive(Clone)]
struct TrainOptions {
    preset: Preset,
    device: Device,
    prepare: bool,
    prepare_only: bool,
    text_path: PathBuf,
    bin_dir: PathBuf,
    train_bin: Option<PathBuf>,
    val_bin: Option<PathBuf>,
    resume: Option<PathBuf>,
    out_dir: PathBuf,
    val_ratio: f32,
    seq_len: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    mlp_hidden_dim: usize,
    micro_batch_size: usize,
    grad_accum_steps: usize,
    max_steps: usize,
    eval_every: usize,
    save_every: usize,
    sample_every: usize,
    eval_batches: usize,
    log_every: usize,
    lr: f64,
    warmup_steps: usize,
    warmdown_ratio: f64,
    final_lr_frac: f64,
    weight_decay: f64,
    betas: (f64, f64),
    max_grad_norm: f64,
    sample_temperature: f32,
    sample_top_k: usize,
    prompt: String,
}

impl TrainOptions {
    fn from_preset(preset: Preset) -> Self {
        match preset {
            Preset::Air16Fast => Self {
                preset,
                device: Device::Cpu,
                prepare: false,
                prepare_only: false,
                text_path: PathBuf::from(DEFAULT_TEXT_PATH),
                bin_dir: PathBuf::from(DEFAULT_BIN_DIR),
                train_bin: None,
                val_bin: None,
                resume: None,
                out_dir: PathBuf::from(DEFAULT_OUT_DIR),
                val_ratio: DEFAULT_VAL_RATIO,
                seq_len: 256,
                n_layer: 6,
                n_head: 6,
                n_embd: 192,
                mlp_hidden_dim: 768,
                micro_batch_size: 4,
                grad_accum_steps: 8,
                max_steps: 12_000,
                eval_every: 250,
                save_every: 50,
                sample_every: 50,
                eval_batches: 20,
                log_every: 10,
                lr: 5e-4,
                warmup_steps: 500,
                warmdown_ratio: 0.65,
                final_lr_frac: 0.1,
                weight_decay: 0.1,
                betas: (0.9, 0.95),
                max_grad_norm: 1.0,
                sample_temperature: DEFAULT_SAMPLE_TEMPERATURE,
                sample_top_k: DEFAULT_SAMPLE_TOP_K,
                prompt: DEFAULT_PROMPT.to_owned(),
            },
            Preset::Air16Overnight => Self {
                preset,
                device: Device::Cpu,
                prepare: false,
                prepare_only: false,
                text_path: PathBuf::from(DEFAULT_TEXT_PATH),
                bin_dir: PathBuf::from(DEFAULT_BIN_DIR),
                train_bin: None,
                val_bin: None,
                resume: None,
                out_dir: PathBuf::from(DEFAULT_OUT_DIR),
                val_ratio: DEFAULT_VAL_RATIO,
                seq_len: 256,
                n_layer: 8,
                n_head: 6,
                n_embd: 256,
                mlp_hidden_dim: 1024,
                micro_batch_size: 4,
                grad_accum_steps: 8,
                max_steps: 20_000,
                eval_every: 250,
                save_every: 50,
                sample_every: 50,
                eval_batches: 20,
                log_every: 10,
                lr: 5e-4,
                warmup_steps: 500,
                warmdown_ratio: 0.65,
                final_lr_frac: 0.1,
                weight_decay: 0.1,
                betas: (0.9, 0.95),
                max_grad_norm: 1.0,
                sample_temperature: DEFAULT_SAMPLE_TEMPERATURE,
                sample_top_k: DEFAULT_SAMPLE_TOP_K,
                prompt: DEFAULT_PROMPT.to_owned(),
            },
        }
    }

    fn gpt_config(&self, vocab_size: usize) -> GPTConfig {
        GPTConfig {
            vocab_size,
            sequence_len: self.seq_len,
            n_layer: self.n_layer,
            n_head: self.n_head,
            n_embd: self.n_embd,
            mlp_hidden_dim: self.mlp_hidden_dim,
            rms_norm_eps: 1e-5,
            rope_base: 10_000.0,
        }
    }
}

fn parse_args() -> TrainOptions {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let preset = scan_preset(&args).unwrap_or(Preset::Air16Fast);
    let mut options = TrainOptions::from_preset(preset);

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--device" => {
                let value = next_arg(&args, &mut index, "--device");
                options.device = parse_device(&value);
            }
            "--preset" => {
                let _ = next_arg(&args, &mut index, "--preset");
            }
            "--prepare" => options.prepare = true,
            "--prepare-only" => {
                options.prepare = true;
                options.prepare_only = true;
            }
            "--text-path" => {
                options.text_path = PathBuf::from(next_arg(&args, &mut index, "--text-path"))
            }
            "--bin-dir" => {
                options.bin_dir = PathBuf::from(next_arg(&args, &mut index, "--bin-dir"))
            }
            "--train-bin" => {
                options.train_bin = Some(PathBuf::from(next_arg(&args, &mut index, "--train-bin")))
            }
            "--val-bin" => {
                options.val_bin = Some(PathBuf::from(next_arg(&args, &mut index, "--val-bin")))
            }
            "--resume" => {
                options.resume = Some(PathBuf::from(next_arg(&args, &mut index, "--resume")))
            }
            "--out-dir" => {
                options.out_dir = PathBuf::from(next_arg(&args, &mut index, "--out-dir"))
            }
            "--val-ratio" => {
                options.val_ratio = parse_f32(&next_arg(&args, &mut index, "--val-ratio"))
            }
            "--seq-len" => options.seq_len = parse_usize(&next_arg(&args, &mut index, "--seq-len")),
            "--n-layer" => options.n_layer = parse_usize(&next_arg(&args, &mut index, "--n-layer")),
            "--n-head" => options.n_head = parse_usize(&next_arg(&args, &mut index, "--n-head")),
            "--n-embd" => {
                options.n_embd = parse_usize(&next_arg(&args, &mut index, "--n-embd"));
            }
            "--mlp-hidden-dim" => {
                options.mlp_hidden_dim =
                    parse_usize(&next_arg(&args, &mut index, "--mlp-hidden-dim"));
            }
            "--micro-batch-size" => {
                options.micro_batch_size =
                    parse_usize(&next_arg(&args, &mut index, "--micro-batch-size"));
            }
            "--grad-accum-steps" => {
                options.grad_accum_steps =
                    parse_usize(&next_arg(&args, &mut index, "--grad-accum-steps"));
            }
            "--max-steps" => {
                options.max_steps = parse_usize(&next_arg(&args, &mut index, "--max-steps"))
            }
            "--eval-every" => {
                options.eval_every = parse_usize(&next_arg(&args, &mut index, "--eval-every"));
            }
            "--save-every" => {
                options.save_every = parse_usize(&next_arg(&args, &mut index, "--save-every"));
            }
            "--sample-every" => {
                options.sample_every = parse_usize(&next_arg(&args, &mut index, "--sample-every"));
            }
            "--eval-batches" => {
                options.eval_batches = parse_usize(&next_arg(&args, &mut index, "--eval-batches"));
            }
            "--log-every" => {
                options.log_every = parse_usize(&next_arg(&args, &mut index, "--log-every"));
            }
            "--lr" => options.lr = parse_f64(&next_arg(&args, &mut index, "--lr")),
            "--warmup-steps" => {
                options.warmup_steps = parse_usize(&next_arg(&args, &mut index, "--warmup-steps"));
            }
            "--weight-decay" => {
                options.weight_decay = parse_f64(&next_arg(&args, &mut index, "--weight-decay"));
            }
            "--max-grad-norm" => {
                options.max_grad_norm = parse_f64(&next_arg(&args, &mut index, "--max-grad-norm"));
            }
            "--sample-temperature" => {
                options.sample_temperature =
                    parse_f32(&next_arg(&args, &mut index, "--sample-temperature"));
            }
            "--sample-top-k" => {
                options.sample_top_k = parse_usize(&next_arg(&args, &mut index, "--sample-top-k"));
            }
            "--prompt" => options.prompt = next_arg(&args, &mut index, "--prompt"),
            "--help" | "-h" => usage(""),
            other => usage(&format!("unexpected argument: {other}")),
        }
        index += 1;
    }

    if options.mlp_hidden_dim == 0 {
        options.mlp_hidden_dim = options.n_embd * 4;
    }
    assert!(options.grad_accum_steps > 0, "grad_accum_steps must be positive");
    assert!(options.micro_batch_size > 0, "micro_batch_size must be positive");
    assert!(options.eval_batches > 0, "eval_batches must be positive");
    assert!(options.sample_temperature > 0.0, "sample_temperature must be positive");
    assert!(options.sample_top_k > 0, "sample_top_k must be positive");
    options
}

fn scan_preset(args: &[String]) -> Option<Preset> {
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--preset" {
            return Some(Preset::from_str(
                args.get(index + 1).unwrap_or_else(|| usage("missing value for --preset")),
            ));
        }
        index += 1;
    }
    None
}

fn next_arg(args: &[String], index: &mut usize, flag: &str) -> String {
    *index += 1;
    args.get(*index).cloned().unwrap_or_else(|| usage(&format!("missing value for {flag}")))
}

fn parse_device(value: &str) -> Device {
    match value {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        "mps" => Device::Mps,
        other => usage(&format!("unsupported device: {other}")),
    }
}

fn parse_usize(value: &str) -> usize {
    value.parse::<usize>().unwrap_or_else(|_| usage(&format!("invalid integer: {value}")))
}

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().unwrap_or_else(|_| usage(&format!("invalid float: {value}")))
}

fn parse_f32(value: &str) -> f32 {
    value.parse::<f32>().unwrap_or_else(|_| usage(&format!("invalid float: {value}")))
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
        eprintln!();
    }

    eprintln!("Usage: cargo run --release --example tinystories_train -- [options]");
    eprintln!("  --device cpu|cuda|mps");
    eprintln!(
        "  note: CUDA builds require `--features cuda`, for example:"
    );
    eprintln!(
        "        cargo run --release --features cuda --example tinystories_train -- --device cuda"
    );
    eprintln!("  --preset air16-fast|air16-overnight");
    eprintln!("  --prepare");
    eprintln!("  --prepare-only");
    eprintln!("  --text-path <path>");
    eprintln!("  --bin-dir <path>");
    eprintln!("  --train-bin <path>");
    eprintln!("  --val-bin <path>");
    eprintln!("  --resume <checkpoint_dir>");
    eprintln!("  --out-dir <path>");
    eprintln!("  --seq-len <usize>");
    eprintln!("  --n-layer <usize>");
    eprintln!("  --n-head <usize>");
    eprintln!("  --n-embd <usize>");
    eprintln!("  --mlp-hidden-dim <usize>");
    eprintln!("  --micro-batch-size <usize>");
    eprintln!("  --grad-accum-steps <usize>");
    eprintln!("  --max-steps <usize>");
    eprintln!("  --eval-every <usize>");
    eprintln!("  --save-every <usize>");
    eprintln!("  --sample-every <usize>");
    eprintln!("  --eval-batches <usize>");
    eprintln!("  --log-every <usize>");
    eprintln!("  --lr <float>");
    eprintln!("  --warmup-steps <usize>");
    eprintln!("  --weight-decay <float>");
    eprintln!("  --max-grad-norm <float>");
    eprintln!("  --sample-temperature <float>");
    eprintln!("  --sample-top-k <usize>");
    eprintln!("  --prompt <text>");
    process::exit(if message.is_empty() { 0 } else { 1 });
}

#[derive(Serialize)]
struct RunMetadata {
    preset: String,
    device: String,
    vocab_size: usize,
    num_params: usize,
    seq_len: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    mlp_hidden_dim: usize,
    micro_batch_size: usize,
    grad_accum_steps: usize,
    max_steps: usize,
    eval_every: usize,
    save_every: usize,
    sample_every: usize,
    eval_batches: usize,
    log_every: usize,
    lr: f64,
    warmup_steps: usize,
    warmdown_ratio: f64,
    final_lr_frac: f64,
    weight_decay: f64,
    betas: (f64, f64),
    max_grad_norm: f64,
    sample_temperature: f32,
    sample_top_k: usize,
    text_path: String,
    bin_dir: String,
    out_dir: String,
    resume: Option<String>,
    prompt: String,
}

#[derive(Serialize, Deserialize)]
struct CheckpointMetadata {
    step: usize,
    train_loss: f32,
    val_loss: Option<f32>,
    lr: f64,
    sequence_len: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    mlp_hidden_dim: usize,
    model_file: String,
    tensor_count: usize,
    optimizer: OptimizerCheckpointMetadata,
}

#[derive(Serialize, Deserialize)]
struct OptimizerCheckpointMetadata {
    step: usize,
    file: String,
    tensor_count: usize,
}
