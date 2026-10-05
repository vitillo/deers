#![cfg(all(feature = "cuda", target_os = "linux"))]

use std::path::PathBuf;

use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor, Var};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen3::{Config as CandleQwen3Config, ModelForCausalLM};
use deers::models::gpt::{KvCache, Qwen3, Qwen3Config};
use deers::nn::ParamStore;
use deers::tokenizer::{ChatMessage, Qwen3Tokenizer, Tokenizer};
use deers::{DType, Device, Tensor, no_grad};
use half::bf16;

#[test]
fn bf16_cuda_operations_match_candle() {
    // Arrange
    if Device::Cuda.check_available().is_err() {
        return;
    }
    let candle_device = CandleDevice::new_cuda(0).unwrap();
    let values: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32 / 4.0 - 0.5)).collect();
    let deers_x = Tensor::from_vec(values.clone(), (2, 3), Device::Cuda);
    let candle_x = CandleTensor::from_vec(values, (2, 3), &candle_device).unwrap();

    // Act
    let deers_result = (&deers_x + &deers_x).matmul(&deers_x.transpose(None));
    let candle_result = (&candle_x + &candle_x).unwrap().matmul(&candle_x.t().unwrap()).unwrap();
    let actual: Vec<f32> =
        deers_result.to_vec::<bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
    let expected: Vec<f32> = candle_result
        .flatten_all()
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();

    // Assert
    assert_eq!(actual, expected);
}

#[test]
fn bf16_cuda_matmul_gradient_matches_candle() {
    // Arrange
    if Device::Cuda.check_available().is_err() {
        return;
    }
    let candle_device = CandleDevice::new_cuda(0).unwrap();
    let values: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32 / 4.0 - 0.5)).collect();
    let deers_x = Tensor::from_vec(values.clone(), (2, 3), Device::Cuda).attach();
    let candle_x = Var::from_vec(values, (2, 3), &candle_device).unwrap();

    // Act
    let deers_loss = (&deers_x + &deers_x).matmul(&deers_x.transpose(None)).sum(vec![0, 1], false);
    let candle_sum = (candle_x.as_tensor() + candle_x.as_tensor()).unwrap();
    let candle_loss = candle_sum.matmul(&candle_x.t().unwrap()).unwrap().sum_all().unwrap();
    let deers_grad =
        deers_loss.backward().unwrap().get(deers_x.id()).unwrap().to_vec::<bf16>().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let candle_grad = candle_grads
        .get(candle_x.as_tensor())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap();

    // Assert
    assert_eq!(deers_grad.len(), candle_grad.len());
    for (actual, expected) in deers_grad.iter().zip(candle_grad.iter()) {
        assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.02);
    }
}

fn candle_config(config: &Qwen3Config) -> CandleQwen3Config {
    CandleQwen3Config {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden,
        intermediate_size: config.mlp_hidden_dim,
        num_hidden_layers: config.n_layers,
        num_attention_heads: config.n_q_heads,
        head_dim: config.head_dim,
        attention_bias: false,
        num_key_value_heads: config.n_kv_heads,
        max_position_embeddings: config.sequence_len,
        sliding_window: None,
        max_window_layers: config.n_layers,
        tie_word_embeddings: true,
        rope_theta: f64::from(config.rope_base),
        rms_norm_eps: config.rms_norm_eps,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

#[test]
fn qwen3_bf16_cuda_prefill_matches_candle() {
    // Arrange
    if Device::Cuda.check_available().is_err() {
        return;
    }
    let Some(dir) = std::env::var_os("QWEN3_06B_DIR").map(PathBuf::from) else {
        eprintln!("set QWEN3_06B_DIR to run the checkpoint parity test");
        return;
    };
    let path = dir.join("model.safetensors");
    assert!(path.exists(), "missing Qwen3 checkpoint at {}", path.display());
    let config = Qwen3Config::qwen3_06b();
    let n_layers = config.n_layers;
    let candle_device = CandleDevice::new_cuda(0).unwrap();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[path], CandleDType::BF16, &candle_device) }
            .unwrap();
    let mut candle_model = ModelForCausalLM::new(&candle_config(&config), vb).unwrap();
    let store = ParamStore::new();
    let mut deers_model = Qwen3::new(config, store.root());
    deers_model.to_dtype(DType::BF16).unwrap();
    deers_model.to_device(Device::Cuda).unwrap();
    store.load_sharded(&dir, Device::Cuda).unwrap();
    let tokenizer = Qwen3Tokenizer::new();
    let prompt = Qwen3Tokenizer::apply_chat_template(
        &[ChatMessage { role: "user", content: "Explain why the sky is blue in one sentence." }],
        true,
    );
    let tokens = tokenizer.encode(&prompt);
    assert!(!tokens.is_empty());
    let long: Vec<u32> = (0..64).map(|i| tokens[i % tokens.len()]).collect();

    // Act
    let mut observations = Vec::new();
    for ids in [tokens, long] {
        candle_model.clear_kv_cache();
        let candle_input =
            CandleTensor::from_vec(ids.clone(), (1, ids.len()), &candle_device).unwrap();
        let candle_logits = candle_model
            .forward(&candle_input, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_dtype(CandleDType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let deers_input = Tensor::from_vec(
            ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
            (1, ids.len()),
            Device::Cuda,
        );
        let mut caches: Vec<KvCache> = (0..n_layers).map(|_| KvCache::new()).collect();
        let deers_logits = no_grad(|| deers_model.prefill(&deers_input, &mut caches).unwrap());
        let actual: Vec<f32> = deers_logits
            .narrow(1, ids.len() - 1, 1)
            .reshape(vec![candle_logits.len()])
            .to_vec::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        observations.push((ids.len(), actual, candle_logits));
    }

    // Assert
    assert_eq!(store.named_parameters().len(), 311);
    for (len, actual, candle_logits) in observations {
        assert_eq!(actual.len(), candle_logits.len());
        let max_abs =
            actual.iter().zip(&candle_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let mean_abs = actual.iter().zip(&candle_logits).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / actual.len() as f32;
        let top =
            |values: &[f32]| values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        eprintln!(
            "BF16 CUDA Qwen3/Candle len={} max_abs={max_abs}, mean_abs={mean_abs}, top1={} vs {}",
            len,
            top(&actual),
            top(&candle_logits)
        );
        assert_eq!(top(&actual), top(&candle_logits));
        assert!(max_abs <= 0.75, "maximum logit error {max_abs}");
        assert!(mean_abs <= 0.15, "mean logit error {mean_abs}");
    }
}
