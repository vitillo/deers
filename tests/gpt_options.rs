use std::collections::BTreeMap;

use deers::loss;
use deers::models::gpt::{self, GPTConfig, GptMlpKind, GptNormKind};
use deers::nn::{ParamStore, Parameter};
use deers::optim::AdamWConfig;
use deers::{Device, Tensor};

fn tiny_config() -> GPTConfig {
    GPTConfig {
        vocab_size: 8,
        sequence_len: 4,
        n_layer: 1,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
        rope_scaling: gpt::RopeScaling::None,
        norm: GptNormKind::RmsNorm,
        mlp: GptMlpKind::ReluSquared,
        tie_embeddings: false,
    }
}

fn token_ids() -> Vec<i64> {
    vec![1, 2, 3, 4, 3, 2]
}

fn targets() -> Vec<i64> {
    vec![2, 3, 4, 3, 2, 1]
}

fn named(store: &ParamStore) -> BTreeMap<String, Parameter> {
    store.named_parameters().into_iter().collect()
}

fn forward_loss(model: &gpt::GPT, batch_size: usize, seq_len: usize) -> Tensor {
    let idx = Tensor::from_vec(token_ids(), (batch_size, seq_len), Device::Cpu);
    let targets = Tensor::from_vec(targets(), (batch_size * seq_len,), Device::Cpu);
    let logits = model.forward(&idx).unwrap();
    loss::cross_entropy(&logits.reshape((batch_size * seq_len, 8)), &targets)
}

fn fill_constant(store: &ParamStore, value: f32) {
    for (_, parameter) in store.named_parameters() {
        let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
        let values = Tensor::from_vec(vec![value; shape.iter().product()], shape, Device::Cpu);
        parameter.set(&values).unwrap();
    }
}

fn fill_pattern(store: &ParamStore) {
    for (_, parameter) in store.named_parameters() {
        let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
        let n: usize = shape.iter().product();
        // Deterministic non-uniform values: a flat constant keeps every
        // activation vector component-equal, which the final RMSNorm then
        // erases, hiding any activation change. Varying values break that
        // symmetry so the activation choice stays observable in the logits.
        let values: Vec<f32> = (0..n).map(|i| 0.02 * ((i % 11) as f32) - 0.1).collect();
        let values = Tensor::from_vec(values, shape, Device::Cpu);
        parameter.set(&values).unwrap();
    }
}

fn grad_norm(grads: &deers::GradientStore, parameter: &Parameter) -> f32 {
    grads
        .get(parameter.id())
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|g| g * g)
        .sum::<f32>()
        .sqrt()
}

#[test]
fn test_default_config_preserves_checkpoint_names() {
    // Arrange
    let store = ParamStore::new();
    let config = tiny_config();

    // Act
    let _model = gpt::GPT::new(config, store.root());
    let names: Vec<String> = store.named_parameters().into_iter().map(|(name, _)| name).collect();

    // Assert: the historical ten tensors, no norm weights, no duplicate head.
    assert_eq!(
        names,
        vec![
            "blocks.0.attn.k_norm.weight",
            "blocks.0.attn.k_proj.weight",
            "blocks.0.attn.out_proj.weight",
            "blocks.0.attn.q_norm.weight",
            "blocks.0.attn.q_proj.weight",
            "blocks.0.attn.v_proj.weight",
            "blocks.0.mlp.down_proj.weight",
            "blocks.0.mlp.up_proj.weight",
            "lm_head.weight",
            "wte.weight",
        ]
    );
}

#[test]
fn test_affine_rmsnorm_registers_scales_and_trains() {
    // Arrange
    let store = ParamStore::new();
    let config = GPTConfig { norm: GptNormKind::AffineRmsNorm, ..tiny_config() };
    let model = gpt::GPT::new(config, store.root());
    let params = named(&store);

    // Act
    let loss = forward_loss(&model, 2, 3);
    let loss_value = loss.to_vec::<f32>().unwrap()[0];
    let grads = loss.backward().unwrap();

    // Assert: the three new scales exist and carry finite nonzero gradients.
    for name in ["blocks.0.norm1.weight", "blocks.0.norm2.weight", "norm.weight"] {
        let norm = grad_norm(&grads, &params[name]);
        assert!(loss_value.is_finite(), "loss is not finite");
        assert!(norm.is_finite() && norm > 0.0, "{name} has no gradient");
    }
}

#[test]
fn test_layernorm_registers_weight_and_bias_and_trains() {
    // Arrange
    let store = ParamStore::new();
    let config = GPTConfig { norm: GptNormKind::LayerNorm, ..tiny_config() };
    let model = gpt::GPT::new(config, store.root());
    let params = named(&store);

    // Act
    let loss = forward_loss(&model, 2, 3);
    let loss_value = loss.to_vec::<f32>().unwrap()[0];
    let grads = loss.backward().unwrap();

    // Assert: weight and bias exist per norm and carry finite nonzero gradients.
    for name in [
        "blocks.0.norm1.weight",
        "blocks.0.norm1.bias",
        "blocks.0.norm2.weight",
        "blocks.0.norm2.bias",
        "norm.weight",
        "norm.bias",
    ] {
        assert!(params.contains_key(name), "{name} was not registered");
        let norm = grad_norm(&grads, &params[name]);
        assert!(loss_value.is_finite(), "loss is not finite");
        assert!(norm.is_finite() && norm > 0.0, "{name} has no gradient");
    }
}

#[test]
fn test_gelu_mlp_differs_from_relu_squared_and_trains() {
    // Arrange: identical patterned weights, so only the activation can
    // differ. The pattern varies per element: a flat constant would keep
    // every activation vector component-equal, which the final RMSNorm
    // erases, hiding the activation change.
    let relu_store = ParamStore::new();
    let relu_model = gpt::GPT::new(tiny_config(), relu_store.root());
    fill_pattern(&relu_store);
    let gelu_store = ParamStore::new();
    let gelu_config = GPTConfig { mlp: GptMlpKind::Gelu, ..tiny_config() };
    let gelu_model = gpt::GPT::new(gelu_config, gelu_store.root());
    fill_pattern(&gelu_store);

    // Act
    let idx = Tensor::from_vec(token_ids(), (2, 3), Device::Cpu);
    let relu_logits = relu_model.forward(&idx).unwrap().to_vec::<f32>().unwrap();
    let gelu_logits = gelu_model.forward(&idx).unwrap().to_vec::<f32>().unwrap();
    let loss = forward_loss(&gelu_model, 2, 3);
    let grads = loss.backward().unwrap();

    // Assert: GELU runs end to end, moves the logits, and trains.
    assert!(gelu_logits.iter().all(|v| v.is_finite()));
    let max_diff = relu_logits
        .iter()
        .zip(gelu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "GELU left the logits unchanged");
    for parameter in gelu_model.parameters() {
        let norm = grad_norm(&grads, &parameter);
        assert!(norm.is_finite() && norm > 0.0, "a GELU parameter has no gradient");
    }
}

#[test]
fn test_tied_embeddings_share_one_parameter_identity() {
    // Arrange
    let store = ParamStore::new();
    let config = GPTConfig { tie_embeddings: true, ..tiny_config() };

    // Act
    let model = gpt::GPT::new(config, store.root());
    let params = named(&store);

    // Assert: both checkpoint names exist behind a single tensor id, listed once.
    assert_eq!(params["wte.weight"].id(), params["lm_head.weight"].id());
    let ids: Vec<_> = model.parameters().iter().map(|p| p.id()).collect();
    assert_eq!(ids.len(), 9, "tied model lists every tensor exactly once");
    assert_eq!(
        ids.iter().filter(|id| **id == params["wte.weight"].id()).count(),
        1,
        "shared tensor appears twice in parameters()"
    );
}

#[test]
fn test_tied_embeddings_backward_flows_through_both_paths() {
    // Arrange
    let store = ParamStore::new();
    let config = GPTConfig { tie_embeddings: true, ..tiny_config() };
    let model = gpt::GPT::new(config, store.root());
    let params = named(&store);

    // Act
    let loss = forward_loss(&model, 2, 3);
    let grads = loss.backward().unwrap();

    // Assert: the shared id carries gradient from the embedding and head paths.
    let norm = grad_norm(&grads, &params["wte.weight"]);
    assert!(norm.is_finite() && norm > 0.0, "shared tensor has no gradient");
}

#[test]
fn test_tied_embeddings_optimizer_step_keeps_weights_in_sync() {
    // Arrange
    let store = ParamStore::new();
    let config = GPTConfig { tie_embeddings: true, ..tiny_config() };
    let model = gpt::GPT::new(config, store.root());
    fill_constant(&store, 0.05);
    let before = named(&store)["wte.weight"].to_vec::<f32>().unwrap();

    // Act: optimize through the deduplicated parameter list, as training does.
    let loss = forward_loss(&model, 2, 3);
    let grads = loss.backward().unwrap();
    let mut opt = AdamWConfig::new(1e-3).build(model.parameters());
    opt.step_with_grads(&grads).unwrap();
    let params = named(&store);

    // Assert: one step moves the shared weight, identically under both names.
    let wte = params["wte.weight"].to_vec::<f32>().unwrap();
    let head = params["lm_head.weight"].to_vec::<f32>().unwrap();
    assert_ne!(wte, before, "optimizer step changed nothing");
    assert_eq!(wte, head, "embedding and head diverged after the step");
    assert!(wte.iter().all(|v| v.is_finite()));
}
