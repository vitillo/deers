use std::collections::BTreeMap;

use deers::models::gpt::{Qwen3, Qwen3Config, RopeScaling};
use deers::nn::{ParamStore, Parameter};
use deers::{Device, Tensor};

fn small_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 64,
        sequence_len: 8,
        n_layers: 2,
        hidden: 16,
        n_q_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        mlp_hidden_dim: 32,
        rms_norm_eps: 1e-6,
        rope_base: 10_000.0,
        rope_scaling: RopeScaling::None,
    }
}

fn small_model() -> (Qwen3, Vec<(String, Parameter)>) {
    let store = ParamStore::new();
    let model = Qwen3::new(small_config(), store.root());
    let named = store.named_parameters();
    (model, named)
}

fn ids(batch: usize, seq: usize, vocab: usize) -> Tensor {
    let values: Vec<i64> = (0..batch * seq).map(|i| (i % vocab) as i64).collect();
    Tensor::from_vec(values, (batch, seq), Device::Cpu)
}

fn layer_names(layer: usize) -> Vec<String> {
    let prefix = format!("blocks.{layer}");
    vec![
        format!("{prefix}.attn.k_norm.weight"),
        format!("{prefix}.attn.k_proj.weight"),
        format!("{prefix}.attn.out_proj.weight"),
        format!("{prefix}.attn.q_norm.weight"),
        format!("{prefix}.attn.q_proj.weight"),
        format!("{prefix}.attn.v_proj.weight"),
        format!("{prefix}.input_layernorm.weight"),
        format!("{prefix}.mlp.down_proj.weight"),
        format!("{prefix}.mlp.gate_proj.weight"),
        format!("{prefix}.mlp.up_proj.weight"),
        format!("{prefix}.post_attention_layernorm.weight"),
    ]
}

#[test]
fn qwen3_layer_count_and_parameter_names_match_qwen_layout() {
    // Arrange: two small decoder layers.
    let (model, named) = small_model();

    // Act
    let mut names: Vec<String> = named.iter().map(|(name, _)| name.clone()).collect();
    names.sort();
    let by_name: BTreeMap<&str, &Parameter> =
        named.iter().map(|(name, parameter)| (name.as_str(), parameter)).collect();

    // Assert
    assert_eq!(model.n_layers(), 2);
    let mut expected = layer_names(0);
    expected.extend(layer_names(1));
    expected.extend(["lm_head.weight", "norm.weight", "wte.weight"].iter().map(|s| s.to_string()));
    expected.sort();
    assert_eq!(names, expected);
    assert_eq!(by_name["wte.weight"].id(), by_name["lm_head.weight"].id());
}

#[test]
fn qwen3_full_dims_construct_and_forward_on_cpu() {
    // Arrange: the published 0.6B dims with a two-token prompt.
    let store = ParamStore::new();
    let model = Qwen3::new(Qwen3Config::qwen3_06b(), store.root());
    let named = store.named_parameters();
    let shapes: BTreeMap<String, Vec<usize>> = named
        .iter()
        .map(|(name, parameter)| {
            (name.clone(), parameter.layout().shape().iter().copied().collect())
        })
        .collect();
    // Constant fill: randn emits rare infs at 155M-element scale, so the
    // end-to-end proof runs on a deterministic finite fill instead.
    for (_, parameter) in &named {
        let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
        let fill = Tensor::from_vec(vec![0.02f32; shape.iter().product()], shape, Device::Cpu);
        parameter.set(&fill).unwrap();
    }
    let idx = Tensor::from_vec(vec![1i64, 2], (1, 2), Device::Cpu);

    // Act
    let logits = model.forward(&idx).unwrap();
    let values = logits.to_vec::<f32>().unwrap();

    // Assert: 28 layers, one name per checkpoint tensor, Qwen leaf shapes.
    assert_eq!(model.n_layers(), 28);
    assert_eq!(named.len(), 311);
    assert_eq!(shapes["wte.weight"], vec![151936, 1024]);
    assert_eq!(shapes["lm_head.weight"], vec![151936, 1024]);
    assert_eq!(shapes["blocks.0.attn.q_proj.weight"], vec![1024, 2048]);
    assert_eq!(shapes["blocks.27.mlp.gate_proj.weight"], vec![1024, 3072]);
    assert_eq!(shapes["norm.weight"], vec![1024]);
    assert_eq!(logits.layout().shape().iter().copied().collect::<Vec<_>>(), vec![1, 2, 151936]);
    assert!(values.iter().all(|v| v.is_finite()));
}

#[test]
fn qwen3_forward_shapes_hold_across_lengths() {
    // Arrange: one small model, three prompt lengths.
    let (model, _) = small_model();

    // Act
    let shapes: Vec<Vec<usize>> = [1, 3, 5]
        .iter()
        .map(|&seq| {
            model.forward(&ids(2, seq, 64)).unwrap().layout().shape().iter().copied().collect()
        })
        .collect();

    // Assert
    assert_eq!(shapes, vec![vec![2, 1, 64], vec![2, 3, 64], vec![2, 5, 64]]);
}

#[test]
fn qwen3_forward_is_deterministic() {
    // Arrange
    let (model, _) = small_model();
    let idx = ids(1, 4, 64);

    // Act
    let first = model.forward(&idx).unwrap().to_vec::<f32>().unwrap();
    let second = model.forward(&idx).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(first, second);
}

#[test]
fn qwen3_backward_reaches_embeddings_and_every_layer() {
    // Arrange
    let (model, named) = small_model();
    let by_name: BTreeMap<&str, &Parameter> =
        named.iter().map(|(name, parameter)| (name.as_str(), parameter)).collect();
    let idx = ids(1, 3, 64);

    // Act
    let loss = model.forward(&idx).unwrap().sum(vec![0, 1, 2], false);
    let grads = loss.backward().unwrap();
    let norm_of = |parameter: &Parameter| {
        grads
            .get(parameter.id())
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt()
    };

    // Assert: the tied embedding/head id carries gradient from both paths.
    let shared = norm_of(by_name["wte.weight"]);
    assert!(shared.is_finite() && shared > 0.0);
    for (name, parameter) in &named {
        let norm = norm_of(parameter);
        assert!(norm.is_finite() && norm > 0.0, "{name} has no gradient");
    }
}

#[test]
fn qwen3_zeroed_model_emits_zero_logits() {
    // Arrange: every registered weight set to zero.
    let (model, named) = small_model();
    for (_, parameter) in &named {
        let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
        let zeros = Tensor::from_vec(vec![0.0f32; shape.iter().product()], shape, Device::Cpu);
        parameter.set(&zeros).unwrap();
    }
    let idx = ids(1, 2, 64);

    // Act
    let logits = model.forward(&idx).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(logits, vec![0.0f32; 2 * 64]);
}
