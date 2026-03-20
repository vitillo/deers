use deers::models::gpt;
use deers::nn::Module;
use deers::{DType, Device, Tensor};

#[test]
fn test_causal_self_attention_shape() {
    // Arrange
    let attn = gpt::CausalSelfAttention::new(4, 2);
    let x = Tensor::from_vec(vec![1.0f32; 24], (2, 3, 4), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(3, 2, 10_000.0, DType::F32, Device::Cpu);

    // Act
    let out = attn.forward(&x, &cos, &sin).unwrap();

    // Assert
    assert_eq!(out.layout().shape, (2, 3, 4).into());
}

#[test]
fn test_causal_self_attention_parameters() {
    // Arrange
    let attn = gpt::CausalSelfAttention::new(4, 2);

    // Act
    let parameters = attn.parameters();

    // Assert
    assert_eq!(parameters.len(), 4);
}

#[test]
fn test_causal_self_attention_to_device() {
    // Arrange
    let attn = gpt::CausalSelfAttention::new(4, 2);

    // Act
    attn.to_device(Device::Mps).unwrap();

    // Assert
    assert!(attn.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_causal_self_attention_is_causal() {
    // Arrange
    let attn = gpt::CausalSelfAttention::new(4, 2);
    let (cos, sin) = gpt::precompute_rotary_embeddings(3, 2, 10_000.0, DType::F32, Device::Cpu);
    let x1 = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (1, 3, 4),
        Device::Cpu,
    );
    let x2 = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        (1, 3, 4),
        Device::Cpu,
    );

    // Act
    let y1 = attn.forward(&x1, &cos, &sin).unwrap().to_vec::<f32>().unwrap();
    let y2 = attn.forward(&x2, &cos, &sin).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(&y1[0..4], &y2[0..4]);
}

#[test]
fn test_mlp_shape() {
    // Arrange
    let mlp = gpt::MLP::new(4, 8);
    let x = Tensor::from_vec(vec![1.0f32; 24], (2, 3, 4), Device::Cpu);

    // Act
    let out = mlp.forward(&x).unwrap();

    // Assert
    assert_eq!(out.layout().shape, (2, 3, 4).into());
}

#[test]
fn test_mlp_parameters() {
    // Arrange
    let mlp = gpt::MLP::new(4, 8);

    // Act
    let parameters = mlp.parameters();

    // Assert
    assert_eq!(parameters.len(), 2);
}

#[test]
fn test_mlp_to_device() {
    // Arrange
    let mlp = gpt::MLP::new(4, 8);

    // Act
    mlp.to_device(Device::Mps).unwrap();

    // Assert
    assert!(mlp.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_block_shape() {
    // Arrange
    let block = gpt::Block::new(4, 2, 8, 1e-5);
    let x = Tensor::from_vec(vec![1.0f32; 24], (2, 3, 4), Device::Cpu);
    let (cos, sin) = gpt::precompute_rotary_embeddings(3, 2, 10_000.0, DType::F32, Device::Cpu);

    // Act
    let out = block.forward(&x, &cos, &sin).unwrap();

    // Assert
    assert_eq!(out.layout().shape, (2, 3, 4).into());
}

#[test]
fn test_block_parameters() {
    // Arrange
    let block = gpt::Block::new(4, 2, 8, 1e-5);

    // Act
    let parameters = block.parameters();

    // Assert
    assert_eq!(parameters.len(), 6);
}

#[test]
fn test_block_to_device() {
    // Arrange
    let block = gpt::Block::new(4, 2, 8, 1e-5);

    // Act
    block.to_device(Device::Mps).unwrap();

    // Assert
    assert!(block.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_block_is_causal() {
    // Arrange
    let block = gpt::Block::new(4, 2, 8, 1e-5);
    let (cos, sin) = gpt::precompute_rotary_embeddings(3, 2, 10_000.0, DType::F32, Device::Cpu);
    let x1 = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (1, 3, 4),
        Device::Cpu,
    );
    let x2 = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        (1, 3, 4),
        Device::Cpu,
    );

    // Act
    let y1 = block.forward(&x1, &cos, &sin).unwrap().to_vec::<f32>().unwrap();
    let y2 = block.forward(&x2, &cos, &sin).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_eq!(&y1[0..4], &y2[0..4]);
}

#[test]
fn test_gpt_shape() {
    // Arrange
    let model = gpt::GPT::new(gpt::GPTConfig {
        vocab_size: 16,
        sequence_len: 8,
        n_layer: 2,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    });
    let idx = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], (2, 3), Device::Cpu);

    // Act
    let logits = model.forward(&idx).unwrap();

    // Assert
    assert_eq!(logits.layout().shape, (2, 3, 16).into());
}

#[test]
fn test_gpt_parameters() {
    // Arrange
    let model = gpt::GPT::new(gpt::GPTConfig {
        vocab_size: 16,
        sequence_len: 8,
        n_layer: 2,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    });

    // Act
    let parameters = model.parameters();

    // Assert
    assert_eq!(parameters.len(), 14);
}

#[test]
fn test_gpt_to_device() {
    // Arrange
    let mut model = gpt::GPT::new(gpt::GPTConfig {
        vocab_size: 16,
        sequence_len: 8,
        n_layer: 2,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    });

    // Act
    model.to_device(Device::Mps).unwrap();

    // Assert
    assert!(model.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_precompute_rotary_embeddings_shape() {
    // Arrange
    let seq_len = 3;
    let head_dim = 4;

    // Act
    let (cos, sin) =
        gpt::precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, Device::Cpu);

    // Assert
    assert_eq!(cos.layout().shape, vec![1, 3, 1, 2].into());
    assert_eq!(sin.layout().shape, vec![1, 3, 1, 2].into());
}

#[test]
fn test_precompute_rotary_embeddings_values() {
    // Arrange
    let seq_len = 3;
    let head_dim = 4;

    // Act
    let (cos, sin) =
        gpt::precompute_rotary_embeddings(seq_len, head_dim, 10_000.0, DType::F32, Device::Cpu);
    let cos_values = cos.to_vec::<f32>().unwrap();
    let sin_values = sin.to_vec::<f32>().unwrap();

    // Assert
    let freqs = [0.0f32, 0.0, 1.0, 0.01, 2.0, 0.02];
    let expected_cos: Vec<f32> = freqs.iter().map(|&x| x.cos()).collect();
    let expected_sin: Vec<f32> = freqs.iter().map(|&x| x.sin()).collect();
    for (actual, expected) in cos_values.iter().zip(expected_cos.iter()) {
        assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
    }
    for (actual, expected) in sin_values.iter().zip(expected_sin.iter()) {
        assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
    }
}

#[test]
fn test_apply_rotary_emb() {
    // Arrange
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 1, 1, 4], Device::Cpu);
    let cos = Tensor::from_vec(vec![0.5f32, 0.25], vec![1, 1, 1, 2], Device::Cpu);
    let sin = Tensor::from_vec(vec![0.75f32, 1.0], vec![1, 1, 1, 2], Device::Cpu);

    // Act
    let out = gpt::apply_rotary_emb(&x, &cos, &sin);
    let values = out.to_vec::<f32>().unwrap();

    // Assert
    let expected = [
        1.0 * 0.5 + 3.0 * 0.75,
        2.0 * 0.25 + 4.0 * 1.0,
        1.0 * -0.75 + 3.0 * 0.5,
        -2.0 + 4.0 * 0.25,
    ];
    for (actual, expected) in values.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5, "expected {expected}, got {actual}");
    }
}
