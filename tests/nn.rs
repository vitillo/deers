use deers::nn::{self, Module};
use deers::optim::SGD;
use deers::{DType, Device, Tensor};

#[test]
fn test_linear_forward() {
    let linear = nn::Linear::new(4, 3);
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 3).into());
}

#[test]
fn test_linear_parameters() {
    let linear = nn::Linear::new(4, 3);
    let parameters = linear.parameters();
    assert_eq!(parameters.len(), 2); // weight + bias
}

#[test]
fn test_linear_no_bias() {
    let linear = nn::Linear::no_bias(4, 3);
    let parameters = linear.parameters();
    assert_eq!(parameters.len(), 1); // weight only
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 3).into());
}

#[test]
fn test_linear_to_device() {
    let linear = nn::Linear::new(4, 3);
    linear.to_device(Device::Mps).unwrap();

    for parameter in linear.parameters() {
        assert_eq!(parameter.device(), Device::Mps);
    }
}

#[test]
fn test_linear_to_same_device_noop() {
    let linear = nn::Linear::new(4, 3);
    linear.to_device(Device::Cpu).unwrap();

    for parameter in linear.parameters() {
        assert_eq!(parameter.device(), Device::Cpu);
    }
}

#[test]
fn test_sequential_forward() {
    let model = nn::seq().add(nn::Linear::new(4, 3)).add(nn::ReLU).add(nn::Linear::new(3, 2));
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = model.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 2).into());
}

#[test]
fn test_sequential_parameters() {
    let model = nn::seq().add(nn::Linear::new(4, 3)).add(nn::ReLU).add(nn::Linear::new(3, 2));
    // 2 Linear layers × 2 parameters each (weight + bias)
    assert_eq!(model.parameters().len(), 4);
}

#[test]
fn test_sequential_to_device() {
    let model = nn::seq().add(nn::Linear::new(4, 3)).add(nn::ReLU).add(nn::Linear::new(3, 2));
    model.to_device(Device::Mps).unwrap();

    assert!(model.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_sequential_trains() {
    let model = nn::seq().add(nn::Linear::new(2, 4)).add(nn::ReLU).add(nn::Linear::new(4, 1));
    let mut sgd = SGD::new(model.parameters(), 0.01);

    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);

    let mut prev_loss = f32::MAX;
    for _ in 0..10 {
        let out = model.forward(&x).unwrap();
        let loss = (&out * &out).sum(vec![0, 1], true);
        let loss_val: Vec<f32> = loss.to_vec().unwrap();
        assert!(loss_val[0] < prev_loss, "loss should decrease");
        prev_loss = loss_val[0];
        sgd.backward_step(&loss).unwrap();
    }
}

#[test]
fn test_embedding_forward() {
    let emb = nn::Embedding::new(10, 4);
    let indices = Tensor::from_vec(vec![0i64, 3, 7], (3,), Device::Cpu);
    let out = emb.forward(&indices).unwrap();
    assert_eq!(out.layout().shape, (3, 4).into());
}

#[test]
fn test_embedding_2d_indices() {
    let emb = nn::Embedding::new(10, 4);
    // batch of 2, sequence length 3
    let indices = Tensor::from_vec(vec![0i64, 1, 2, 3, 4, 5], (2, 3), Device::Cpu);
    let out = emb.forward(&indices).unwrap();
    assert_eq!(out.layout().shape, (2, 3, 4).into());
}

#[test]
fn test_embedding_selects_correct_rows() {
    let emb = nn::Embedding::new(4, 3);
    // Look up indices 2 then 0
    let indices = Tensor::from_vec(vec![2i64, 0], (2,), Device::Cpu);
    let out = emb.forward(&indices).unwrap();
    let weight_data: Vec<f32> = emb.parameters()[0].to_vec().unwrap();
    let out_data: Vec<f32> = out.to_vec().unwrap();
    // Row 2 of weight should be first row of output
    assert_eq!(&out_data[0..3], &weight_data[6..9]);
    // Row 0 of weight should be second row of output
    assert_eq!(&out_data[3..6], &weight_data[0..3]);
}

#[test]
fn test_embedding_mps() {
    let emb = nn::Embedding::new(10, 4);
    emb.to_device(Device::Mps).unwrap();
    let indices = Tensor::from_vec(vec![0i64, 3, 7], (3,), Device::Mps);
    let out = emb.forward(&indices).unwrap();
    assert_eq!(out.layout().shape, (3, 4).into());
}

#[test]
fn test_causal_self_attention_shape() {
    // Arrange
    let attn = nn::CausalSelfAttention::new(4, 2);
    let x = Tensor::from_vec(vec![1.0f32; 24], (2, 3, 4), Device::Cpu);
    let (cos, sin) =
        nn::functional::precompute_rotary_embeddings(3, 2, 10_000.0, DType::F32, Device::Cpu);

    // Act
    let out = attn.forward(&x, &cos, &sin).unwrap();

    // Assert
    assert_eq!(out.layout().shape, (2, 3, 4).into());
}

#[test]
fn test_causal_self_attention_parameters() {
    // Arrange
    let attn = nn::CausalSelfAttention::new(4, 2);

    // Act
    let parameters = attn.parameters();

    // Assert
    assert_eq!(parameters.len(), 4);
}

#[test]
fn test_causal_self_attention_to_device() {
    // Arrange
    let attn = nn::CausalSelfAttention::new(4, 2);

    // Act
    attn.to_device(Device::Mps).unwrap();

    // Assert
    assert!(attn.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_causal_self_attention_is_causal() {
    // Arrange
    let attn = nn::CausalSelfAttention::new(4, 2);
    let (cos, sin) =
        nn::functional::precompute_rotary_embeddings(3, 2, 10_000.0, DType::F32, Device::Cpu);
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
fn test_rms_norm_shape() {
    let norm = nn::RMSNorm::new(1e-5);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 4), Device::Cpu);
    let out = norm.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 4).into());
}

#[test]
fn test_rms_norm_normalizes() {
    let norm = nn::RMSNorm::new(1e-5);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu);
    let out = norm.forward(&x).unwrap();
    // RMSNorm: x / sqrt(mean(x²) + eps)
    // mean(x²) = (1+4+9)/3 = 14/3, sqrt(14/3) ≈ 2.1602
    // [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
    let result: Vec<f32> = out.to_vec().unwrap();
    let rms = (14.0f32 / 3.0).sqrt();
    let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }
}

#[test]
fn test_rms_norm_parameters() {
    let norm = nn::RMSNorm::new(1e-5);
    assert!(norm.parameters().is_empty());
}

#[test]
fn test_precompute_rotary_embeddings_shape() {
    // Arrange
    let seq_len = 3;
    let head_dim = 4;

    // Act
    let (cos, sin) = nn::functional::precompute_rotary_embeddings(
        seq_len,
        head_dim,
        10_000.0,
        DType::F32,
        Device::Cpu,
    );

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
    let (cos, sin) = nn::functional::precompute_rotary_embeddings(
        seq_len,
        head_dim,
        10_000.0,
        DType::F32,
        Device::Cpu,
    );
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
    let out = nn::functional::apply_rotary_emb(&x, &cos, &sin);
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

#[test]
fn test_causal_mask() {
    let mask = nn::functional::causal_mask(2, 3, 0, DType::F32, Device::Cpu);

    assert_eq!(mask.layout().shape, vec![2, 1, 3, 3].into());
    assert_eq!(
        mask.to_vec::<f32>().unwrap(),
        vec![
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
        ]
    );
}

#[test]
fn test_causal_mask_with_offset() {
    let mask = nn::functional::causal_mask(1, 2, 3, DType::F32, Device::Cpu);

    assert_eq!(mask.layout().shape, vec![1, 1, 2, 5].into());
    assert_eq!(
        mask.to_vec::<f32>().unwrap(),
        vec![0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0,]
    );
}

#[test]
fn test_causal_mask_mps() {
    let mask = nn::functional::causal_mask(1, 3, 0, DType::F32, Device::Mps);

    assert_eq!(mask.device(), Device::Mps);
    assert_eq!(
        mask.to_vec::<f32>().unwrap(),
        vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0,]
    );
}
