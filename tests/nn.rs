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
