use deers::nn::{self, Module};
use deers::optim::SGD;
use deers::{DType, Device, Tensor};

fn root() -> nn::ParamBuilder {
    nn::ParamStore::new().root()
}

#[test]
fn test_linear_forward() {
    let linear = nn::Linear::new(root(), 4, 3);
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 3).into());
}

#[test]
fn test_linear_parameters() {
    let linear = nn::Linear::new(root(), 4, 3);
    let parameters = linear.parameters();
    assert_eq!(parameters.len(), 2); // weight + bias
}

#[test]
fn test_linear_no_bias() {
    let linear = nn::Linear::no_bias(root(), 4, 3);
    let parameters = linear.parameters();
    assert_eq!(parameters.len(), 1); // weight only
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 3).into());
}

#[test]
fn test_linear_to_device() {
    if !Device::Mps.is_available() {
        return;
    }

    let linear = nn::Linear::new(root(), 4, 3);
    linear.to_device(Device::Mps).unwrap();

    for parameter in linear.parameters() {
        assert_eq!(parameter.device(), Device::Mps);
    }
}

#[test]
fn test_linear_to_same_device_noop() {
    let linear = nn::Linear::new(root(), 4, 3);
    linear.to_device(Device::Cpu).unwrap();

    for parameter in linear.parameters() {
        assert_eq!(parameter.device(), Device::Cpu);
    }
}

#[test]
fn test_sequential_forward() {
    let model = nn::seq().add(nn::Linear::new(root(), 4, 3)).add(nn::ReLU).add(nn::Linear::new(
        root(),
        3,
        2,
    ));
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = model.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 2).into());
}

#[test]
fn test_sequential_parameters() {
    let model = nn::seq().add(nn::Linear::new(root(), 4, 3)).add(nn::ReLU).add(nn::Linear::new(
        root(),
        3,
        2,
    ));
    // 2 Linear layers × 2 parameters each (weight + bias)
    assert_eq!(model.parameters().len(), 4);
}

#[test]
fn test_sequential_to_device() {
    if !Device::Mps.is_available() {
        return;
    }

    let model = nn::seq().add(nn::Linear::new(root(), 4, 3)).add(nn::ReLU).add(nn::Linear::new(
        root(),
        3,
        2,
    ));
    model.to_device(Device::Mps).unwrap();

    assert!(model.parameters().iter().all(|parameter| parameter.device() == Device::Mps));
}

#[test]
fn test_sequential_trains() {
    let model = nn::seq().add(nn::Linear::new(root(), 2, 4)).add(nn::ReLU).add(nn::Linear::new(
        root(),
        4,
        1,
    ));
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
    let emb = nn::Embedding::new(root(), 10, 4);
    let indices = Tensor::from_vec(vec![0i64, 3, 7], (3,), Device::Cpu);
    let out = emb.forward(&indices).unwrap();
    assert_eq!(out.layout().shape, (3, 4).into());
}

#[test]
fn test_embedding_2d_indices() {
    let emb = nn::Embedding::new(root(), 10, 4);
    // batch of 2, sequence length 3
    let indices = Tensor::from_vec(vec![0i64, 1, 2, 3, 4, 5], (2, 3), Device::Cpu);
    let out = emb.forward(&indices).unwrap();
    assert_eq!(out.layout().shape, (2, 3, 4).into());
}

#[test]
fn test_embedding_selects_correct_rows() {
    let emb = nn::Embedding::new(root(), 4, 3);
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
    if !Device::Mps.is_available() {
        return;
    }

    let emb = nn::Embedding::new(root(), 10, 4);
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
fn test_linear_train_eval_toggles_flag() {
    // Arrange
    let linear = nn::Linear::new(root(), 4, 3);

    // Act
    let initially_training = linear.is_training();
    linear.eval();
    let after_eval = linear.is_training();
    linear.train();
    let after_train = linear.is_training();

    // Assert
    assert!(initially_training);
    assert!(!after_eval);
    assert!(after_train);
}

#[test]
fn test_sequential_train_eval_toggles_flag() {
    // Arrange
    let model = nn::seq().add(nn::Linear::new(root(), 4, 3)).add(nn::ReLU).add(nn::Linear::new(
        root(),
        3,
        2,
    ));

    // Act
    let initially_training = model.is_training();
    model.eval();
    let after_eval = model.is_training();
    model.train();
    let after_train = model.is_training();

    // Assert
    assert!(initially_training);
    assert!(!after_eval);
    assert!(after_train);
}

#[test]
fn test_norms_train_eval_toggles_flag() {
    // Arrange
    let rms = nn::RMSNorm::new_affine(root(), 3, 1e-5);
    let norm = nn::LayerNorm::new(root(), 3, 1e-5);

    // Act
    rms.eval();
    norm.eval();
    let rms_eval = rms.is_training();
    let norm_eval = norm.is_training();
    rms.train();
    norm.train();

    // Assert
    assert!(!rms_eval);
    assert!(!norm_eval);
    assert!(rms.is_training());
    assert!(norm.is_training());
}

#[test]
fn test_rms_norm_affine_registers_weight() {
    // Arrange
    let store = nn::ParamStore::new();

    // Act
    let norm = nn::RMSNorm::new_affine(store.root().pp("norm"), 4, 1e-5);

    // Assert
    assert_eq!(norm.parameters().len(), 1);
    assert_eq!(norm.parameters()[0].to_vec::<f32>().unwrap(), vec![1.0; 4]);
    let names = store.named_parameters().into_iter().map(|(name, _)| name).collect::<Vec<_>>();
    assert_eq!(names, vec!["norm.weight"]);
}

#[test]
fn test_rms_norm_affine_scales_output() {
    // Arrange
    let norm = nn::RMSNorm::new_affine(root(), 3, 1e-5);
    norm.parameters()[0].set(&Tensor::from_vec(vec![2.0f32, 2.0, 2.0], (3,), Device::Cpu)).unwrap();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu);

    // Act
    let result: Vec<f32> = norm.forward(&x).unwrap().to_vec().unwrap();

    // Assert
    let rms = (14.0f32 / 3.0 + 1e-5).sqrt();
    let expected = [2.0 / rms, 4.0 / rms, 6.0 / rms];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }
}

#[test]
fn test_layer_norm_parameters() {
    // Arrange
    let store = nn::ParamStore::new();

    // Act
    let norm = nn::LayerNorm::new(store.root().pp("norm"), 3, 1e-5);

    // Assert
    assert_eq!(norm.parameters().len(), 2);
    assert_eq!(norm.parameters()[0].to_vec::<f32>().unwrap(), vec![1.0; 3]);
    assert_eq!(norm.parameters()[1].to_vec::<f32>().unwrap(), vec![0.0; 3]);
    let names = store.named_parameters().into_iter().map(|(name, _)| name).collect::<Vec<_>>();
    assert_eq!(names, vec!["norm.bias", "norm.weight"]);
}

#[test]
fn test_layer_norm_matches_expected_values() {
    // Arrange
    let norm = nn::LayerNorm::new(root(), 3, 1e-5);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu);

    // Act
    let result: Vec<f32> = norm.forward(&x).unwrap().to_vec().unwrap();

    // Assert
    let inv = 1.0 / (2.0f32 / 3.0 + 1e-5).sqrt();
    let expected = [-inv, 0.0, inv];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }
}

#[test]
fn test_layer_norm_no_bias_forward() {
    // Arrange
    let norm = nn::LayerNorm::no_bias(root(), 2, 1e-5);
    let x = Tensor::from_vec(vec![3.0f32, 1.0], (1, 2), Device::Cpu);

    // Act
    let result: Vec<f32> = norm.forward(&x).unwrap().to_vec().unwrap();

    // Assert
    assert_eq!(norm.parameters().len(), 1);
    let inv = 1.0 / (1.0f32 + 1e-5).sqrt();
    let expected = [inv, -inv];
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }
}

#[test]
fn test_layer_norm_affine_params_receive_gradients() {
    // Arrange
    let norm = nn::LayerNorm::new(root(), 3, 1e-5);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), Device::Cpu);

    // Act
    let loss = norm.forward(&x).unwrap().sum(vec![0, 1], false);
    let grads = loss.backward().unwrap();

    // Assert
    assert!(grads.get(norm.parameters()[0].id()).is_some());
    assert!(grads.get(norm.parameters()[1].id()).is_some());
}

#[test]
fn test_dropout_eval_is_identity() {
    // Arrange
    let dropout = nn::Dropout::new(0.5);
    dropout.eval();
    let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.5], (2, 2), Device::Cpu);

    // Act
    let result: Vec<f32> = dropout.forward(&x).unwrap().to_vec().unwrap();

    // Assert
    assert_eq!(result, vec![1.0, -2.0, 3.0, 0.5]);
}

#[test]
fn test_functional_dropout_not_training_is_identity() {
    // Arrange
    let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.5], (2, 2), Device::Cpu);

    // Act
    let result: Vec<f32> = nn::functional::dropout(&x, 0.5, false).to_vec().unwrap();

    // Assert
    assert_eq!(result, vec![1.0, -2.0, 3.0, 0.5]);
}

#[test]
fn test_dropout_zero_p_is_identity_in_train() {
    // Arrange
    let dropout = nn::Dropout::new(0.0);
    let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.5], (2, 2), Device::Cpu);

    // Act
    let result: Vec<f32> = dropout.forward(&x).unwrap().to_vec().unwrap();

    // Assert
    assert!(dropout.is_training());
    assert_eq!(result, vec![1.0, -2.0, 3.0, 0.5]);
}

#[test]
fn test_dropout_train_zeroes_half_and_scales_survivors() {
    // Arrange
    let dropout = nn::Dropout::new(0.5);
    let x = Tensor::from_vec(vec![1.0f32; 20000], (20000,), Device::Cpu);

    // Act
    let result: Vec<f32> = dropout.forward(&x).unwrap().to_vec().unwrap();

    // Assert
    let zeros = result.iter().filter(|&&v| v == 0.0).count();
    let zero_frac = zeros as f32 / result.len() as f32;
    assert!((zero_frac - 0.5).abs() < 0.05, "zero fraction {zero_frac} far from 0.5");
    assert!(result.iter().all(|&v| v == 0.0 || v == 2.0));
}

#[test]
fn test_dropout_backward_masks_gradient() {
    // Arrange
    let dropout = nn::Dropout::new(0.5);
    let x = Tensor::from_vec(vec![1.0f32; 64], (64,), Device::Cpu).attach();

    // Act
    let out = dropout.forward(&x).unwrap();
    let out_vals: Vec<f32> = out.to_vec().unwrap();
    let grads = out.sum(vec![0], false).backward().unwrap();

    // Assert
    let grad_vals: Vec<f32> = grads.get(x.id()).unwrap().to_vec().unwrap();
    assert_eq!(grad_vals, out_vals);
}

#[test]
fn test_dropout_train_eval_toggles_flag() {
    // Arrange
    let dropout = nn::Dropout::new(0.3);

    // Act
    let initially_training = dropout.is_training();
    dropout.eval();
    let after_eval = dropout.is_training();
    dropout.train();
    let after_train = dropout.is_training();

    // Assert
    assert!(initially_training);
    assert!(!after_eval);
    assert!(after_train);
    assert!(dropout.parameters().is_empty());
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
    if !Device::Mps.is_available() {
        return;
    }

    let mask = nn::functional::causal_mask(1, 3, 0, DType::F32, Device::Mps);

    assert_eq!(mask.device(), Device::Mps);
    assert_eq!(
        mask.to_vec::<f32>().unwrap(),
        vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0,]
    );
}
