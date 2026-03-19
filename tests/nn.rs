use deers::nn::{self, Module};
use deers::optim::SGD;
use deers::{Device, Tensor};

#[test]
fn test_linear_forward() {
    let linear = nn::Linear::new(4, 3);
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 3).into());
}

#[test]
fn test_linear_vars() {
    let linear = nn::Linear::new(4, 3);
    let vars = linear.vars();
    assert_eq!(vars.len(), 2); // weight + bias
}

#[test]
fn test_linear_no_bias() {
    let linear = nn::Linear::no_bias(4, 3);
    let vars = linear.vars();
    assert_eq!(vars.len(), 1); // weight only
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = linear.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 3).into());
}

#[test]
fn test_linear_to_device() {
    let linear = nn::Linear::new(4, 3);
    linear.to_device(Device::Mps).unwrap();

    for var in linear.vars() {
        assert_eq!(var.device(), Device::Mps);
    }
}

#[test]
fn test_linear_to_same_device_noop() {
    let linear = nn::Linear::new(4, 3);
    linear.to_device(Device::Cpu).unwrap();

    for var in linear.vars() {
        assert_eq!(var.device(), Device::Cpu);
    }
}

#[test]
fn test_sequential_forward() {
    let model = nn::seq()
        .add(nn::Linear::new(4, 3))
        .add(nn::ReLU)
        .add(nn::Linear::new(3, 2));
    let x = Tensor::from_vec(vec![1.0f32; 8], (2, 4), Device::Cpu);
    let out = model.forward(&x).unwrap();
    assert_eq!(out.layout().shape, (2, 2).into());
}

#[test]
fn test_sequential_vars() {
    let model = nn::seq()
        .add(nn::Linear::new(4, 3))
        .add(nn::ReLU)
        .add(nn::Linear::new(3, 2));
    // 2 Linear layers × 2 vars each (weight + bias)
    assert_eq!(model.vars().len(), 4);
}

#[test]
fn test_sequential_to_device() {
    let model = nn::seq()
        .add(nn::Linear::new(4, 3))
        .add(nn::ReLU)
        .add(nn::Linear::new(3, 2));
    model.to_device(Device::Mps).unwrap();

    assert!(model.vars().iter().all(|var| var.device() == Device::Mps));
}

#[test]
fn test_sequential_trains() {
    let model = nn::seq()
        .add(nn::Linear::new(2, 4))
        .add(nn::ReLU)
        .add(nn::Linear::new(4, 1));
    let mut sgd = SGD::new(model.vars(), 0.01);

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
