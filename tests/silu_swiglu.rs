use candle_core::{Device as CDevice, Tensor as CTensor, Var};
use deers::models::gpt::{self, MlpKind};
use deers::nn::{self, Module};
use deers::{Device, Tensor};

const TOL: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");

    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < TOL, "{label}[{index}]: got {a}, expected {e}");
    }
}

fn devices() -> Vec<Device> {
    [Device::Cpu, Device::Cuda, Device::Mps]
        .into_iter()
        .filter(|device| device.is_available())
        .collect()
}

fn candle_var(data: Vec<f32>, shape: &[usize]) -> Var {
    let tensor = CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap();
    Var::from_tensor(&tensor).unwrap()
}

fn candle_grad(grads: &candle_core::backprop::GradStore, var: &Var) -> Vec<f32> {
    grads.get(var.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn silu_forward_conforms_with_candle() {
    // Arrange
    let data = vec![-2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 4], &CDevice::Cpu).unwrap();
    let expected = candle_input.silu().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 4), device);
            let values = input.silu().to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("silu forward on {:?}", device));
    }
}

#[test]
fn silu_backward_conforms_with_candle() {
    // Arrange
    let data = vec![-2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 4]);
    let candle_loss = candle_input.silu().unwrap().sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 4), device).attach();
            let loss = input.silu().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("silu backward on {:?}", device));
    }
}

#[test]
fn silu_backward_matches_finite_differences() {
    // Arrange
    let data = vec![-1.5f32, -0.5, 0.25, 1.0, 2.0];
    let eps = 1e-3f32;
    let input = Tensor::from_vec(data.clone(), (1, 5), Device::Cpu).attach();
    let loss = input.silu().sum(vec![0, 1], true);
    let grads = loss.backward().unwrap();
    let analytic = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();

    // Act
    let numeric: Vec<f32> = data
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut plus = data.clone();
            let mut minus = data.clone();
            plus[index] += eps;
            minus[index] -= eps;
            let plus_sum: f32 =
                Tensor::from_vec(plus, (1, 5), Device::Cpu).silu().to_vec::<f32>().unwrap().iter().sum();
            let minus_sum: f32 =
                Tensor::from_vec(minus, (1, 5), Device::Cpu).silu().to_vec::<f32>().unwrap().iter().sum();
            (plus_sum - minus_sum) / (2.0 * eps)
        })
        .collect();

    // Assert
    assert_close(&analytic, &numeric, "silu finite-difference gradient");
}

#[test]
fn swiglu_forward_matches_candle_reference() {
    // Arrange
    let store = nn::ParamStore::new();
    let swiglu = nn::SwiGLU::new(store.root(), 4, 6, 4);
    let params = swiglu.parameters();
    assert_eq!(params.len(), 3);
    let shapes: Vec<Vec<usize>> =
        params.iter().map(|p| p.layout().shape().iter().copied().collect()).collect();
    assert_eq!(shapes, vec![vec![4, 6], vec![4, 6], vec![6, 4]]);
    let weights: Vec<Vec<f32>> = params.iter().map(|p| p.to_vec::<f32>().unwrap()).collect();
    let gate = CTensor::from_vec(weights[0].clone(), &[4, 6], &CDevice::Cpu).unwrap();
    let up = CTensor::from_vec(weights[1].clone(), &[4, 6], &CDevice::Cpu).unwrap();
    let down = CTensor::from_vec(weights[2].clone(), &[6, 4], &CDevice::Cpu).unwrap();
    let input_data: Vec<f32> = (0..24).map(|v| v as f32 * 0.1 - 0.5).collect();
    let candle_x = CTensor::from_vec(input_data.clone(), &[2, 3, 4], &CDevice::Cpu).unwrap();
    let flat = candle_x.reshape(&[6, 4]).unwrap();
    let expected = flat
        .matmul(&gate)
        .unwrap()
        .silu()
        .unwrap()
        .broadcast_mul(&flat.matmul(&up).unwrap())
        .unwrap()
        .matmul(&down)
        .unwrap()
        .reshape(&[2, 3, 4])
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let x = Tensor::from_vec(input_data, (2, 3, 4), Device::Cpu);
    let actual = swiglu.forward(&x).unwrap().to_vec::<f32>().unwrap();

    // Assert
    assert_close(&actual, &expected, "swiglu forward");
}

#[test]
fn swiglu_backward_flows_to_all_projections() {
    // Arrange
    let store = nn::ParamStore::new();
    let swiglu = nn::SwiGLU::new(store.root(), 4, 6, 4);
    let params = swiglu.parameters();
    let x = Tensor::from_vec(vec![0.25f32; 24], (2, 3, 4), Device::Cpu);

    // Act
    let loss = swiglu.forward(&x).unwrap().sum(vec![0, 1, 2], false);
    let grads = loss.backward().unwrap();
    let norms: Vec<f32> = params
        .iter()
        .map(|p| {
            grads
                .get(p.id())
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    // Assert
    assert_eq!(norms.len(), 3);
    for (index, norm) in norms.iter().enumerate() {
        assert!(norm.is_finite() && *norm > 0.0, "swiglu projection {index} has no gradient");
    }
}

fn tiny_config() -> gpt::GPTConfig {
    gpt::GPTConfig {
        vocab_size: 8,
        sequence_len: 4,
        n_layer: 1,
        n_head: 2,
        n_embd: 4,
        mlp_hidden_dim: 8,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
    }
}

#[test]
fn gpt_swiglu_option_runs_with_default_behavior_unchanged() {
    // Arrange
    let default_model = gpt::GPT::new(tiny_config(), nn::ParamStore::new().root());
    let default_params = default_model.parameters().len();
    let swiglu_model =
        gpt::GPT::new_with_kind(tiny_config(), nn::ParamStore::new().root(), MlpKind::SwiGlu);
    let idx = Tensor::from_vec(vec![1i64, 2, 3, 4, 3, 2], (2, 3), Device::Cpu);

    // Act
    let default_logits = default_model.forward(&idx).unwrap();
    let swiglu_logits = swiglu_model.forward(&idx).unwrap();

    // Assert
    let default_shape: Vec<usize> =
        default_logits.layout().shape().iter().copied().collect();
    let swiglu_shape: Vec<usize> = swiglu_logits.layout().shape().iter().copied().collect();
    assert_eq!(default_shape, vec![2, 3, 8]);
    assert_eq!(swiglu_shape, vec![2, 3, 8]);
    assert_eq!(default_params, 8);
    assert_eq!(swiglu_model.parameters().len(), default_params + 1);
    assert_eq!(MlpKind::default(), MlpKind::ReluSquared);
}
