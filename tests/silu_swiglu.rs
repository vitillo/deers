use candle_core::{Device as CDevice, Tensor as CTensor};
use deers::nn::{self, Module};
use deers::{Device, Tensor};

const TOL: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");

    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < TOL, "{label}[{index}]: got {a}, expected {e}");
    }
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
fn swiglu_forward_supports_nonsquare_output() {
    // Arrange
    let store = nn::ParamStore::new();
    let swiglu = nn::SwiGLU::new(store.root(), 4, 6, 5);
    let params = swiglu.parameters();
    let weights: Vec<Vec<f32>> = params.iter().map(|p| p.to_vec::<f32>().unwrap()).collect();
    let gate = CTensor::from_vec(weights[0].clone(), &[4, 6], &CDevice::Cpu).unwrap();
    let up = CTensor::from_vec(weights[1].clone(), &[4, 6], &CDevice::Cpu).unwrap();
    let down = CTensor::from_vec(weights[2].clone(), &[6, 5], &CDevice::Cpu).unwrap();
    let input_data: Vec<f32> = (0..8).map(|v| v as f32 * 0.1 - 0.5).collect();
    let candle_x = CTensor::from_vec(input_data.clone(), &[2, 4], &CDevice::Cpu).unwrap();
    let expected = candle_x
        .matmul(&gate)
        .unwrap()
        .silu()
        .unwrap()
        .broadcast_mul(&candle_x.matmul(&up).unwrap())
        .unwrap()
        .matmul(&down)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let x = Tensor::from_vec(input_data, (2, 4), Device::Cpu);
    let out = swiglu.forward(&x).unwrap();

    // Assert
    assert_eq!(out.layout().shape().as_slice(), &[2, 5]);
    assert_close(&out.to_vec::<f32>().unwrap(), &expected, "swiglu nonsquare forward");
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
