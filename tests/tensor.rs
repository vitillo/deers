use candle_core::{Device as CDevice, Tensor as CTensor, Var};
use deers::{Device, Tensor};
use half::{bf16, f16};

const TOL: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");

    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < TOL, "{label}[{index}]: got {a}, expected {e}");
    }
}

fn candle_var(data: Vec<f32>, shape: &[usize]) -> Var {
    let tensor = CTensor::from_vec(data, shape, &CDevice::Cpu).unwrap();
    Var::from_tensor(&tensor).unwrap()
}

fn candle_grad(grads: &candle_core::backprop::GradStore, var: &Var) -> Vec<f32> {
    grads.get(var.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

fn devices() -> Vec<Device> {
    [Device::Cpu, Device::Cuda, Device::Mps]
        .into_iter()
        .filter(|device| device.is_available())
        .collect()
}

/// Host devices for CPU-only selectors. `argmax`, `topk`, and `sort` refuse
/// accelerator tensors, so their conformance tests run here while refusal
/// coverage lives in the accelerator tests below.
fn host_devices() -> Vec<Device> {
    vec![Device::Cpu]
}

/// Devices with on-device selection kernels. `where_cond` and `masked_fill`
/// compute on the host and on CUDA; MPS refuses them, so its coverage lives
/// in the refusal tests below.
fn select_devices() -> Vec<Device> {
    devices().into_iter().filter(|device| !matches!(device, Device::Mps)).collect()
}

#[test]
fn relu_forward_conforms() {
    // Arrange
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let shape = (2, 3);
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_input.flatten_all().unwrap().relu().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), shape, device);
            let output = input.relu();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("relu forward on {:?}", device));
    }
}

#[test]
fn relu_backward_conforms() {
    // Arrange
    let data = vec![-2.0f32, -1.0, 0.5, 1.0, -0.5, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_input.relu().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.relu().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("relu backward on {:?}", device));
    }
}

#[test]
fn matmul_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[3, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_lhs.matmul(&candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 3), device);
            let rhs = Tensor::from_vec(rhs.clone(), (3, 2), device);
            let output = lhs.matmul(&rhs);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("matmul forward on {:?}", device));
    }
}

#[test]
fn matmul_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 3]);
    let candle_rhs = candle_var(rhs.clone(), &[3, 2]);
    let candle_output = candle_lhs.matmul(&candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 3), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (3, 2), device).attach();
            let loss = lhs.matmul(&rhs).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(
            &lhs_grad,
            &expected_lhs_grad,
            &format!("matmul backward lhs on {:?}", device),
        );
        assert_close(
            &rhs_grad,
            &expected_rhs_grad,
            &format!("matmul backward rhs on {:?}", device),
        );
    }
}

#[test]
fn matmul_non_square_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[3, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_lhs.matmul(&candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 3), device);
            let rhs = Tensor::from_vec(rhs.clone(), (3, 2), device);
            let output = lhs.matmul(&rhs);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("matmul non-square forward on {:?}", device));
    }
}

#[test]
fn matmul_non_square_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 3]);
    let candle_rhs = candle_var(rhs.clone(), &[3, 2]);
    let candle_output = candle_lhs.matmul(&candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 3), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (3, 2), device).attach();
            let loss = lhs.matmul(&rhs).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(
            &lhs_grad,
            &expected_lhs_grad,
            &format!("matmul non-square backward lhs on {:?}", device),
        );
        assert_close(
            &rhs_grad,
            &expected_rhs_grad,
            &format!("matmul non-square backward rhs on {:?}", device),
        );
    }
}

#[test]
fn matmul_batched_3d_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let rhs = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 2, 3], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[2, 3, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_lhs.matmul(&candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), vec![2, 2, 3], device);
            let rhs = Tensor::from_vec(rhs.clone(), vec![2, 3, 2], device);
            let output = lhs.matmul(&rhs);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("matmul batched 3d forward on {:?}", device));
    }
}

#[test]
fn matmul_batched_3d_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let rhs = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 2, 3]);
    let candle_rhs = candle_var(rhs.clone(), &[2, 3, 2]);
    let candle_output = candle_lhs.matmul(&candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), vec![2, 2, 3], device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), vec![2, 3, 2], device).attach();
            let loss = lhs.matmul(&rhs).sum(vec![0, 1, 2], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(
            &lhs_grad,
            &expected_lhs_grad,
            &format!("matmul batched 3d backward lhs on {:?}", device),
        );
        assert_close(
            &rhs_grad,
            &expected_rhs_grad,
            &format!("matmul batched 3d backward rhs on {:?}", device),
        );
    }
}

#[test]
fn log_softmax_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 1.5, 0.5, -1.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_nn::ops::log_softmax(&candle_input, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.log_softmax(1);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("log_softmax forward on {:?}", device));
    }
}

#[test]
fn log_softmax_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 1.5, 0.5, -1.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_nn::ops::log_softmax(&candle_input, 1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.log_softmax(1).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(
            &actual_grad,
            &expected_grad,
            &format!("log_softmax backward on {:?}", device),
        );
    }
}

#[test]
fn log_forward_conforms() {
    // Arrange
    let data = vec![0.5f32, 1.0, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.log().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.log();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("log forward on {:?}", device));
    }
}

#[test]
fn log_backward_conforms() {
    // Arrange
    let data = vec![0.5f32, 1.0, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.log().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.log().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("log backward on {:?}", device));
    }
}

#[test]
fn exp_forward_conforms() {
    // Arrange
    let data = vec![0.5f32, 1.0, -0.5, 2.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.exp().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.exp();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("exp forward on {:?}", device));
    }
}

#[test]
fn exp_backward_conforms() {
    // Arrange
    let data = vec![0.5f32, 1.0, -0.5, 2.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.exp().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.exp().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("exp backward on {:?}", device));
    }
}

#[test]
fn scalar_powf_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[3], &CDevice::Cpu).unwrap();
    let expected = candle_input.powf(3.0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3,), device);
            let output = input.scalar_powf(3.0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("scalar_powf forward on {:?}", device));
    }
}

#[test]
fn scalar_powf_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[3]);
    let candle_output = candle_input.powf(3.0).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3,), device).attach();
            let loss = input.scalar_powf(3.0).sum(vec![0], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(
            &actual_grad,
            &expected_grad,
            &format!("scalar_powf backward on {:?}", device),
        );
    }
}

#[test]
fn sum_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_input.sum(1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.sum(vec![1], false);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("sum forward on {:?}", device));
    }
}

#[test]
fn sum_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_input.sum(1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.sum(vec![1], false).sum(vec![0], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("sum backward on {:?}", device));
    }
}

#[test]
fn broadcast_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[1, 3], &CDevice::Cpu).unwrap();
    let expected = candle_input
        .broadcast_as(&[2, 3])
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (1, 3), device);
            let output = input.broadcast((2, 3));
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("broadcast forward on {:?}", device));
    }
}

#[test]
fn broadcast_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[1, 3]);
    let candle_output = candle_input.broadcast_as(&[2, 3]).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (1, 3), device).attach();
            let loss = input.broadcast((2, 3)).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("broadcast backward on {:?}", device));
    }
}

#[test]
fn max_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.max(1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.max(vec![1], false);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("max forward on {:?}", device));
    }
}

#[test]
fn max_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.max(1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.max(vec![1], false).sum(vec![0], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("max backward on {:?}", device));
    }
}

#[test]
fn permute_forward_conforms() {
    // Arrange
    let data = (0..24).map(|v| v as f32).collect::<Vec<_>>();
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3, 4], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.permute((1, 2, 0)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3, 4), device);
            let output = input.permute(vec![1, 2, 0]);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("permute forward on {:?}", device));
    }
}

#[test]
fn permute_backward_conforms() {
    // Arrange
    let data = (0..24).map(|v| v as f32).collect::<Vec<_>>();
    let grad = (0..24).map(|v| v as f32).collect::<Vec<_>>();
    let candle_input = candle_var(data.clone(), &[2, 3, 4]);
    let candle_grad_tensor = CTensor::from_vec(grad.clone(), &[3, 4, 2], &CDevice::Cpu).unwrap();
    let candle_output = candle_input.permute((1, 2, 0)).unwrap();
    let candle_loss = candle_output.mul(&candle_grad_tensor).unwrap().sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3, 4), device).attach();
            let grad = Tensor::from_vec(grad.clone(), (3, 4, 2), device);
            let loss = (&input.permute(vec![1, 2, 0]) * &grad).sum(vec![0, 1, 2], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("permute backward on {:?}", device));
    }
}

#[test]
fn neg_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, -2.0, 3.0, -4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.neg().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = -&input;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("neg forward on {:?}", device));
    }
}

#[test]
fn neg_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, -2.0, 3.0, -4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.neg().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = (-&input).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("neg backward on {:?}", device));
    }
}

#[test]
fn add_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        (&candle_lhs + &candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device);
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device);
            let output = &lhs + &rhs;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("add forward on {:?}", device));
    }
}

#[test]
fn add_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 2]);
    let candle_rhs = candle_var(rhs.clone(), &[2, 2]);
    let candle_output = (&*candle_lhs + &*candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device).attach();
            let loss = (&lhs + &rhs).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(&lhs_grad, &expected_lhs_grad, &format!("add backward lhs on {:?}", device));
        assert_close(&rhs_grad, &expected_rhs_grad, &format!("add backward rhs on {:?}", device));
    }
}

#[test]
fn sub_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        (&candle_lhs - &candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device);
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device);
            let output = &lhs - &rhs;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("sub forward on {:?}", device));
    }
}

#[test]
fn sub_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 2]);
    let candle_rhs = candle_var(rhs.clone(), &[2, 2]);
    let candle_output = (&*candle_lhs - &*candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device).attach();
            let loss = (&lhs - &rhs).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(&lhs_grad, &expected_lhs_grad, &format!("sub backward lhs on {:?}", device));
        assert_close(&rhs_grad, &expected_rhs_grad, &format!("sub backward rhs on {:?}", device));
    }
}

#[test]
fn mul_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        (&candle_lhs * &candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device);
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device);
            let output = &lhs * &rhs;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("mul forward on {:?}", device));
    }
}

#[test]
fn mul_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 2]);
    let candle_rhs = candle_var(rhs.clone(), &[2, 2]);
    let candle_output = (&*candle_lhs * &*candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device).attach();
            let loss = (&lhs * &rhs).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(&lhs_grad, &expected_lhs_grad, &format!("mul backward lhs on {:?}", device));
        assert_close(&rhs_grad, &expected_rhs_grad, &format!("mul backward rhs on {:?}", device));
    }
}

#[test]
fn div_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        (&candle_lhs / &candle_rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device);
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device);
            let output = &lhs / &rhs;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("div forward on {:?}", device));
    }
}

#[test]
fn div_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 2]);
    let candle_rhs = candle_var(rhs.clone(), &[2, 2]);
    let candle_output = (&*candle_lhs / &*candle_rhs).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device).attach();
            let loss = (&lhs / &rhs).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(&lhs_grad, &expected_lhs_grad, &format!("div backward lhs on {:?}", device));
        assert_close(&rhs_grad, &expected_rhs_grad, &format!("div backward rhs on {:?}", device));
    }
}

#[test]
fn scalar_add_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = (&candle_input + 2.0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = &input + 2.0;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("scalar add forward on {:?}", device));
    }
}

#[test]
fn scalar_add_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = (&*candle_input + 2.0).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = (&input + 2.0).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("scalar add backward on {:?}", device));
    }
}

#[test]
fn scalar_sub_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = (&candle_input - 2.0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = &input - 2.0;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("scalar sub forward on {:?}", device));
    }
}

#[test]
fn scalar_sub_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = (&*candle_input - 2.0).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = (&input - 2.0).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("scalar sub backward on {:?}", device));
    }
}

#[test]
fn scalar_mul_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = (&candle_input * 2.0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = &input * 2.0;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("scalar mul forward on {:?}", device));
    }
}

#[test]
fn scalar_mul_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = (&*candle_input * 2.0).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = (&input * 2.0).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("scalar mul backward on {:?}", device));
    }
}

#[test]
fn scalar_div_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = (&candle_input / 2.0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = &input / 2.0;
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("scalar div forward on {:?}", device));
    }
}

#[test]
fn scalar_div_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = (&*candle_input / 2.0).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = (&input / 2.0).sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("scalar div backward on {:?}", device));
    }
}

#[test]
fn zeros_forward_conforms() {
    // Arrange
    let expected = CTensor::from_vec(vec![0.0f32; 6], &[2, 3], &CDevice::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let output = Tensor::zeros((2, 3), deers::DType::F32, device);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("zeros forward on {:?}", device));
    }
}

#[test]
fn ones_forward_conforms() {
    // Arrange
    let expected = CTensor::from_vec(vec![1.0f32; 6], &[2, 3], &CDevice::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let output = Tensor::ones((2, 3), deers::DType::F32, device);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("ones forward on {:?}", device));
    }
}

#[test]
fn from_vec_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let expected = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let output = Tensor::from_vec(data.clone(), (2, 2), device);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("from_vec forward on {:?}", device));
    }
}

#[test]
fn to_device_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let expected = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), Device::Cpu);
            let output = input.to_device(device).unwrap();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("to_device forward on {:?}", device));
    }
}

#[test]
fn ones_like_forward_conforms() {
    // Arrange
    let data = vec![4.0f32, 5.0, 6.0, 7.0];
    let expected = CTensor::from_vec(vec![1.0f32; 4], &[2, 2], &CDevice::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.ones_like();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("ones_like forward on {:?}", device));
    }
}

#[test]
fn zeros_like_forward_conforms() {
    // Arrange
    let data = vec![4.0f32, 5.0, 6.0, 7.0];
    let expected = CTensor::from_vec(vec![0.0f32; 4], &[2, 2], &CDevice::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.zeros_like();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("zeros_like forward on {:?}", device));
    }
}

#[test]
fn rand_forward_has_valid_range() {
    // Arrange
    let len = 64;

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let output = Tensor::rand((len,), deers::DType::F32, device);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual.len(), len, "rand length on {:?}", device);
        assert!(actual.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    }
}

#[test]
fn randn_forward_has_finite_values() {
    // Arrange
    let len = 64;

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let output = Tensor::randn((len,), deers::DType::F32, device);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual.len(), len, "randn length on {:?}", device);
        assert!(actual.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn narrow_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.narrow(1, 1, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.narrow(1, 1, 1);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("narrow forward on {:?}", device));
    }
}

#[test]
fn narrow_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.narrow(1, 1, 1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.narrow(1, 1, 1).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("narrow backward on {:?}", device));
    }
}

#[test]
fn reshape_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = CTensor::from_vec(data.clone(), &[3, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.reshape((2, 3)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 2), device);
            let output = input.reshape((2, 3));
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("reshape forward on {:?}", device));
    }
}

#[test]
fn reshape_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = candle_var(data.clone(), &[3, 2]);
    let candle_output = candle_input.reshape((2, 3)).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 2), device).attach();
            let loss = input.reshape((2, 3)).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("reshape backward on {:?}", device));
    }
}

#[test]
fn transpose_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.transpose(0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.transpose(None);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("transpose forward on {:?}", device));
    }
}

#[test]
fn transpose_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.transpose(0, 1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.transpose(None).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("transpose backward on {:?}", device));
    }
}

#[test]
fn compact_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.transpose(0, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.transpose(None).compact();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("compact forward on {:?}", device));
    }
}

#[test]
fn compact_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let grad = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_grad_tensor = CTensor::from_vec(grad.clone(), &[3, 2], &CDevice::Cpu).unwrap();
    let candle_output = candle_input.transpose(0, 1).unwrap();
    let candle_loss = candle_output.mul(&candle_grad_tensor).unwrap().sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let grad = Tensor::from_vec(grad.clone(), (3, 2), device);
            let loss = (&input.transpose(None).compact() * &grad).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("compact backward on {:?}", device));
    }
}

#[test]
fn powf_forward_conforms() {
    // Arrange
    let base = vec![1.0f32, 2.0, 3.0];
    let exp = vec![2.0f32, 3.0, 1.5];
    let candle_base = CTensor::from_vec(base.clone(), &[3], &CDevice::Cpu).unwrap();
    let candle_exp = CTensor::from_vec(exp.clone(), &[3], &CDevice::Cpu).unwrap();
    let expected =
        candle_base.pow(&candle_exp).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let base = Tensor::from_vec(base.clone(), (3,), device);
            let exp = Tensor::from_vec(exp.clone(), (3,), device);
            let output = base.powf(&exp);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("powf forward on {:?}", device));
    }
}

#[test]
fn powf_backward_conforms() {
    // Arrange
    let base = vec![1.0f32, 2.0, 3.0];
    let exp = vec![2.0f32, 3.0, 1.5];
    let candle_base = candle_var(base.clone(), &[3]);
    let candle_exp = candle_var(exp.clone(), &[3]);
    let candle_output = candle_base.broadcast_pow(&candle_exp).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_base_grad = candle_grad(&candle_grads, &candle_base);
    let expected_exp_grad = candle_grad(&candle_grads, &candle_exp);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let base = Tensor::from_vec(base.clone(), (3,), device).attach();
            let exp = Tensor::from_vec(exp.clone(), (3,), device).attach();
            let loss = base.powf(&exp).sum(vec![0], false);
            let grads = loss.backward().unwrap();
            let base_grad = grads.get(base.id()).unwrap().to_vec::<f32>().unwrap();
            let exp_grad = grads.get(exp.id()).unwrap().to_vec::<f32>().unwrap();
            (device, base_grad, exp_grad)
        })
        .collect();

    // Assert
    for (device, base_grad, exp_grad) in results {
        assert_close(
            &base_grad,
            &expected_base_grad,
            &format!("powf backward base on {:?}", device),
        );
        assert_close(&exp_grad, &expected_exp_grad, &format!("powf backward exp on {:?}", device));
    }
}

#[test]
fn sqrt_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 4.0, 9.0, 16.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.sqrt().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.sqrt();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("sqrt forward on {:?}", device));
    }
}

#[test]
fn sqrt_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 4.0, 9.0, 16.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.sqrt().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.sqrt().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("sqrt backward on {:?}", device));
    }
}

#[test]
fn sin_forward_conforms() {
    // Arrange
    let data = vec![0.0f32, 1.0, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.sin().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.sin();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("sin forward on {:?}", device));
    }
}

#[test]
fn sin_backward_conforms() {
    // Arrange
    let data = vec![0.0f32, 1.0, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.sin().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.sin().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("sin backward on {:?}", device));
    }
}

#[test]
fn cos_forward_conforms() {
    // Arrange
    let data = vec![0.0f32, 1.0, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.cos().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.cos();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("cos forward on {:?}", device));
    }
}

#[test]
fn cos_backward_conforms() {
    // Arrange
    let data = vec![0.0f32, 1.0, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.cos().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.cos().sum(vec![0, 1], true);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("cos backward on {:?}", device));
    }
}

#[test]
fn log_sum_exp_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 1.5, 0.5, -1.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.log_sum_exp(1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.log_sum_exp(vec![1]);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("log_sum_exp forward on {:?}", device));
    }
}

#[test]
fn log_sum_exp_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 1.5, 0.5, -1.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_input.log_sum_exp(1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.log_sum_exp(vec![1]).sum(vec![0], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(
            &actual_grad,
            &expected_grad,
            &format!("log_sum_exp backward on {:?}", device),
        );
    }
}

#[test]
fn cat_forward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = CTensor::from_vec(lhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_rhs = CTensor::from_vec(rhs.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = CTensor::cat(&[&candle_lhs, &candle_rhs], 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device);
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device);
            let output = Tensor::cat(&[lhs, rhs], 0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("cat forward on {:?}", device));
    }
}

#[test]
fn cat_backward_conforms() {
    // Arrange
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let candle_lhs = candle_var(lhs.clone(), &[2, 2]);
    let candle_rhs = candle_var(rhs.clone(), &[2, 2]);
    let candle_output = CTensor::cat(&[&*candle_lhs, &*candle_rhs], 0).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_lhs_grad = candle_grad(&candle_grads, &candle_lhs);
    let expected_rhs_grad = candle_grad(&candle_grads, &candle_rhs);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let lhs = Tensor::from_vec(lhs.clone(), (2, 2), device).attach();
            let rhs = Tensor::from_vec(rhs.clone(), (2, 2), device).attach();
            let loss = Tensor::cat(&[lhs.clone(), rhs.clone()], 0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let lhs_grad = grads.get(lhs.id()).unwrap().to_vec::<f32>().unwrap();
            let rhs_grad = grads.get(rhs.id()).unwrap().to_vec::<f32>().unwrap();
            (device, lhs_grad, rhs_grad)
        })
        .collect();

    // Assert
    for (device, lhs_grad, rhs_grad) in results {
        assert_close(&lhs_grad, &expected_lhs_grad, &format!("cat backward lhs on {:?}", device));
        assert_close(&rhs_grad, &expected_rhs_grad, &format!("cat backward rhs on {:?}", device));
    }
}

#[test]
fn sigmoid_forward_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.0, 1.0, 2.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_nn::ops::sigmoid(&candle_input)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.sigmoid();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("sigmoid forward on {:?}", device));
    }
}

#[test]
fn sigmoid_backward_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.0, 1.0, 2.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_nn::ops::sigmoid(&candle_input).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.sigmoid().sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("sigmoid backward on {:?}", device));
    }
}

#[test]
fn tanh_forward_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.0, 1.0, 2.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_input.tanh().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.tanh();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("tanh forward on {:?}", device));
    }
}

#[test]
fn tanh_backward_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.0, 1.0, 2.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.tanh().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.tanh().sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("tanh backward on {:?}", device));
    }
}

#[test]
fn gelu_forward_conforms() {
    // Arrange
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_input.gelu().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.gelu();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("gelu forward on {:?}", device));
    }
}

#[test]
fn gelu_backward_conforms() {
    // Arrange
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_input.gelu().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.gelu().sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("gelu backward on {:?}", device));
    }
}

#[test]
fn silu_forward_conforms() {
    // Arrange
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_input.silu().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.silu();
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("silu forward on {:?}", device));
    }
}

#[test]
fn silu_backward_conforms() {
    // Arrange
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_input.silu().unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.silu().sum(vec![0, 1], false);
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
fn softmax_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 1.5, 0.5, -1.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_nn::ops::softmax(&candle_input, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.softmax(1);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("softmax forward on {:?}", device));
    }
}

#[test]
fn softmax_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 1.5, 0.5, -1.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_nn::ops::softmax(&candle_input, 1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.softmax(1).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("softmax backward on {:?}", device));
    }
}

#[test]
fn mean_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected = candle_input.mean(1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.mean(vec![1], false);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("mean forward on {:?}", device));
    }
}

#[test]
fn mean_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_output = candle_input.mean(1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let loss = input.mean(vec![1], false).sum(vec![0], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("mean backward on {:?}", device));
    }
}

#[test]
fn gather_forward_conforms() {
    // Arrange
    let data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let indices = vec![1i64, 2];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let candle_indices = CTensor::from_vec(indices.clone(), &[2, 1], &CDevice::Cpu).unwrap();
    let expected = candle_input
        .gather(&candle_indices, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let indices = Tensor::from_vec(indices.clone(), (2, 1), device);
            let output = input.gather(1, &indices);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("gather forward on {:?}", device));
    }
}

#[test]
fn gather_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![0i64, 2];
    let candle_input = candle_var(data.clone(), &[2, 3]);
    let candle_indices = CTensor::from_vec(indices.clone(), &[2, 1], &CDevice::Cpu).unwrap();
    let candle_output = candle_input.gather(&candle_indices, 1).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device).attach();
            let indices = Tensor::from_vec(indices.clone(), (2, 1), device);
            let loss = input.gather(1, &indices).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("gather backward on {:?}", device));
    }
}

#[test]
fn index_select_forward_conforms() {
    // Arrange
    let data = vec![10.0f32, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0];
    let indices = vec![2i64, 0, 3];
    let candle_input = CTensor::from_vec(data.clone(), &[4, 3], &CDevice::Cpu).unwrap();
    let candle_indices = CTensor::from_vec(indices.clone(), &[3], &CDevice::Cpu).unwrap();
    let expected = candle_input
        .index_select(&candle_indices, 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (4, 3), device);
            let indices = Tensor::from_vec(indices.clone(), (3,), device);
            let output = input.index_select(0, &indices);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("index_select forward on {:?}", device));
    }
}

#[test]
fn index_select_forward_dim_conforms() {
    // Arrange
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let indices = vec![2i64, 0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2, 3], &CDevice::Cpu).unwrap();
    let candle_indices = CTensor::from_vec(indices.clone(), &[2], &CDevice::Cpu).unwrap();
    let expected = candle_input
        .index_select(&candle_indices, 2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2, 3), device);
            let indices = Tensor::from_vec(indices.clone(), (2,), device);
            let output = input.index_select(2, &indices);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("index_select forward on {:?}", device));
    }
}

#[test]
fn index_select_backward_conforms() {
    // Arrange
    let data = vec![10.0f32, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0];
    let indices = vec![2i64, 0, 2];
    let expected_grad = vec![1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (4, 3), device).attach();
            let indices = Tensor::from_vec(indices.clone(), (3,), device);
            let loss = input.index_select(0, &indices).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(
            &actual_grad,
            &expected_grad,
            &format!("index_select backward on {:?}", device),
        );
    }
}

#[test]
fn argmax_forward_drops_dim() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0, 0.0, 5.0];
    let expected = vec![1i64, 2];

    // Act
    let results: Vec<(Device, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.argmax(1, false);
            let values = output.to_vec::<i64>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual, expected, "argmax forward on {:?}", device);
    }
}

#[test]
fn argmax_forward_keep_dims() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0, 0.0, 5.0];
    let expected = vec![1i64, 2];

    // Act
    let results: Vec<(Device, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.argmax(1, true);
            let values = output.to_vec::<i64>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual, expected, "argmax keep_dims on {:?}", device);
    }
}

#[test]
fn topk_forward_values_and_indices() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0, 0.0, 5.0];
    let expected_values = vec![3.0f32, 2.0, 5.0, 4.0];
    let expected_indices = vec![1i64, 2, 2, 0];

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let (values, indices) = input.topk(2, 1);
            let values = values.to_vec::<f32>().unwrap();
            let indices = indices.to_vec::<i64>().unwrap();
            (device, values, indices)
        })
        .collect();

    // Assert
    for (device, values, indices) in results {
        assert_close(&values, &expected_values, &format!("topk values on {:?}", device));
        assert_eq!(indices, expected_indices, "topk indices on {:?}", device);
    }
}

#[test]
fn sort_forward_ascending() {
    // Arrange
    let data = vec![3.0f32, 1.0, 2.0, 5.0, 4.0, 0.0];
    let expected_values = vec![1.0f32, 2.0, 3.0, 0.0, 4.0, 5.0];
    let expected_indices = vec![1i64, 2, 0, 2, 1, 0];

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let (values, indices) = input.sort(1, false);
            let values = values.to_vec::<f32>().unwrap();
            let indices = indices.to_vec::<i64>().unwrap();
            (device, values, indices)
        })
        .collect();

    // Assert
    for (device, values, indices) in results {
        assert_close(&values, &expected_values, &format!("sort values on {:?}", device));
        assert_eq!(indices, expected_indices, "sort indices on {:?}", device);
    }
}

#[test]
fn sort_forward_descending() {
    // Arrange
    let data = vec![3.0f32, 1.0, 2.0];
    let expected_values = vec![3.0f32, 2.0, 1.0];
    let expected_indices = vec![0i64, 2, 1];

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3,), device);
            let (values, indices) = input.sort(0, true);
            let values = values.to_vec::<f32>().unwrap();
            let indices = indices.to_vec::<i64>().unwrap();
            (device, values, indices)
        })
        .collect();

    // Assert
    for (device, values, indices) in results {
        assert_close(&values, &expected_values, &format!("sort desc values on {:?}", device));
        assert_eq!(indices, expected_indices, "sort desc indices on {:?}", device);
    }
}

#[test]
fn clamp_forward_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.5, 1.5, 3.0];
    let expected = vec![0.0f32, 0.5, 1.5, 2.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.clamp(0.0, 2.0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("clamp forward on {:?}", device));
    }
}

#[test]
fn clamp_backward_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.5, 2.0, 3.0];
    let expected_grad = vec![0.0f32, 1.0, 1.0, 0.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.clamp(0.0, 2.0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("clamp backward on {:?}", device));
    }
}

#[test]
fn where_forward_conforms() {
    // Arrange
    let cond = vec![1i64, 0, 0, 1];
    let on_true = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false = vec![10.0f32, 20.0, 30.0, 40.0];
    let expected = vec![1.0f32, 20.0, 30.0, 4.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = select_devices()
        .into_iter()
        .map(|device| {
            let cond = Tensor::from_vec(cond.clone(), (2, 2), device);
            let on_true = Tensor::from_vec(on_true.clone(), (2, 2), device);
            let on_false = Tensor::from_vec(on_false.clone(), (2, 2), device);
            let output = cond.where_cond(&on_true, &on_false);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("where forward on {:?}", device));
    }
}

#[test]
fn where_backward_conforms() {
    // Arrange
    let cond = vec![1i64, 0, 0, 1];
    let on_true = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false = vec![10.0f32, 20.0, 30.0, 40.0];
    let expected_true_grad = vec![1.0f32, 0.0, 0.0, 1.0];
    let expected_false_grad = vec![0.0f32, 1.0, 1.0, 0.0];

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = select_devices()
        .into_iter()
        .map(|device| {
            let cond = Tensor::from_vec(cond.clone(), (2, 2), device);
            let on_true = Tensor::from_vec(on_true.clone(), (2, 2), device).attach();
            let on_false = Tensor::from_vec(on_false.clone(), (2, 2), device).attach();
            let loss = cond.where_cond(&on_true, &on_false).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let true_grad = grads.get(on_true.id()).unwrap().to_vec::<f32>().unwrap();
            let false_grad = grads.get(on_false.id()).unwrap().to_vec::<f32>().unwrap();
            (device, true_grad, false_grad)
        })
        .collect();

    // Assert
    for (device, true_grad, false_grad) in results {
        assert_close(
            &true_grad,
            &expected_true_grad,
            &format!("where backward true on {:?}", device),
        );
        assert_close(
            &false_grad,
            &expected_false_grad,
            &format!("where backward false on {:?}", device),
        );
    }
}

#[test]
fn masked_fill_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0i64, 1, 0, 1];
    let expected = vec![1.0f32, 0.0, 3.0, 0.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = select_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
            let output = input.masked_fill(&mask, 0.0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("masked_fill forward on {:?}", device));
    }
}

#[test]
fn masked_fill_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0i64, 1, 0, 1];
    let expected_grad = vec![1.0f32, 0.0, 1.0, 0.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = select_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
            let loss = input.masked_fill(&mask, 0.0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(
            &actual_grad,
            &expected_grad,
            &format!("masked_fill backward on {:?}", device),
        );
    }
}

#[test]
fn tril_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected = vec![1.0f32, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 3), device);
            let output = input.tril(0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("tril forward on {:?}", device));
    }
}

#[test]
fn tril_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let expected_grad = vec![1.0f32, 0.0, 1.0, 1.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.tril(0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("tril backward on {:?}", device));
    }
}

#[test]
fn triu_forward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected = vec![1.0f32, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 3), device);
            let output = input.triu(0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("triu forward on {:?}", device));
    }
}

#[test]
fn triu_backward_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let expected_grad = vec![1.0f32, 1.0, 0.0, 1.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.triu(0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("triu backward on {:?}", device));
    }
}

#[test]
fn argmax_forward_candle_conforms() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0, 0.0, 5.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected: Vec<i64> = candle_input
        .argmax(1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
        .iter()
        .map(|v| *v as i64)
        .collect();

    // Act
    let results: Vec<(Device, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.argmax(1, false);
            let values = output.to_vec::<i64>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual, expected, "argmax forward on {:?}", device);
    }
}

#[test]
fn argmax_forward_dim0_candle_conforms() {
    // Arrange
    let data = vec![1.0f32, 3.0, 2.0, 4.0, 0.0, 5.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let expected: Vec<i64> = candle_input
        .argmax(0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
        .iter()
        .map(|v| *v as i64)
        .collect();

    // Act
    let results: Vec<(Device, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let output = input.argmax(0, false);
            let values = output.to_vec::<i64>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual, expected, "argmax dim0 forward on {:?}", device);
    }
}

#[test]
fn clamp_forward_candle_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.5, 1.5, 3.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected =
        candle_input.clamp(0f32, 2f32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device);
            let output = input.clamp(0.0, 2.0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("clamp forward on {:?}", device));
    }
}

#[test]
fn clamp_backward_candle_conforms() {
    // Arrange
    let data = vec![-1.0f32, 0.5, 1.5, 3.0];
    let candle_input = candle_var(data.clone(), &[2, 2]);
    let candle_output = candle_input.clamp(0f32, 2f32).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
            let loss = input.clamp(0.0, 2.0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("clamp backward on {:?}", device));
    }
}

#[test]
fn where_forward_candle_conforms() {
    // Arrange
    let cond = vec![1i64, 0, 0, 1];
    let on_true = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false = vec![10.0f32, 20.0, 30.0, 40.0];
    let candle_cond = CTensor::from_vec(cond.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_true = CTensor::from_vec(on_true.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_false = CTensor::from_vec(on_false.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let expected = candle_cond
        .where_cond(&candle_true, &candle_false)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = select_devices()
        .into_iter()
        .map(|device| {
            let cond = Tensor::from_vec(cond.clone(), (2, 2), device);
            let on_true = Tensor::from_vec(on_true.clone(), (2, 2), device);
            let on_false = Tensor::from_vec(on_false.clone(), (2, 2), device);
            let output = cond.where_cond(&on_true, &on_false);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("where forward on {:?}", device));
    }
}

#[test]
fn where_backward_candle_conforms() {
    // Arrange
    let cond = vec![1i64, 0, 0, 1];
    let on_true = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false = vec![10.0f32, 20.0, 30.0, 40.0];
    let candle_cond = CTensor::from_vec(cond.clone(), &[2, 2], &CDevice::Cpu).unwrap();
    let candle_true = candle_var(on_true.clone(), &[2, 2]);
    let candle_false = candle_var(on_false.clone(), &[2, 2]);
    let candle_output = candle_cond.where_cond(&candle_true, &candle_false).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_true_grad = candle_grad(&candle_grads, &candle_true);
    let expected_false_grad = candle_grad(&candle_grads, &candle_false);

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<f32>)> = select_devices()
        .into_iter()
        .map(|device| {
            let cond = Tensor::from_vec(cond.clone(), (2, 2), device);
            let on_true = Tensor::from_vec(on_true.clone(), (2, 2), device).attach();
            let on_false = Tensor::from_vec(on_false.clone(), (2, 2), device).attach();
            let loss = cond.where_cond(&on_true, &on_false).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let true_grad = grads.get(on_true.id()).unwrap().to_vec::<f32>().unwrap();
            let false_grad = grads.get(on_false.id()).unwrap().to_vec::<f32>().unwrap();
            (device, true_grad, false_grad)
        })
        .collect();

    // Assert
    for (device, true_grad, false_grad) in results {
        assert_close(
            &true_grad,
            &expected_true_grad,
            &format!("where backward true on {:?}", device),
        );
        assert_close(
            &false_grad,
            &expected_false_grad,
            &format!("where backward false on {:?}", device),
        );
    }
}

#[test]
fn sort_last_dim_candle_conforms() {
    // Arrange
    let data = vec![3.0f32, 1.0, 2.0, 5.0, 4.0, 0.0];
    let candle_input = CTensor::from_vec(data.clone(), &[2, 3], &CDevice::Cpu).unwrap();
    let (candle_values, candle_indices) = candle_input.sort_last_dim(true).unwrap();
    let expected_values = candle_values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected_indices: Vec<i64> = candle_indices
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
        .iter()
        .map(|v| *v as i64)
        .collect();

    // Act
    let results: Vec<(Device, Vec<f32>, Vec<i64>)> = host_devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (2, 3), device);
            let (values, indices) = input.sort(1, false);
            let values = values.to_vec::<f32>().unwrap();
            let indices = indices.to_vec::<i64>().unwrap();
            (device, values, indices)
        })
        .collect();

    // Assert
    for (device, values, indices) in results {
        assert_close(&values, &expected_values, &format!("sort values on {:?}", device));
        assert_eq!(indices, expected_indices, "sort indices on {:?}", device);
    }
}

#[test]
fn tril_forward_candle_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let candle_input = CTensor::from_vec(data.clone(), &[3, 3], &CDevice::Cpu).unwrap();
    let mask = CTensor::tril2(3, candle_core::DType::F32, &CDevice::Cpu).unwrap();
    let expected =
        candle_input.mul(&mask).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 3), device);
            let output = input.tril(0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("tril forward on {:?}", device));
    }
}

#[test]
fn tril_backward_candle_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mask = CTensor::tril2(3, candle_core::DType::F32, &CDevice::Cpu).unwrap();
    let candle_input = candle_var(data.clone(), &[3, 3]);
    let candle_output = candle_input.mul(&mask).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 3), device).attach();
            let loss = input.tril(0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("tril backward on {:?}", device));
    }
}

#[test]
fn triu_forward_candle_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let candle_input = CTensor::from_vec(data.clone(), &[3, 3], &CDevice::Cpu).unwrap();
    let mask = CTensor::triu2(3, candle_core::DType::F32, &CDevice::Cpu).unwrap();
    let expected =
        candle_input.mul(&mask).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 3), device);
            let output = input.triu(0);
            let values = output.to_vec::<f32>().unwrap();
            (device, values)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_close(&actual, &expected, &format!("triu forward on {:?}", device));
    }
}

#[test]
fn triu_backward_candle_conforms() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mask = CTensor::triu2(3, candle_core::DType::F32, &CDevice::Cpu).unwrap();
    let candle_input = candle_var(data.clone(), &[3, 3]);
    let candle_output = candle_input.mul(&mask).unwrap();
    let candle_loss = candle_output.sum_all().unwrap();
    let candle_grads = candle_loss.backward().unwrap();
    let expected_grad = candle_grad(&candle_grads, &candle_input);

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3, 3), device).attach();
            let loss = input.triu(0).sum(vec![0, 1], false);
            let grads = loss.backward().unwrap();
            let grad = grads.get(input.id()).unwrap().to_vec::<f32>().unwrap();
            (device, grad)
        })
        .collect();

    // Assert
    for (device, actual_grad) in results {
        assert_close(&actual_grad, &expected_grad, &format!("triu backward on {:?}", device));
    }
}

#[test]
fn scalar_compare_i64_conforms() {
    // Arrange
    let targets = vec![1i64, -100, 2, -100, 0];

    // Act
    let results: Vec<(Device, Vec<i64>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(targets.clone(), (5,), device);
            let ne = input.ne_scalar_i64(-100).to_vec::<i64>().unwrap();
            (device, ne)
        })
        .collect();

    // Assert
    for (device, ne) in results {
        assert_eq!(ne, vec![1, 0, 1, 0, 1], "ne_scalar on {:?}", device);
    }
}

#[test]
fn scalar_compare_f32_conforms() {
    // Arrange
    let data = vec![0.0f32, 1.5, 1.5, -2.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (4,), device);
            let ne = input.ne_scalar(1.5).to_vec::<f32>().unwrap();
            (device, ne)
        })
        .collect();

    // Assert
    for (device, ne) in results {
        assert_close(&ne, &[1.0, 0.0, 0.0, 1.0], &format!("ne_scalar f32 on {:?}", device));
    }
}

#[test]
fn scalar_compare_strided_cpu() {
    // Arrange
    let input = Tensor::from_vec(vec![1i64, 2, -100, 4, -100, 6], (2, 3), Device::Cpu);
    let transposed = input.transpose(None);

    // Act
    let ne = transposed.ne_scalar_i64(-100).to_vec::<i64>().unwrap();

    // Assert: the strided (3, 2) view compares in logical order.
    assert_eq!(ne, vec![1, 1, 1, 0, 0, 1]);
}

#[test]
fn scalar_compare_carries_no_gradient() {
    // Arrange
    let input = Tensor::from_vec(vec![1i64, -100, 2], (3,), Device::Cpu);

    // Act
    let mask = input.ne_scalar_i64(-100);

    // Assert
    assert!(!mask.requires_grad());
    assert!(mask.op().is_none());
}

#[test]
fn where_i64_branches_conform() {
    // Arrange: integer selection, as used for gather-safe targets.
    let cond = vec![1i64, 0, 0, 1];
    let on_true = vec![5i64, 6, 7, 8];
    let on_false = vec![0i64, 0, 0, 0];
    let expected = vec![5i64, 0, 0, 8];

    // Act
    let results: Vec<(Device, Vec<i64>)> = devices()
        .into_iter()
        .map(|device| {
            let cond = Tensor::from_vec(cond.clone(), (4,), device);
            let on_true = Tensor::from_vec(on_true.clone(), (4,), device);
            let on_false = Tensor::from_vec(on_false.clone(), (4,), device);
            let output = cond.where_cond(&on_true, &on_false).to_vec::<i64>().unwrap();
            (device, output)
        })
        .collect();

    // Assert
    for (device, actual) in results {
        assert_eq!(actual, expected, "where i64 on {:?}", device);
    }
}
// Selection ops keep whole tensors on device or refuse loudly. The tests
// below pin both halves: host behavior across dtypes, CUDA kernel parity
// with the host, and loud refusal off host where no kernel exists.

/// Asserts an off-host call refused loudly with a message containing `wanted`.
fn assert_refusal<T>(result: std::thread::Result<T>, op: &str, device: Device, wanted: &str) {
    match result {
        Ok(_) => panic!("{op} must refuse {device:?} tensors"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .expect("refusal must carry a message");
            assert!(
                message.contains(wanted),
                "{op} refusal on {device:?} must contain {wanted:?}, got: {message}"
            );
        }
    }
}

fn catch_argmax(device: Device) -> std::thread::Result<Tensor> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Tensor::from_vec(vec![1.0f32, 2.0], (2,), device).argmax(0, false)
    }))
}

fn catch_topk(device: Device) -> std::thread::Result<(Tensor, Tensor)> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Tensor::from_vec(vec![1.0f32, 2.0], (2,), device).topk(1, 0)
    }))
}

fn catch_sort(device: Device) -> std::thread::Result<(Tensor, Tensor)> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Tensor::from_vec(vec![1.0f32, 2.0], (2,), device).sort(0, false)
    }))
}

#[test]
fn argmax_refuses_accelerator_tensors() {
    // Arrange
    let targets: Vec<Device> = devices().into_iter().filter(|d| *d != Device::Cpu).collect();

    // Act
    let refusals: Vec<(Device, std::thread::Result<Tensor>)> =
        targets.into_iter().map(|device| (device, catch_argmax(device))).collect();

    // Assert
    for (device, result) in refusals {
        assert_refusal(result, "argmax", device, "host");
    }
}

#[test]
fn topk_refuses_accelerator_tensors() {
    // Arrange
    let targets: Vec<Device> = devices().into_iter().filter(|d| *d != Device::Cpu).collect();

    // Act
    let refusals: Vec<(Device, std::thread::Result<(Tensor, Tensor)>)> =
        targets.into_iter().map(|device| (device, catch_topk(device))).collect();

    // Assert
    for (device, result) in refusals {
        assert_refusal(result, "topk", device, "host");
    }
}

#[test]
fn sort_refuses_accelerator_tensors() {
    // Arrange
    let targets: Vec<Device> = devices().into_iter().filter(|d| *d != Device::Cpu).collect();

    // Act
    let refusals: Vec<(Device, std::thread::Result<(Tensor, Tensor)>)> =
        targets.into_iter().map(|device| (device, catch_sort(device))).collect();

    // Assert
    for (device, result) in refusals {
        assert_refusal(result, "sort", device, "host");
    }
}

#[test]
fn where_cond_refuses_mps_tensors() {
    // Arrange: MPS is the only backend without a select kernel.
    let targets: Vec<Device> = devices().into_iter().filter(|d| matches!(d, Device::Mps)).collect();

    // Act
    let refusals: Vec<(Device, std::thread::Result<Tensor>)> = targets
        .into_iter()
        .map(|device| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let cond = Tensor::from_vec(vec![1i64, 0], (2,), device);
                let on_true = Tensor::from_vec(vec![1.0f32, 2.0], (2,), device);
                let on_false = Tensor::from_vec(vec![10.0f32, 20.0], (2,), device);
                cond.where_cond(&on_true, &on_false)
            }));
            (device, result)
        })
        .collect();

    // Assert
    for (device, result) in refusals {
        assert_refusal(result, "where_cond", device, "host");
    }
}

#[test]
fn masked_fill_refuses_mps_tensors() {
    // Arrange: MPS is the only backend without a select kernel.
    let targets: Vec<Device> = devices().into_iter().filter(|d| matches!(d, Device::Mps)).collect();

    // Act
    let refusals: Vec<(Device, std::thread::Result<Tensor>)> = targets
        .into_iter()
        .map(|device| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let input = Tensor::from_vec(vec![1.0f32, 2.0], (2,), device);
                let mask = Tensor::from_vec(vec![0i64, 1], (2,), device);
                input.masked_fill(&mask, 0.0)
            }));
            (device, result)
        })
        .collect();

    // Assert
    for (device, result) in refusals {
        assert_refusal(result, "masked_fill", device, "host");
    }
}

#[test]
fn where_cond_refuses_mixed_devices() {
    // Arrange
    if !Device::Cuda.is_available() {
        return;
    }
    let cond = Tensor::from_vec(vec![1i64, 0], (2,), Device::Cpu);
    let on_true = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cuda);
    let on_false = Tensor::from_vec(vec![10.0f32, 20.0], (2,), Device::Cuda);

    // Act
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cond.where_cond(&on_true, &on_false)
    }));

    // Assert
    assert_refusal(result, "where_cond", Device::Cuda, "DeviceMismatch");
}

#[test]
fn masked_fill_refuses_mixed_devices() {
    // Arrange
    if !Device::Cuda.is_available() {
        return;
    }
    let input = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cuda);
    let mask = Tensor::from_vec(vec![0i64, 1], (2,), Device::Cpu);

    // Act
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| input.masked_fill(&mask, 0.0)));

    // Assert
    assert_refusal(result, "masked_fill", Device::Cuda, "DeviceMismatch");
}

#[test]
fn argmax_host_supports_all_dtypes() {
    // Arrange
    let device = Device::Cpu;
    let expected = vec![1i64];

    // Act
    let f32_out = Tensor::from_vec(vec![1.0f32, 3.0, 2.0], (3,), device).argmax(0, false);
    let f16_out = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(3.0), f16::from_f32(2.0)],
        (3,),
        device,
    )
    .argmax(0, false);
    let bf16_out = Tensor::from_vec(
        vec![bf16::from_f32(1.0), bf16::from_f32(3.0), bf16::from_f32(2.0)],
        (3,),
        device,
    )
    .argmax(0, false);
    let i64_out = Tensor::from_vec(vec![1i64, 3, 2], (3,), device).argmax(0, false);

    // Assert
    for (label, output) in
        [("f32", f32_out), ("f16", f16_out), ("bf16", bf16_out), ("i64", i64_out)]
    {
        assert_eq!(output.to_vec::<i64>().unwrap(), expected, "argmax host {label}");
    }
}

#[test]
fn topk_host_supports_all_dtypes() {
    // Arrange
    let device = Device::Cpu;
    let expected_indices = vec![1i64, 2];

    // Act
    let (f32_values, f32_indices) =
        Tensor::from_vec(vec![1.0f32, 3.0, 2.0], (3,), device).topk(2, 0);
    let (f16_values, f16_indices) = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(3.0), f16::from_f32(2.0)],
        (3,),
        device,
    )
    .topk(2, 0);
    let (bf16_values, bf16_indices) = Tensor::from_vec(
        vec![bf16::from_f32(1.0), bf16::from_f32(3.0), bf16::from_f32(2.0)],
        (3,),
        device,
    )
    .topk(2, 0);
    let (i64_values, i64_indices) = Tensor::from_vec(vec![1i64, 3, 2], (3,), device).topk(2, 0);

    // Assert
    assert_close(&f32_values.to_vec::<f32>().unwrap(), &[3.0, 2.0], "topk host f32 values");
    assert_eq!(f32_indices.to_vec::<i64>().unwrap(), expected_indices, "topk host f32 indices");
    let f16_values: Vec<f32> =
        f16_values.to_vec::<f16>().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_close(&f16_values, &[3.0, 2.0], "topk host f16 values");
    assert_eq!(f16_indices.to_vec::<i64>().unwrap(), expected_indices, "topk host f16 indices");
    let bf16_values: Vec<f32> =
        bf16_values.to_vec::<bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_close(&bf16_values, &[3.0, 2.0], "topk host bf16 values");
    assert_eq!(bf16_indices.to_vec::<i64>().unwrap(), expected_indices, "topk host bf16 indices");
    assert_eq!(i64_values.to_vec::<i64>().unwrap(), vec![3, 2], "topk host i64 values");
    assert_eq!(i64_indices.to_vec::<i64>().unwrap(), expected_indices, "topk host i64 indices");
}

#[test]
fn sort_host_supports_all_dtypes() {
    // Arrange
    let device = Device::Cpu;
    let expected_indices = vec![1i64, 2, 0];

    // Act
    let (f32_values, f32_indices) =
        Tensor::from_vec(vec![3.0f32, 1.0, 2.0], (3,), device).sort(0, false);
    let (f16_values, f16_indices) = Tensor::from_vec(
        vec![f16::from_f32(3.0), f16::from_f32(1.0), f16::from_f32(2.0)],
        (3,),
        device,
    )
    .sort(0, false);
    let (bf16_values, bf16_indices) = Tensor::from_vec(
        vec![bf16::from_f32(3.0), bf16::from_f32(1.0), bf16::from_f32(2.0)],
        (3,),
        device,
    )
    .sort(0, false);
    let (i64_values, i64_indices) = Tensor::from_vec(vec![3i64, 1, 2], (3,), device).sort(0, false);

    // Assert
    assert_close(&f32_values.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0], "sort host f32 values");
    assert_eq!(f32_indices.to_vec::<i64>().unwrap(), expected_indices, "sort host f32 indices");
    let f16_values: Vec<f32> =
        f16_values.to_vec::<f16>().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_close(&f16_values, &[1.0, 2.0, 3.0], "sort host f16 values");
    assert_eq!(f16_indices.to_vec::<i64>().unwrap(), expected_indices, "sort host f16 indices");
    let bf16_values: Vec<f32> =
        bf16_values.to_vec::<bf16>().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_close(&bf16_values, &[1.0, 2.0, 3.0], "sort host bf16 values");
    assert_eq!(bf16_indices.to_vec::<i64>().unwrap(), expected_indices, "sort host bf16 indices");
    assert_eq!(i64_values.to_vec::<i64>().unwrap(), vec![1, 2, 3], "sort host i64 values");
    assert_eq!(i64_indices.to_vec::<i64>().unwrap(), expected_indices, "sort host i64 indices");
}

#[test]
fn where_host_supports_all_cond_dtypes() {
    // Arrange
    let device = Device::Cpu;
    let on_true = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false = vec![10.0f32, 20.0, 30.0, 40.0];
    let expected = vec![1.0f32, 20.0, 30.0, 4.0];

    // Act
    let f32_cond = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), device);
    let f16_cond = Tensor::from_vec(
        vec![f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(1.0)],
        (2, 2),
        device,
    );
    let bf16_cond = Tensor::from_vec(
        vec![bf16::from_f32(1.0), bf16::from_f32(0.0), bf16::from_f32(0.0), bf16::from_f32(1.0)],
        (2, 2),
        device,
    );
    let i64_cond = Tensor::from_vec(vec![1i64, 0, 0, 1], (2, 2), device);
    let outputs = [f32_cond, f16_cond, bf16_cond, i64_cond].into_iter().map(|cond| {
        let on_true = Tensor::from_vec(on_true.clone(), (2, 2), device);
        let on_false = Tensor::from_vec(on_false.clone(), (2, 2), device);
        cond.where_cond(&on_true, &on_false).to_vec::<f32>().unwrap()
    });

    // Assert
    for (dtype, actual) in ["f32", "f16", "bf16", "i64"].into_iter().zip(outputs) {
        assert_close(&actual, &expected, &format!("where host cond {dtype}"));
    }
}

#[test]
fn masked_fill_host_supports_all_cond_dtypes() {
    // Arrange
    let device = Device::Cpu;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let expected = vec![1.0f32, 9.0, 3.0, 9.0];

    // Act
    let f32_mask = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 1.0], (2, 2), device);
    let f16_mask = Tensor::from_vec(
        vec![f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(1.0)],
        (2, 2),
        device,
    );
    let bf16_mask = Tensor::from_vec(
        vec![bf16::from_f32(0.0), bf16::from_f32(1.0), bf16::from_f32(0.0), bf16::from_f32(1.0)],
        (2, 2),
        device,
    );
    let i64_mask = Tensor::from_vec(vec![0i64, 1, 0, 1], (2, 2), device);
    let outputs = [f32_mask, f16_mask, bf16_mask, i64_mask].into_iter().map(|mask| {
        let input = Tensor::from_vec(data.clone(), (2, 2), device);
        input.masked_fill(&mask, 9.0).to_vec::<f32>().unwrap()
    });

    // Assert
    for (dtype, actual) in ["f32", "f16", "bf16", "i64"].into_iter().zip(outputs) {
        assert_close(&actual, &expected, &format!("masked_fill host mask {dtype}"));
    }
}

#[test]
fn where_forward_cuda_matches_host_across_dtypes() {
    // Arrange
    if !Device::Cuda.is_available() {
        return;
    }

    // Act: identical inputs on the host and on CUDA, including a transposed
    // (non-compact) view that forces the device compact path.
    let cond = vec![1i64, 0, 0, 1];
    let on_true_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false_f32 = vec![10.0f32, 20.0, 30.0, 40.0];
    let host = Tensor::from_vec(cond.clone(), (2, 2), Device::Cpu).where_cond(
        &Tensor::from_vec(on_true_f32.clone(), (2, 2), Device::Cpu),
        &Tensor::from_vec(on_false_f32.clone(), (2, 2), Device::Cpu),
    );
    let accel = Tensor::from_vec(cond.clone(), (2, 2), Device::Cuda).where_cond(
        &Tensor::from_vec(on_true_f32.clone(), (2, 2), Device::Cuda),
        &Tensor::from_vec(on_false_f32.clone(), (2, 2), Device::Cuda),
    );
    let host_t = Tensor::from_vec(on_true_f32.clone(), (2, 2), Device::Cpu).transpose(None);
    let host_t = Tensor::from_vec(cond.clone(), (2, 2), Device::Cpu).where_cond(
        &host_t,
        &Tensor::from_vec(on_false_f32.clone(), (2, 2), Device::Cpu).transpose(None),
    );
    let accel_t = Tensor::from_vec(on_true_f32.clone(), (2, 2), Device::Cuda).transpose(None);
    let accel_t = Tensor::from_vec(cond.clone(), (2, 2), Device::Cuda).where_cond(
        &accel_t,
        &Tensor::from_vec(on_false_f32.clone(), (2, 2), Device::Cuda).transpose(None),
    );
    let host_i64 = Tensor::from_vec(cond.clone(), (2, 2), Device::Cpu).where_cond(
        &Tensor::from_vec(vec![1i64, 2, 3, 4], (2, 2), Device::Cpu),
        &Tensor::from_vec(vec![10i64, 20, 30, 40], (2, 2), Device::Cpu),
    );
    let accel_i64 = Tensor::from_vec(cond.clone(), (2, 2), Device::Cuda).where_cond(
        &Tensor::from_vec(vec![1i64, 2, 3, 4], (2, 2), Device::Cuda),
        &Tensor::from_vec(vec![10i64, 20, 30, 40], (2, 2), Device::Cuda),
    );

    // Assert
    assert_eq!(accel.device(), Device::Cuda, "where output must stay on CUDA");
    assert_close(
        &accel.to_vec::<f32>().unwrap(),
        &host.to_vec::<f32>().unwrap(),
        "where cuda f32 matches host",
    );
    assert_close(
        &accel_t.to_vec::<f32>().unwrap(),
        &host_t.to_vec::<f32>().unwrap(),
        "where cuda transposed matches host",
    );
    assert_eq!(
        accel_i64.to_vec::<i64>().unwrap(),
        host_i64.to_vec::<i64>().unwrap(),
        "where cuda i64 matches host"
    );
}

#[test]
fn where_backward_cuda_matches_host() {
    // Arrange
    if !Device::Cuda.is_available() {
        return;
    }
    let cond = vec![1i64, 0, 0, 1];
    let on_true = vec![1.0f32, 2.0, 3.0, 4.0];
    let on_false = vec![10.0f32, 20.0, 30.0, 40.0];

    // Act
    let run = |device: Device| {
        let cond = Tensor::from_vec(cond.clone(), (2, 2), device);
        let on_true = Tensor::from_vec(on_true.clone(), (2, 2), device).attach();
        let on_false = Tensor::from_vec(on_false.clone(), (2, 2), device).attach();
        let loss = cond.where_cond(&on_true, &on_false).sum(vec![0, 1], false);
        let grads = loss.backward().unwrap();
        (
            grads.get(on_true.id()).unwrap().to_vec::<f32>().unwrap(),
            grads.get(on_false.id()).unwrap().to_vec::<f32>().unwrap(),
        )
    };
    let (host_true, host_false) = run(Device::Cpu);
    let (accel_true, accel_false) = run(Device::Cuda);

    // Assert
    assert_close(&accel_true, &host_true, "where cuda true grad matches host");
    assert_close(&accel_false, &host_false, "where cuda false grad matches host");
}

#[test]
fn masked_fill_forward_cuda_matches_host_across_dtypes() {
    // Arrange
    if !Device::Cuda.is_available() {
        return;
    }
    let mask = vec![0i64, 1, 0, 1];

    // Act
    let run_f32 = |device: Device| {
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device);
        let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
        input.masked_fill(&mask, 9.0).to_vec::<f32>().unwrap()
    };
    let run_i64 = |device: Device| {
        let input = Tensor::from_vec(vec![1i64, 2, 3, 4], (2, 2), device);
        let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
        input.masked_fill(&mask, 7.0).to_vec::<i64>().unwrap()
    };
    let run_f16 = |device: Device| {
        let input = Tensor::from_vec(
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
            (2, 2),
            device,
        );
        let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
        input
            .masked_fill(&mask, 9.0)
            .to_vec::<f16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect::<Vec<f32>>()
    };
    let run_bf16 = |device: Device| {
        let input = Tensor::from_vec(
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
            (2, 2),
            device,
        );
        let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
        input
            .masked_fill(&mask, 9.0)
            .to_vec::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect::<Vec<f32>>()
    };

    // Assert
    assert_close(&run_f32(Device::Cuda), &run_f32(Device::Cpu), "masked_fill cuda f32");
    assert_eq!(run_i64(Device::Cuda), run_i64(Device::Cpu), "masked_fill cuda i64");
    assert_close(&run_f16(Device::Cuda), &run_f16(Device::Cpu), "masked_fill cuda f16");
    assert_close(&run_bf16(Device::Cuda), &run_bf16(Device::Cpu), "masked_fill cuda bf16");
}

#[test]
fn masked_fill_backward_cuda_matches_host() {
    // Arrange
    if !Device::Cuda.is_available() {
        return;
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0i64, 1, 0, 1];

    // Act
    let run = |device: Device| {
        let input = Tensor::from_vec(data.clone(), (2, 2), device).attach();
        let mask = Tensor::from_vec(mask.clone(), (2, 2), device);
        let loss = input.masked_fill(&mask, 0.0).sum(vec![0, 1], false);
        let grads = loss.backward().unwrap();
        grads.get(input.id()).unwrap().to_vec::<f32>().unwrap()
    };

    // Assert
    assert_close(&run(Device::Cuda), &run(Device::Cpu), "masked_fill cuda grad matches host");
}
