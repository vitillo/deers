use candle_core::{Device as CDevice, Tensor as CTensor, Var};
use deers::{Device, Tensor};

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

fn devices() -> [Device; 2] {
    [Device::Cpu, Device::Mps]
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
            let indices = Tensor::from_vec(indices.clone(), (2,), device);
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
            let indices = Tensor::from_vec(indices.clone(), (2,), device);
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
            let output = input.index_select(&indices);
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
fn index_select_backward_is_not_implemented() {
    // Arrange
    let data = vec![10.0f32, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0];
    let indices = vec![2i64, 0, 3];

    // Act
    let results: Vec<(Device, String)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (4, 3), device).attach();
            let indices = Tensor::from_vec(indices.clone(), (3,), device);
            let result = input.index_select(&indices).sum(vec![0, 1], false).backward();
            (device, format!("{:?}", result.unwrap_err()))
        })
        .collect();

    // Assert
    for (device, error) in results {
        assert!(
            error.contains("NotImplemented"),
            "index_select backward on {:?}: {}",
            device,
            error
        );
    }
}
