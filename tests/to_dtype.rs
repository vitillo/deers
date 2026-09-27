use half::{bf16, f16};

use deers::{DType, Device, Tensor};

fn devices() -> Vec<Device> {
    [Device::Cpu, Device::Cuda, Device::Mps]
        .into_iter()
        .filter(|device| device.is_available())
        .collect()
}

#[test]
fn f32_to_f16_keeps_exact_values() {
    // Arrange
    let input = Tensor::from_vec(vec![0.0f32, 1.0, -2.0, 1.5], (4,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::F16).unwrap();

    // Assert
    assert_eq!(output.dtype(), DType::F16);
    assert_eq!(
        output.to_vec::<f16>().unwrap(),
        vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(-2.0),
            f16::from_f32(1.5),
        ]
    );
}

#[test]
fn f32_to_bf16_keeps_exact_values() {
    // Arrange
    let input = Tensor::from_vec(vec![0.0f32, 1.0, -2.0, 1.5, 100.0], (5,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::BF16).unwrap();

    // Assert
    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(
        output.to_vec::<bf16>().unwrap(),
        vec![
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(-2.0),
            bf16::from_f32(1.5),
            bf16::from_f32(100.0),
        ]
    );
}

#[test]
fn f32_to_f16_rounds_like_half_crate() {
    // Arrange: 0.1 has no exact F16 representation.
    let input = Tensor::from_vec(vec![0.1f32], (1,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::F16).unwrap();

    // Assert
    assert_eq!(output.to_vec::<f16>().unwrap(), vec![f16::from_f32(0.1)]);
}

#[test]
fn f16_to_f32_restores_values() {
    // Arrange
    let input = Tensor::from_vec(vec![f16::from_f32(1.5), f16::from_f32(-0.25)], (2,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::F32).unwrap();

    // Assert
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.to_vec::<f32>().unwrap(), vec![1.5, -0.25]);
}

#[test]
fn bf16_to_f32_restores_values() {
    // Arrange
    let input =
        Tensor::from_vec(vec![bf16::from_f32(3.140625), bf16::from_f32(-2.0)], (2,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::F32).unwrap();

    // Assert
    assert_eq!(output.to_vec::<f32>().unwrap(), vec![3.140625, -2.0]);
}

#[test]
fn f32_to_i64_truncates_toward_zero() {
    // Arrange
    let input = Tensor::from_vec(vec![1.9f32, -1.9, 2.0, -0.5, 0.0], (5,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::I64).unwrap();

    // Assert
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.to_vec::<i64>().unwrap(), vec![1, -1, 2, 0, 0]);
}

#[test]
fn i64_to_f32_widens_exactly() {
    // Arrange: small ints and a large power of two are all exact in F32.
    let input = Tensor::from_vec(vec![0i64, 1, -2, 1 << 30], (4,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::F32).unwrap();

    // Assert
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.to_vec::<f32>().unwrap(), vec![0.0, 1.0, -2.0, (1 << 30) as f32]);
}

#[test]
fn i64_to_f16_widens_small_ints() {
    // Arrange
    let input = Tensor::from_vec(vec![1i64, -2, 3], (3,), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::F16).unwrap();

    // Assert
    assert_eq!(
        output.to_vec::<f16>().unwrap(),
        vec![f16::from_f32(1.0), f16::from_f32(-2.0), f16::from_f32(3.0)]
    );
}

#[test]
fn same_dtype_conversion_is_noop() {
    // Arrange
    let input = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu).attach();

    // Act
    let output = input.to_dtype(DType::F32).unwrap();

    // Assert: no copy, no new graph node — the same tensor comes back.
    assert_eq!(output.id(), input.id());
    assert!(output.requires_grad());
    assert_eq!(output.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn conversion_preserves_shape_and_device() {
    // Arrange
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

    // Act
    let output = input.to_dtype(DType::BF16).unwrap();

    // Assert
    assert_eq!(output.layout().shape().as_slice(), &[2, 3]);
    assert_eq!(output.device(), Device::Cpu);
    assert_eq!(output.dtype(), DType::BF16);
}

#[test]
fn conversion_reads_strided_views() {
    // Arrange: a permuted (non-compact) view of [[1, 2], [3, 4]].
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu);
    let transposed = input.permute(vec![1, 0]);

    // Act
    let output = transposed.to_dtype(DType::F16).unwrap();

    // Assert: values follow the view, not the backing buffer order.
    let values: Vec<f32> =
        output.to_vec::<f16>().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn float_conversion_keeps_autograd_edge() {
    // Arrange
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 4.0], (3,), Device::Cpu).attach();

    // Act
    let cast = input.to_dtype(DType::F16).unwrap();
    assert!(cast.requires_grad());
    assert!(cast.op().is_some());
    let grads = cast.sum(vec![0], false).backward().unwrap();

    // Assert: d(sum)/dx is ones, cast back to the input dtype.
    let grad = grads.get(input.id()).unwrap();
    assert_eq!(grad.dtype(), DType::F32);
    assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn float_conversion_backward_scales_through_chain() {
    // Arrange
    let input = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu).attach();

    // Act
    let cast = input.to_dtype(DType::BF16).unwrap();
    let scaled = &cast.to_dtype(DType::F32).unwrap() * 2.0;
    let grads = scaled.sum(vec![0], false).backward().unwrap();

    // Assert
    assert_eq!(grads.get(input.id()).unwrap().to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
}

#[test]
fn float_conversion_backward_casts_grad_to_input_dtype() {
    // Arrange: F16 input so the incoming F32-ones gradient must narrow.
    let input =
        Tensor::from_vec(vec![f16::from_f32(1.0), f16::from_f32(2.0)], (2,), Device::Cpu).attach();

    // Act
    let cast = input.to_dtype(DType::F32).unwrap();
    let grads = cast.sum(vec![0], false).backward().unwrap();

    // Assert
    let grad = grads.get(input.id()).unwrap();
    assert_eq!(grad.dtype(), DType::F16);
    assert_eq!(grad.to_vec::<f16>().unwrap(), vec![f16::from_f32(1.0), f16::from_f32(1.0)]);
}

#[test]
fn conversion_to_i64_does_not_track_gradients() {
    // Arrange
    let input = Tensor::from_vec(vec![1.5f32, 2.5], (2,), Device::Cpu).attach();

    // Act
    let output = input.to_dtype(DType::I64).unwrap();

    // Assert: detached output, and the input keeps no edge from the cast.
    assert!(!output.requires_grad());
    assert!(output.op().is_none());
    assert_eq!(output.to_vec::<i64>().unwrap(), vec![1, 2]);
}

#[test]
fn conversion_to_i64_leaves_upstream_graph_intact() {
    // Arrange
    let input = Tensor::from_vec(vec![1.5f32, 2.5], (2,), Device::Cpu).attach();
    let doubled = &input * 2.0;

    // Act
    let _ids = doubled.to_dtype(DType::I64).unwrap();
    let grads = doubled.sum(vec![0], false).backward().unwrap();

    // Assert: the detached cast did not disturb gradients through `doubled`.
    assert_eq!(grads.get(input.id()).unwrap().to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
}

#[test]
fn forward_converts_on_every_available_device() {
    // Arrange
    let data = vec![1.5f32, -2.25, 100.0];

    // Act
    let results: Vec<(Device, Vec<bf16>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3,), device);
            let output = input.to_dtype(DType::BF16).unwrap();
            assert_eq!(output.device(), device);
            (device, output.to_vec::<bf16>().unwrap())
        })
        .collect();

    // Assert
    let expected: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    assert!(!results.is_empty());
    for (device, actual) in results {
        assert_eq!(actual, expected, "forward on {device:?}");
    }
}

#[test]
fn backward_flows_on_every_available_device() {
    // Arrange
    let data = vec![1.0f32, 2.0, 3.0];

    // Act
    let results: Vec<(Device, Vec<f32>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3,), device).attach();
            let cast = input.to_dtype(DType::F16).unwrap();
            let grads = cast.sum(vec![0], false).backward().unwrap();
            (device, grads.get(input.id()).unwrap().to_vec::<f32>().unwrap())
        })
        .collect();

    // Assert
    assert!(!results.is_empty());
    for (device, actual) in results {
        assert_eq!(actual, vec![1.0, 1.0, 1.0], "backward on {device:?}");
    }
}

#[test]
fn i64_roundtrip_on_every_available_device() {
    // Arrange
    let data = vec![1.9f32, -1.9, 2.0];

    // Act
    let results: Vec<(Device, Vec<i64>)> = devices()
        .into_iter()
        .map(|device| {
            let input = Tensor::from_vec(data.clone(), (3,), device);
            let output = input.to_dtype(DType::I64).unwrap();
            assert!(!output.requires_grad());
            (device, output.to_vec::<i64>().unwrap())
        })
        .collect();

    // Assert
    assert!(!results.is_empty());
    for (device, actual) in results {
        assert_eq!(actual, vec![1, -1, 2], "i64 cast on {device:?}");
    }
}
