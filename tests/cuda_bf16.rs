//! BF16 coverage for the CUDA backend: every kernel and conversion path that
//! carries BF16 data on CUDA must compute instead of hitting a dtype guard.

#![cfg(all(feature = "cuda", target_os = "linux"))]

use std::collections::BTreeMap;

use half::bf16;

use deers::models::gpt::{Qwen3, Qwen3Config, RopeScaling};
use deers::nn::ParamStore;
use deers::{DType, Device, Tensor};

const TOL: f32 = 2e-2;

fn require_cuda() -> bool {
    Device::Cuda.is_available()
}

fn bf16_cuda(values: Vec<f32>, shape: (usize, usize)) -> Tensor {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    Tensor::from_vec(data, shape, Device::Cuda)
}

fn bf16_cpu(values: Vec<f32>, shape: (usize, usize)) -> Tensor {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    Tensor::from_vec(data, shape, Device::Cpu)
}

fn to_f32(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<bf16>().unwrap().iter().map(|v| v.to_f32()).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < TOL, "{label}[{index}]: got {a}, expected {e}");
    }
}

/// Relative-tolerance comparison for end-to-end model logits, where BF16
/// rounding accumulates through the transformer stack. Measured worst case on
/// the constant-fill small Qwen3 is 3e-5, so a 5% bound with a 0.05 floor
/// passes with wide margin while still catching wrong-kernel failures, which
/// deviate by orders of magnitude more.
fn assert_close_rel(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let bound = 0.05 * e.abs().max(1.0);
        assert!((a - e).abs() < bound, "{label}[{index}]: got {a}, expected {e}");
    }
}

#[test]
fn bf16_zeros_and_ones_allocate_on_cuda() {
    // Arrange
    if !require_cuda() {
        return;
    }

    // Act
    let zeros = Tensor::zeros((2, 2), DType::BF16, Device::Cuda);
    let ones = Tensor::ones((2, 2), DType::BF16, Device::Cuda);

    // Assert
    assert_eq!(zeros.dtype(), DType::BF16);
    assert_eq!(ones.dtype(), DType::BF16);
    assert_eq!(zeros.to_vec::<bf16>().unwrap(), vec![bf16::ZERO; 4]);
    assert_eq!(ones.to_vec::<bf16>().unwrap(), vec![bf16::ONE; 4]);
}

#[test]
fn bf16_host_to_device_roundtrip_preserves_bits() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0), bf16::from_f32(3.140625)];

    // Act
    let tensor = Tensor::from_vec(values.clone(), (3,), Device::Cuda);
    let back: Vec<bf16> = tensor.to_vec().unwrap();

    // Assert
    assert_eq!(tensor.dtype(), DType::BF16);
    assert_eq!(back, values);
}

#[test]
fn bf16_to_device_roundtrip_both_directions() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![bf16::from_f32(0.5), bf16::from_f32(100.0)];

    // Act
    let cpu = Tensor::from_vec(values.clone(), (2,), Device::Cpu);
    let back: Vec<bf16> =
        cpu.to_device(Device::Cuda).unwrap().to_device(Device::Cpu).unwrap().to_vec().unwrap();

    // Assert
    assert_eq!(back, values);
}

#[test]
fn bf16_add_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![1.0, 2.0, -4.0, 0.5], (2, 2));
    let b = bf16_cuda(vec![0.5, 0.25, 8.0, -0.5], (2, 2));

    // Act
    let actual = to_f32(&(&a + &b));

    // Assert
    assert_close(&actual, &[1.5, 2.25, 4.0, 0.0], "bf16 add on cuda");
}

#[test]
fn bf16_broadcast_add_matches_cpu_and_preserves_mask_gradient() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let shape = vec![2, 3, 4, 33];
    let scores: Vec<bf16> =
        (0..2 * 3 * 4 * 33).map(|i| bf16::from_f32((i % 31) as f32 / 2.0)).collect();
    let mask: Vec<bf16> =
        (0..4 * 33).map(|i| bf16::from_f32(if i % 3 == 0 { -2.0 } else { 0.5 })).collect();
    let cpu = Tensor::from_vec(scores.clone(), shape.clone(), Device::Cpu);
    let cuda = Tensor::from_vec(scores, shape.clone(), Device::Cuda);
    let cpu_mask = Tensor::from_vec(mask.clone(), vec![1, 1, 4, 33], Device::Cpu).attach();
    let cuda_mask = Tensor::from_vec(mask, vec![1, 1, 4, 33], Device::Cuda).attach();

    // Act
    let cpu_out = &cpu + &cpu_mask.broadcast(shape.clone());
    let cuda_out = &cuda + &cuda_mask.broadcast(shape);
    let cpu_grad =
        cpu_out.sum(vec![0, 1, 2, 3], false).backward().unwrap().get(cpu_mask.id()).unwrap();
    let cuda_grad =
        cuda_out.sum(vec![0, 1, 2, 3], false).backward().unwrap().get(cuda_mask.id()).unwrap();

    // Assert
    assert_eq!(to_f32(&cuda_out), to_f32(&cpu_out));
    assert_eq!(to_f32(&cuda_grad), to_f32(&cpu_grad));
    assert!(to_f32(&cuda_grad).iter().all(|&v| v == 6.0));
}

#[test]
fn bf16_sub_and_div_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![1.0, 2.0, 8.0, 3.0], (2, 2));
    let b = bf16_cuda(vec![0.5, 0.5, 2.0, 2.0], (2, 2));

    // Act
    let diff = to_f32(&(&a - &b));
    let quot = to_f32(&(&a / &b));

    // Assert
    assert_close(&diff, &[0.5, 1.5, 6.0, 1.0], "bf16 sub on cuda");
    assert_close(&quot, &[2.0, 4.0, 4.0, 1.5], "bf16 div on cuda");
}

#[test]
fn bf16_mul_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![2.0, 3.0, -1.0, 0.5], (2, 2));
    let b = bf16_cuda(vec![4.0, 0.5, -8.0, 2.0], (2, 2));

    // Act
    let actual = to_f32(&(&a * &b));

    // Assert
    assert_close(&actual, &[8.0, 1.5, 8.0, 1.0], "bf16 mul on cuda");
}

#[test]
fn bf16_powf_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let base = bf16_cuda(vec![2.0, 4.0, 9.0, 1.0], (2, 2));
    let exp = bf16_cuda(vec![2.0, 0.5, 0.5, 5.0], (2, 2));
    let expected = to_f32(
        &bf16_cpu(vec![2.0, 4.0, 9.0, 1.0], (2, 2))
            .powf(bf16_cpu(vec![2.0, 0.5, 0.5, 5.0], (2, 2))),
    );

    // Act
    let actual = to_f32(&base.powf(&exp));

    // Assert
    assert_close(&actual, &expected, "bf16 powf on cuda");
}

#[test]
fn bf16_scalar_add_mul_div_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![1.0, 2.0, 4.0, 8.0], (2, 2));

    // Act
    let added = to_f32(&(&a + 1.0));
    let scaled = to_f32(&(&a * 2.0));
    let halved = to_f32(&(&a / 2.0));

    // Assert
    assert_close(&added, &[2.0, 3.0, 5.0, 9.0], "bf16 scalar add on cuda");
    assert_close(&scaled, &[2.0, 4.0, 8.0, 16.0], "bf16 scalar mul on cuda");
    assert_close(&halved, &[0.5, 1.0, 2.0, 4.0], "bf16 scalar div on cuda");
}

#[test]
fn bf16_scalar_powf_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![4.0, 9.0, 16.0, 1.0], (2, 2));
    let expected = to_f32(&bf16_cpu(vec![4.0, 9.0, 16.0, 1.0], (2, 2)).scalar_powf(0.5));

    // Act
    let actual = to_f32(&a.scalar_powf(0.5));

    // Assert
    assert_close(&actual, &expected, "bf16 scalar powf on cuda");
}

#[test]
fn bf16_neg_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![1.5, -2.0, 0.0, 100.0], (2, 2));

    // Act
    let actual = to_f32(&(-&a));

    // Assert
    assert_close(&actual, &[-1.5, 2.0, 0.0, -100.0], "bf16 neg on cuda");
}

#[test]
fn bf16_exp_and_log_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![0.0, 1.0, 2.0, 0.5], (2, 2));
    let b = bf16_cuda(vec![1.0, 2.0, 4.0, 0.5], (2, 2));
    let expected_exp = to_f32(&bf16_cpu(vec![0.0, 1.0, 2.0, 0.5], (2, 2)).exp());
    let expected_log = to_f32(&bf16_cpu(vec![1.0, 2.0, 4.0, 0.5], (2, 2)).log());

    // Act
    let actual_exp = to_f32(&a.exp());
    let actual_log = to_f32(&b.log());

    // Assert
    assert_close(&actual_exp, &expected_exp, "bf16 exp on cuda");
    assert_close(&actual_log, &expected_log, "bf16 log on cuda");
}

#[test]
fn bf16_sin_cos_tanh_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![0.0, 0.5, 1.0, -1.0];
    let a = bf16_cuda(values.clone(), (2, 2));
    let cpu = bf16_cpu(values, (2, 2));
    let expected_sin = to_f32(&cpu.sin());
    let expected_cos = to_f32(&cpu.cos());
    let expected_tanh = to_f32(&cpu.tanh());

    // Act
    let actual_sin = to_f32(&a.sin());
    let actual_cos = to_f32(&a.cos());
    let actual_tanh = to_f32(&a.tanh());

    // Assert
    assert_close(&actual_sin, &expected_sin, "bf16 sin on cuda");
    assert_close(&actual_cos, &expected_cos, "bf16 cos on cuda");
    assert_close(&actual_tanh, &expected_tanh, "bf16 tanh on cuda");
}

#[test]
fn bf16_relu_forward_and_backward_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![-2.0, -0.5, 0.0, 1.5];
    let cpu = bf16_cpu(values.clone(), (2, 2)).attach();
    let cuda = bf16_cuda(values, (2, 2)).attach();

    // Act
    let expected_fwd = to_f32(&cpu.relu());
    let actual_fwd = to_f32(&cuda.relu());
    let cpu_loss = cpu.relu().sum(vec![0, 1], true);
    let cuda_loss = cuda.relu().sum(vec![0, 1], true);
    let expected_grad = to_f32(&cpu_loss.backward().unwrap().get(cpu.id()).unwrap());
    let actual_grad = to_f32(&cuda_loss.backward().unwrap().get(cuda.id()).unwrap());

    // Assert
    assert_close(&actual_fwd, &expected_fwd, "bf16 relu fwd on cuda");
    assert_close(&actual_grad, &expected_grad, "bf16 relu bwd on cuda");
}

#[test]
fn bf16_reduce_sum_and_max_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cuda = bf16_cuda(values.clone(), (2, 3));
    let cpu = bf16_cpu(values, (2, 3));

    // Act
    let actual_sum = to_f32(&cuda.sum(vec![1], false));
    let expected_sum = to_f32(&cpu.sum(vec![1], false));
    let actual_max = to_f32(&cuda.max(vec![1], false));
    let expected_max = to_f32(&cpu.max(vec![1], false));

    // Assert
    assert_close(&actual_sum, &expected_sum, "bf16 reduce sum on cuda");
    assert_close(&actual_max, &expected_max, "bf16 reduce max on cuda");
}

#[test]
fn bf16_max_backward_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![1.0, 4.0, 2.0, 3.0];
    let cpu = bf16_cpu(values.clone(), (2, 2)).attach();
    let cuda = bf16_cuda(values, (2, 2)).attach();

    // Act
    let cpu_loss = cpu.max(vec![1], true).sum(vec![0, 1], true);
    let cuda_loss = cuda.max(vec![1], true).sum(vec![0, 1], true);
    let expected = to_f32(&cpu_loss.backward().unwrap().get(cpu.id()).unwrap());
    let actual = to_f32(&cuda_loss.backward().unwrap().get(cuda.id()).unwrap());

    // Assert
    assert_close(&actual, &expected, "bf16 max backward on cuda");
}

#[test]
fn bf16_matmul_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let lhs = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let rhs = bf16_cuda(vec![5.0, 6.0, 7.0, 8.0], (2, 2));

    // Act
    let out = lhs.matmul(&rhs);

    // Assert
    assert_eq!(out.dtype(), DType::BF16);
    assert_close(&to_f32(&out), &[19.0, 22.0, 43.0, 50.0], "bf16 matmul on cuda");
}

#[test]
fn bf16_matmul_transposed_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let lhs = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let rhs = bf16_cuda(vec![1.0, 0.0, 0.0, 1.0], (2, 2));
    let cpu_lhs = bf16_cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));

    // Act
    let actual = to_f32(&lhs.transpose(None).matmul(&rhs));
    let expected =
        to_f32(&cpu_lhs.transpose(None).matmul(&bf16_cpu(vec![1.0, 0.0, 0.0, 1.0], (2, 2))));

    // Assert
    assert_close(&actual, &expected, "bf16 transposed matmul on cuda");
}

#[test]
fn bf16_batched_matmul_with_offset_and_transpose_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let lhs_values = vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // batch 0, excluded by narrow
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1
        2.0, 0.0, 1.0, 0.0, 3.0, 1.0, // batch 2
    ];
    let rhs_values = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0, 1.0];
    let cuda_lhs = Tensor::from_vec(
        lhs_values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>(),
        (3, 2, 3),
        Device::Cuda,
    );
    let cuda_rhs = Tensor::from_vec(
        rhs_values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>(),
        (2, 2, 3),
        Device::Cuda,
    );
    let cpu_lhs = Tensor::from_vec(
        lhs_values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        (3, 2, 3),
        Device::Cpu,
    );
    let cpu_rhs = Tensor::from_vec(
        rhs_values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        (2, 2, 3),
        Device::Cpu,
    );

    // Act
    let actual = to_f32(&cuda_lhs.narrow(0, 1, 2).matmul(&cuda_rhs.transpose(None)));
    let expected = to_f32(&cpu_lhs.narrow(0, 1, 2).matmul(&cpu_rhs.transpose(None)));

    // Assert
    assert_close(&actual, &expected, "bf16 batched offset/transpose matmul on cuda");
}

#[test]
fn bf16_batched_matmul_with_strided_rows_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let lhs_values: Vec<_> = (0..24).map(|i| bf16::from_f32(i as f32 / 4.0)).collect();
    let rhs_values: Vec<_> = (0..16).map(|i| bf16::from_f32((i % 5) as f32 / 4.0)).collect();
    let cuda_lhs = Tensor::from_vec(lhs_values.clone(), (2, 3, 4), Device::Cuda);
    let cuda_rhs = Tensor::from_vec(rhs_values.clone(), (2, 4, 2), Device::Cuda);
    let cpu_lhs = Tensor::from_vec(lhs_values, (2, 3, 4), Device::Cpu);
    let cpu_rhs = Tensor::from_vec(rhs_values, (2, 4, 2), Device::Cpu);

    // Act
    let actual = to_f32(&cuda_lhs.narrow(1, 1, 2).matmul(&cuda_rhs));
    let expected = to_f32(&cpu_lhs.narrow(1, 1, 2).matmul(&cpu_rhs));

    // Assert
    assert_close(&actual, &expected, "bf16 strided row matmul on cuda");
}

#[test]
fn bf16_matmul_backward_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let cpu_lhs = bf16_cpu(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).attach();
    let cpu_rhs = bf16_cpu(vec![5.0, 6.0, 7.0, 8.0], (2, 2)).attach();
    let cuda_lhs = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).attach();
    let cuda_rhs = bf16_cuda(vec![5.0, 6.0, 7.0, 8.0], (2, 2)).attach();

    // Act
    let cpu_loss = cpu_lhs.matmul(&cpu_rhs).sum(vec![0, 1], true);
    let cuda_loss = cuda_lhs.matmul(&cuda_rhs).sum(vec![0, 1], true);
    let expected = to_f32(&cpu_loss.backward().unwrap().get(cpu_lhs.id()).unwrap());
    let actual = to_f32(&cuda_loss.backward().unwrap().get(cuda_lhs.id()).unwrap());

    // Assert
    assert_close(&actual, &expected, "bf16 matmul backward on cuda");
}

#[test]
fn bf16_log_sum_exp_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![1.0, 2.0, 3.0, 0.5, -1.0, 0.0];
    let cuda = bf16_cuda(values.clone(), (2, 3));
    let expected = to_f32(&bf16_cpu(values, (2, 3)).log_sum_exp(vec![1]));

    // Act
    let actual = to_f32(&cuda.log_sum_exp(vec![1]));

    // Assert
    assert_close(&actual, &expected, "bf16 log_sum_exp on cuda");
}

#[test]
fn bf16_log_softmax_forward_and_backward_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![1.0, 2.0, 3.0, 0.5, -1.0, 0.0];
    let cpu = bf16_cpu(values.clone(), (2, 3)).attach();
    let cuda = bf16_cuda(values, (2, 3)).attach();
    let expected_fwd = to_f32(&cpu.log_softmax(1));

    // Act
    let actual_fwd = to_f32(&cuda.log_softmax(1));
    let cpu_loss = cpu.log_softmax(1).sum(vec![0, 1], true);
    let cuda_loss = cuda.log_softmax(1).sum(vec![0, 1], true);
    let expected_grad = to_f32(&cpu_loss.backward().unwrap().get(cpu.id()).unwrap());
    let actual_grad = to_f32(&cuda_loss.backward().unwrap().get(cuda.id()).unwrap());

    // Assert
    assert_close(&actual_fwd, &expected_fwd, "bf16 log_softmax fwd on cuda");
    assert_close(&actual_grad, &expected_grad, "bf16 log_softmax bwd on cuda");
}

#[test]
fn bf16_softmax_forward_and_backward_match_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values = vec![1.0, 2.0, 3.0, 0.5, -1.0, 0.0];
    let weights = vec![1.0, 0.0, -1.0, -1.0, 2.0, 1.0];
    let cpu = bf16_cpu(values.clone(), (2, 3)).attach();
    let cuda = bf16_cuda(values, (2, 3)).attach();
    let expected = to_f32(&cpu.softmax(1));

    // Act
    let actual = to_f32(&cuda.softmax(1));
    let cpu_loss = (&cpu.softmax(1) * &bf16_cpu(weights.clone(), (2, 3))).sum(vec![0, 1], true);
    let cuda_loss = (&cuda.softmax(1) * &bf16_cuda(weights, (2, 3))).sum(vec![0, 1], true);
    let expected_grad = to_f32(&cpu_loss.backward().unwrap().get(cpu.id()).unwrap());
    let actual_grad = to_f32(&cuda_loss.backward().unwrap().get(cuda.id()).unwrap());

    // Assert
    assert_close(&actual, &expected, "bf16 softmax fwd on cuda");
    assert_close(&actual_grad, &expected_grad, "bf16 softmax bwd on cuda");
}

#[test]
fn bf16_wide_softmax_and_log_softmax_match_f32_reference() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let width = 2048;
    let values: Vec<bf16> = (0..2 * width)
        .map(|i| {
            if i >= width && i % width > 1024 {
                bf16::NEG_INFINITY
            } else {
                bf16::from_f32(((i * 79 % 101) as f32 - 50.0) / 9.0)
            }
        })
        .collect();
    let input = Tensor::from_vec(values.clone(), (2, width), Device::Cuda);
    let mut expected_softmax = Vec::new();
    let mut expected_log_softmax = Vec::new();
    for row in values.chunks(width) {
        let max = row.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v.to_f32() - max).exp()).sum();
        for v in row {
            let log_prob = v.to_f32() - max - sum.ln();
            expected_log_softmax.push(bf16::from_f32(log_prob).to_f32());
            expected_softmax.push(bf16::from_f32(log_prob.exp()).to_f32());
        }
    }

    // Act
    let softmax = to_f32(&input.softmax(1));
    let log_softmax = to_f32(&input.log_softmax(1));

    // Assert
    for (i, (actual, expected)) in softmax.iter().zip(expected_softmax.iter()).enumerate() {
        assert!((actual - expected).abs() < 0.002, "softmax[{i}]: {actual} != {expected}");
    }
    for (i, (actual, expected)) in log_softmax.iter().zip(expected_log_softmax.iter()).enumerate() {
        assert!(
            actual == expected || (actual - expected).abs() < 0.063,
            "log_softmax[{i}]: {actual} != {expected}"
        );
    }
}

#[test]
fn bf16_gather_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let src = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let idx_cpu = Tensor::from_vec(vec![2i64, 0], (2, 1), Device::Cpu);
    let idx_cuda = Tensor::from_vec(vec![2i64, 0], (2, 1), Device::Cuda);

    // Act
    let actual = to_f32(&src.gather(1, &idx_cuda));
    let expected =
        to_f32(&bf16_cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)).gather(1, &idx_cpu));

    // Assert
    assert_close(&actual, &expected, "bf16 gather on cuda");
}

#[test]
fn bf16_gather_backward_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let cpu = bf16_cpu(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).attach();
    let cuda = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).attach();
    let idx_cpu = Tensor::from_vec(vec![1i64, 0], (2, 1), Device::Cpu);
    let idx_cuda = Tensor::from_vec(vec![1i64, 0], (2, 1), Device::Cuda);

    // Act
    let cpu_loss = cpu.gather(1, &idx_cpu).sum(vec![0, 1], true);
    let cuda_loss = cuda.gather(1, &idx_cuda).sum(vec![0, 1], true);
    let expected = to_f32(&cpu_loss.backward().unwrap().get(cpu.id()).unwrap());
    let actual = to_f32(&cuda_loss.backward().unwrap().get(cuda.id()).unwrap());

    // Assert
    assert_close(&actual, &expected, "bf16 gather backward on cuda");
}

#[test]
fn bf16_index_select_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let src = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
    let idx_cuda = Tensor::from_vec(vec![2i64, 0], (2,), Device::Cuda);
    let idx_cpu = Tensor::from_vec(vec![2i64, 0], (2,), Device::Cpu);
    let expected =
        to_f32(&bf16_cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)).index_select(0, &idx_cpu));

    // Act
    let actual = to_f32(&src.index_select(0, &idx_cuda));

    // Assert
    assert_close(&actual, &expected, "bf16 index_select on cuda");
}

#[test]
fn bf16_index_select_backward_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let cpu = bf16_cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)).attach();
    let cuda = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)).attach();
    let idx_cpu = Tensor::from_vec(vec![2i64, 0], (2,), Device::Cpu);
    let idx_cuda = Tensor::from_vec(vec![2i64, 0], (2,), Device::Cuda);

    // Act
    let cpu_loss = cpu.index_select(0, &idx_cpu).sum(vec![0, 1], true);
    let cuda_loss = cuda.index_select(0, &idx_cuda).sum(vec![0, 1], true);
    let expected = to_f32(&cpu_loss.backward().unwrap().get(cpu.id()).unwrap());
    let actual = to_f32(&cuda_loss.backward().unwrap().get(cuda.id()).unwrap());

    // Assert
    assert_close(&actual, &expected, "bf16 index_select backward on cuda");
}

#[test]
fn bf16_cat_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![1.0, 2.0], (1, 2));
    let b = bf16_cuda(vec![3.0, 4.0], (1, 2));

    // Act
    let actual = to_f32(&Tensor::cat(&[a, b], 0));

    // Assert
    assert_close(&actual, &[1.0, 2.0, 3.0, 4.0], "bf16 cat on cuda");
}

#[test]
fn bf16_upload_and_cat_keep_queued_copies_ordered() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let mut results = Vec::new();

    // Act
    for i in 0..256 {
        let input = Tensor::from_vec(vec![bf16::from_f32(i as f32); 128], (128,), Device::Cuda);
        let left = input.narrow(0, 0, 64);
        let right = input.narrow(0, 64, 64);
        results.push(Tensor::cat(&[left, right], 0));
    }
    Device::Cuda.synchronize();

    // Assert
    for (i, result) in results.iter().enumerate() {
        assert!(result.to_vec::<bf16>().unwrap().iter().all(|v| v.to_f32() == i as f32));
    }
}

#[test]
fn bf16_strided_view_computes_on_cuda() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let a = bf16_cuda(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let b = bf16_cuda(vec![0.5, 0.5, 0.5, 0.5], (2, 2));

    // Act: the transpose is non-compact, so the add compacts through copy_compact_bf16.
    let actual = to_f32(&(&a.transpose(None) + &b));

    // Assert
    assert_close(&actual, &[1.5, 3.5, 2.5, 4.5], "bf16 strided add on cuda");
}

fn small_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 32,
        sequence_len: 8,
        n_layers: 1,
        hidden: 16,
        n_q_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        mlp_hidden_dim: 32,
        rms_norm_eps: 1e-6,
        rope_base: 10_000.0,
        rope_scaling: RopeScaling::None,
    }
}

#[test]
fn bf16_model_converts_and_runs_forward_on_cuda() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let store = ParamStore::new();
    let mut model = Qwen3::new(small_config(), store.root());
    let ids = Tensor::from_vec(vec![1i64, 2, 3, 4], (1, 4), Device::Cpu);

    // Act
    model.to_dtype(DType::BF16).unwrap();
    model.to_device(Device::Cuda).unwrap();
    let logits = model.forward(&ids.to_device(Device::Cuda).unwrap()).unwrap();

    // Assert
    assert_eq!(logits.dtype(), DType::BF16);
    let shape: Vec<usize> = logits.layout().shape().iter().copied().collect();
    assert_eq!(shape, vec![1, 4, 32]);
    let flat = to_f32(&logits.reshape(vec![4 * 32]));
    assert!(flat.iter().all(|v| v.is_finite()), "bf16 cuda logits must be finite");
}

#[test]
fn bf16_model_cuda_matches_cpu_forward() {
    // Arrange: constant fill keeps the comparison deterministic across runs;
    // randn init would let small-magnitude logits amplify cross-device noise.
    if !require_cuda() {
        return;
    }
    let cpu_store = ParamStore::new();
    let mut cpu_model = Qwen3::new(small_config(), cpu_store.root());
    for (_, parameter) in cpu_store.named_parameters().iter() {
        let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
        let fill = Tensor::from_vec(vec![0.02f32; shape.iter().product()], shape, Device::Cpu);
        parameter.set(&fill).unwrap();
    }
    cpu_model.to_dtype(DType::BF16).unwrap();
    let cuda_store = ParamStore::new();
    let mut cuda_model = Qwen3::new(small_config(), cuda_store.root());
    for ((_, cpu_param), (_, cuda_param)) in
        cpu_store.named_parameters().iter().zip(cuda_store.named_parameters().iter())
    {
        cuda_param.set(&cpu_param.detach()).unwrap();
    }
    cuda_model.to_dtype(DType::BF16).unwrap();
    cuda_model.to_device(Device::Cuda).unwrap();
    let ids = Tensor::from_vec(vec![1i64, 2, 3, 4], (1, 4), Device::Cpu);
    let cuda_ids = ids.to_device(Device::Cuda).unwrap();

    // Act
    let expected = to_f32(&cpu_model.forward(&ids).unwrap().reshape(vec![4 * 32]));
    let first = cuda_model.forward(&cuda_ids).unwrap();
    let second = cuda_model.forward(&cuda_ids).unwrap();
    let actual = to_f32(&first.reshape(vec![4 * 32]));

    // Assert: same weights agree across devices, and CUDA repeats bitwise.
    assert_close_rel(&actual, &expected, "bf16 qwen3 cuda vs cpu forward");
    assert_eq!(
        first.to_vec::<bf16>().unwrap(),
        second.to_vec::<bf16>().unwrap(),
        "bf16 qwen3 cuda forward must be deterministic"
    );
}

#[test]
fn bf16_checkpoint_loads_onto_cuda() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let dir = std::env::temp_dir();
    let path = dir.join(format!("deers-cuda-bf16-load-{}.safetensors", std::process::id()));
    let values = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0)];
    let saved: BTreeMap<String, Tensor> =
        [(String::from("w"), Tensor::from_vec(values.clone(), (2,), Device::Cpu))]
            .into_iter()
            .collect();
    deers::checkpoint::save_tensors(&path, &saved).unwrap();

    // Act
    let loaded = deers::checkpoint::load_tensors(&path, Device::Cuda).unwrap();
    let _ = std::fs::remove_file(&path);

    // Assert
    assert_eq!(loaded["w"].dtype(), DType::BF16);
    assert_eq!(loaded["w"].to_vec::<bf16>().unwrap(), values);
}

/// Primitive RMSNorm decomposition, mirroring the pre-fusion `RMSNorm::forward`.
fn unfused_rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Tensor {
    let last = x.layout().ndim() - 1;
    let mean_sq = (x * x).mean(vec![last], true);
    let inv_norm = (mean_sq + eps).scalar_powf(-0.5);
    let normed = x * &inv_norm;
    match weight {
        Some(w) => &normed * w,
        None => normed,
    }
}

#[test]
fn bf16_rms_norm_matches_cpu_with_and_without_weight_and_gradient() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values: Vec<f32> = (0..2 * 128).map(|i| (i % 29) as f32 / 17.0 - 0.8).collect();
    let weights: Vec<f32> = (0..128).map(|i| (i % 7) as f32 / 8.0 + 0.5).collect();
    let x_cpu = bf16_cpu(values.clone(), (2, 128)).attach();
    let x_cuda = bf16_cuda(values, (2, 128)).attach();
    let w_cpu = Tensor::from_vec(
        weights.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
        (128,),
        Device::Cpu,
    );
    let w_cuda = w_cpu.to_device(Device::Cuda).unwrap();

    // Act
    let expected = x_cpu.fused_rms_norm(Some(&w_cpu), 1e-5);
    let actual = x_cuda.fused_rms_norm(Some(&w_cuda), 1e-5);
    let expected_plain = x_cpu.fused_rms_norm(None, 1e-5);
    let actual_plain = x_cuda.fused_rms_norm(None, 1e-5);
    let expected_grad =
        expected.sum(vec![0, 1], false).backward().unwrap().get(x_cpu.id()).unwrap();
    let actual_grad = actual.sum(vec![0, 1], false).backward().unwrap().get(x_cuda.id()).unwrap();

    // Assert
    assert_close(&to_f32(&actual), &to_f32(&expected), "bf16 weighted RMSNorm");
    assert_close(&to_f32(&actual_plain), &to_f32(&expected_plain), "bf16 plain RMSNorm");
    assert_close(&to_f32(&actual_grad), &to_f32(&expected_grad), "bf16 RMSNorm gradient");
}

#[test]
fn f32_rms_norm_matches_cpu_with_and_without_weight() {
    // Arrange
    if !require_cuda() {
        return;
    }
    for (rows, cols) in [(3, 300), (2, 128), (1, 7)] {
        let values: Vec<f32> = (0..rows * cols).map(|i| (i % 17) as f32 / 9.0 - 1.0).collect();
        let weights: Vec<f32> = (0..cols).map(|i| (i % 5) as f32 / 4.0 + 0.5).collect();
        let x_cpu = Tensor::from_vec(values.clone(), (rows, cols), Device::Cpu);
        let x_cuda = Tensor::from_vec(values, (rows, cols), Device::Cuda);
        let w_cpu = Tensor::from_vec(weights.clone(), (cols,), Device::Cpu);
        let w_cuda = Tensor::from_vec(weights, (cols,), Device::Cuda);

        // Act
        let expected_weighted = x_cpu.fused_rms_norm(Some(&w_cpu), 1e-5).to_vec::<f32>().unwrap();
        let actual_weighted = x_cuda.fused_rms_norm(Some(&w_cuda), 1e-5).to_vec::<f32>().unwrap();
        let expected_plain = x_cpu.fused_rms_norm(None, 1e-5).to_vec::<f32>().unwrap();
        let actual_plain = x_cuda.fused_rms_norm(None, 1e-5).to_vec::<f32>().unwrap();

        // Assert
        assert_close(&actual_weighted, &expected_weighted, "f32 weighted RMSNorm");
        assert_close(&actual_plain, &expected_plain, "f32 weightless RMSNorm");
    }
}

#[test]
fn f32_rms_norm_backward_matches_cpu() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let values: Vec<f32> = (0..256).map(|i| (i % 13) as f32 / 8.0 - 0.5).collect();
    let weights: Vec<f32> = (0..128).map(|i| (i % 7) as f32 / 6.0 + 0.5).collect();
    let x_cpu = Tensor::from_vec(values.clone(), (2, 128), Device::Cpu).attach();
    let x_cuda = Tensor::from_vec(values, (2, 128), Device::Cuda).attach();
    let w_cpu = Tensor::from_vec(weights.clone(), (128,), Device::Cpu).attach();
    let w_cuda = Tensor::from_vec(weights, (128,), Device::Cuda).attach();

    // Act
    let cpu_grads =
        x_cpu.fused_rms_norm(Some(&w_cpu), 1e-5).sum(vec![0, 1], true).backward().unwrap();
    let cuda_grads =
        x_cuda.fused_rms_norm(Some(&w_cuda), 1e-5).sum(vec![0, 1], true).backward().unwrap();

    // Assert
    assert_close(
        &cuda_grads.get(x_cuda.id()).unwrap().to_vec::<f32>().unwrap(),
        &cpu_grads.get(x_cpu.id()).unwrap().to_vec::<f32>().unwrap(),
        "f32 RMSNorm input gradient",
    );
    assert_close(
        &cuda_grads.get(w_cuda.id()).unwrap().to_vec::<f32>().unwrap(),
        &cpu_grads.get(w_cpu.id()).unwrap().to_vec::<f32>().unwrap(),
        "f32 RMSNorm weight gradient",
    );
}

#[test]
fn fused_rms_norm_forward_and_backward_match_unfused() {
    // Arrange: inner width past the 256-thread block with a tail, plus a
    // small even-width case for the QK-norm shape family.
    if !require_cuda() {
        return;
    }
    for (outer, inner) in [(3usize, 300usize), (2, 128)] {
        let values: Vec<f32> =
            (0..outer * inner).map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0).collect();
        let weights: Vec<f32> = (0..inner).map(|i| 0.5 + ((i * 13) % 7) as f32 * 0.1).collect();
        let x_cpu = Tensor::from_vec(
            values.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
            vec![outer, inner],
            Device::Cpu,
        )
        .attach();
        let w_cpu = Tensor::from_vec(
            weights.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
            vec![inner],
            Device::Cpu,
        )
        .attach();
        let x_cuda = Tensor::from_vec(
            values.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
            vec![outer, inner],
            Device::Cuda,
        )
        .attach();
        let w_cuda = Tensor::from_vec(
            weights.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>(),
            vec![inner],
            Device::Cuda,
        )
        .attach();

        // Act: fused on CUDA against the primitive chain on CPU, forward and
        // backward (the fused op reuses the unfused backward decomposition).
        let expected_fwd = to_f32(&unfused_rms_norm(&x_cpu, Some(&w_cpu), 1e-6));
        let actual_fwd = to_f32(&x_cuda.fused_rms_norm(Some(&w_cuda), 1e-6));
        let cpu_loss = unfused_rms_norm(&x_cpu, Some(&w_cpu), 1e-6).sum(vec![0, 1], true);
        let cuda_loss = x_cuda.fused_rms_norm(Some(&w_cuda), 1e-6).sum(vec![0, 1], true);
        let expected_gx = to_f32(&cpu_loss.backward().unwrap().get(x_cpu.id()).unwrap());
        let actual_gx = to_f32(&cuda_loss.backward().unwrap().get(x_cuda.id()).unwrap());
        let expected_gw = to_f32(&cpu_loss.backward().unwrap().get(w_cpu.id()).unwrap());
        let actual_gw = to_f32(&cuda_loss.backward().unwrap().get(w_cuda.id()).unwrap());

        // Weightless forward must also match (scale of 1, no second input).
        let expected_plain = to_f32(&unfused_rms_norm(&x_cpu, None, 1e-5));
        let actual_plain = to_f32(&x_cuda.fused_rms_norm(None, 1e-5));

        // Assert
        let label = format!("fused rms_norm [{outer}, {inner}]");
        assert_close(&actual_fwd, &expected_fwd, &format!("{label} fwd"));
        assert_close(&actual_gx, &expected_gx, &format!("{label} bwd input grad"));
        assert_close(&actual_gw, &expected_gw, &format!("{label} bwd weight grad"));
        assert_close(&actual_plain, &expected_plain, &format!("{label} weightless fwd"));
    }
}
