//! BF16 CUDA storage and the Qwen3 BF16 path on the accelerator.
//!
//! Every test skips unless the CUDA backend is available, so the CPU-only
//! suite stays green without a GPU.

use deers::models::gpt::{Qwen3, Qwen3Config, RopeScaling};
use deers::nn::{ParamStore, Parameter};
use deers::sample::SamplingConfig;
use deers::{DType, Device, Tensor, no_grad};
use half::bf16;

// Accelerator kernels reorder floating-point summation: the established
// cross-backend tolerance from tests/gpt.rs (exact on CPU, 2e-3 on GPUs).
const ACCEL_TOL: f32 = 2e-3;

fn cuda_device() -> Option<Device> {
    Device::Cuda.is_available().then_some(Device::Cuda)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{label}[{index}]: got {a}, expected {e}");
    }
}

fn small_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 64,
        sequence_len: 8,
        n_layers: 2,
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

fn constant_model(fill: f32) -> (Qwen3, Vec<(String, Parameter)>) {
    let store = ParamStore::new();
    let model = Qwen3::new(small_config(), store.root());
    let named = store.named_parameters();
    for (_, parameter) in &named {
        let shape: Vec<usize> = parameter.layout().shape().iter().copied().collect();
        let values = Tensor::from_vec(vec![fill; shape.iter().product()], shape, Device::Cpu);
        parameter.set(&values).unwrap();
    }
    (model, named)
}

fn to_f32(values: Vec<bf16>) -> Vec<f32> {
    values.iter().map(|v| v.to_f32()).collect()
}

#[test]
fn cuda_bf16_elementwise_matches_cpu() {
    // Arrange
    let Some(device) = cuda_device() else { return };
    let values: Vec<bf16> = [0.5f32, -1.25, 2.0, 3.75].iter().map(|&v| bf16::from_f32(v)).collect();
    let cpu = Tensor::from_vec(values.clone(), (2, 2), Device::Cpu);
    let gpu = Tensor::from_vec(values, (2, 2), device);

    // Act
    let cpu_out = to_f32((&cpu * &cpu).exp().to_vec().unwrap());
    let gpu_out = to_f32((&gpu * &gpu).exp().to_vec().unwrap());

    // Assert
    assert!(cpu_out.iter().all(|v| v.is_finite()));
    assert!(gpu_out.iter().all(|v| v.is_finite()));
    assert_close(&gpu_out, &cpu_out, ACCEL_TOL, "bf16 square-then-exp");
}

#[test]
fn cuda_bf16_qwen3_forward_matches_cpu() {
    // Arrange: constant weights in the published checkpoint dtype.
    let Some(device) = cuda_device() else { return };
    let (mut model, _) = constant_model(0.02);
    model.to_dtype(DType::BF16).unwrap();
    let prompt = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), Device::Cpu);

    // Act
    let expected = to_f32(no_grad(|| model.forward(&prompt).unwrap()).to_vec().unwrap());
    model.to_device(device).unwrap();
    let gpu_prompt = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), device);
    let actual = to_f32(no_grad(|| model.forward(&gpu_prompt).unwrap()).to_vec().unwrap());

    // Assert: same logits within the accelerator tolerance.
    assert!(expected.iter().all(|v| v.is_finite()));
    assert!(actual.iter().all(|v| v.is_finite()));
    assert_close(&actual, &expected, ACCEL_TOL, "bf16 qwen3 forward");
}

#[test]
fn cuda_generate_matches_cpu_greedy() {
    // Arrange: `generate` must keep its index tensors on the model device;
    // before the fix this failed the rotary-cache device assertion.
    let Some(device) = cuda_device() else { return };
    let (mut model, _) = constant_model(0.02);
    model.to_dtype(DType::BF16).unwrap();
    let prompt = vec![1u32, 2, 3];
    let config = SamplingConfig { temperature: 0.0, ..SamplingConfig::new() };

    // Act
    let expected = model.generate(&prompt, 2, &config).unwrap();
    model.to_device(device).unwrap();
    let actual = model.generate(&prompt, 2, &config).unwrap();

    // Assert: two greedy ids, identical across backends on this probe.
    assert_eq!(expected.len(), 2);
    assert_eq!(actual, expected);
}
