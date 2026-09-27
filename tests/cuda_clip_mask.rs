//! On-device proof for the host-readback removals: F16 gradient clipping and
//! the ignore-index mask must compute on CUDA with a single scalar download.

#![cfg(all(feature = "cuda", target_os = "linux"))]

use half::f16;

use deers::loss::{Reduction, nll_loss_with_options};
use deers::nn::Parameter;
use deers::optim::clip_grad_norm;
use deers::{Device, GradientStore, Tensor};

fn require_cuda() -> bool {
    Device::Cuda.is_available()
}

#[test]
fn cuda_f16_clip_norm_above_f16_range() {
    // Arrange: squares (9e8, 1.6e9) overflow F16, so the norm must accumulate
    // in F32 on device; an F16 accumulation would report infinity.
    if !require_cuda() {
        return;
    }
    let device = Device::Cuda;
    let x = Parameter::new(Tensor::from_vec(
        vec![f16::from_f32(0.0), f16::from_f32(0.0)],
        (2,),
        device,
    ));
    let mut grads = GradientStore::new();
    grads.insert(
        x.id(),
        Tensor::from_vec(vec![f16::from_f32(30_000.0), f16::from_f32(40_000.0)], (2,), device),
    );

    // Act
    let norm = clip_grad_norm(std::slice::from_ref(&x), &mut grads, 1.0).unwrap();
    let clipped: Vec<f32> = grads
        .get(x.id())
        .unwrap()
        .to_vec::<f16>()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();

    // Assert
    assert!((norm - 50_000.0).abs() < 5.0, "norm={norm}");
    assert!((clipped[0] - 0.6).abs() < 1e-2, "clipped={clipped:?}");
    assert!((clipped[1] - 0.8).abs() < 1e-2, "clipped={clipped:?}");
}

#[test]
fn cuda_ignore_mask_mean_and_all_ignored() {
    // Arrange
    if !require_cuda() {
        return;
    }
    let device = Device::Cuda;
    let log_probs = Tensor::from_vec(
        vec![-0.9f32, -1.2, -2.4, -0.4, -1.9, -1.5],
        (2, 3),
        device,
    )
    .attach();
    let targets = Tensor::from_vec(vec![1i64, -100], (2,), device);
    let all_ignored = Tensor::from_vec(vec![-100i64, -100], (2,), device);

    // Act
    let loss = nll_loss_with_options(&log_probs, &targets, Reduction::Mean, Some(-100))
        .to_vec::<f32>()
        .unwrap();
    let grads =
        nll_loss_with_options(&log_probs, &targets, Reduction::Mean, Some(-100)).backward().unwrap();
    let grad: Vec<f32> = grads.get(log_probs.id()).unwrap().to_vec().unwrap();
    let zero = nll_loss_with_options(&log_probs, &all_ignored, Reduction::Mean, Some(-100))
        .to_vec::<f32>()
        .unwrap();

    // Assert: the ignored position contributes no loss and no gradient,
    // and an all-ignored batch returns zero instead of NaN.
    assert!((loss[0] - 1.2).abs() < 1e-4, "loss={loss:?}");
    assert_eq!(grad, vec![0.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(zero[0], 0.0);
}
