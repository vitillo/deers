#![cfg(all(feature = "cuda", target_os = "linux"))]

use deers::{DType, Device, ProfilerConfig, Tensor, profile};

#[test]
fn cuda_profile_reports_device_time_for_adds() {
    if Device::Cuda.check_available().is_err() {
        return;
    }

    // Arrange
    let lhs = Tensor::ones((4096,), DType::F32, Device::Cuda);
    let rhs = Tensor::ones((4096,), DType::F32, Device::Cuda);

    // Act
    let (out, report) = profile(ProfilerConfig::default(), || {
        let mut out = &lhs + &rhs;
        for _ in 0..15 {
            out = &out + &rhs;
        }
        out
    });

    // Assert
    assert_eq!(out.to_vec::<f32>().unwrap(), vec![17.0; 4096]);
    let row = report.rows().iter().find(|row| row.name == "add").unwrap();
    assert_eq!(row.calls, 16);
    assert!(row.device_time_ns > 0, "CUDA add timings should be collected at profile finish");
}

#[test]
fn cuda_profile_reports_device_time_for_blas() {
    if Device::Cuda.check_available().is_err() {
        return;
    }

    // Arrange
    let lhs = Tensor::ones((64, 64), DType::F32, Device::Cuda);
    let rhs = Tensor::ones((64, 64), DType::F32, Device::Cuda);

    // Act
    let (out, report) = profile(ProfilerConfig::default(), || lhs.matmul(&rhs));

    // Assert
    assert_eq!(out.to_vec::<f32>().unwrap(), vec![64.0; 64 * 64]);
    let row = report.rows().iter().find(|row| row.name == "matmul").unwrap();
    assert_eq!(row.calls, 1);
    assert!(row.device_time_ns > 0, "CUDA matmul timing should be collected at profile finish");
}
