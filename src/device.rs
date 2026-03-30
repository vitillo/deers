//! Compute devices (CPU, CUDA, MPS) and backend availability checks.

#![allow(dead_code)]

use crate::{
    dtype::DType,
    error::{Error, Result},
    storage::{CpuStorage, CudaStorage, MpsStorage, Storage},
};

/// The compute device where tensor data is stored and operations run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Host CPU backend.
    Cpu,
    /// Placeholder CUDA backend.
    Cuda,
    /// Apple Metal Performance Shaders backend.
    Mps,
}

impl Device {
    /// Returns whether this device backend is usable in the current process.
    pub fn is_available(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Cuda => CudaStorage::is_available(),
            #[cfg(target_os = "macos")]
            Device::Mps => metal::Device::system_default().is_some(),
            #[cfg(not(target_os = "macos"))]
            Device::Mps => false,
        }
    }

    /// Returns an error describing why this backend is unavailable.
    pub fn check_available(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda => crate::storage::cuda::availability(),
            #[cfg(target_os = "macos")]
            Device::Mps => {
                if metal::Device::system_default().is_some() {
                    Ok(())
                } else {
                    Err(Error::NotImplemented("mps backend is unavailable"))
                }
            }
            #[cfg(not(target_os = "macos"))]
            Device::Mps => Err(Error::NotImplemented("mps backend is unavailable")),
        }
    }

    /// Waits for pending work on this device to finish.
    pub fn synchronize(&self) {
        match self {
            Device::Cpu => {}
            Device::Cuda => crate::storage::cuda::synchronize(),
            Device::Mps => crate::storage::mps::synchronize(),
        }
    }

    /// Allocates a zero-filled storage buffer of `size` elements.
    pub fn zeros(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F16) => {
                Storage::Cpu(CpuStorage::F16(vec![half::f16::from_f32(0.0); size]))
            }
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![0.0; size])),
            (Device::Cpu, DType::I64) => Storage::Cpu(CpuStorage::I64(vec![0; size])),
            (Device::Mps, DType::F16) => Storage::Mps(MpsStorage::zeros(size, DType::F16)),
            (Device::Mps, DType::F32) => Storage::Mps(MpsStorage::zeros(size, DType::F32)),
            (Device::Mps, DType::I64) => Storage::Mps(MpsStorage::zeros(size, DType::I64)),
            (Device::Cuda, DType::F16) => Storage::Cuda(CudaStorage::zeros(size, DType::F16)),
            (Device::Cuda, DType::F32) => Storage::Cuda(CudaStorage::zeros(size, DType::F32)),
            (Device::Cuda, DType::I64) => Storage::Cuda(CudaStorage::zeros(size, DType::I64)),
        }
    }

    /// Allocates a storage buffer of `size` elements filled with ones.
    pub fn ones(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F16) => {
                Storage::Cpu(CpuStorage::F16(vec![half::f16::from_f32(1.0); size]))
            }
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![1.0; size])),
            (Device::Cpu, DType::I64) => Storage::Cpu(CpuStorage::I64(vec![1; size])),
            (Device::Mps, DType::F16) => Storage::Mps(MpsStorage::ones(size, DType::F16)),
            (Device::Mps, DType::F32) => Storage::Mps(MpsStorage::ones(size, DType::F32)),
            (Device::Mps, DType::I64) => Storage::Mps(MpsStorage::ones(size, DType::I64)),
            (Device::Cuda, DType::F16) => Storage::Cuda(CudaStorage::ones(size, DType::F16)),
            (Device::Cuda, DType::F32) => Storage::Cuda(CudaStorage::ones(size, DType::F32)),
            (Device::Cuda, DType::I64) => Storage::Cuda(CudaStorage::ones(size, DType::I64)),
        }
    }
}
