#![allow(dead_code)]

use crate::{
    dtype::DType,
    storage::{CpuStorage, MpsStorage, Storage},
};

/// The compute device where tensor data is stored and operations run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Mps,
}

impl Device {
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
            (Device::Cuda, _) => todo!(),
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
            (Device::Cuda, _) => todo!(),
        }
    }
}
