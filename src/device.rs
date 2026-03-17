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
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![0.0; size])),
            (Device::Cpu, DType::F64) => Storage::Cpu(CpuStorage::F64(vec![0.0; size])),
            (Device::Cpu, DType::U32) => Storage::Cpu(CpuStorage::U32(vec![0; size])),
            (Device::Mps, DType::F32) => Storage::Mps(MpsStorage::from(vec![0.0f32; size])),
            (Device::Mps, DType::F64) => todo!(),
            (Device::Mps, DType::U32) => todo!(),
            (Device::Mps, _) => todo!(),
            (Device::Cpu, _) => todo!(),
            (Device::Cuda, _) => todo!(),
        }
    }

    /// Allocates a storage buffer of `size` elements filled with ones.
    pub fn ones(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![1.0; size])),
            (Device::Cpu, DType::F64) => Storage::Cpu(CpuStorage::F64(vec![1.0; size])),
            (Device::Cpu, DType::U32) => Storage::Cpu(CpuStorage::U32(vec![1; size])),
            (Device::Mps, DType::F32) => Storage::Mps(MpsStorage::from(vec![1.0f32; size])),
            (Device::Mps, DType::F64) => todo!(),
            (Device::Mps, DType::U32) => todo!(),
            (Device::Mps, _) => todo!(),
            (Device::Cpu, _) => todo!(),
            (Device::Cuda, _) => todo!(),
        }
    }
}
