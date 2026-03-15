#![allow(dead_code)]

use crate::{
    dtype::DType,
    storage::{CpuStorage, Storage},
};

/// The compute device where tensor data is stored and operations run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    /// Allocates a zero-filled storage buffer of `size` elements.
    pub fn zeros(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![0.0; size])),
            (Device::Cpu, DType::F64) => Storage::Cpu(CpuStorage::F64(vec![0.0; size])),
            (Device::Cpu, _) => todo!(),
            (Device::Cuda, _) => todo!(),
        }
    }

    /// Allocates a storage buffer of `size` elements filled with ones.
    pub fn ones(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![1.0; size])),
            (Device::Cpu, DType::F64) => Storage::Cpu(CpuStorage::F64(vec![1.0; size])),
            (Device::Cpu, _) => todo!(),
            (Device::Cuda, _) => todo!(),
        }
    }
}
