#![allow(dead_code)]

use crate::{
    dtype::DType,
    storage::{CpuStorage, Storage},
};

/// A device is responsible for allocating memory for arrays
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn zeros(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![0.0; size])),
            (Device::Cpu, DType::F64) => Storage::Cpu(CpuStorage::F64(vec![0.0; size])),
            (Device::Cpu, _) => todo!(),
            (Device::Cuda, _) => todo!(),
        }
    }

    pub fn ones(&self, size: usize, dtype: DType) -> Storage {
        match (self, dtype) {
            (Device::Cpu, DType::F32) => Storage::Cpu(CpuStorage::F32(vec![1.0; size])),
            (Device::Cpu, DType::F64) => Storage::Cpu(CpuStorage::F64(vec![1.0; size])),
            (Device::Cpu, _) => todo!(),
            (Device::Cuda, _) => todo!(),
        }
    }
}
