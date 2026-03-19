#![allow(dead_code)]

use half::f16;

use crate::storage::CpuStorage;

/// Supported tensor element types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    I64,
}

/// Trait implemented by Rust types that can be stored in a tensor.
pub trait WithDType: Sized + Copy {
    fn to_vec(storage: &CpuStorage) -> Vec<Self>;
    fn as_slice(storage: &CpuStorage) -> &[Self];
}

impl WithDType for f16 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F16(vec) => vec.clone(),
            _ => todo!(), // type mismatch error
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F16(vec) => vec.as_slice(),
            _ => todo!(), // type mismatch error
        }
    }
}

impl WithDType for f32 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F32(vec) => vec.clone(),
            _ => todo!(), // type mismatch error
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F32(vec) => vec.as_slice(),
            _ => todo!(), // type mismatch error
        }
    }
}

impl WithDType for i64 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::I64(vec) => vec.clone(),
            _ => todo!(),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::I64(vec) => vec.as_slice(),
            _ => todo!(),
        }
    }
}
