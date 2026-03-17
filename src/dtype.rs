#![allow(dead_code)]

use crate::storage::CpuStorage;

/// Supported tensor element types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    F64,
    U32,
}

/// Trait implemented by Rust types that can be stored in a tensor (f32, f64).
pub trait WithDType: Sized + Copy {
    fn to_vec(storage: &CpuStorage) -> Vec<Self>;
    fn as_slice(storage: &CpuStorage) -> &[Self];
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

impl WithDType for f64 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F64(vec) => vec.clone(),
            _ => todo!(),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F64(vec) => vec.as_slice(),
            _ => todo!(),
        }
    }
}

impl WithDType for u32 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::U32(vec) => vec.clone(),
            _ => todo!(),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::U32(vec) => vec.as_slice(),
            _ => todo!(),
        }
    }
}
