//! Element types (`F16`, `F32`, `I64`) and the [`WithDType`] trait for
//! projecting Rust scalars into and out of tensor storage.

#![allow(dead_code)]

use std::fmt;

use half::f16;

use crate::storage::{BackendStorage, CpuStorage};

/// Supported tensor element types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE half-precision floating point.
    F16,
    /// IEEE single-precision floating point.
    F32,
    /// 64-bit signed integer.
    I64,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::F32 => write!(f, "f32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}

impl DType {
    /// Returns the byte width of one element of this dtype.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F16 => std::mem::size_of::<f16>(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::I64 => std::mem::size_of::<i64>(),
        }
    }
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
            other => panic!("expected F16 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F16(vec) => vec.as_slice(),
            other => panic!("expected F16 storage but got {:?}", other.dtype()),
        }
    }
}

impl WithDType for f32 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F32(vec) => vec.clone(),
            other => panic!("expected F32 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F32(vec) => vec.as_slice(),
            other => panic!("expected F32 storage but got {:?}", other.dtype()),
        }
    }
}

impl WithDType for i64 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::I64(vec) => vec.clone(),
            other => panic!("expected I64 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::I64(vec) => vec.as_slice(),
            other => panic!("expected I64 storage but got {:?}", other.dtype()),
        }
    }
}
