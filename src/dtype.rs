#![allow(dead_code)]

use crate::storage::CpuStorage;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    F64,
}

pub trait WithDType: Sized {
    fn to_vec(storage: &CpuStorage) -> Vec<Self>;
}

impl WithDType for f32 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F32(vec) => vec.clone(),
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
}
