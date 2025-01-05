#![allow(dead_code)]

use crate::{dtype::DType, error::Result};

pub trait UnaryOp {
    fn f32(v: f32) -> f32;
    fn f64(v: f64) -> f64;
}

pub struct Neg;

impl UnaryOp for Neg {
    fn f32(v: f32) -> f32 {
        -v
    }

    fn f64(v: f64) -> f64 {
        -v
    }
}

pub trait BinaryOp {
    fn f32(v: f32, w: f32) -> f32;
    fn f64(v: f64, w: f64) -> f64;
}

macro_rules! impl_binary_op {
    ($id:ident, $op:ident) => {
        pub struct $id;

        impl BinaryOp for $id {
            fn f32(v: f32, w: f32) -> f32 {
                use std::ops::*;
                v.$op(w)
            }

            fn f64(v: f64, w: f64) -> f64 {
                use std::ops::*;
                v.$op(w)
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);

#[derive(Debug, PartialEq, PartialOrd)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn unary_op<O: UnaryOp>(&self) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_op::<O>()?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    pub fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Storage::Cpu(storage), Storage::Cpu(other)) => {
                let storage = storage.binary_op::<O>(other)?;
                Ok(Self::Cpu(storage))
            }
        }
    }
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(storage) => storage.dtype(),
        }
    }
}

trait BackendStorage: Sized {
    fn unary_op<O: UnaryOp>(&self) -> Result<Self>;
    fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self>;
}

#[derive(Debug, PartialEq, PartialOrd)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
        }
    }
}

impl From<Vec<f32>> for CpuStorage {
    fn from(v: Vec<f32>) -> Self {
        Self::F32(v)
    }
}

impl From<Vec<f64>> for CpuStorage {
    fn from(v: Vec<f64>) -> Self {
        Self::F64(v)
    }
}

impl BackendStorage for CpuStorage {
    fn unary_op<O: UnaryOp>(&self) -> Result<Self> {
        match self {
            CpuStorage::F32(data) => {
                let array = data.iter().map(|v| O::f32(*v)).collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            CpuStorage::F64(data) => {
                let array = data.iter().map(|v| O::f64(*v)).collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
        }
    }

    fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (CpuStorage::F32(v), CpuStorage::F32(w)) => {
                let array = v
                    .iter()
                    .zip(w.iter())
                    .map(|(v, w)| O::f32(*v, *w))
                    .collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            (CpuStorage::F64(v), CpuStorage::F64(w)) => {
                let array = v
                    .iter()
                    .zip(w.iter())
                    .map(|(v, w)| O::f64(*v, *w))
                    .collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
            (CpuStorage::F32(_), CpuStorage::F64(_)) => unreachable!(),
            (CpuStorage::F64(_), CpuStorage::F32(_)) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neg() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);

        let storage = storage.unary_op::<Neg>().unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![-1.0, -2.0, -3.0]))
        );
    }

    #[test]
    fn test_add() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<Add>(&other).unwrap();

        assert_eq!(storage, Storage::Cpu(CpuStorage::F32(vec![5.0, 7.0, 9.0])));
    }

    #[test]
    fn test_mul() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<Mul>(&other).unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![4.0, 10.0, 18.0]))
        );
    }
}
