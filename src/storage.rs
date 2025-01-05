#![allow(dead_code)]

use crate::{dtype::DType, error::Result};

pub trait UnaryOp {
    fn f32(&self, v: f32) -> f32;
    fn f64(&self, v: f64) -> f64;
}

pub struct Neg;

impl UnaryOp for Neg {
    fn f32(&self, v: f32) -> f32 {
        -v
    }

    fn f64(&self, v: f64) -> f64 {
        -v
    }
}

pub struct ScalarAdd(pub f64);

impl UnaryOp for ScalarAdd {
    fn f32(&self, v: f32) -> f32 {
        v + self.0 as f32
    }

    fn f64(&self, v: f64) -> f64 {
        v + self.0
    }
}

pub struct ScalarMul(pub f64);

impl UnaryOp for ScalarMul {
    fn f64(&self, v: f64) -> f64 {
        v * self.0
    }

    fn f32(&self, v: f32) -> f32 {
        v * self.0 as f32
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
                #[allow(unused_imports)]
                use std::ops::*;
                v.$op(w)
            }

            fn f64(v: f64, w: f64) -> f64 {
                #[allow(unused_imports)]
                use std::ops::*;
                v.$op(w)
            }
        }
    };
}

impl_binary_op!(EWiseAdd, add);
impl_binary_op!(EWiseSub, sub);
impl_binary_op!(EWiseMul, mul);
impl_binary_op!(EWisePow, powf);

#[derive(Debug, PartialEq, PartialOrd)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn powf(&self, e: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.powf(e)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    pub fn ewise_log(&self) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.log()?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    pub fn unary_op<O: UnaryOp>(&self, op: O) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_op(op)?;
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
    fn powf(&self, e: f64) -> Result<Self>;
    fn log(&self) -> Result<Self>;
    fn unary_op<O: UnaryOp>(&self, op: O) -> Result<Self>;
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
    fn powf(&self, e: f64) -> Result<Self> {
        match self {
            CpuStorage::F32(data) => {
                let array = data.iter().map(|v| v.powf(e as f32)).collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            CpuStorage::F64(data) => {
                let array = data.iter().map(|v| v.powf(e)).collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
        }
    }

    fn log(&self) -> Result<Self> {
        match self {
            CpuStorage::F32(data) => {
                let array = data.iter().map(|v| v.ln()).collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            CpuStorage::F64(data) => {
                let array = data.iter().map(|v| v.ln()).collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O) -> Result<Self> {
        match self {
            CpuStorage::F32(data) => {
                let array = data.iter().map(|v| op.f32(*v)).collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            CpuStorage::F64(data) => {
                let array = data.iter().map(|v| op.f64(*v)).collect::<Vec<_>>();
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

        let storage = storage.unary_op(Neg).unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![-1.0, -2.0, -3.0]))
        );
    }

    #[test]
    fn test_ewise_add() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<EWiseAdd>(&other).unwrap();

        assert_eq!(storage, Storage::Cpu(CpuStorage::F32(vec![5.0, 7.0, 9.0])));
    }

    #[test]
    fn test_ewise_sub() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<EWiseSub>(&other).unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![-3.0, -3.0, -3.0]))
        );
    }

    #[test]
    fn test_ewise_mul() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<EWiseMul>(&other).unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![4.0, 10.0, 18.0]))
        );
    }

    #[test]
    fn test_ewise_pow() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<EWisePow>(&other).unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![1.0, 32.0, 729.0]))
        );
    }

    #[test]
    fn test_scalar_add() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0]));

        let storage = storage.unary_op(ScalarAdd(10.0)).unwrap();

        assert_eq!(
            storage,
            Storage::Cpu(CpuStorage::F32(vec![11.0, 12.0, 13.0]))
        );
    }

    #[test]
    fn test_powf() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);

        let storage = storage.powf(2.0).unwrap();

        assert_eq!(storage, Storage::Cpu(CpuStorage::F32(vec![1.0, 4.0, 9.0])));
    }
}
