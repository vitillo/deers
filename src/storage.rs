#![allow(dead_code)]

use crate::error::Result;

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

pub struct Add;

impl BinaryOp for Add {
    fn f32(v: f32, w: f32) -> f32 {
        v + w
    }

    fn f64(v: f64, w: f64) -> f64 {
        v + w
    }
}

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

trait BackendStorage: Sized {
    fn unary_op<O: UnaryOp>(&self) -> Result<Self>;
    fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self>;
}

#[derive(Debug, PartialEq, PartialOrd)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
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
}
