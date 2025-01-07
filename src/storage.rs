#![allow(dead_code)]

mod cpu;

use crate::{
    dtype::{DType, WithDType},
    error::Result,
};

pub use cpu::*;

pub trait UnaryOp {
    const KERNEL: &'static str;

    fn f32(&self, v: f32) -> f32;
    fn f64(&self, v: f64) -> f64;
}

pub struct Neg;

impl UnaryOp for Neg {
    const KERNEL: &'static str = "Neg";

    fn f32(&self, v: f32) -> f32 {
        -v
    }

    fn f64(&self, v: f64) -> f64 {
        -v
    }
}

macro_rules! unary_op {
    ($op:ident, $name:literal, $a:ident, $e:expr) => {
        pub struct $op;

        impl UnaryOp for $op {
            const KERNEL: &'static str = $name;
            fn f32(&self, $a: f32) -> f32 {
                $e
            }

            fn f64(&self, $a: f64) -> f64 {
                $e
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp());
unary_op!(Log, "log", v, v.ln());

macro_rules! scalar_op {
    ($op:ident, $name:literal, $e:tt) => {
        pub struct $op(pub f64);

        impl UnaryOp for $op {
            const KERNEL: &'static str = $name;
            fn f32(&self, v: f32) -> f32 {
                v $e self.0 as f32
            }

            fn f64(&self, v: f64) -> f64 {
                v $e self.0
            }
        }
    };
}

scalar_op!(ScalarAdd, "scalar_add", +);
scalar_op!(ScalarMul, "scalar_mul", *);
scalar_op!(ScalarDiv, "scalar_div", /);

pub trait BinaryOp {
    const KERNEL: &'static str;

    fn f32(v: f32, w: f32) -> f32;
    fn f64(v: f64, w: f64) -> f64;
}

macro_rules! impl_binary_op {
    ($op:ident, $name: literal, $e:ident) => {
        pub struct $op;

        impl BinaryOp for $op {
            const KERNEL: &'static str = $name;

            fn f32(v: f32, w: f32) -> f32 {
                #[allow(unused_imports)]
                use std::ops::*;
                v.$e(w)
            }

            fn f64(v: f64, w: f64) -> f64 {
                #[allow(unused_imports)]
                use std::ops::*;
                v.$e(w)
            }
        }
    };
}

impl_binary_op!(EWiseAdd, "add", add);
impl_binary_op!(EWiseSub, "sub", sub);
impl_binary_op!(EWiseMul, "mul", mul);
impl_binary_op!(EWiseDiv, "div", div);
impl_binary_op!(EWisePow, "powf", powf);

pub trait BackendStorage: Sized {
    fn ewise_powf(&self, e: f64) -> Result<Self>;
    //fn ewise_log(&self) -> Result<Self>;
    //fn ewise_exp(&self) -> Result<Self>;
    fn unary_op<O: UnaryOp>(&self, op: O) -> Result<Self>;
    fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self>;
    fn dtype(&self) -> DType;
    fn to_vec<D: WithDType>(&self) -> Vec<D>;
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl BackendStorage for Storage {
    fn ewise_powf(&self, e: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.ewise_powf(e)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_op(op)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Storage::Cpu(storage), Storage::Cpu(other)) => {
                let storage = storage.binary_op::<O>(other)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(storage) => storage.dtype(),
        }
    }

    fn to_vec<D: WithDType>(&self) -> Vec<D> {
        match self {
            Storage::Cpu(cpu_storage) => cpu_storage.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32;

    use crate::tests::Approx;

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
    fn test_ewise_div() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);

        let storage = storage.binary_op::<EWiseDiv>(&other).unwrap();

        assert_eq!(storage, Storage::Cpu(CpuStorage::F32(vec![0.25, 0.4, 0.5])));
    }

    #[test]
    fn test_ewise_log() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);

        let storage = storage.unary_op(Log).unwrap();

        assert!(storage
            .to_vec::<f32>()
            .approx_eq(vec![0.0, f32::consts::LN_2, 1.0986]));
    }

    #[test]
    fn test_ewise_exp() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);

        let storage = storage.unary_op(Exp).unwrap();

        assert!(storage
            .to_vec::<f32>()
            .approx_eq(vec![f32::consts::E, 7.3891, 20.0855]));
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

        let storage = storage.ewise_powf(2.0).unwrap();

        assert_eq!(storage, Storage::Cpu(CpuStorage::F32(vec![1.0, 4.0, 9.0])));
    }
}
