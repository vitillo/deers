#![allow(dead_code)]

mod cpu;

use std::borrow::Borrow;

use crate::{
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
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
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self>;
    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self>;
    fn binary_op<O: BinaryOp>(
        &self,
        layout: &Layout,
        other: &Self,
        layout_other: &Layout,
    ) -> Result<Self>;
    fn dtype(&self) -> DType;
    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D>;
    fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl BackendStorage for Storage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.ewise_powf(e, l)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_op(op, l)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    fn binary_op<O: BinaryOp>(
        &self,
        layout: &Layout,
        other: &Self,
        other_layout: &Layout,
    ) -> Result<Self> {
        match (self, other) {
            (Storage::Cpu(storage), Storage::Cpu(other)) => {
                let storage = storage.binary_op::<O>(layout, other, other_layout)?;
                Ok(Self::Cpu(storage))
            }
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(storage) => storage.dtype(),
        }
    }

    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
        match self {
            Storage::Cpu(cpu_storage) => cpu_storage.to_vec(layout),
        }
    }

    fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
        match (self, dst) {
            (Storage::Cpu(src), Storage::Cpu(dst)) => {
                src.copy_compact(src_layout, dst)?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32;

    use crate::{
        layout::{Shape, Strides},
        tests::Approx,
    };

    use super::*;

    #[test]
    fn test_neg() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0]));
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage.unary_op(Neg, &layout).unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_ewise_add() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage
            .binary_op::<EWiseAdd>(&layout, &other, &layout)
            .unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_ewise_sub() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage
            .binary_op::<EWiseSub>(&layout, &other, &layout)
            .unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_ewise_mul() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage
            .binary_op::<EWiseMul>(&layout, &other, &layout)
            .unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_ewise_pow() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage
            .binary_op::<EWisePow>(&layout, &other, &layout)
            .unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![1.0, 32.0, 729.0]);
    }

    #[test]
    fn test_ewise_div() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
        let storage = Storage::Cpu(storage);
        let other = CpuStorage::F32(vec![4.0, 5.0, 6.0]);
        let other = Storage::Cpu(other);
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage
            .binary_op::<EWiseDiv>(&layout, &other, &layout)
            .unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![0.25, 0.4, 0.5]);
    }

    #[test]
    fn test_ewise_log() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0]));
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage.unary_op(Log, &layout).unwrap();

        Vec::<_>::assert_approx_eq(
            storage.to_vec::<f32>(&layout),
            vec![0.0, f32::consts::LN_2, 1.0986],
        );
    }

    #[test]
    fn test_ewise_exp() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0]));
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage.unary_op(Exp, &layout).unwrap();

        Vec::<_>::assert_approx_eq(
            storage.to_vec::<f32>(&layout),
            vec![f32::consts::E, 7.3891, 20.0855],
        );
    }

    #[test]
    fn test_scalar_add() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0]));
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage.unary_op(ScalarAdd(10.0), &layout).unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_powf() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0]));
        let layout = Layout::from(Shape::from((3,)));

        let storage = storage.ewise_powf(2.0, &layout).unwrap();

        assert_eq!(storage.to_vec::<f32>(&layout), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_copy_strided_src() {
        let src = Storage::Cpu(CpuStorage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let mut dst = Storage::Cpu(CpuStorage::F32(vec![0.0; 6]));
        let layout = Layout::new(Shape::from((3, 2)), Strides(vec![1, 3]), 0);

        src.copy_compact(&layout, &mut dst).unwrap();

        assert_eq!(
            dst.to_vec::<f32>(Layout::from(Shape::from((3, 2)))),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn test_to_vec() {
        let storage = Storage::Cpu(CpuStorage::F32(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]));
        let layout = Layout::new(Shape::from((3, 2)), Strides(vec![1, 3]), 0);

        let result = storage.to_vec::<f32>(&layout);

        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
