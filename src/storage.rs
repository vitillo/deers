#![allow(dead_code)]

mod cpu;
mod mps;

use std::borrow::Borrow;

use crate::{
    dtype::{DType, WithDType},
    error::{Error, Result},
    layout::Layout,
};

pub use cpu::*;
pub use mps::*;

/// An element-wise operation applied to a single tensor.
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
unary_op!(Tanh, "tanh", v, v.tanh());
pub struct Relu;

impl UnaryOp for Relu {
    const KERNEL: &'static str = "relu";

    fn f32(&self, v: f32) -> f32 {
        v.max(0.0)
    }

    fn f64(&self, v: f64) -> f64 {
        v.max(0.0)
    }
}

pub struct ReluBackward;

impl UnaryOp for ReluBackward {
    const KERNEL: &'static str = "relu_backward";

    fn f32(&self, v: f32) -> f32 {
        if v > 0.0 { 1.0 } else { 0.0 }
    }

    fn f64(&self, v: f64) -> f64 {
        if v > 0.0 { 1.0 } else { 0.0 }
    }
}

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

/// An element-wise operation applied to two tensors of the same shape.
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

pub struct EWiseEq;

impl BinaryOp for EWiseEq {
    const KERNEL: &'static str = "eq";

    fn f32(v: f32, w: f32) -> f32 {
        if v == w { 1.0 } else { 0.0 }
    }

    fn f64(v: f64, w: f64) -> f64 {
        if v == w { 1.0 } else { 0.0 }
    }
}

/// A reduction operation that combines elements (e.g. sum, max).
pub trait ReduceOp {
    const KERNEL: &'static str;

    fn f32(v: f32, w: f32) -> f32;
    fn f64(v: f64, w: f64) -> f64;
}

pub struct ReduceSum;
impl ReduceOp for ReduceSum {
    const KERNEL: &'static str = "reduce_sum";

    fn f32(acc: f32, x: f32) -> f32 {
        acc + x
    }

    fn f64(acc: f64, x: f64) -> f64 {
        acc + x
    }
}

pub struct ReduceMax;
impl ReduceOp for ReduceMax {
    const KERNEL: &'static str = "reduce_max";

    fn f32(acc: f32, x: f32) -> f32 {
        acc.max(x)
    }

    fn f64(acc: f64, x: f64) -> f64 {
        acc.max(x)
    }
}

/// Trait that storage backends (CPU, future GPU) must implement.
pub trait BackendStorage: Sized {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self>;
    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self>;
    fn binary_op<O: BinaryOp>(
        &self,
        layout: &Layout,
        other: &Self,
        layout_other: &Layout,
    ) -> Result<Self>;
    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()>;
    fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self>;
    /// Gathers values along `dim` using integer indices.
    /// Input must be compact. Returns a new storage with one value per index.
    fn gather(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
    ) -> Result<Self>;
    /// Scatters `src` values into a zero-initialized tensor of size `full_size`,
    /// placing each value at the corresponding index along `dim`.
    fn scatter(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        full_shape: &[usize],
    ) -> Result<Self>;
    fn dtype(&self) -> DType;
    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D>;
    fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()>;
}

/// Backend-agnostic storage enum.
#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
    Mps(MpsStorage),
}

impl BackendStorage for Storage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.ewise_powf(e, l)?;
                Ok(Self::Cpu(storage))
            }
            Storage::Mps(storage) => {
                let storage = storage.ewise_powf(e, l)?;
                Ok(Self::Mps(storage))
            }
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_op(op, l)?;
                Ok(Self::Cpu(storage))
            }
            Storage::Mps(storage) => {
                let storage = storage.unary_op(op, l)?;
                Ok(Self::Mps(storage))
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
            (Storage::Mps(storage), Storage::Mps(other)) => {
                let storage = storage.binary_op::<O>(layout, other, other_layout)?;
                Ok(Self::Mps(storage))
            }
            _ => Err(Error::DeviceMismatch { op: O::KERNEL }),
        }
    }

    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
        match (self, dst) {
            (Storage::Cpu(storage), Storage::Cpu(dst)) => storage.reduce::<O>(layout, dst),
            (Storage::Mps(storage), Storage::Mps(dst)) => storage.reduce::<O>(layout, dst),
            _ => Err(Error::DeviceMismatch { op: O::KERNEL }),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(storage) => storage.dtype(),
            Storage::Mps(storage) => storage.dtype(),
        }
    }

    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
        match self {
            Storage::Cpu(cpu_storage) => cpu_storage.to_vec(layout),
            Storage::Mps(mps_storage) => mps_storage.to_vec(layout),
        }
    }

    fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
        match (self, dst) {
            (Storage::Cpu(src), Storage::Cpu(dst)) => {
                src.copy_compact(src_layout, dst)?;
                Ok(())
            }
            (Storage::Mps(src), Storage::Mps(dst)) => {
                src.copy_compact(src_layout, dst)?;
                Ok(())
            }
            _ => Err(Error::DeviceMismatch { op: "compact" }),
        }
    }

    fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
        match (self, other) {
            (Storage::Cpu(storage), Storage::Cpu(other)) => {
                let storage = storage.matmul(layout, other, layout_other)?;
                Ok(Self::Cpu(storage))
            }
            (Storage::Mps(storage), Storage::Mps(other)) => {
                let storage = storage.matmul(layout, other, layout_other)?;
                Ok(Self::Mps(storage))
            }
            _ => Err(Error::DeviceMismatch { op: "matmul" }),
        }
    }

    fn gather(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
    ) -> Result<Self> {
        match (self, indices) {
            (Storage::Cpu(storage), Storage::Cpu(indices)) => {
                Ok(Self::Cpu(storage.gather(layout, dim, indices, indices_layout)?))
            }
            (Storage::Mps(storage), Storage::Mps(indices)) => {
                Ok(Self::Mps(storage.gather(layout, dim, indices, indices_layout)?))
            }
            _ => Err(Error::DeviceMismatch { op: "gather" }),
        }
    }

    fn scatter(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        full_shape: &[usize],
    ) -> Result<Self> {
        match (self, indices) {
            (Storage::Cpu(storage), Storage::Cpu(indices)) => {
                Ok(Self::Cpu(storage.scatter(layout, dim, indices, indices_layout, full_shape)?))
            }
            (Storage::Mps(storage), Storage::Mps(indices)) => {
                Ok(Self::Mps(storage.scatter(layout, dim, indices, indices_layout, full_shape)?))
            }
            _ => Err(Error::DeviceMismatch { op: "scatter" }),
        }
    }
}

impl Storage {
    /// Concatenates compact storages into a single contiguous storage.
    /// All inputs must be compact and on the same device.
    pub fn cat(parts: &[(&Storage, usize)]) -> Result<Self> {
        assert!(!parts.is_empty());
        match parts[0].0 {
            Storage::Cpu(_) => {
                let cpu_parts: Vec<_> = parts
                    .iter()
                    .map(|(s, len)| match s {
                        Storage::Cpu(cpu) => (cpu, *len),
                        _ => panic!("mixed devices in cat"),
                    })
                    .collect();
                Ok(Storage::Cpu(CpuStorage::cat(&cpu_parts)))
            }
            Storage::Mps(_) => {
                let mps_parts: Vec<_> = parts
                    .iter()
                    .map(|(s, len)| match s {
                        Storage::Mps(mps) => (mps, *len),
                        _ => panic!("mixed devices in cat"),
                    })
                    .collect();
                Ok(Storage::Mps(MpsStorage::cat(&mps_parts)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32;

    use crate::layout::{Shape, Strides};

    use super::*;

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
