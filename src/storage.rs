#![allow(dead_code)]

mod cpu;
pub(crate) mod mps;

use std::borrow::Borrow;

use half::f16;

use crate::{
    dtype::{DType, WithDType},
    error::{Error, Result},
    layout::Layout,
};

pub use cpu::*;
pub use mps::*;

pub(crate) fn synchronize_all() {
    mps::synchronize();
    // cuda::synchronize() when added
}

/// An element-wise kernel shared by storage backends.
///
/// Backends implement the storage traversal; the op only defines the scalar
/// computation for each supported dtype.
pub trait UnaryOp {
    const KERNEL: &'static str;

    fn f16(&self, v: f16) -> f16;
    fn f32(&self, v: f32) -> f32;
}

pub struct Neg;

impl UnaryOp for Neg {
    const KERNEL: &'static str = "Neg";

    fn f16(&self, v: f16) -> f16 {
        -v
    }

    fn f32(&self, v: f32) -> f32 {
        -v
    }
}

macro_rules! unary_op {
    ($op:ident, $name:literal, $a:ident, $e:expr) => {
        pub struct $op;

        impl UnaryOp for $op {
            const KERNEL: &'static str = $name;
            fn f16(&self, $a: f16) -> f16 {
                let $a = $a.to_f32();
                f16::from_f32($e)
            }

            fn f32(&self, $a: f32) -> f32 {
                $e
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp());
unary_op!(Log, "log", v, v.ln());
unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Tanh, "tanh", v, v.tanh());
pub struct Relu;

impl UnaryOp for Relu {
    const KERNEL: &'static str = "relu";

    fn f16(&self, v: f16) -> f16 {
        v.max(f16::from_f32(0.0))
    }

    fn f32(&self, v: f32) -> f32 {
        v.max(0.0)
    }
}

pub struct ReluBackward;

impl UnaryOp for ReluBackward {
    const KERNEL: &'static str = "relu_backward";

    fn f16(&self, v: f16) -> f16 {
        if v > f16::from_f32(0.0) { f16::from_f32(1.0) } else { f16::from_f32(0.0) }
    }

    fn f32(&self, v: f32) -> f32 {
        if v > 0.0 { 1.0 } else { 0.0 }
    }
}

macro_rules! scalar_op {
    ($op:ident, $name:literal, $e:tt) => {
        pub struct $op(pub f64);

        impl UnaryOp for $op {
            const KERNEL: &'static str = $name;
            fn f16(&self, v: f16) -> f16 {
                v $e f16::from_f32(self.0 as f32)
            }

            fn f32(&self, v: f32) -> f32 {
                v $e self.0 as f32
            }
        }
    };
}

scalar_op!(ScalarAdd, "scalar_add", +);
scalar_op!(ScalarMul, "scalar_mul", *);
scalar_op!(ScalarDiv, "scalar_div", /);

/// An element-wise kernel shared by storage backends for two input tensors.
pub trait BinaryOp {
    const KERNEL: &'static str;

    fn f16(v: f16, w: f16) -> f16;
    fn f32(v: f32, w: f32) -> f32;
}

macro_rules! impl_binary_op {
    ($op:ident, $name: literal, $e:ident) => {
        pub struct $op;

        impl BinaryOp for $op {
            const KERNEL: &'static str = $name;

            fn f16(v: f16, w: f16) -> f16 {
                #[allow(unused_imports)]
                use std::ops::*;
                v.$e(w)
            }

            fn f32(v: f32, w: f32) -> f32 {
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
pub struct EWisePow;

impl BinaryOp for EWisePow {
    const KERNEL: &'static str = "powf";

    fn f16(v: f16, w: f16) -> f16 {
        f16::from_f32(v.to_f32().powf(w.to_f32()))
    }

    fn f32(v: f32, w: f32) -> f32 {
        v.powf(w)
    }
}

pub struct EWiseEq;

impl BinaryOp for EWiseEq {
    const KERNEL: &'static str = "eq";

    fn f16(v: f16, w: f16) -> f16 {
        if v == w { f16::from_f32(1.0) } else { f16::from_f32(0.0) }
    }

    fn f32(v: f32, w: f32) -> f32 {
        if v == w { 1.0 } else { 0.0 }
    }
}

/// A reduction kernel shared by storage backends.
pub trait ReduceOp {
    const KERNEL: &'static str;

    fn f16(v: f16, w: f16) -> f16;
    fn f32(v: f32, w: f32) -> f32;
}

pub struct ReduceSum;
impl ReduceOp for ReduceSum {
    const KERNEL: &'static str = "reduce_sum";

    fn f16(acc: f16, x: f16) -> f16 {
        acc + x
    }

    fn f32(acc: f32, x: f32) -> f32 {
        acc + x
    }
}

pub struct ReduceMax;
impl ReduceOp for ReduceMax {
    const KERNEL: &'static str = "reduce_max";

    fn f16(acc: f16, x: f16) -> f16 {
        acc.max(x)
    }

    fn f32(acc: f32, x: f32) -> f32 {
        acc.max(x)
    }
}

/// Trait implemented by concrete storage backends such as CPU and MPS.
///
/// Layout arguments describe how to interpret the existing buffer contents.
/// Methods that return a new storage produce a compact output buffer unless
/// documented otherwise.
pub trait BackendStorage: Sized {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self>;
    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self>;
    /// Applies an element-wise binary kernel to two layouts with the same shape.
    fn binary_op<O: BinaryOp>(
        &self,
        layout: &Layout,
        other: &Self,
        layout_other: &Layout,
    ) -> Result<Self>;
    /// Reduces `layout` into the already-allocated compact destination storage `dst`.
    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()>;
    /// Matrix multiplication for layouts whose shapes are compatible under matmul rules.
    fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self>;
    /// Gathers values along `dim` using compact integer indices.
    /// `indices` must have the same rank as `layout` and the same shape on every
    /// non-indexed dimension. The returned compact storage matches `indices`.
    fn gather(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
    ) -> Result<Self>;
    /// Scatter-adds `src` values into a zero-initialized tensor of shape `dst_shape`,
    /// accumulating each value at the corresponding index along `dim`.
    fn scatter_add(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        dst_shape: &[usize],
    ) -> Result<Self>;
    /// Selects slices along `dim` using a compact 1-D integer index tensor.
    /// The output matches `layout` except the length at `dim` becomes `indices.len()`.
    fn index_select(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
    ) -> Result<Self>;
    /// Adds source slices into a zero-initialized tensor of shape `dst_shape`
    /// along `dim` using a compact 1-D integer index tensor.
    fn index_add(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        dst_shape: &[usize],
    ) -> Result<Self>;
    fn dtype(&self) -> DType;
    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D>;
    fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()>;
}

/// Backend-agnostic storage wrapper.
///
/// This keeps tensor code generic while still making device dispatch explicit at
/// the storage boundary.
#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
    Mps(MpsStorage),
}

impl BackendStorage for Storage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.ewise_powf(e, l)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.ewise_powf(e, l)?)),
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.unary_op(op, l)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.unary_op(op, l)?)),
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
                Ok(Self::Cpu(storage.binary_op::<O>(layout, other, other_layout)?))
            }
            (Storage::Mps(storage), Storage::Mps(other)) => {
                Ok(Self::Mps(storage.binary_op::<O>(layout, other, other_layout)?))
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
                Ok(Self::Cpu(storage.matmul(layout, other, layout_other)?))
            }
            (Storage::Mps(storage), Storage::Mps(other)) => {
                Ok(Self::Mps(storage.matmul(layout, other, layout_other)?))
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

    fn scatter_add(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        dst_shape: &[usize],
    ) -> Result<Self> {
        match (self, indices) {
            (Storage::Cpu(storage), Storage::Cpu(indices)) => Ok(Self::Cpu(storage.scatter_add(
                layout,
                dim,
                indices,
                indices_layout,
                dst_shape,
            )?)),
            (Storage::Mps(storage), Storage::Mps(indices)) => Ok(Self::Mps(storage.scatter_add(
                layout,
                dim,
                indices,
                indices_layout,
                dst_shape,
            )?)),
            _ => Err(Error::DeviceMismatch { op: "scatter_add" }),
        }
    }

    fn index_select(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
    ) -> Result<Self> {
        match (self, indices) {
            (Storage::Cpu(storage), Storage::Cpu(indices)) => {
                Ok(Self::Cpu(storage.index_select(layout, dim, indices, indices_layout)?))
            }
            (Storage::Mps(storage), Storage::Mps(indices)) => {
                Ok(Self::Mps(storage.index_select(layout, dim, indices, indices_layout)?))
            }
            _ => Err(Error::DeviceMismatch { op: "index_select" }),
        }
    }

    fn index_add(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        dst_shape: &[usize],
    ) -> Result<Self> {
        match (self, indices) {
            (Storage::Cpu(storage), Storage::Cpu(indices)) => {
                Ok(Self::Cpu(storage.index_add(layout, dim, indices, indices_layout, dst_shape)?))
            }
            (Storage::Mps(storage), Storage::Mps(indices)) => {
                Ok(Self::Mps(storage.index_add(layout, dim, indices, indices_layout, dst_shape)?))
            }
            _ => Err(Error::DeviceMismatch { op: "index_add" }),
        }
    }
}

impl Storage {
    /// Concatenates compact storages into a single contiguous storage.
    /// All inputs must be compact and on the same device.
    /// Each `usize` is the number of valid elements contributed by that storage.
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
                Ok(Storage::Cpu(CpuStorage::cat(&cpu_parts)?))
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
