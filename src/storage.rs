//! Backend storage dispatch (CPU, CUDA, MPS) and the [`BackendStorage`] trait
//! that each backend implements.

#![allow(dead_code)]

mod cpu;
pub(crate) mod cuda;
pub(crate) mod mps;

use std::borrow::Borrow;

use half::f16;

use crate::{
    device::Device,
    dtype::{DType, WithDType},
    error::{Error, Result},
    layout::Layout,
};

pub use cpu::*;
pub use cuda::*;
pub use mps::*;

pub(crate) fn synchronize_all() {
    mps::synchronize();
    cuda::synchronize();
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
    /// Element-wise `!= scalar`: 1 where different, else 0, in the input dtype.
    /// Returns compact storage and carries no gradient.
    fn ne_scalar(&self, layout: &Layout, scalar: f64) -> Result<Self>;
    /// Element-wise `!= scalar` for integer storage: 1 where different, else 0.
    /// Returns compact storage and carries no gradient.
    fn ne_scalar_i64(&self, layout: &Layout, scalar: i64) -> Result<Self>;
    /// Picks from `on_true` where `cond` is nonzero, else from `on_false`.
    /// All three share one dtype and shape; returns compact storage.
    fn select(
        &self,
        cond_layout: &Layout,
        on_true: &Self,
        true_layout: &Layout,
        on_false: &Self,
        false_layout: &Layout,
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
    /// Fused log-sum-exp: log(sum(exp(x - max(x)))) + max(x) per row.
    /// `layout` must be compact. Reduces `reduce_size` elements per row
    /// into `outer_size` output elements, where `outer_size * reduce_size == layout.size()`.
    fn log_sum_exp(&self, layout: &Layout, outer_size: usize, reduce_size: usize) -> Result<Self>;
    /// Fused log-softmax forward: `dst[i] = src[i] - log(sum_j exp(src[j]))` per row.
    ///
    /// `outer_size * inner_size` must equal `layout.size()`. The layout must be compact.
    fn log_softmax_fwd(
        &self,
        layout: &Layout,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<Self>;
    /// Fused log-softmax backward: `grad_input[i] = grad[i] - exp(lsm[i]) * sum_j grad[j]`
    /// per row, where `lsm` is the saved log-softmax output from the forward pass.
    fn log_softmax_bwd(
        &self,
        grad_layout: &Layout,
        lsm: &Self,
        lsm_layout: &Layout,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<Self>;
    /// Fused RMSNorm forward: `dst = src * rsqrt(mean(src^2) + eps) * w` per row.
    ///
    /// `outer_size * inner_size` must equal `layout.size()`. The fused kernel
    /// reads a compacted input; `weight` (if any) holds `inner_size` elements
    /// of the same dtype.
    fn rms_norm_fwd(
        &self,
        layout: &Layout,
        weight: Option<(&Self, &Layout)>,
        outer_size: usize,
        inner_size: usize,
        eps: f32,
    ) -> Result<Self>;
    /// Fused RoPE forward over `[B, T, H, D]` rows: `y1 = x1*cos - x2*sin`,
    /// `y2 = x1*sin + x2*cos`, with the cos/sin row selected per token.
    ///
    /// `outer_size * head_dim` must equal `layout.size()`. Cos/sin hold
    /// `cos_t_len` rows of `head_dim / 2` in the same dtype as the input.
    #[allow(clippy::too_many_arguments)]
    fn rope_fwd(
        &self,
        layout: &Layout,
        cos: &Self,
        cos_layout: &Layout,
        sin: &Self,
        sin_layout: &Layout,
        outer_size: usize,
        head_dim: usize,
        n_heads: usize,
        t_len: usize,
        cos_t_len: usize,
    ) -> Result<Self>;
    /// Fused SiLU forward: `dst[i] = src[i] / (1 + exp(-src[i]))`.
    ///
    /// `layout.size()` elements are read in compact order and written to a
    /// fresh compact buffer.
    fn silu_fwd(&self, layout: &Layout) -> Result<Self>;
    /// Copies `blocks` contiguous `block_len`-element runs from compact `src`
    /// into `self`, where destination run `n` starts at
    /// `dst_base + n * dst_stride`.
    ///
    /// `src_layout.size()` must equal `blocks * block_len`. A strided source
    /// is compacted first, so callers pass any layout.
    fn copy_blocks_into(
        &mut self,
        src: &Self,
        src_layout: &Layout,
        blocks: usize,
        block_len: usize,
        dst_base: usize,
        dst_stride: usize,
    ) -> Result<()>;
    /// Fused SiLU-gate product: `dst[i] = silu(gate[i]) * up[i]`.
    ///
    /// Both inputs share `layout.size()` elements, read in compact order.
    fn silu_mul_fwd(&self, layout: &Layout, up: &Self, up_layout: &Layout) -> Result<Self>;
    /// Fused scaled masked softmax: `dst = softmax(scores*scale + mask)` rows.
    ///
    /// `outer_size * inner_size` must equal `layout.size()`. The mask holds
    /// `mask_t_len` rows of `inner_size`, one per query position.
    #[allow(clippy::too_many_arguments)]
    fn masked_softmax_fwd(
        &self,
        layout: &Layout,
        mask: &Self,
        mask_layout: &Layout,
        outer_size: usize,
        inner_size: usize,
        scale: f32,
        t_len: usize,
        mask_t_len: usize,
    ) -> Result<Self>;
    /// Converts `layout` to `dtype` without leaving the device.
    ///
    /// The cast reads strided source elements and writes a compact output buffer:
    /// floats round to nearest-even when narrowing and truncate toward zero when
    /// targeting `I64`; integers widen exactly into floats. Backends implement this
    /// with native on-device kernels, never a host roundtrip.
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self>;
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
    Cuda(CudaStorage),
    Mps(MpsStorage),
}

impl BackendStorage for Storage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.ewise_powf(e, l)?)),
            Storage::Cuda(storage) => Ok(Self::Cuda(storage.ewise_powf(e, l)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.ewise_powf(e, l)?)),
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.unary_op(op, l)?)),
            Storage::Cuda(storage) => Ok(Self::Cuda(storage.unary_op(op, l)?)),
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
            (Storage::Cuda(storage), Storage::Cuda(other)) => {
                Ok(Self::Cuda(storage.binary_op::<O>(layout, other, other_layout)?))
            }
            (Storage::Mps(storage), Storage::Mps(other)) => {
                Ok(Self::Mps(storage.binary_op::<O>(layout, other, other_layout)?))
            }
            _ => Err(Error::DeviceMismatch { op: O::KERNEL }),
        }
    }

    fn ne_scalar(&self, layout: &Layout, scalar: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.ne_scalar(layout, scalar)?)),
            Storage::Cuda(storage) => Ok(Self::Cuda(storage.ne_scalar(layout, scalar)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.ne_scalar(layout, scalar)?)),
        }
    }

    fn ne_scalar_i64(&self, layout: &Layout, scalar: i64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.ne_scalar_i64(layout, scalar)?)),
            Storage::Cuda(storage) => Ok(Self::Cuda(storage.ne_scalar_i64(layout, scalar)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.ne_scalar_i64(layout, scalar)?)),
        }
    }

    fn select(
        &self,
        cond_layout: &Layout,
        on_true: &Self,
        true_layout: &Layout,
        on_false: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        match (self, on_true, on_false) {
            (Storage::Cpu(cond), Storage::Cpu(t), Storage::Cpu(f)) => {
                Ok(Self::Cpu(cond.select(cond_layout, t, true_layout, f, false_layout)?))
            }
            (Storage::Cuda(cond), Storage::Cuda(t), Storage::Cuda(f)) => {
                Ok(Self::Cuda(cond.select(cond_layout, t, true_layout, f, false_layout)?))
            }
            (Storage::Mps(cond), Storage::Mps(t), Storage::Mps(f)) => {
                Ok(Self::Mps(cond.select(cond_layout, t, true_layout, f, false_layout)?))
            }
            _ => Err(Error::DeviceMismatch { op: "select" }),
        }
    }

    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
        match (self, dst) {
            (Storage::Cpu(storage), Storage::Cpu(dst)) => storage.reduce::<O>(layout, dst),
            (Storage::Cuda(storage), Storage::Cuda(dst)) => storage.reduce::<O>(layout, dst),
            (Storage::Mps(storage), Storage::Mps(dst)) => storage.reduce::<O>(layout, dst),
            _ => Err(Error::DeviceMismatch { op: O::KERNEL }),
        }
    }

    fn log_sum_exp(&self, layout: &Layout, outer_size: usize, reduce_size: usize) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                Ok(Self::Cpu(storage.log_sum_exp(layout, outer_size, reduce_size)?))
            }
            Storage::Cuda(storage) => {
                Ok(Self::Cuda(storage.log_sum_exp(layout, outer_size, reduce_size)?))
            }
            Storage::Mps(storage) => {
                Ok(Self::Mps(storage.log_sum_exp(layout, outer_size, reduce_size)?))
            }
        }
    }

    fn log_softmax_fwd(
        &self,
        layout: &Layout,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                Ok(Self::Cpu(storage.log_softmax_fwd(layout, outer_size, inner_size)?))
            }
            Storage::Cuda(storage) => {
                Ok(Self::Cuda(storage.log_softmax_fwd(layout, outer_size, inner_size)?))
            }
            Storage::Mps(storage) => {
                Ok(Self::Mps(storage.log_softmax_fwd(layout, outer_size, inner_size)?))
            }
        }
    }

    fn log_softmax_bwd(
        &self,
        grad_layout: &Layout,
        lsm: &Self,
        lsm_layout: &Layout,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<Self> {
        match (self, lsm) {
            (Storage::Cpu(grad), Storage::Cpu(lsm)) => Ok(Self::Cpu(
                grad.log_softmax_bwd(grad_layout, lsm, lsm_layout, outer_size, inner_size)?,
            )),
            (Storage::Cuda(grad), Storage::Cuda(lsm)) => Ok(Self::Cuda(
                grad.log_softmax_bwd(grad_layout, lsm, lsm_layout, outer_size, inner_size)?,
            )),
            (Storage::Mps(grad), Storage::Mps(lsm)) => Ok(Self::Mps(
                grad.log_softmax_bwd(grad_layout, lsm, lsm_layout, outer_size, inner_size)?,
            )),
            _ => Err(Error::DeviceMismatch { op: "log_softmax_bwd" }),
        }
    }

    fn rms_norm_fwd(
        &self,
        layout: &Layout,
        weight: Option<(&Self, &Layout)>,
        outer_size: usize,
        inner_size: usize,
        eps: f32,
    ) -> Result<Self> {
        match (self, weight) {
            (Storage::Cpu(storage), None) => Ok(Self::Cpu(
                storage.rms_norm_fwd(layout, None, outer_size, inner_size, eps)?,
            )),
            (Storage::Cpu(storage), Some((Storage::Cpu(w), w_layout))) => {
                Ok(Self::Cpu(storage.rms_norm_fwd(
                    layout,
                    Some((w, w_layout)),
                    outer_size,
                    inner_size,
                    eps,
                )?))
            }
            (Storage::Cuda(storage), None) => Ok(Self::Cuda(
                storage.rms_norm_fwd(layout, None, outer_size, inner_size, eps)?,
            )),
            (Storage::Cuda(storage), Some((Storage::Cuda(w), w_layout))) => {
                Ok(Self::Cuda(storage.rms_norm_fwd(
                    layout,
                    Some((w, w_layout)),
                    outer_size,
                    inner_size,
                    eps,
                )?))
            }
            (Storage::Mps(storage), None) => Ok(Self::Mps(
                storage.rms_norm_fwd(layout, None, outer_size, inner_size, eps)?,
            )),
            (Storage::Mps(storage), Some((Storage::Mps(w), w_layout))) => {
                Ok(Self::Mps(storage.rms_norm_fwd(
                    layout,
                    Some((w, w_layout)),
                    outer_size,
                    inner_size,
                    eps,
                )?))
            }
            _ => Err(Error::DeviceMismatch { op: "rms_norm_fwd" }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_fwd(
        &self,
        layout: &Layout,
        cos: &Self,
        cos_layout: &Layout,
        sin: &Self,
        sin_layout: &Layout,
        outer_size: usize,
        head_dim: usize,
        n_heads: usize,
        t_len: usize,
        cos_t_len: usize,
    ) -> Result<Self> {
        match (self, cos, sin) {
            (Storage::Cpu(x), Storage::Cpu(c), Storage::Cpu(s)) => Ok(Self::Cpu(x.rope_fwd(
                layout, c, cos_layout, s, sin_layout,
                outer_size, head_dim, n_heads, t_len, cos_t_len,
            )?)),
            (Storage::Cuda(x), Storage::Cuda(c), Storage::Cuda(s)) => Ok(Self::Cuda(x.rope_fwd(
                layout, c, cos_layout, s, sin_layout,
                outer_size, head_dim, n_heads, t_len, cos_t_len,
            )?)),
            (Storage::Mps(x), Storage::Mps(c), Storage::Mps(s)) => Ok(Self::Mps(x.rope_fwd(
                layout, c, cos_layout, s, sin_layout,
                outer_size, head_dim, n_heads, t_len, cos_t_len,
            )?)),
            _ => Err(Error::DeviceMismatch { op: "rope_fwd" }),
        }
    }

    fn silu_fwd(&self, layout: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.silu_fwd(layout)?)),
            Storage::Cuda(storage) => Ok(Self::Cuda(storage.silu_fwd(layout)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.silu_fwd(layout)?)),
        }
    }

    fn copy_blocks_into(
        &mut self,
        src: &Self,
        src_layout: &Layout,
        blocks: usize,
        block_len: usize,
        dst_base: usize,
        dst_stride: usize,
    ) -> Result<()> {
        match (self, src) {
            (Storage::Cpu(dst), Storage::Cpu(s)) => {
                dst.copy_blocks_into(s, src_layout, blocks, block_len, dst_base, dst_stride)
            }
            (Storage::Cuda(dst), Storage::Cuda(s)) => {
                dst.copy_blocks_into(s, src_layout, blocks, block_len, dst_base, dst_stride)
            }
            (Storage::Mps(dst), Storage::Mps(s)) => {
                dst.copy_blocks_into(s, src_layout, blocks, block_len, dst_base, dst_stride)
            }
            _ => Err(Error::DeviceMismatch { op: "copy_blocks_into" }),
        }
    }

    fn silu_mul_fwd(&self, layout: &Layout, up: &Self, up_layout: &Layout) -> Result<Self> {
        match (self, up) {
            (Storage::Cpu(gate), Storage::Cpu(u)) => {
                Ok(Self::Cpu(gate.silu_mul_fwd(layout, u, up_layout)?))
            }
            (Storage::Cuda(gate), Storage::Cuda(u)) => {
                Ok(Self::Cuda(gate.silu_mul_fwd(layout, u, up_layout)?))
            }
            (Storage::Mps(gate), Storage::Mps(u)) => {
                Ok(Self::Mps(gate.silu_mul_fwd(layout, u, up_layout)?))
            }
            _ => Err(Error::DeviceMismatch { op: "silu_mul_fwd" }),
        }
    }

    fn masked_softmax_fwd(
        &self,
        layout: &Layout,
        mask: &Self,
        mask_layout: &Layout,
        outer_size: usize,
        inner_size: usize,
        scale: f32,
        t_len: usize,
        mask_t_len: usize,
    ) -> Result<Self> {
        match (self, mask) {
            (Storage::Cpu(scores), Storage::Cpu(m)) => Ok(Self::Cpu(scores
                .masked_softmax_fwd(
                    layout, m, mask_layout,
                    outer_size, inner_size, scale, t_len, mask_t_len,
                )?)),
            (Storage::Cuda(scores), Storage::Cuda(m)) => Ok(Self::Cuda(scores
                .masked_softmax_fwd(
                    layout, m, mask_layout,
                    outer_size, inner_size, scale, t_len, mask_t_len,
                )?)),
            (Storage::Mps(scores), Storage::Mps(m)) => Ok(Self::Mps(scores
                .masked_softmax_fwd(
                    layout, m, mask_layout,
                    outer_size, inner_size, scale, t_len, mask_t_len,
                )?)),
            _ => Err(Error::DeviceMismatch { op: "masked_softmax_fwd" }),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(storage) => storage.dtype(),
            Storage::Cuda(storage) => storage.dtype(),
            Storage::Mps(storage) => storage.dtype(),
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => Ok(Self::Cpu(storage.to_dtype(layout, dtype)?)),
            Storage::Cuda(storage) => Ok(Self::Cuda(storage.to_dtype(layout, dtype)?)),
            Storage::Mps(storage) => Ok(Self::Mps(storage.to_dtype(layout, dtype)?)),
        }
    }

    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
        match self {
            Storage::Cpu(cpu_storage) => cpu_storage.to_vec(layout),
            Storage::Cuda(cuda_storage) => cuda_storage.to_vec(layout),
            Storage::Mps(mps_storage) => mps_storage.to_vec(layout),
        }
    }

    fn copy_compact(&self, src_layout: &Layout, dst: &mut Self) -> Result<()> {
        match (self, dst) {
            (Storage::Cpu(src), Storage::Cpu(dst)) => {
                src.copy_compact(src_layout, dst)?;
                Ok(())
            }
            (Storage::Cuda(src), Storage::Cuda(dst)) => {
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
            (Storage::Cuda(storage), Storage::Cuda(other)) => {
                Ok(Self::Cuda(storage.matmul(layout, other, layout_other)?))
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
            (Storage::Cuda(storage), Storage::Cuda(indices)) => {
                Ok(Self::Cuda(storage.gather(layout, dim, indices, indices_layout)?))
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
            (Storage::Cuda(storage), Storage::Cuda(indices)) => Ok(Self::Cuda(
                storage.scatter_add(layout, dim, indices, indices_layout, dst_shape)?,
            )),
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
            (Storage::Cuda(storage), Storage::Cuda(indices)) => {
                Ok(Self::Cuda(storage.index_select(layout, dim, indices, indices_layout)?))
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
            (Storage::Cuda(storage), Storage::Cuda(indices)) => Ok(Self::Cuda(storage.index_add(
                layout,
                dim,
                indices,
                indices_layout,
                dst_shape,
            )?)),
            (Storage::Mps(storage), Storage::Mps(indices)) => {
                Ok(Self::Mps(storage.index_add(layout, dim, indices, indices_layout, dst_shape)?))
            }
            _ => Err(Error::DeviceMismatch { op: "index_add" }),
        }
    }
}

impl Storage {
    /// Copies the `layout` region of this storage onto `device`.
    ///
    /// The returned storage is compact and holds exactly `layout.size()`
    /// elements. Same-backend moves copy between device buffers with no host
    /// traffic. Moves to or from the host perform a single upload or download,
    /// borrowing compact host sources instead of staging through a temporary.
    ///
    /// The remaining host-transit case is cross-vendor accelerator moves (CUDA
    /// to MPS or the reverse), which stage through one host buffer: the two
    /// vendor APIs expose no peer-DMA path, and the backends never coexist on
    /// one machine, so no direct copy exists to implement.
    pub fn transfer(&self, layout: &Layout, device: Device) -> Result<Self> {
        match (self, device) {
            (Storage::Cpu(src), Device::Cpu) => {
                let Storage::Cpu(mut dst) = Device::Cpu.zeros(layout.size(), src.dtype()) else {
                    unreachable!("cpu zeros returned non-cpu storage");
                };
                src.copy_compact(layout, &mut dst)?;
                Ok(Self::Cpu(dst))
            }
            (Storage::Cuda(src), Device::Cuda) => Ok(Self::Cuda(src.copy_to_device(layout)?)),
            (Storage::Mps(src), Device::Mps) => Ok(Self::Mps(src.copy_to_device(layout)?)),
            (Storage::Cpu(src), Device::Cuda) => {
                Ok(Self::Cuda(CudaStorage::copy_from_cpu(src, layout)?))
            }
            (Storage::Cpu(src), Device::Mps) => {
                Ok(Self::Mps(MpsStorage::copy_from_cpu(src, layout)?))
            }
            (Storage::Cuda(src), Device::Cpu) => Ok(Self::Cpu(src.copy_to_cpu(layout)?)),
            (Storage::Mps(src), Device::Cpu) => Ok(Self::Cpu(src.copy_to_cpu(layout)?)),
            (Storage::Cuda(src), Device::Mps) => {
                let cpu = src.copy_to_cpu(layout)?;
                let compact =
                    Layout::new(layout.shape().clone(), layout.shape().compact_strides(), 0);
                Ok(Self::Mps(MpsStorage::copy_from_cpu(&cpu, &compact)?))
            }
            (Storage::Mps(src), Device::Cuda) => {
                let cpu = src.copy_to_cpu(layout)?;
                let compact =
                    Layout::new(layout.shape().clone(), layout.shape().compact_strides(), 0);
                Ok(Self::Cuda(CudaStorage::copy_from_cpu(&cpu, &compact)?))
            }
        }
    }

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
            Storage::Cuda(_) => {
                let cuda_parts: Vec<_> = parts
                    .iter()
                    .map(|(s, len)| match s {
                        Storage::Cuda(cuda) => (cuda, *len),
                        _ => panic!("mixed devices in cat"),
                    })
                    .collect();
                Ok(Storage::Cuda(CudaStorage::cat(&cuda_parts)?))
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
