//! The central [`Tensor`] type: a reference-counted, multi-dimensional array
//! with lazy autograd graph construction and operator overloading.

#![allow(dead_code)]

use std::borrow::Borrow;
use std::cell::Cell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

thread_local! {
    static NO_GRAD: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that disables autograd tracking for all tensor operations
/// within its scope. Used during backward to avoid building secondary graphs.
pub(crate) struct NoGradGuard;

impl NoGradGuard {
    pub fn new() -> Self {
        NO_GRAD.with(|f| f.set(true));
        NoGradGuard
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD.with(|f| f.set(false));
    }
}

use half::f16;
use rand::RngExt;

use crate::device::Device;
use crate::dtype::{DType, WithDType};
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::ops::{self, TensorOp};
use crate::storage::{BackendStorage, CpuStorage, Storage};

/// Unique identifier for a tensor, used to look up gradients after [`Tensor::backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(usize);

/// Shared backing state for a [`Tensor`].
///
/// Most code should use [`Tensor`] rather than interacting with this type
/// directly. It stores the tensor's storage, layout, autograd metadata,
/// and unique identifier behind the `Arc` owned by [`Tensor`].
#[derive(Debug)]
struct TensorInternal {
    id: TensorId,
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    op: Option<Box<dyn TensorOp>>,
    requires_grad: bool,
}

/// A multi-dimensional array with optional automatic differentiation.
///
/// Tensors are reference-counted (`Arc`-wrapped) so cloning is cheap and
/// shares the underlying storage. Operations on tensors build a computation
/// graph that can be traversed with [`backward`](Tensor::backward) to
/// compute gradients.
///
/// Arithmetic operators (`+`, `-`, `*`, `/`) work element-wise on tensors
/// of the same shape, or between a tensor and an `f64` scalar.
#[derive(Clone, Debug)]
pub struct Tensor(Arc<TensorInternal>);

impl From<TensorInternal> for Tensor {
    fn from(value: TensorInternal) -> Self {
        Self(Arc::new(value))
    }
}

impl Tensor {
    fn from_plain_storage(storage: Storage, shape: Shape) -> Self {
        let layout: Layout = shape.into();
        Tensor::new(Arc::new(RwLock::new(storage)), layout, false, None)
    }

    pub(crate) fn new(
        storage: Arc<RwLock<Storage>>,
        layout: Layout,
        requires_grad: bool,
        op: Option<Box<dyn TensorOp>>,
    ) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = TensorId(COUNTER.fetch_add(1, Ordering::Relaxed));

        // When no_grad is active (e.g. during backward), skip autograd tracking.
        let (requires_grad, op) = if NO_GRAD.with(|f| f.get()) {
            (false, None)
        } else {
            let requires_grad = requires_grad
                || op
                    .as_ref()
                    .map(|o| o.dependencies().iter().any(|dep| dep.requires_grad()))
                    .unwrap_or(false);
            let op = if requires_grad { op } else { None };
            (requires_grad, op)
        };

        TensorInternal { id, storage, layout, op, requires_grad }.into()
    }

    /// Returns the operation that produced this tensor, if any.
    pub fn op(&self) -> &Option<Box<dyn TensorOp>> {
        &self.0.op
    }

    /// Returns this tensor's unique identifier.
    pub fn id(&self) -> TensorId {
        self.0.id
    }

    /// Returns a read guard to the underlying storage.
    pub fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        self.0.storage.read().unwrap()
    }

    pub(crate) fn storage_clone(&self) -> Arc<RwLock<Storage>> {
        self.0.storage.clone()
    }

    /// Returns a write guard to the underlying storage.
    pub(crate) fn storage_mut(&self) -> RwLockWriteGuard<'_, Storage> {
        self.0.storage.write().unwrap()
    }

    /// Returns the layout (shape, strides, offset) of this tensor.
    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    /// Returns the tensor element type.
    pub fn dtype(&self) -> DType {
        self.storage().dtype()
    }

    /// Returns the device (CPU, MPS) where this tensor is stored.
    pub fn device(&self) -> Device {
        match &*self.storage() {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Cuda(_) => Device::Cuda,
            Storage::Mps(_) => Device::Mps,
        }
    }

    /// Returns whether this tensor tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    /// Enables gradient tracking for this tensor. This is the equivalent of
    /// PyTorch's `requires_grad_(True)`. Tensors produced by operations on an
    /// attached tensor will also track gradients.
    pub fn attach(self) -> Tensor {
        if self.0.requires_grad {
            return self.clone();
        }

        Tensor::new(self.0.storage.clone(), self.0.layout.clone(), true, None)
    }

    /// Returns a view of this tensor that does not track gradients.
    /// Operations on the detached tensor will not build a computation graph.
    /// This is the equivalent of PyTorch's `Tensor.detach()`.
    pub fn detach(&self) -> Tensor {
        if !self.0.requires_grad {
            return self.clone();
        }
        Tensor::new(self.0.storage.clone(), self.0.layout.clone(), false, None)
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        Tensor::from_plain_storage(device.zeros(shape.size(), dtype), shape)
    }

    /// Creates a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        Tensor::from_plain_storage(device.ones(shape.size(), dtype), shape)
    }

    /// Creates a tensor from a CPU vector with the given shape.
    pub fn from_vec(vec: impl Into<CpuStorage>, shape: impl Into<Shape>, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        Tensor::from_plain_storage(vec.into().to(device), shape)
    }

    /// Create a tensor with values drawn from a uniform distribution in [0, 1).
    pub fn rand(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let mut rng = rand::rng();
        let storage = match dtype {
            DType::F16 => {
                let data: Vec<f16> =
                    (0..shape.size()).map(|_| f16::from_f32(rng.random::<f32>())).collect();
                CpuStorage::from(data).to(device)
            }
            DType::F32 => {
                let data: Vec<f32> = (0..shape.size()).map(|_| rng.random()).collect();
                CpuStorage::from(data).to(device)
            }
            _ => unimplemented!(),
        };
        Tensor::from_plain_storage(storage, shape)
    }

    /// Create a tensor with values drawn from a standard normal distribution.
    pub fn randn(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let mut rng = rand::rng();
        // Box-Muller transform
        let storage = match dtype {
            DType::F16 => {
                let data: Vec<f16> = (0..shape.size())
                    .map(|_| {
                        let u1: f32 = rng.random();
                        let u2: f32 = rng.random();
                        let z =
                            (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                        f16::from_f32(z)
                    })
                    .collect();
                CpuStorage::from(data).to(device)
            }
            DType::F32 => {
                let data: Vec<f32> = (0..shape.size())
                    .map(|_| {
                        let u1: f32 = rng.random();
                        let u2: f32 = rng.random();
                        (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
                    })
                    .collect();
                CpuStorage::from(data).to(device)
            }
            _ => unimplemented!(),
        };
        Tensor::from_plain_storage(storage, shape)
    }

    /// Copies the tensor data into a flat `Vec`, respecting strides.
    pub fn to_vec<S: WithDType>(&self) -> Result<Vec<S>> {
        Ok(self.storage().to_vec(&self.0.layout))
    }

    /// Returns a copy of this tensor on the target device.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if self.device() == device {
            return Ok(self.clone());
        }

        let shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        let out = match self.dtype() {
            DType::F16 => Tensor::from_vec(self.to_vec::<f16>()?, shape, device),
            DType::F32 => Tensor::from_vec(self.to_vec::<f32>()?, shape, device),
            DType::I64 => Tensor::from_vec(self.to_vec::<i64>()?, shape, device),
        };
        Ok(out)
    }

    /// Creates a tensor of ones with the same shape, dtype, and device.
    pub fn ones_like(&self) -> Tensor {
        Tensor::ones(self.layout().shape().clone(), self.dtype(), self.device())
    }

    /// Creates a tensor of zeros with the same shape, dtype, and device.
    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.layout().shape().clone(), self.dtype(), self.device())
    }

    /// Returns a no-copy view over a contiguous range on the given dimension.
    ///
    /// This is intended for dataset-style batching and other simple slicing.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        ops::Narrow::new(self.clone(), dim, start, len).unwrap().forward().unwrap()
    }

    /// Reorders the dimensions of the tensor. Does not copy data.
    pub fn permute(&self, axis: impl Into<Shape>) -> Tensor {
        ops::Permute::new(self.clone(), axis.into()).unwrap().forward().unwrap()
    }

    /// Expands dimensions of size 1 to match `new_shape`. Does not copy data.
    pub fn broadcast(&self, new_shape: impl Into<Shape>) -> Tensor {
        ops::Broadcast::new(self.clone(), new_shape.into()).unwrap().forward().unwrap()
    }

    /// Returns a view with a different shape but the same total number of elements.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Tensor {
        ops::Reshape::new(self.clone(), new_shape.into()).unwrap().forward().unwrap()
    }

    /// Swaps two dimensions. Defaults to the last two if `axes` is `None`.
    pub fn transpose(&self, axes: Option<(usize, usize)>) -> Tensor {
        assert!(self.layout().ndim() >= 2, "transpose requires ndim >= 2");
        let axes = axes.unwrap_or((self.layout().ndim() - 2, self.layout().ndim() - 1));
        assert!(
            axes.0 < self.layout().ndim() && axes.1 < self.layout().ndim(),
            "Transpose axes must be less than tensor dimensions",
        );
        let mut reshaped_axes: Vec<_> = (0..self.0.layout.ndim()).collect();
        (reshaped_axes[axes.0], reshaped_axes[axes.1]) =
            (reshaped_axes[axes.1], reshaped_axes[axes.0]);
        self.permute(reshaped_axes)
    }

    /// Sums elements along the given axes. If `keep_dims`, reduced axes become size 1.
    pub fn sum(&self, axis: Vec<usize>, keep_dims: bool) -> Tensor {
        ops::Sum::new(self.clone(), axis, keep_dims).unwrap().forward().unwrap()
    }

    /// Returns the maximum along the given axes.
    pub fn max(&self, axis: Vec<usize>, keep_dims: bool) -> Tensor {
        ops::Max::new(self.clone(), axis, keep_dims).unwrap().forward().unwrap()
    }

    /// Returns true if elements are contiguous in memory (row-major strides).
    pub fn is_contiguous(&self) -> bool {
        self.0.layout.is_contiguous()
    }

    /// Returns true if contiguous with zero offset.
    pub fn is_compact(&self) -> bool {
        self.0.layout.is_compact()
    }

    /// Returns a contiguous copy of the tensor. No-op if already compact.
    pub fn compact(&self) -> Tensor {
        if self.is_compact() {
            return self.clone();
        }
        ops::Compact::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Element-wise power: `self^e`.
    pub fn powf<B: Borrow<Tensor>>(&self, e: B) -> Tensor {
        ops::EWisePowf::new(self.clone(), e.borrow().clone()).unwrap().forward().unwrap()
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Tensor {
        ops::EWiseLog::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Tensor {
        self.scalar_powf(0.5)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Tensor {
        ops::EWiseExp::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Tensor {
        ops::EWiseSin::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Tensor {
        ops::EWiseCos::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Raises every element to the scalar power `e`.
    pub fn scalar_powf(&self, e: f64) -> Tensor {
        ops::ScalarPowf::new(self.clone(), e).unwrap().forward().unwrap()
    }

    /// Matrix multiplication: `[..., m, k] @ [..., k, n] -> [..., m, n]`.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        ops::MatMul::new(self.clone(), other.clone()).unwrap().forward().unwrap()
    }

    /// Element-wise ReLU: `max(0, x)`.
    pub fn relu(&self) -> Tensor {
        ops::Relu::new(self.clone()).unwrap().forward().unwrap()
    }

    pub(crate) fn eq(&self, other: &Tensor) -> Tensor {
        let (a, b) = broadcast_pair(self, other);
        let storage = Arc::new(RwLock::new(
            a.storage()
                .binary_op::<crate::storage::EWiseEq>(a.layout(), &b.storage(), b.layout())
                .unwrap(),
        ));
        Tensor::new(storage, Layout::from(a.layout().shape().clone()), false, None)
    }

    /// Numerically stable `log(sum(exp(x)))` along the given axes.
    pub fn log_sum_exp(&self, axes: Vec<usize>) -> Tensor {
        ops::LogSumExp::new(self.clone(), axes).unwrap().forward().unwrap()
    }

    /// Numerically stable log-softmax along the given axis.
    ///
    /// On CUDA with a compact last-axis layout this uses a fused single-kernel path
    /// that avoids materialising the broadcast LSE intermediate. All other cases fall
    /// back to the primitive decomposition (log_sum_exp → reshape → broadcast → sub).
    pub fn log_softmax(&self, axis: usize) -> Tensor {
        // Fused CUDA path: single kernel reads x twice and writes output once,
        // skipping the separate log_sum_exp + broadcast + sub chain.
        let last_axis = self.layout().ndim() - 1;
        if axis == last_axis
            && self.device() == crate::device::Device::Cuda
            && self.is_compact()
        {
            return ops::FusedLogSoftmax::new(self.clone(), axis).unwrap().forward().unwrap();
        }
        // Primitive fallback.
        let lse = self.log_sum_exp(vec![axis]);
        let mut shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        shape[axis] = 1;
        let lse = lse.reshape(shape);
        let lse = lse.broadcast(self.layout().shape().clone());
        self - &lse
    }

    /// Concatenates tensors along the given dimension.
    pub fn cat(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        if tensors.len() == 1 {
            return tensors[0].clone();
        }
        // For non-zero dim: transpose to bring cat dim first, cat along 0, transpose back
        if dim != 0 {
            let transposed: Vec<_> = tensors.iter().map(|t| t.transpose(Some((0, dim)))).collect();
            let cat0 = Self::cat(&transposed, 0);
            return cat0.transpose(Some((0, dim)));
        }
        ops::Cat::new(tensors.to_vec()).unwrap().forward().unwrap()
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(&self) -> Tensor {
        let denom = (-self).exp() + 1.0;
        let one = Tensor::ones(vec![1], self.dtype(), self.device())
            .broadcast(self.layout().shape().clone());
        &one / &denom
    }

    /// Element-wise tanh.
    pub fn tanh(&self) -> Tensor {
        ops::Tanh::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Numerically stable softmax along the given axis.
    pub fn softmax(&self, axis: usize) -> Tensor {
        self.log_softmax(axis).exp()
    }

    /// Mean along the given axes. If `keep_dims`, reduced axes become size 1.
    pub fn mean(&self, axes: Vec<usize>, keep_dims: bool) -> Tensor {
        let n: usize = axes.iter().map(|&a| self.layout().shape()[a]).product();
        let s = self.sum(axes, keep_dims);
        &s * (1.0 / n as f64)
    }

    /// Gathers values along `dim` using integer indices.
    ///
    /// The index tensor must have the same rank as `self` and the same shape on
    /// every non-indexed dimension. The output shape matches `indices`, so each
    /// index element picks one value from `self` along `dim`.
    pub fn gather(&self, dim: usize, indices: &Tensor) -> Tensor {
        ops::Gather::new(self.clone(), dim, indices.clone()).unwrap().forward().unwrap()
    }

    /// Selects slices along `dim` using a 1-D integer index tensor.
    ///
    /// The output shape matches `self` except the size at `dim` becomes
    /// `indices.len()`.
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Tensor {
        ops::IndexSelect::new(self.clone(), dim, indices.clone()).unwrap().forward().unwrap()
    }
}

/// Computes the broadcast-compatible output shape for two shapes (numpy-style).
fn broadcast_shape(a: &Shape, b: &Shape) -> Shape {
    let ndim = a.ndim().max(b.ndim());
    let mut out = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let da = if i < ndim - a.ndim() { 1 } else { a[i - (ndim - a.ndim())] };
        let db = if i < ndim - b.ndim() { 1 } else { b[i - (ndim - b.ndim())] };
        assert!(da == db || da == 1 || db == 1, "incompatible shapes for broadcast");
        out.push(da.max(db));
    }
    out.into()
}

/// Auto-broadcasts two tensors to the same shape if needed.
fn broadcast_pair(a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    if a.layout().shape() == b.layout().shape() {
        return (a.clone(), b.clone());
    }
    let out_shape = broadcast_shape(a.layout().shape(), b.layout().shape());
    // Only create Broadcast nodes for tensors that don't already have the target shape.
    // This avoids trivial Broadcast nodes (shape already matches) whose backward would
    // call sum([]) — launching a CUDA reduce kernel for every no-op broadcast.
    let a_out = if a.layout().shape() == &out_shape { a.clone() } else { a.broadcast(out_shape.clone()) };
    let b_out = if b.layout().shape() == &out_shape { b.clone() } else { b.broadcast(out_shape) };
    (a_out, b_out)
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.layout().shape == other.layout().shape
            && match (self.dtype(), other.dtype()) {
                (DType::F16, DType::F16) => {
                    self.to_vec::<f16>().unwrap() == other.to_vec::<f16>().unwrap()
                }
                (DType::F32, DType::F32) => {
                    self.to_vec::<f32>().unwrap() == other.to_vec::<f32>().unwrap()
                }
                (DType::I64, DType::I64) => {
                    self.to_vec::<i64>().unwrap() == other.to_vec::<i64>().unwrap()
                }
                _ => false,
            }
    }
}

impl Eq for Tensor {}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        ops::Neg::new(self.clone()).unwrap().forward().unwrap()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        ops::Neg::new(self.clone()).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Add<B> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(&self, rhs.borrow());
        ops::EWiseAdd::new(a, b).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Add<B> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(self, rhs.borrow());
        ops::EWiseAdd::new(a, b).unwrap().forward().unwrap()
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        ops::ScalarAdd::new(self.clone(), rhs).unwrap().forward().unwrap()
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        ops::ScalarAdd::new(self.clone(), rhs).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Sub<B> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(&self, rhs.borrow());
        ops::EWiseSub::new(a, b).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Sub<B> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(self, rhs.borrow());
        ops::EWiseSub::new(a, b).unwrap().forward().unwrap()
    }
}

impl Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        self + -rhs
    }
}

impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        self + -rhs
    }
}

impl<B: Borrow<Tensor>> Mul<B> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(&self, rhs.borrow());
        ops::EWiseMul::new(a, b).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Mul<B> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(self, rhs.borrow());
        ops::EWiseMul::new(a, b).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Div<B> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(&self, rhs.borrow());
        ops::EWiseDiv::new(a, b).unwrap().forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Div<B> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: B) -> Self::Output {
        let (a, b) = broadcast_pair(self, rhs.borrow());
        ops::EWiseDiv::new(a, b).unwrap().forward().unwrap()
    }
}

impl Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Div<f64> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        ops::ScalarMul::new(self.clone(), rhs).unwrap().forward().unwrap()
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        ops::ScalarMul::new(self.clone(), rhs).unwrap().forward().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep this module focused on DEERS-specific API behavior that Candle cannot validate
    // meaningfully, such as metadata, graph attachment, and local invariants. Public tensor
    // math/data semantics live in tests/tensor_conformance.rs so CPU and MPS can both be checked
    // against the same Candle reference.

    #[test]
    fn test_op() {
        // Arrange
        let a = Tensor::ones((2,), DType::F32, Device::Cpu);
        let b = Tensor::ones((2,), DType::F32, Device::Cpu);

        // Act
        let c = &a + &b;

        // Assert — tensors that do not require grad should not retain op history.
        assert!(a.op().is_none());
        assert!(c.op().is_none());
    }

    #[test]
    fn test_attached_tensor_keeps_op_history() {
        // Arrange
        let a = Tensor::ones((2,), DType::F32, Device::Cpu).attach();
        let b = Tensor::ones((2,), DType::F32, Device::Cpu);

        // Act
        let c = &a + &b;

        // Assert
        assert!(c.requires_grad());
        assert!(c.op().is_some());
    }

    #[test]
    fn test_id() {
        // Arrange
        let a = Tensor::ones((1,), DType::F32, Device::Cpu);
        let b = Tensor::ones((1,), DType::F32, Device::Cpu);

        // Act
        let (a_id, b_id) = (a.id(), b.id());

        // Assert
        assert_ne!(a_id, b_id);
    }

    #[test]
    fn test_storage() {
        // Arrange
        let tensor = Tensor::ones((2,), DType::F32, Device::Cpu);

        // Act
        let dtype = tensor.storage().dtype();

        // Assert
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_layout() {
        // Arrange
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        // Act
        let layout = tensor.layout();

        // Assert
        assert_eq!(layout.shape, (2, 3).into());
        assert_eq!(layout.strides, (3, 1).into());
    }

    #[test]
    fn test_dtype() {
        // Arrange
        let tensor = Tensor::ones((2,), DType::I64, Device::Cpu);

        // Act
        let dtype = tensor.dtype();

        // Assert
        assert_eq!(dtype, DType::I64);
    }

    #[test]
    fn test_device() {
        // Arrange
        let tensor = Tensor::ones((2,), DType::F32, Device::Cpu);

        // Act
        let device = tensor.device();

        // Assert
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn test_requires_grad() {
        // Arrange
        let tensor = Tensor::ones((2,), DType::F32, Device::Cpu);

        // Act
        let detached = tensor.requires_grad();
        let attached = tensor.attach().requires_grad();

        // Assert
        assert!(!detached);
        assert!(attached);
    }

    #[test]
    fn test_attach() {
        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu);

        // Act
        let attached = tensor.attach();

        // Assert
        assert!(attached.requires_grad());
        assert_eq!(attached.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_attach_backward() {
        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], (2,), Device::Cpu).attach();

        // Act
        let grads = tensor.sum(vec![0], false).backward().unwrap();

        // Assert
        assert_eq!(grads.get(tensor.id()).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_to_vec() {
        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

        // Act
        let values = tensor.to_vec::<f32>().unwrap();

        // Assert
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_device() {
        if !Device::Mps.is_available() {
            return;
        }

        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

        // Act
        let moved = tensor.to_device(Device::Mps).unwrap();

        // Assert
        assert_eq!(moved.device(), Device::Mps);
        assert_eq!(moved.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_device_backward() {
        if !Device::Mps.is_available() {
            return;
        }

        // Arrange
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Mps).attach();

        // Act
        let moved = tensor.to_device(Device::Cpu).unwrap().attach();
        let grads = moved.sum(vec![0], false).backward().unwrap();

        // Assert
        assert_eq!(grads.get(tensor.id()), None);
        assert_eq!(grads.get(moved.id()).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_ones_like() {
        // Arrange
        let tensor = Tensor::zeros((2, 2), DType::F32, Device::Cpu);

        // Act
        let ones = tensor.ones_like();

        // Assert
        assert_eq!(ones.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn test_ones_like_backward() {
        // Arrange
        let tensor = Tensor::zeros((2, 2), DType::F32, Device::Cpu).attach();

        // Act
        let grads = tensor.ones_like().attach().sum(vec![0, 1], false).backward().unwrap();

        // Assert
        assert_eq!(grads.get(tensor.id()), None);
    }

    #[test]
    fn test_zeros_like() {
        // Arrange
        let tensor = Tensor::ones((2, 2), DType::F32, Device::Cpu);

        // Act
        let zeros = tensor.zeros_like();

        // Assert
        assert_eq!(zeros.to_vec::<f32>().unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn test_zeros_like_backward() {
        // Arrange
        let tensor = Tensor::ones((2, 2), DType::F32, Device::Cpu).attach();

        // Act
        let grads = tensor.zeros_like().attach().sum(vec![0, 1], false).backward().unwrap();

        // Assert
        assert_eq!(grads.get(tensor.id()), None);
    }

    #[test]
    fn test_is_compact() {
        // Arrange
        let compact = Tensor::ones((2, 2), DType::F32, Device::Cpu);
        let non_compact = compact.permute(vec![1, 0]);

        // Act
        let compact_flag = compact.is_compact();
        let non_compact_flag = non_compact.is_compact();

        // Assert
        assert!(compact_flag);
        assert!(!non_compact_flag);
    }
}
