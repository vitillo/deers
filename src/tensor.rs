#![allow(dead_code)]

use std::borrow::Borrow;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rand::Rng;

use crate::device::Device;
use crate::dtype::{DType, WithDType};
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::ops::{self, TensorOp};
use crate::storage::{BackendStorage, CpuStorage, Storage};

/// Unique identifier for a tensor, used to look up gradients after [`Tensor::backward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(usize);

#[derive(Debug)]
pub struct TensorInternal {
    id: TensorId,
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    device: Device,
    dtype: DType,
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

impl Deref for Tensor {
    type Target = TensorInternal;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub(crate) fn new(
        storage: Arc<RwLock<Storage>>,
        layout: Layout,
        device: Device,
        dtype: DType,
        requires_grad: bool,
        op: Option<Box<dyn TensorOp>>,
    ) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = TensorId(COUNTER.fetch_add(1, Ordering::Relaxed));

        let requires_grad = requires_grad
            || op
                .as_ref()
                .map(|o| o.dependencies().iter().any(|dep| dep.requires_grad()))
                .unwrap_or(false);

        TensorInternal {
            id,
            storage,
            layout,
            device,
            dtype,
            op,
            requires_grad,
        }
        .into()
    }

    pub(crate) fn with_op(self, op: Box<dyn TensorOp>) -> Self {
        let tensor = Arc::into_inner(self.0).unwrap();
        Tensor::new(
            tensor.storage,
            tensor.layout,
            tensor.device,
            tensor.dtype,
            false,
            Some(op),
        )
    }

    /// Returns the operation that produced this tensor, if any.
    pub fn op(&self) -> &Option<Box<dyn TensorOp>> {
        &self.op
    }

    /// Returns this tensor's unique identifier.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Returns a read guard to the underlying storage.
    pub fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }

    pub(crate) fn storage_clone(&self) -> Arc<RwLock<Storage>> {
        self.storage.clone()
    }

    /// Returns a write guard to the underlying storage.
    pub(crate) fn storage_mut(&self) -> RwLockWriteGuard<'_, Storage> {
        self.storage.write().unwrap()
    }

    /// Returns the layout (shape, strides, offset) of this tensor.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the data type (F32, F64) of this tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the device (CPU, CUDA) where this tensor is stored.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns whether this tensor tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Enables gradient tracking for this tensor. This is the equivalent of
    /// PyTorch's `requires_grad_(True)`. Tensors produced by operations on an
    /// attached tensor will also track gradients.
    pub fn attach(self) -> Tensor {
        if self.requires_grad {
            return self.clone();
        }

        Tensor::new(
            self.storage.clone(),
            self.layout.clone(),
            self.device,
            self.dtype,
            true,
            None,
        )
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = Arc::new(RwLock::new(device.zeros(shape.size(), dtype)));
        let layout: Layout = shape.into();
        Tensor::new(storage, layout, device, dtype, false, None)
    }

    /// Creates a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = Arc::new(RwLock::new(device.ones(shape.size(), dtype)));
        let layout: Layout = shape.into();
        Tensor::new(storage, layout, device, dtype, false, None)
    }

    /// Creates a tensor from a `Vec<f32>` or `Vec<f64>` with the given shape.
    pub fn from_vec(vec: impl Into<CpuStorage>, shape: impl Into<Shape>, device: Device) -> Tensor {
        assert_eq!(
            device,
            Device::Cpu,
            "TODO: implement load for other devices"
        );
        let shape: Shape = shape.into();
        let storage = Storage::Cpu(vec.into());
        let layout: Layout = shape.into();
        let dtype = storage.dtype();
        let storage = Arc::new(RwLock::new(storage));
        Tensor::new(storage, layout, device, dtype, false, None)
    }

    /// Create a tensor with values drawn from a uniform distribution in [0, 1).
    pub fn rand(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let mut rng = rand::thread_rng();
        let storage = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..shape.size()).map(|_| rng.gen()).collect();
                Storage::Cpu(CpuStorage::from(data))
            }
            DType::F64 => {
                let data: Vec<f64> = (0..shape.size()).map(|_| rng.gen()).collect();
                Storage::Cpu(CpuStorage::from(data))
            }
            _ => unimplemented!(),
        };
        let layout: Layout = shape.into();
        Tensor::new(
            Arc::new(RwLock::new(storage)),
            layout,
            device,
            dtype,
            false,
            None,
        )
    }

    /// Create a tensor with values drawn from a standard normal distribution.
    pub fn randn(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let mut rng = rand::thread_rng();
        // Box-Muller transform
        let storage = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..shape.size())
                    .map(|_| {
                        let u1: f32 = rng.gen();
                        let u2: f32 = rng.gen();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                    })
                    .collect();
                Storage::Cpu(CpuStorage::from(data))
            }
            DType::F64 => {
                let data: Vec<f64> = (0..shape.size())
                    .map(|_| {
                        let u1: f64 = rng.gen();
                        let u2: f64 = rng.gen();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    })
                    .collect();
                Storage::Cpu(CpuStorage::from(data))
            }
            _ => unimplemented!(),
        };
        let layout: Layout = shape.into();
        Tensor::new(
            Arc::new(RwLock::new(storage)),
            layout,
            device,
            dtype,
            false,
            None,
        )
    }

    /// Copies the tensor data into a flat `Vec`, respecting strides.
    pub fn to_vec<S: WithDType>(&self) -> Result<Vec<S>> {
        Ok(self.storage().to_vec(&self.layout))
    }

    /// Creates a tensor of ones with the same shape, dtype, and device.
    pub fn ones_like(&self) -> Tensor {
        Tensor::new(
            Arc::new(RwLock::new(
                self.device.ones(self.layout.size(), self.dtype),
            )),
            self.layout.clone(),
            self.device,
            self.dtype,
            false,
            None,
        )
    }

    /// Creates a tensor of zeros with the same shape, dtype, and device.
    pub fn zeros_like(&self) -> Tensor {
        Tensor::new(
            Arc::new(RwLock::new(
                self.device.zeros(self.layout.size(), self.dtype),
            )),
            self.layout.clone(),
            self.device,
            self.dtype,
            false,
            None,
        )
    }

    /// Reorders the dimensions of the tensor. Does not copy data.
    pub fn permute(&self, axis: impl Into<Shape>) -> Tensor {
        ops::Permute::new(self.clone(), axis.into())
            .forward()
            .unwrap()
    }

    /// Expands dimensions of size 1 to match `new_shape`. Does not copy data.
    pub fn broadcast(&self, new_shape: impl Into<Shape>) -> Tensor {
        ops::Broadcast::new(self.clone(), new_shape.into())
            .forward()
            .unwrap()
    }

    /// Returns a view with a different shape but the same total number of elements.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Tensor {
        ops::Reshape::new(self.clone(), new_shape.into())
            .forward()
            .unwrap()
    }

    /// Swaps two dimensions. Defaults to the last two if `axes` is `None`.
    pub fn transpose(&self, axes: Option<(usize, usize)>) -> Tensor {
        let axes = axes.unwrap_or((self.layout().ndim() - 2, self.layout().ndim() - 1));
        assert!(
            axes.0 < self.layout().ndim() && axes.1 < self.layout().ndim(),
            "Transpose axes must be less than tensor dimensions",
        );
        let mut reshaped_axes: Vec<_> = (0..self.layout.ndim()).collect();
        (reshaped_axes[axes.0], reshaped_axes[axes.1]) =
            (reshaped_axes[axes.1], reshaped_axes[axes.0]);
        self.permute(reshaped_axes)
    }

    /// Sums elements along the given axes. If `keep_dims`, reduced axes become size 1.
    pub fn sum(&self, axis: Vec<usize>, keep_dims: bool) -> Tensor {
        ops::Sum::new(self.clone(), axis, keep_dims)
            .forward()
            .unwrap()
    }

    /// Returns the maximum along the given axes.
    pub fn max(&self, axis: Vec<usize>, keep_dims: bool) -> Tensor {
        ops::Max::new(self.clone(), axis, keep_dims)
            .forward()
            .unwrap()
    }

    /// Returns true if the tensor's memory layout is contiguous (row-major).
    pub fn is_compact(&self) -> bool {
        self.layout.is_compact()
    }

    /// Returns a contiguous copy of the tensor. No-op if already compact.
    pub fn compact(&self) -> Tensor {
        if self.is_compact() {
            return self.clone();
        }
        ops::Compact::new(self.clone()).forward().unwrap()
    }

    /// Element-wise power: `self^e`.
    pub fn powf<B: Borrow<Tensor>>(&self, e: B) -> Tensor {
        ops::EWisePowf::new(self.clone(), e.borrow().clone())
            .unwrap()
            .forward()
            .unwrap()
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Tensor {
        ops::EWiseLog::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Tensor {
        ops::EWiseExp::new(self.clone()).unwrap().forward().unwrap()
    }

    /// Raises every element to the scalar power `e`.
    pub fn scalar_powf(&self, e: f64) -> Tensor {
        ops::ScalarPowf::new(self.clone(), e)
            .unwrap()
            .forward()
            .unwrap()
    }

    /// 2-D matrix multiplication: `(m, n) @ (n, p) -> (m, p)`.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        ops::MatMul::new(self.clone(), other.clone())
            .unwrap()
            .forward()
            .unwrap()
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
        Tensor::new(
            storage,
            Layout::from(a.layout().shape().clone()),
            a.device(),
            a.dtype(),
            false,
            None,
        )
    }

    /// Numerically stable `log(sum(exp(x)))` along the given axes.
    pub fn log_sum_exp(&self, axes: Vec<usize>) -> Tensor {
        ops::LogSumExp::new(self.clone(), axes).forward().unwrap()
    }

    /// Numerically stable log-softmax along the given axis.
    ///
    /// Computes `x - logsumexp(x, axis)` with the logsumexp result
    /// broadcast back to the original shape.
    pub fn log_softmax(&self, axis: usize) -> Tensor {
        let lse = self.log_sum_exp(vec![axis]);
        // Reshape lse to have size 1 in the reduced axis so we can broadcast
        let mut shape: Vec<usize> = self.layout().shape().iter().copied().collect();
        shape[axis] = 1;
        let lse = lse.reshape(shape);
        let lse = lse.broadcast(self.layout().shape().clone());
        self - &lse
    }

    /// Gathers values along `dim` using integer indices.
    ///
    /// For a 2D tensor of shape `(rows, cols)` with `dim=1` and indices of shape `(rows,)`,
    /// returns shape `(rows, 1)` where `out[i, 0] = self[i, indices[i]]`.
    pub fn gather(&self, dim: usize, indices: &Tensor) -> Tensor {
        let idx: Vec<usize> = indices
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| {
                assert!(v.is_finite(), "gather indices must be finite");
                assert!(v >= 0.0, "gather indices must be non-negative");
                assert!(v.fract() == 0.0, "gather indices must be integers");
                v as usize
            })
            .collect();
        ops::Gather::new(self.clone(), dim, idx).forward().unwrap()
    }
}

/// Computes the broadcast-compatible output shape for two shapes (numpy-style).
fn broadcast_shape(a: &Shape, b: &Shape) -> Shape {
    let ndim = a.ndim().max(b.ndim());
    let mut out = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let da = if i < ndim - a.ndim() {
            1
        } else {
            a[i - (ndim - a.ndim())]
        };
        let db = if i < ndim - b.ndim() {
            1
        } else {
            b[i - (ndim - b.ndim())]
        };
        assert!(
            da == db || da == 1 || db == 1,
            "incompatible shapes for broadcast"
        );
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
    (a.broadcast(out_shape.clone()), b.broadcast(out_shape))
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.layout().shape == other.layout().shape
            && match (self.dtype(), other.dtype()) {
                (DType::F32, DType::F32) => {
                    self.to_vec::<f32>().unwrap() == other.to_vec::<f32>().unwrap()
                }
                (DType::F64, DType::F64) => {
                    self.to_vec::<f64>().unwrap() == other.to_vec::<f64>().unwrap()
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
        ops::ScalarAdd::new(self.clone(), rhs)
            .unwrap()
            .forward()
            .unwrap()
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        ops::ScalarAdd::new(self.clone(), rhs)
            .unwrap()
            .forward()
            .unwrap()
    }
}

impl<B: Borrow<Tensor>> Sub<B> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: B) -> Self::Output {
        self + (-rhs.borrow())
    }
}

impl<B: Borrow<Tensor>> Sub<B> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: B) -> Self::Output {
        self + -rhs.borrow()
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

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        ops::ScalarMul::new(self.clone(), rhs)
            .unwrap()
            .forward()
            .unwrap()
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        ops::ScalarMul::new(self.clone(), rhs)
            .unwrap()
            .forward()
            .unwrap()
    }
}
