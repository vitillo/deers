#![allow(dead_code)]

use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::backprop::BackpropOp;
use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::storage::{CpuStorage, Neg, Storage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(usize);

#[derive(Debug)]
pub struct TensorInternal {
    id: TensorId,
    storage: Storage,
    layout: Layout,
    device: Device,
    dtype: DType,
    op: Option<BackpropOp>,
}

impl TensorInternal {
    pub fn new(
        storage: Storage,
        layout: Layout,
        device: Device,
        dtype: DType,
        op: Option<BackpropOp>,
    ) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = TensorId(COUNTER.fetch_add(1, Ordering::Relaxed));

        Self {
            id,
            storage,
            layout,
            device,
            dtype,
            op,
        }
    }
}

impl PartialEq for TensorInternal {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
            && self.layout == other.layout
            && self.device == other.device
            && self.dtype == other.dtype
    }
}

impl Eq for TensorInternal {}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub fn op(&self) -> &Option<BackpropOp> {
        &self.op
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = device.zeros(shape.size(), dtype);
        let layout: Layout = shape.into();
        let op = None;
        TensorInternal::new(storage, layout, device, dtype, op).into()
    }

    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = device.ones(shape.size(), dtype);
        let layout: Layout = shape.into();
        let op = None;
        TensorInternal::new(storage, layout, device, dtype, op).into()
    }

    pub fn load_vec(vec: impl Into<CpuStorage>, shape: impl Into<Shape>, device: Device) -> Tensor {
        assert_eq!(
            device,
            Device::Cpu,
            "TODO: implement load for other devices"
        );
        let shape: Shape = shape.into();
        let storage = Storage::Cpu(vec.into());
        let layout: Layout = shape.into();
        let dtype = storage.dtype();
        let op = None;
        TensorInternal::new(storage, layout, device, dtype, op).into()
    }

    pub fn ones_like(&self) -> Tensor {
        TensorInternal::new(
            self.device.ones(self.layout.size(), self.dtype),
            self.layout.clone(),
            self.device,
            self.dtype,
            None,
        )
        .into()
    }

    pub fn zeros_like(&self) -> Tensor {
        TensorInternal::new(
            self.device.zeros(self.layout.size(), self.dtype),
            self.layout.clone(),
            self.device,
            self.dtype,
            None,
        )
        .into()
    }

    pub fn neg(&self) -> Result<Tensor> {
        let storage = self.storage.unary_op::<Neg>()?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Neg(self.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn add(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::Add>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Add(self.clone(), rhs.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn sub(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::Sub>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Sub(self.clone(), rhs.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn mul(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::Mul>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Mul(self.clone(), rhs.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::CpuStorage;

    use super::*;

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros((2, 3), DType::F32, Device::Cpu);

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, vec![3, 1].into());
        assert_eq!(tensor.storage, Storage::Cpu(CpuStorage::F32(vec![0.0; 6])));
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, vec![3, 1].into());
        assert_eq!(tensor.storage, Storage::Cpu(CpuStorage::F32(vec![1.0; 6])));
    }

    #[test]
    fn test_ones_like() {
        let tensor = Tensor::zeros((2, 3), DType::F32, Device::Cpu);
        let tensor = tensor.ones_like();

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, vec![3, 1].into());
        assert_eq!(tensor.storage, Storage::Cpu(CpuStorage::F32(vec![1.0; 6])));
    }

    #[test]
    fn test_zeros_like() {
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let tensor = tensor.zeros_like();

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, vec![3, 1].into());
        assert_eq!(tensor.storage, Storage::Cpu(CpuStorage::F32(vec![0.0; 6])));
    }

    #[test]
    fn test_neg() {
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let tensor = tensor.neg().unwrap();

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, vec![3, 1].into());
        assert_eq!(tensor.storage, Storage::Cpu(CpuStorage::F32(vec![-1.0; 6])));
    }

    #[test]
    fn test_add() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let c = a.add(&b).unwrap();

        assert_eq!(c.layout.shape, (2, 3).into());
        assert_eq!(c.layout.strides, vec![3, 1].into());
        assert_eq!(c.storage, Storage::Cpu(CpuStorage::F32(vec![2.0; 6])));
    }

    #[test]
    fn test_mul() {
        let a = Tensor::load_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
        let b = Tensor::load_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

        let c = a.mul(&b).unwrap();

        assert_eq!(
            c.storage,
            Storage::Cpu(CpuStorage::F32(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]))
        );
    }
}
