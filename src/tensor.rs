#![allow(dead_code)]

use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::backprop::BackpropOp;
use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::storage::{Neg, Storage};

#[derive(Debug)]
pub struct TensorInternal {
    storage: Storage,
    layout: Layout,
    device: Device,
    dtype: DType,
    op: Option<BackpropOp>,
    grad: Arc<Mutex<Option<Tensor>>>,
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

    pub fn grad(&self) -> Arc<Mutex<Option<Tensor>>> {
        self.grad.clone()
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = device.zeros(shape.size(), dtype);
        let layout: Layout = shape.into();
        let op = None;
        let grad = Arc::new(Mutex::new(None));
        TensorInternal {
            storage,
            layout,
            dtype,
            device,
            op,
            grad,
        }
        .into()
    }

    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = device.ones(shape.size(), dtype);
        let layout: Layout = shape.into();
        let op = None;
        let grad = Arc::new(Mutex::new(None));
        TensorInternal {
            storage,
            layout,
            dtype,
            device,
            op,
            grad,
        }
        .into()
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor(Arc::new(TensorInternal {
            storage: self.device.ones(self.layout.size(), self.dtype),
            layout: self.layout.clone(),
            dtype: self.dtype,
            device: self.device,
            op: None,
            grad: Arc::new(Mutex::new(None)),
        }))
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor(Arc::new(TensorInternal {
            storage: self.device.zeros(self.layout.size(), self.dtype),
            layout: self.layout.clone(),
            dtype: self.dtype,
            device: self.device,
            op: None,
            grad: Arc::new(Mutex::new(None)),
        }))
    }

    pub fn neg(&self) -> Result<Tensor> {
        let storage = self.storage.unary_op::<Neg>()?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Neg(self.clone()));
        let grad = Arc::new(Mutex::new(None));
        Ok(TensorInternal {
            storage,
            layout,
            dtype: self.dtype,
            device: self.device,
            op,
            grad,
        }
        .into())
    }

    pub fn add(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::Add>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Add(self.clone(), rhs.clone()));
        let grad = Arc::new(Mutex::new(None));
        Ok(TensorInternal {
            storage,
            layout,
            dtype: self.dtype,
            device: self.device,
            op,
            grad,
        }
        .into())
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
}
