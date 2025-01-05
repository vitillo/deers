#![allow(dead_code)]

use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::backprop::BackpropOp;
use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::storage::{CpuStorage, EWiseAdd, Neg, ScalarAdd, ScalarMul, Storage};

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
        let storage = self.storage.unary_op(Neg)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Neg(self.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn ewise_add(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::EWiseAdd>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::EWiseAdd(self.clone(), rhs.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn ewise_sub(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::EWiseSub>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::EWiseSub(self.clone(), rhs.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn ewise_mul(&self, rhs: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::EWiseMul>(&rhs.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::EWiseMul(self.clone(), rhs.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn ewise_powf(&self, e: &Self) -> Result<Tensor> {
        let storage = self
            .storage
            .binary_op::<crate::storage::EWisePow>(&e.storage)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::EWisePow(self.clone(), e.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn ewise_log(&self) -> Result<Tensor> {
        let storage = self.storage.ewise_log()?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::EWiseLog(self.clone()));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn scalar_add(&self, scalar: f64) -> Result<Tensor> {
        let storage = self.storage.unary_op(ScalarAdd(scalar))?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::ScalarAdd(self.clone(), scalar));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn scalar_mul(&self, scalar: f64) -> Result<Tensor> {
        let storage = self.storage.unary_op(ScalarMul(scalar))?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::ScalarMul(self.clone(), scalar));
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, op).into())
    }

    pub fn scalar_powf(&self, e: f64) -> Result<Tensor> {
        let storage = self.storage.powf(e)?;
        let layout = self.layout.clone();
        let op = Some(BackpropOp::Powf(self.clone(), e));
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
    fn test_ewise_add() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let c = a.ewise_add(&b).unwrap();

        assert_eq!(c.layout.shape, (2, 3).into());
        assert_eq!(c.layout.strides, vec![3, 1].into());
        assert_eq!(c.storage, Storage::Cpu(CpuStorage::F32(vec![2.0; 6])));
    }

    #[test]
    fn test_ewise_sub() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let c = a.ewise_sub(&b).unwrap();

        assert_eq!(c.layout.shape, (2, 3).into());
        assert_eq!(c.layout.strides, vec![3, 1].into());
        assert_eq!(c.storage, Storage::Cpu(CpuStorage::F32(vec![0.0; 6])));
    }

    #[test]
    fn test_ewise_mul() {
        let a = Tensor::load_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
        let b = Tensor::load_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

        let c = a.ewise_mul(&b).unwrap();

        assert_eq!(
            c.storage,
            Storage::Cpu(CpuStorage::F32(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]))
        );
    }

    #[test]
    fn test_ewise_powf() {
        let a = Tensor::load_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
        let b = Tensor::load_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

        let c = a.ewise_powf(&b).unwrap();

        assert_eq!(
            c.storage,
            Storage::Cpu(CpuStorage::F32(vec![
                1.0, 4.0, 27.0, 256.0, 3125.0, 46656.0
            ]))
        );
    }

    #[test]
    fn test_ewise_log() {
        let a = Tensor::load_vec(vec![1.0f32, 1.0, 1.0], (3,), Device::Cpu);

        let b = a.ewise_log().unwrap();

        assert_eq!(
            b.storage,
            Storage::Cpu(CpuStorage::F32(vec![0.0f32, 0.0, 0.0]))
        );
    }

    #[test]
    fn test_scalar_add() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let b = a.scalar_add(2.0).unwrap();

        assert_eq!(b.storage, Storage::Cpu(CpuStorage::F32(vec![3.0; 6])));
    }

    #[test]
    fn test_scalar_sub() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let b = a.scalar_add(-2.0).unwrap();

        assert_eq!(b.storage, Storage::Cpu(CpuStorage::F32(vec![-1.0; 6])));
    }

    #[test]
    fn test_scalar_mul() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let b = a.scalar_mul(2.0).unwrap();

        assert_eq!(b.storage, Storage::Cpu(CpuStorage::F32(vec![2.0; 6])));
    }

    #[test]
    fn test_scalar_powf() {
        let a = Tensor::load_vec(vec![1.0f32, 2.0, 3.0], (2, 3), Device::Cpu);

        let b = a.scalar_powf(2.0).unwrap();

        assert_eq!(
            b.storage,
            Storage::Cpu(CpuStorage::F32(vec![1.0f32, 4.0, 9.0]))
        );
    }
}
