#![allow(dead_code)]

use std::borrow::Borrow;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard};

use crate::device::Device;
use crate::dtype::{DType, WithDType};
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::ops::{self, TensorOp};
use crate::storage::{BackendStorage, CpuStorage, Storage};

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
}

impl TensorInternal {
    pub fn new(
        storage: Arc<RwLock<Storage>>,
        layout: Layout,
        device: Device,
        dtype: DType,
        op: Option<Box<dyn TensorOp>>,
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
    pub fn op(&self) -> &Option<Box<dyn TensorOp>> {
        &self.op
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = Arc::new(RwLock::new(device.zeros(shape.size(), dtype)));
        let layout: Layout = shape.into();
        TensorInternal::new(storage, layout, device, dtype, None).into()
    }

    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: Device) -> Tensor {
        let shape: Shape = shape.into();
        let storage = Arc::new(RwLock::new(device.ones(shape.size(), dtype)));
        let layout: Layout = shape.into();
        TensorInternal::new(storage, layout, device, dtype, None).into()
    }

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
        TensorInternal::new(storage, layout, device, dtype, None).into()
    }

    pub fn to_vec<S: WithDType>(&self) -> Result<Vec<S>> {
        Ok(self.storage().to_vec(&self.layout))
    }

    pub fn ones_like(&self) -> Tensor {
        TensorInternal::new(
            Arc::new(RwLock::new(
                self.device.ones(self.layout.size(), self.dtype),
            )),
            self.layout.clone(),
            self.device,
            self.dtype,
            None,
        )
        .into()
    }

    pub fn zeros_like(&self) -> Tensor {
        TensorInternal::new(
            Arc::new(RwLock::new(
                self.device.zeros(self.layout.size(), self.dtype),
            )),
            self.layout.clone(),
            self.device,
            self.dtype,
            None,
        )
        .into()
    }

    pub fn permute(&self, axis: impl AsRef<[usize]>) -> Result<Tensor> {
        let axis = axis.as_ref();
        let storage = self.storage.clone();
        let layout = self.layout.permute(axis);
        Ok(TensorInternal::new(storage, layout, self.device, self.dtype, None).into())
    }

    pub fn is_compact(&self) -> bool {
        self.layout.is_compact()
    }

    pub fn compact(&self) -> Result<Tensor> {
        if self.is_compact() {
            return Ok(self.clone());
        }

        let mut storage = self.device().zeros(self.layout.size(), self.dtype);
        self.storage().copy_compact(&self.layout, &mut storage)?;
        Ok(TensorInternal::new(
            Arc::new(RwLock::new(storage)),
            self.layout.clone(),
            self.device,
            self.dtype,
            None,
        )
        .into())
    }

    pub fn powf<B: Borrow<Tensor>>(&self, e: B) -> Tensor {
        ops::EWisePowf(self.clone(), e.borrow().clone())
            .forward()
            .unwrap()
    }

    pub fn log(&self) -> Tensor {
        ops::EWiseLog(self.clone()).forward().unwrap()
    }

    pub fn exp(&self) -> Tensor {
        ops::EWiseExp(self.clone()).forward().unwrap()
    }

    pub fn scalar_powf(&self, e: f64) -> Tensor {
        ops::ScalarPowf(self.clone(), e).forward().unwrap()
    }
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
        ops::Neg(self.clone()).forward().unwrap()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        ops::Neg(self.clone()).forward().unwrap()
    }
}

impl<B: Borrow<Tensor>> Add<B> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: B) -> Self::Output {
        ops::EWiseAdd(self.clone(), rhs.borrow().clone())
            .forward()
            .unwrap()
    }
}

impl<B: Borrow<Tensor>> Add<B> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: B) -> Self::Output {
        ops::EWiseAdd(self.clone(), rhs.borrow().clone())
            .forward()
            .unwrap()
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        ops::ScalarAdd(self.clone(), rhs).forward().unwrap()
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        ops::ScalarAdd(self.clone(), rhs).forward().unwrap()
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
        ops::EWiseMul(self.clone(), rhs.borrow().clone())
            .forward()
            .unwrap()
    }
}

impl<B: Borrow<Tensor>> Mul<B> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: B) -> Self::Output {
        ops::EWiseMul(self.clone(), rhs.borrow().clone())
            .forward()
            .unwrap()
    }
}

impl<B: Borrow<Tensor>> Div<B> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: B) -> Self::Output {
        ops::EWiseDiv(self.clone(), rhs.borrow().clone())
            .forward()
            .unwrap()
    }
}

impl<B: Borrow<Tensor>> Div<B> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: B) -> Self::Output {
        ops::EWiseDiv(self.clone(), rhs.borrow().clone())
            .forward()
            .unwrap()
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        ops::ScalarMul(self.clone(), rhs).forward().unwrap()
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        ops::ScalarMul(self.clone(), rhs).forward().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::error::Result;

    use super::*;

    #[test]
    fn test_zeros() -> Result<()> {
        let tensor = Tensor::zeros((2, 3), DType::F32, Device::Cpu);

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, (3, 1).into());
        assert_eq!(tensor.to_vec::<f32>()?, vec![0.0f32; 6]);
        Ok(())
    }

    #[test]
    fn test_ones() -> Result<()> {
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, (3, 1).into());
        assert_eq!(tensor.to_vec::<f32>()?, vec![1.0f32; 6]);
        Ok(())
    }

    #[test]
    fn test_ones_like() -> Result<()> {
        let tensor = Tensor::zeros((2, 3), DType::F32, Device::Cpu);
        let tensor = tensor.ones_like();

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, (3, 1).into());
        assert_eq!(tensor.to_vec::<f32>()?, vec![1.0f32; 6]);
        Ok(())
    }

    #[test]
    fn test_zeros_like() -> Result<()> {
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let tensor = tensor.zeros_like();

        assert_eq!(tensor.layout.shape, (2, 3).into());
        assert_eq!(tensor.layout.strides, (3, 1).into());
        assert_eq!(tensor.to_vec::<f32>()?, vec![0.0f32; 6]);
        Ok(())
    }

    #[test]
    fn test_neg() -> Result<()> {
        let tensor = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let tensor = tensor.neg();

        assert_eq!(tensor.to_vec::<f32>()?, vec![-1.0f32; 6]);
        Ok(())
    }

    #[test]
    fn test_ewise_add() -> Result<()> {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let c = a + b;

        assert_eq!(c.to_vec::<f32>()?, vec![2.0f32; 6]);
        Ok(())
    }

    #[test]
    fn test_ewise_sub() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let c = a - b;

        assert_eq!(c.to_vec::<f32>().unwrap(), vec![0.0; 6]);
    }

    #[test]
    fn test_ewise_mul() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

        let c = a * b;

        assert_eq!(
            c.to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
        );
    }

    #[test]
    fn test_ewise_powf() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

        let c = a.powf(b);

        assert_eq!(
            c.to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 27.0, 256.0, 3125.0, 46656.0]
        );
    }

    #[test]
    fn test_ewise_log() {
        let a = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (3,), Device::Cpu);

        let b = a.log();

        assert_eq!(b.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scalar_add() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let b = a + 2.0;

        assert_eq!(b.to_vec::<f32>().unwrap(), vec![3.0; 6]);
    }

    #[test]
    fn test_scalar_sub() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let b = a - 2.0;

        assert_eq!(b.to_vec::<f32>().unwrap(), vec![-1.0; 6]);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu);

        let b = a * 2.0;

        assert_eq!(b.to_vec::<f32>().unwrap(), vec![2.0; 6]);
    }

    #[test]
    fn test_scalar_powf() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), Device::Cpu);

        let b = a.scalar_powf(2.0);

        assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_permute() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), Device::Cpu);

        let b = a.permute(vec![1, 0]).unwrap();

        assert_eq!(
            b.to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        )
    }
}
