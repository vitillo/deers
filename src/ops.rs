#![allow(dead_code)]

use std::fmt;

use crate::backprop::GradientStore;
use crate::error::Result;
use crate::storage;
use crate::tensor::{Tensor, TensorInternal};

pub trait TensorOp: fmt::Debug + Send + Sync {
    fn forward(self) -> Result<Tensor>;

    /// Compute partial adjoint for each input value for a given output adjoint
    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()>;

    fn dependencies(&self) -> Vec<&Tensor>;
}

#[derive(Debug)]
pub struct Neg(pub Tensor);

impl TensorOp for Neg {
    fn forward(self) -> Result<Tensor> {
        let storage = self.0.storage().unary_op(storage::Neg)?;
        let layout = self.0.layout().clone();
        Ok(TensorInternal::new(
            storage,
            layout,
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let grad_sum = store.get_or_insert_zero(&self.0);
        *grad_sum = &*grad_sum - out_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0]
    }
}

#[derive(Debug)]
pub struct EWiseAdd(pub Tensor, pub Tensor);

impl TensorOp for EWiseAdd {
    fn forward(self) -> Result<Tensor> {
        let storage = self
            .0
            .storage()
            .binary_op::<storage::EWiseAdd>(self.1.storage())?;
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let grad_sum = store.get_or_insert_zero(&self.0);
        *grad_sum = &*grad_sum + out_grad;

        let grad_sum = store.get_or_insert_zero(&self.1);
        *grad_sum = &*grad_sum + out_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0, &self.1]
    }
}

#[derive(Debug)]
pub struct EWiseMul(pub Tensor, pub Tensor);

impl TensorOp for EWiseMul {
    fn forward(self) -> Result<Tensor> {
        let storage = self
            .0
            .storage()
            .binary_op::<storage::EWiseMul>(self.1.storage())?;
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = store.get_or_insert_zero(&self.0);
        *arg1_grad_sum = &*arg1_grad_sum + &self.1 * out_grad;

        let arg2_grad_sum = store.get_or_insert_zero(&self.1);
        *arg2_grad_sum = &*arg2_grad_sum + &self.0 * out_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0, &self.1]
    }
}

#[derive(Debug)]
pub struct EWiseDiv(pub Tensor, pub Tensor);

impl TensorOp for EWiseDiv {
    fn forward(self) -> Result<Tensor> {
        let storage = self
            .0
            .storage()
            .binary_op::<storage::EWiseDiv>(self.1.storage())?;
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = store.get_or_insert_zero(&self.0);
        *arg1_grad_sum = &*arg1_grad_sum + out_grad / &self.1;

        let arg2_grad_sum = store.get_or_insert_zero(&self.1);
        *arg2_grad_sum = &*arg2_grad_sum + -out_grad * &self.0 / (&self.1.scalar_powf(2.0));
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0, &self.1]
    }
}

#[derive(Debug)]
pub struct EWisePowf(pub Tensor, pub Tensor);

impl TensorOp for EWisePowf {
    fn forward(self) -> Result<Tensor> {
        let storage = self
            .0
            .storage()
            .binary_op::<storage::EWisePow>(self.1.storage())?;
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * &self.1 * self.0.powf(&self.1 - 1.0);
        let arg_grad_sum = store.get_or_insert_zero(&self.0);
        *arg_grad_sum = &*arg_grad_sum + arg_grad;

        let e_grad = out_grad * self.0.powf(&self.1) * self.0.log();
        let e_grad_sum = store.get_or_insert_zero(&self.1);
        *e_grad_sum = &*e_grad_sum + e_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0, &self.1]
    }
}

#[derive(Debug)]
pub struct EWiseLog(pub Tensor);

impl TensorOp for EWiseLog {
    fn forward(self) -> Result<Tensor> {
        Ok(TensorInternal::new(
            self.0.storage().ewise_log()?,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = grads.get_or_insert_zero(&self.0);
        *arg1_grad_sum = &*arg1_grad_sum + out_grad / &self.0;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0]
    }
}

#[derive(Debug)]
pub struct ScalarAdd(pub Tensor, pub f64);

impl TensorOp for ScalarAdd {
    fn forward(self) -> Result<Tensor> {
        Ok(TensorInternal::new(
            self.0.storage().unary_op(storage::ScalarAdd(self.1))?,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let grad_sum = store.get_or_insert_zero(&self.0);
        *grad_sum = &*grad_sum + out_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0]
    }
}

#[derive(Debug)]
pub struct ScalarMul(pub Tensor, pub f64);

impl TensorOp for ScalarMul {
    fn forward(self) -> Result<Tensor> {
        Ok(TensorInternal::new(
            self.0.storage().unary_op(storage::ScalarMul(self.1))?,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * self.1;
        let grad_sum = store.get_or_insert_zero(&self.0);
        *grad_sum = &*grad_sum + arg_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0]
    }
}

#[derive(Debug)]
pub struct ScalarPowf(pub Tensor, pub f64);

impl TensorOp for ScalarPowf {
    fn forward(self) -> Result<Tensor> {
        Ok(TensorInternal::new(
            self.0.storage().powf(self.1)?,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = (out_grad * &self.0 * self.1).scalar_powf(self.1 - 1.0);
        let sum_grad = store.get_or_insert_zero(&self.0);
        *sum_grad = &*sum_grad + arg_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0]
    }
}

#[cfg(test)]
mod tests {
    use crate::{device::Device, dtype::DType, error::Result, tests::Approx};

    use super::*;

    #[test]
    fn test_ewise_add_backward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu);
        let c = &a + b;

        let grads = c.backward().unwrap();

        assert_eq!(grads.get(a.id()).unwrap(), a.ones_like());
    }

    #[test]
    fn test_backward_ewise_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu);
        let c = &a * &b;

        let grads = c.backward().unwrap();

        assert_eq!(grads.get(a.id()).unwrap(), b);
        assert_eq!(grads.get(b.id()).unwrap(), a);
    }

    #[test]
    fn test_backward_ewise_div() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu);
        let c = &a / &b;

        let grads = c.backward().unwrap();

        assert!(grads
            .get(a.id())
            .unwrap()
            .to_vec()
            .unwrap()
            .approx_eq(vec![0.2500, 0.2000, 0.1667]));
        assert!(grads
            .get(b.id())
            .unwrap()
            .to_vec()
            .unwrap()
            .approx_eq(vec![-0.0625, -0.0800, -0.0833]));
    }

    #[test]
    fn test_backward_ewise_powf() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu);
        let c = a.powf(&b);

        let grads = c.backward()?;

        assert_eq!(
            grads.get(a.id()).unwrap(),
            Tensor::from_vec(vec![4.0, 80., 1458.], (3,), Device::Cpu)
        );
        assert!(grads
            .get(b.id())
            .unwrap()
            .to_vec()?
            .approx_eq(vec![0., 22.1807, 800.8884]));
        Ok(())
    }

    #[test]
    fn test_backward_ewise_log() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = &a.log();

        let grads = b.backward().unwrap();

        assert!(grads
            .get(a.id())
            .unwrap()
            .to_vec()
            .unwrap()
            .approx_eq(vec![1.0, 0.5, 1.0 / 3.0]));
    }

    #[test]
    fn test_neg_backward() {
        let a = Tensor::zeros((3,), DType::F32, Device::Cpu);
        let op = Neg(a);
        let out_grad = Tensor::ones((3,), DType::F32, Device::Cpu);
        let mut grads = GradientStore::new();

        op.backward(&mut grads, &out_grad).unwrap();

        assert_eq!(grads.get(op.0.id()).unwrap(), -out_grad);
    }
}
