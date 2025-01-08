#![allow(dead_code)]

use std::fmt;
use std::iter::zip;
use std::sync::{Arc, RwLock};

use crate::backprop::GradientStore;
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::storage::{self, BackendStorage};
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().unary_op(storage::Neg, self.0.layout())?,
        ));
        let layout = self.0.layout().clone();
        Ok(TensorInternal::new(
            storage,
            layout,
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().binary_op::<storage::EWiseAdd>(
                self.0.layout(),
                &self.1.storage(),
                self.1.layout(),
            )?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().binary_op::<storage::EWiseMul>(
                self.0.layout(),
                &self.1.storage(),
                self.1.layout(),
            )?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().binary_op::<storage::EWiseDiv>(
                self.0.layout(),
                &self.1.storage(),
                self.1.layout(),
            )?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().binary_op::<storage::EWisePow>(
                self.0.layout(),
                &self.1.storage(),
                self.1.layout(),
            )?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().unary_op(storage::Log, self.0.layout())?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
pub struct EWiseExp(pub Tensor);

impl TensorOp for EWiseExp {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.0.storage().unary_op(storage::Exp, self.0.layout())?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = grads.get_or_insert_zero(&self.0);
        *arg1_grad_sum = &*arg1_grad_sum + out_grad * &self.0.exp();
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
        let storage = Arc::new(RwLock::new(
            self.0
                .storage()
                .unary_op(storage::ScalarAdd(self.1), self.0.layout())?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0
                .storage()
                .unary_op(storage::ScalarMul(self.1), self.0.layout())?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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
        let storage = Arc::new(RwLock::new(
            self.0.storage().ewise_powf(self.1, self.0.layout())?,
        ));
        Ok(TensorInternal::new(
            storage,
            self.0.layout().clone(),
            self.0.device(),
            self.0.dtype(),
            false,
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

#[derive(Debug)]
pub struct Permute(pub Tensor, pub Vec<usize>);

impl TensorOp for Permute {
    fn forward(self) -> Result<Tensor> {
        assert_eq!(self.0.layout().ndim(), self.1.len());
        let storage = self.0.storage.clone();
        let layout = self.0.layout().permute(&self.1);
        Ok(TensorInternal::new(
            storage,
            layout,
            self.0.device(),
            self.0.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad.permute(&self.1);
        let sum_grad = store.get_or_insert_zero(&self.0);
        *sum_grad = &*sum_grad + arg_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.0]
    }
}

#[derive(Debug)]
pub struct Broadcast {
    pub arg: Tensor,
    pub new_shape: Shape,
}

impl TensorOp for Broadcast {
    fn forward(self) -> Result<Tensor> {
        assert!(self.new_shape.ndim() >= self.arg.layout().ndim());
        let shape_diff = self.new_shape.ndim() - self.arg.layout().ndim();

        let mut old_shape = Vec::with_capacity(self.new_shape.ndim());
        old_shape.extend((0..shape_diff).map(|_| 1));
        old_shape.extend(self.arg.layout().shape().iter());

        let mut new_strides = Vec::with_capacity(self.new_shape.ndim());
        new_strides.extend((0..shape_diff).map(|_| 0));
        new_strides.extend(self.arg.layout().strides().iter());

        for (i, (new_dim, old_dim)) in zip(self.new_shape.iter(), old_shape.iter()).enumerate() {
            if *old_dim == 1 {
                new_strides[i] = 0;
            } else if old_dim != new_dim {
                panic!("Invalid shape");
            }
        }

        let layout = Layout::new(self.new_shape.clone(), new_strides, 0);
        let storage = self.arg.storage.clone();
        Ok(TensorInternal::new(
            storage,
            layout,
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        todo!()
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}
