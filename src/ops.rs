#![allow(dead_code)]

use std::iter::zip;
use std::sync::{Arc, RwLock};
use std::{fmt, iter};

use crate::backprop::GradientStore;
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::storage::{self, BackendStorage};
use crate::storage::{ReduceMax, ReduceSum};
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
pub struct Permute(Tensor, Shape);

impl Permute {
    pub fn new(arg: Tensor, shape: Shape) -> Self {
        // TODO add constructors to check pre-requisites
        assert!(arg.layout().ndim() == shape.ndim());
        Self(arg, shape)
    }
}

impl TensorOp for Permute {
    fn forward(self) -> Result<Tensor> {
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
        let arg_grad = out_grad.permute(self.1.clone());
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

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape_diff = self.new_shape.ndim() - self.arg.layout().ndim();

        // Shape without broadcasting
        let shape: Vec<_> = iter::repeat_n(1, shape_diff)
            .chain(self.arg.layout().shape().iter().copied())
            .collect();

        // Find axes that were broadcasted
        let axes: Vec<_> = zip(shape, self.new_shape.iter().copied())
            .enumerate()
            .filter(|(_, (o, n))| o != n)
            .map(|(i, _)| i)
            .collect();

        // Sum out broadcasted axes
        let out_grad = out_grad
            .sum(axes, false)
            .reshape(self.arg.layout().shape().clone());

        let sum_grad = grads.get_or_insert_zero(&self.arg);
        *sum_grad = &*sum_grad + out_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Sum {
    pub arg: Tensor,
    pub axis: Vec<usize>,
    pub keep_dims: bool,
}

impl TensorOp for Sum {
    fn forward(self) -> Result<Tensor> {
        assert!(self.axis.iter().all(|&i| i < self.arg.layout().ndim()));
        let new_shape: Shape = if self.keep_dims {
            self.arg
                .layout()
                .shape()
                .iter()
                .enumerate()
                .map(|(ref i, &v)| if self.axis.contains(i) { 1 } else { v })
                .collect::<Vec<usize>>()
                .into()
        } else {
            self.arg
                .layout()
                .shape()
                .iter()
                .enumerate()
                .filter(|&(i, _)| !self.axis.contains(&i))
                .map(|(_, &dim)| dim)
                .collect::<Vec<usize>>()
                .into()
        };

        let permuted_dims: Vec<usize> = (0..self.arg.layout().ndim())
            .filter(|i| !self.axis.contains(i))
            .chain(self.axis.iter().copied())
            .collect();

        let view = self.arg.permute(permuted_dims).compact()?;
        let mut out_storage = self.arg.device().zeros(new_shape.size(), self.arg.dtype());
        view.storage()
            .reduce::<ReduceSum>(view.layout(), &mut out_storage)?;
        let storage = Arc::new(RwLock::new(out_storage));
        Ok(TensorInternal::new(
            storage,
            Layout::from(new_shape),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape: Shape = self
            .arg
            .layout()
            .shape()
            .iter()
            .enumerate()
            .map(|(ref i, &v)| if self.axis.contains(i) { 1 } else { v })
            .collect::<Vec<usize>>()
            .into();

        let out_grad = out_grad
            .reshape(shape)
            .broadcast(self.arg.layout().shape().clone())
            .compact()?; // TODO: support striped add
        let sum_grad = grads.get_or_insert_zero(&self.arg);
        *sum_grad = &*sum_grad + out_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Max {
    arg: Tensor,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Max {
    pub fn new(arg: Tensor, axis: Vec<usize>, keep_dims: bool) -> Self {
        Self {
            arg,
            axis,
            keep_dims,
        }
    }
}

impl TensorOp for Max {
    fn forward(self) -> Result<Tensor> {
        // TODO: refactor with Sum forward implementation
        assert!(self.axis.iter().all(|&i| i < self.arg.layout().ndim()));
        let new_shape: Shape = if self.keep_dims {
            self.arg
                .layout()
                .shape()
                .iter()
                .enumerate()
                .map(|(ref i, &v)| if self.axis.contains(i) { 1 } else { v })
                .collect::<Vec<usize>>()
                .into()
        } else {
            self.arg
                .layout()
                .shape()
                .iter()
                .enumerate()
                .filter(|&(i, _)| !self.axis.contains(&i))
                .map(|(_, &dim)| dim)
                .collect::<Vec<usize>>()
                .into()
        };

        let permuted_dims: Vec<usize> = (0..self.arg.layout().ndim())
            .filter(|i| !self.axis.contains(i))
            .chain(self.axis.iter().copied())
            .collect();

        let view = self.arg.permute(permuted_dims).compact()?;
        let mut out_storage = self.arg.device().zeros(new_shape.size(), self.arg.dtype());
        view.storage()
            .reduce::<ReduceMax>(view.layout(), &mut out_storage)?;
        let storage = Arc::new(RwLock::new(out_storage));
        Ok(TensorInternal::new(
            storage,
            Layout::from(new_shape),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, _: &mut GradientStore, _: &Tensor) -> Result<()> {
        todo!()
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Reshape {
    pub arg: Tensor,
    pub new_shape: Shape,
}

impl TensorOp for Reshape {
    fn forward(self) -> Result<Tensor> {
        assert!(self.arg.layout().size() == self.new_shape.size());
        let storage = self.arg.storage.clone();

        Ok(TensorInternal::new(
            storage,
            Layout::from(self.new_shape.clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let out_grad = out_grad.reshape(self.arg.layout().shape().clone());
        let sum_grad = grads.get_or_insert_zero(&self.arg);
        *sum_grad = &*sum_grad + out_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct MatMul {
    arg1: Tensor,
    arg2: Tensor,
}

impl MatMul {
    pub fn new(arg1: Tensor, arg2: Tensor) -> Self {
        assert!(arg1.layout().ndim() == 2 && arg2.layout().ndim() == 2);
        assert!(arg1.layout().shape()[1] == arg2.layout().shape()[0]);
        Self { arg1, arg2 }
    }
}

impl TensorOp for MatMul {
    fn forward(self) -> Result<Tensor> {
        let shape: Shape = (self.arg1.layout().shape[1], self.arg2.layout().shape[0]).into();
        let storage = Arc::new(RwLock::new(self.arg1.storage().matmul(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));

        Ok(TensorInternal::new(
            storage,
            shape.into(),
            self.arg1.device(),
            self.arg1.dtype(),
            false,
            Some(Box::new(self)),
        )
        .into())
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        // a * b, da = b, db = a
        // a = n x p
        // b = p x k
        // out = n x k
        // out @ b^t = n x p
        // a^t @ out = p x k

        // TODO: get rid of compactions
        let arg1_grad = out_grad.matmul(&self.arg2.transpose(None).compact()?);
        let sum_grad = grads.get_or_insert_zero(&self.arg1);
        *sum_grad = &*sum_grad + arg1_grad;

        let arg2_grad = self.arg1.transpose(None).compact()?.matmul(out_grad);
        let sum_grad = grads.get_or_insert_zero(&self.arg2);
        *sum_grad = &*sum_grad + arg2_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg1, &self.arg2]
    }
}

#[derive(Debug)]
struct LogSoftmax {
    arg: Tensor,
}

impl LogSoftmax {
    pub fn new(arg: Tensor) -> Self {
        Self { arg }
    }
}

impl TensorOp for LogSoftmax {
    fn forward(self) -> Result<Tensor> {
        todo!()
    }

    fn backward(&self, _: &mut GradientStore, _: &Tensor) -> Result<()> {
        todo!()
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}
