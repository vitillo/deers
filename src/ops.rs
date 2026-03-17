#![allow(dead_code)]

use std::sync::{Arc, RwLock};
use std::{fmt, iter};

use crate::backprop::GradientStore;
use crate::error::Result;
use crate::layout::{Layout, Shape};
use crate::storage::{self, BackendStorage, ReduceMax, ReduceSum};
use crate::tensor::Tensor;

/// An operator in the computation graph with forward and backward passes.
///
/// Each operator stores its input tensors and implements:
/// - `forward`: computes the output tensor and records itself in the graph.
/// - `backward`: given the output gradient, accumulates gradients for each input.
/// - `dependencies`: returns references to input tensors (for topological sorting).
pub trait TensorOp: fmt::Debug + Send + Sync {
    /// Executes the forward computation and returns the result tensor.
    fn forward(self) -> Result<Tensor>;

    /// Computes partial gradients for each input given the output gradient `out_grad`,
    /// accumulating them in `store`.
    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()>;

    /// Returns references to the input tensors this op depends on.
    fn dependencies(&self) -> Vec<&Tensor>;
}

#[derive(Debug)]
pub struct Neg {
    arg: Tensor,
}

impl Neg {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for Neg {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::Neg, self.arg.layout())?,
        ));
        let layout = Layout::from(self.arg.layout().shape().clone());
        Ok(Tensor::new(
            storage,
            layout,
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let grad_sum = store.get_or_insert_zero(&self.arg);
        *grad_sum = &*grad_sum - out_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct EWiseAdd {
    arg1: Tensor,
    arg2: Tensor,
}

impl EWiseAdd {
    pub fn new(arg1: Tensor, arg2: Tensor) -> Result<Self> {
        assert!(arg1.layout().shape() == arg2.layout().shape());
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseAdd {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg1.storage().binary_op::<storage::EWiseAdd>(
                self.arg1.layout(),
                &self.arg2.storage(),
                self.arg2.layout(),
            )?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            self.arg1.device(),
            self.arg1.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let grad_sum = store.get_or_insert_zero(&self.arg1);
        *grad_sum = &*grad_sum + out_grad;

        let grad_sum = store.get_or_insert_zero(&self.arg2);
        *grad_sum = &*grad_sum + out_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg1, &self.arg2]
    }
}

#[derive(Debug)]
pub struct EWiseMul {
    arg1: Tensor,
    arg2: Tensor,
}

impl EWiseMul {
    pub fn new(arg1: Tensor, arg2: Tensor) -> Result<Self> {
        assert!(arg1.layout().shape() == arg2.layout().shape());
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseMul {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg1.storage().binary_op::<storage::EWiseMul>(
                self.arg1.layout(),
                &self.arg2.storage(),
                self.arg2.layout(),
            )?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            self.arg1.device(),
            self.arg1.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = store.get_or_insert_zero(&self.arg1);
        *arg1_grad_sum = &*arg1_grad_sum + &self.arg2 * out_grad;

        let arg2_grad_sum = store.get_or_insert_zero(&self.arg2);
        *arg2_grad_sum = &*arg2_grad_sum + &self.arg1 * out_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg1, &self.arg2]
    }
}

#[derive(Debug)]
pub struct EWiseDiv {
    arg1: Tensor,
    arg2: Tensor,
}

impl EWiseDiv {
    pub fn new(arg1: Tensor, arg2: Tensor) -> Result<Self> {
        assert!(arg1.layout().shape() == arg2.layout().shape());
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseDiv {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg1.storage().binary_op::<storage::EWiseDiv>(
                self.arg1.layout(),
                &self.arg2.storage(),
                self.arg2.layout(),
            )?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            self.arg1.device(),
            self.arg1.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = store.get_or_insert_zero(&self.arg1);
        *arg1_grad_sum = &*arg1_grad_sum + out_grad / &self.arg2;

        let arg2_grad_sum = store.get_or_insert_zero(&self.arg2);
        *arg2_grad_sum = &*arg2_grad_sum + -out_grad * &self.arg1 / (&self.arg2.scalar_powf(2.0));
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg1, &self.arg2]
    }
}

#[derive(Debug)]
pub struct EWisePowf {
    arg: Tensor,
    e: Tensor,
}

impl EWisePowf {
    pub fn new(arg: Tensor, e: Tensor) -> Result<Self> {
        assert!(arg.layout().shape() == e.layout().shape());
        Ok(Self { arg, e })
    }
}

impl TensorOp for EWisePowf {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg.storage().binary_op::<storage::EWisePow>(
                self.arg.layout(),
                &self.e.storage(),
                self.e.layout(),
            )?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * &self.e * self.arg.powf(&self.e - 1.0);
        let arg_grad_sum = store.get_or_insert_zero(&self.arg);
        *arg_grad_sum = &*arg_grad_sum + arg_grad;

        let e_grad = out_grad * self.arg.powf(&self.e) * self.arg.log();
        let e_grad_sum = store.get_or_insert_zero(&self.e);
        *e_grad_sum = &*e_grad_sum + e_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg, &self.e]
    }
}

#[derive(Debug)]
pub struct EWiseLog {
    arg: Tensor,
}

impl EWiseLog {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for EWiseLog {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::Log, self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = grads.get_or_insert_zero(&self.arg);
        *arg1_grad_sum = &*arg1_grad_sum + out_grad / &self.arg;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct EWiseExp {
    arg: Tensor,
}

impl EWiseExp {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for EWiseExp {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::Exp, self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg1_grad_sum = grads.get_or_insert_zero(&self.arg);
        *arg1_grad_sum = &*arg1_grad_sum + out_grad * &self.arg.exp();
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Relu {
    arg: Tensor,
}

impl Relu {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for Relu {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::Relu, self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        // grad is out_grad * (input > 0)
        let mask_storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::ReluBackward, self.arg.layout())?,
        ));
        let mask = Tensor::new(
            mask_storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            None,
        );
        let grad_sum = grads.get_or_insert_zero(&self.arg);
        *grad_sum = &*grad_sum + out_grad * &mask;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct ScalarAdd {
    arg: Tensor,
    scalar: f64,
}

impl ScalarAdd {
    pub fn new(arg: Tensor, scalar: f64) -> Result<Self> {
        Ok(Self { arg, scalar })
    }
}

impl TensorOp for ScalarAdd {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::ScalarAdd(self.scalar), self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let grad_sum = store.get_or_insert_zero(&self.arg);
        *grad_sum = &*grad_sum + out_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct ScalarMul {
    arg: Tensor,
    scalar: f64,
}

impl ScalarMul {
    pub fn new(arg: Tensor, scalar: f64) -> Result<Self> {
        Ok(Self { arg, scalar })
    }
}

impl TensorOp for ScalarMul {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg
                .storage()
                .unary_op(storage::ScalarMul(self.scalar), self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * self.scalar;
        let grad_sum = store.get_or_insert_zero(&self.arg);
        *grad_sum = &*grad_sum + arg_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct ScalarPowf {
    arg: Tensor,
    e: f64,
}

impl ScalarPowf {
    pub fn new(arg: Tensor, e: f64) -> Result<Self> {
        Ok(Self { arg, e })
    }
}

impl TensorOp for ScalarPowf {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(
            self.arg.storage().ewise_powf(self.e, self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * self.e * self.arg.scalar_powf(self.e - 1.0);
        let sum_grad = store.get_or_insert_zero(&self.arg);
        *sum_grad = &*sum_grad + arg_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Permute(Tensor, Shape);

impl Permute {
    pub fn new(arg: Tensor, shape: Shape) -> Self {
        assert!(arg.layout().ndim() == shape.ndim());
        Self(arg, shape)
    }
}

impl TensorOp for Permute {
    fn forward(self) -> Result<Tensor> {
        let storage = self.0.storage_clone();
        let layout = self.0.layout().permute(&self.1);
        Ok(Tensor::new(
            storage,
            layout,
            self.0.device(),
            self.0.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, store: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let mut inverse = vec![0; self.1.ndim()];
        for (new_axis, &old_axis) in self.1.iter().enumerate() {
            inverse[old_axis] = new_axis;
        }

        let arg_grad = out_grad.permute(inverse);
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
    arg: Tensor,
    new_shape: Shape,
}

impl Broadcast {
    pub fn new(arg: Tensor, new_shape: Shape) -> Self {
        assert!(new_shape.ndim() >= arg.layout().ndim());
        Self { arg, new_shape }
    }
}

impl TensorOp for Broadcast {
    fn forward(self) -> Result<Tensor> {
        let shape_diff = self.new_shape.ndim() - self.arg.layout().ndim();

        let mut old_shape = Vec::with_capacity(self.new_shape.ndim());
        old_shape.extend((0..shape_diff).map(|_| 1));
        old_shape.extend(self.arg.layout().shape().iter());

        let mut new_strides = Vec::with_capacity(self.new_shape.ndim());
        new_strides.extend((0..shape_diff).map(|_| 0));
        new_strides.extend(self.arg.layout().strides().iter());

        for (i, (new_dim, old_dim)) in self.new_shape.iter().zip(old_shape.iter()).enumerate() {
            if *old_dim == 1 {
                new_strides[i] = 0;
            } else if old_dim != new_dim {
                panic!("Invalid shape");
            }
        }

        let layout = Layout::new(self.new_shape.clone(), new_strides, 0);
        let storage = self.arg.storage_clone();
        Ok(Tensor::new(
            storage,
            layout,
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape_diff = self.new_shape.ndim() - self.arg.layout().ndim();

        // Shape without broadcasting
        let shape: Vec<_> = iter::repeat_n(1, shape_diff)
            .chain(self.arg.layout().shape().iter().copied())
            .collect();

        // Find axes that were broadcasted
        let axes: Vec<_> = shape.into_iter().zip(self.new_shape.iter().copied())
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
    arg: Tensor,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Sum {
    pub fn new(arg: Tensor, axis: Vec<usize>, keep_dims: bool) -> Self {
        assert!(axis.iter().all(|&i| i < arg.layout().ndim()));
        Self {
            arg,
            axis,
            keep_dims,
        }
    }
}

impl TensorOp for Sum {
    fn forward(self) -> Result<Tensor> {
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

        let view = self.arg.permute(permuted_dims).compact();
        let mut out_storage = self.arg.device().zeros(new_shape.size(), self.arg.dtype());
        view.storage()
            .reduce::<ReduceSum>(view.layout(), &mut out_storage)?;
        let storage = Arc::new(RwLock::new(out_storage));
        Ok(Tensor::new(
            storage,
            Layout::from(new_shape),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
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
            .broadcast(self.arg.layout().shape().clone());
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
        // TODO: refactor with Sum forward implementation
        assert!(axis.iter().all(|&i| i < arg.layout().ndim()));
        Self {
            arg,
            axis,
            keep_dims,
        }
    }
}

impl TensorOp for Max {
    fn forward(self) -> Result<Tensor> {
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

        let view = self.arg.permute(permuted_dims).compact();
        let mut out_storage = self.arg.device().zeros(new_shape.size(), self.arg.dtype());
        view.storage()
            .reduce::<ReduceMax>(view.layout(), &mut out_storage)?;
        let storage = Arc::new(RwLock::new(out_storage));
        Ok(Tensor::new(
            storage,
            Layout::from(new_shape),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
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
    arg: Tensor,
    new_shape: Shape,
}

impl Reshape {
    pub fn new(arg: Tensor, new_shape: Shape) -> Self {
        assert!(arg.layout().size() == new_shape.size());
        Self {
            arg: arg.compact(),
            new_shape,
        }
    }
}

impl TensorOp for Reshape {
    fn forward(self) -> Result<Tensor> {
        let storage = self.arg.storage_clone();

        Ok(Tensor::new(
            storage,
            Layout::from(self.new_shape.clone()),
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
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
    pub fn new(arg1: Tensor, arg2: Tensor) -> Result<Self> {
        assert!(arg1.layout().ndim() == 2 && arg2.layout().ndim() == 2);
        assert!(arg1.layout().shape()[1] == arg2.layout().shape()[0]);
        Ok(Self {
            arg1: arg1.compact(),
            arg2: arg2.compact(),
        })
    }
}

impl TensorOp for MatMul {
    fn forward(self) -> Result<Tensor> {
        let shape: Shape = (self.arg1.layout().shape[0], self.arg2.layout().shape[1]).into();
        let storage = Arc::new(RwLock::new(self.arg1.storage().matmul(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));

        Ok(Tensor::new(
            storage,
            shape.into(),
            self.arg1.device(),
            self.arg1.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        // a * b, da = b, db = a
        // a = n x p
        // b = p x k
        // out = n x k
        // out @ b^t = n x p
        // a^t @ out = p x k

        let arg1_grad = out_grad.matmul(&self.arg2.transpose(None));
        let sum_grad = grads.get_or_insert_zero(&self.arg1);
        *sum_grad = &*sum_grad + arg1_grad;

        let arg2_grad = self.arg1.transpose(None).matmul(out_grad);
        let sum_grad = grads.get_or_insert_zero(&self.arg2);
        *sum_grad = &*sum_grad + arg2_grad;

        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg1, &self.arg2]
    }
}

#[derive(Debug)]
pub struct LogSumExp {
    arg: Tensor,
    axes: Vec<usize>,
}

impl LogSumExp {
    pub fn new(arg: Tensor, axes: Vec<usize>) -> Self {
        Self { arg, axes }
    }
}

impl TensorOp for LogSumExp {
    fn forward(self) -> Result<Tensor> {
        // https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        let max_z = self.arg.max(self.axes.clone(), true);
        let broadcast_max_z = max_z.broadcast(self.arg.layout().shape().clone());
        let sum_z = (&self.arg - &broadcast_max_z)
            .exp()
            .sum(self.axes.clone(), false);
        let logsumexp = max_z.reshape(sum_z.layout().shape.clone()) + sum_z.log();
        // TODO: This is required to avoid creating a computational graph for the above operations
        // How do I prevent this from happening for all ops??
        Ok(logsumexp.with_op(Box::new(self)))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let max_z = self.arg.max(self.axes.clone(), true);
        let broadcast_max_z = max_z.broadcast(self.arg.layout().shape().clone());
        let exp_z = (&self.arg - broadcast_max_z).exp();
        let sum_z = exp_z.sum(self.axes.clone(), false);

        let expand_shape = {
            let mut shape = self.arg.layout().shape().clone();
            for axis in &self.axes {
                shape[*axis] = 1;
            }
            shape
        };
        let grad_sum_z = (out_grad / sum_z)
            .reshape(expand_shape)
            .broadcast(self.arg.layout().shape().clone());
        let arg_grad = grad_sum_z * exp_z;

        let sum_grad = grads.get_or_insert_zero(&self.arg);
        *sum_grad = &*sum_grad + arg_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Compact {
    arg: Tensor,
}

impl Compact {
    pub fn new(arg: Tensor) -> Self {
        Self { arg }
    }
}

impl TensorOp for Compact {
    fn forward(self) -> Result<Tensor> {
        let mut storage = self
            .arg
            .device()
            .zeros(self.arg.layout().size(), self.arg.dtype());
        self.arg
            .storage()
            .copy_compact(self.arg.layout(), &mut storage)?;
        let strides = self.arg.layout().shape().compact_strides();
        let layout = Layout::new(self.arg.layout().shape().clone(), strides, 0);
        Ok(Tensor::new(
            Arc::new(RwLock::new(storage)),
            layout,
            self.arg.device(),
            self.arg.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let sum_grad = grads.get_or_insert_zero(&self.arg);
        *sum_grad = &*sum_grad + out_grad;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

/// Gathers values along an axis using integer indices.
///
/// For a 2D input of shape `(rows, cols)` with `dim=1` and indices of shape `(rows,)`,
/// produces output of shape `(rows, 1)` where `out[i, 0] = input[i, indices[i]]`.
///
/// Currently only supports 2D input with 1D indices along dim 1.
#[derive(Debug)]
pub struct Gather {
    input: Tensor,
    dim: usize,
    indices: Vec<usize>,
}

impl Gather {
    pub fn new(input: Tensor, dim: usize, indices: Vec<usize>) -> Self {
        assert_eq!(input.layout().ndim(), 2, "gather only supports 2D input");
        assert_eq!(dim, 1, "gather only supports dim=1");
        assert_eq!(indices.len(), input.layout().shape()[0]);
        Self {
            input: input.compact(),
            dim,
            indices,
        }
    }
}

impl TensorOp for Gather {
    fn forward(self) -> Result<Tensor> {
        let rows = self.input.layout().shape()[0];
        let storage = Arc::new(RwLock::new(
            self.input
                .storage()
                .gather(self.input.layout(), self.dim, &self.indices)?,
        ));
        let shape: Shape = (rows, 1).into();
        Ok(Tensor::new(
            storage,
            shape.into(),
            self.input.device(),
            self.input.dtype(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let full_shape = self.input.layout().shape();
        let grad_storage = out_grad.compact().storage().scatter(
            out_grad.layout(),
            self.dim,
            &self.indices,
            &full_shape.iter().copied().collect::<Vec<_>>(),
        )?;
        let grad_tensor = Tensor::new(
            Arc::new(RwLock::new(grad_storage)),
            full_shape.clone().into(),
            self.input.device(),
            self.input.dtype(),
            false,
            None,
        );
        let sum_grad = grads.get_or_insert_zero(&self.input);
        *sum_grad = &*sum_grad + &grad_tensor;
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}
