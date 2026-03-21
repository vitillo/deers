#![allow(dead_code)]

use std::sync::{Arc, RwLock};
use std::{fmt, iter};

use crate::backprop::GradientStore;
use crate::error::{Error, Result};
use crate::layout::{Layout, Shape};
use crate::storage::{self, BackendStorage, MpsStorage, ReduceMax, ReduceSum, Storage};
use crate::tensor::Tensor;

/// Computes the output shape for a reduction along the given axes.
fn reduce_shape(shape: &Shape, axes: &[usize], keep_dims: bool) -> Shape {
    if keep_dims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &v)| if axes.contains(&i) { 1 } else { v })
            .collect::<Vec<usize>>()
            .into()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| !axes.contains(&i))
            .map(|(_, &dim)| dim)
            .collect::<Vec<usize>>()
            .into()
    }
}

/// Prepares a tensor for reduction: permutes non-reduced axes first, then compacts.
fn reduce_view(arg: &Tensor, axes: &[usize]) -> Tensor {
    let permuted_dims: Vec<usize> = (0..arg.layout().ndim())
        .filter(|i| !axes.contains(i))
        .chain(axes.iter().copied())
        .collect();
    arg.permute(permuted_dims).compact()
}

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
    /// accumulating them in `grads`.
    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()>;

    /// Returns references to the input tensors this op depends on.
    fn dependencies(&self) -> Vec<&Tensor>;

    /// Short name for profiling (e.g. "MatMul", "Add"). Defaults to the struct name.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>().rsplit("::").next().unwrap_or("?")
    }
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
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Neg, self.arg.layout())?));
        let layout = Layout::from(self.arg.layout().shape().clone());
        Ok(Tensor::new(storage, layout, false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, -out_grad);
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
        let storage = Arc::new(RwLock::new(self.arg1.storage().binary_op::<storage::EWiseAdd>(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg1, out_grad.clone());
        grads.accumulate(&self.arg2, out_grad.clone());
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg1, &self.arg2]
    }
}

#[derive(Debug)]
pub struct EWiseSub {
    arg1: Tensor,
    arg2: Tensor,
}

impl EWiseSub {
    pub fn new(arg1: Tensor, arg2: Tensor) -> Result<Self> {
        assert!(arg1.layout().shape() == arg2.layout().shape());
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseSub {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(self.arg1.storage().binary_op::<storage::EWiseSub>(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg1, out_grad.clone());
        grads.accumulate(&self.arg2, -out_grad);
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
        let storage = Arc::new(RwLock::new(self.arg1.storage().binary_op::<storage::EWiseMul>(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg1, &self.arg2 * out_grad);
        grads.accumulate(&self.arg2, &self.arg1 * out_grad);

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
        let storage = Arc::new(RwLock::new(self.arg1.storage().binary_op::<storage::EWiseDiv>(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg1.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg1, out_grad / &self.arg2);
        grads.accumulate(&self.arg2, -out_grad * &self.arg1 / (&self.arg2.scalar_powf(2.0)));
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
        let storage = Arc::new(RwLock::new(self.arg.storage().binary_op::<storage::EWisePow>(
            self.arg.layout(),
            &self.e.storage(),
            self.e.layout(),
        )?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * &self.e * self.arg.powf(&self.e - 1.0);
        grads.accumulate(&self.arg, arg_grad);

        let e_grad = out_grad * self.arg.powf(&self.e) * self.arg.log();
        grads.accumulate(&self.e, e_grad);

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
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Log, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, out_grad / &self.arg);
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
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Exp, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, out_grad * &self.arg.exp());
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct EWiseSin {
    arg: Tensor,
}

impl EWiseSin {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for EWiseSin {
    fn forward(self) -> Result<Tensor> {
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Sin, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, out_grad * &self.arg.cos());
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct EWiseCos {
    arg: Tensor,
}

impl EWiseCos {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for EWiseCos {
    fn forward(self) -> Result<Tensor> {
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Cos, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, out_grad * &self.arg.sin() * -1.0);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Tanh {
    arg: Tensor,
}

impl Tanh {
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for Tanh {
    fn forward(self) -> Result<Tensor> {
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Tanh, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let out = self.arg.tanh();
        let arg_grad = out_grad * (&(&out * &out) * -1.0 + 1.0);
        grads.accumulate(&self.arg, arg_grad);
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
        let storage =
            Arc::new(RwLock::new(self.arg.storage().unary_op(storage::Relu, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        // grad is out_grad * (input > 0)
        let mask_storage = Arc::new(RwLock::new(
            self.arg.storage().unary_op(storage::ReluBackward, self.arg.layout())?,
        ));
        let mask =
            Tensor::new(mask_storage, Layout::from(self.arg.layout().shape().clone()), false, None);
        grads.accumulate(&self.arg, out_grad * &mask);
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
            self.arg.storage().unary_op(storage::ScalarAdd(self.scalar), self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, out_grad.clone());
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
            self.arg.storage().unary_op(storage::ScalarMul(self.scalar), self.arg.layout())?,
        ));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * self.scalar;
        grads.accumulate(&self.arg, arg_grad);

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
        let storage =
            Arc::new(RwLock::new(self.arg.storage().ewise_powf(self.e, self.arg.layout())?));
        Ok(Tensor::new(
            storage,
            Layout::from(self.arg.layout().shape().clone()),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_grad = out_grad * self.e * self.arg.scalar_powf(self.e - 1.0);
        grads.accumulate(&self.arg, arg_grad);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Permute {
    arg: Tensor,
    axes: Shape,
}

impl Permute {
    pub fn new(arg: Tensor, axes: Shape) -> Result<Self> {
        if arg.layout().ndim() != axes.ndim() {
            return Err(Error::LayoutMismatch(format!(
                "permute: tensor has {} dims but got {} axes",
                arg.layout().ndim(),
                axes.ndim()
            )));
        }
        Ok(Self { arg, axes })
    }
}

impl TensorOp for Permute {
    fn forward(self) -> Result<Tensor> {
        let storage = self.arg.storage_clone();
        let layout = self.arg.layout().permute(&self.axes);
        Ok(Tensor::new(storage, layout, false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let mut inverse = vec![0; self.axes.ndim()];
        for (new_axis, &old_axis) in self.axes.iter().enumerate() {
            inverse[old_axis] = new_axis;
        }

        let arg_grad = out_grad.permute(inverse);
        grads.accumulate(&self.arg, arg_grad);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Broadcast {
    arg: Tensor,
    new_shape: Shape,
}

impl Broadcast {
    pub fn new(arg: Tensor, new_shape: Shape) -> Result<Self> {
        if new_shape.ndim() < arg.layout().ndim() {
            return Err(Error::LayoutMismatch(format!(
                "broadcast: target ndim {} < source ndim {}",
                new_shape.ndim(),
                arg.layout().ndim()
            )));
        }
        Ok(Self { arg, new_shape })
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
                return Err(Error::LayoutMismatch(format!(
                    "broadcast: dimension {} is {} but target is {}",
                    i, old_dim, new_dim
                )));
            }
        }

        let layout = Layout::new(self.new_shape.clone(), new_strides, self.arg.layout().offset);
        let storage = self.arg.storage_clone();
        Ok(Tensor::new(storage, layout, false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape_diff = self.new_shape.ndim() - self.arg.layout().ndim();

        // Shape without broadcasting
        let shape: Vec<_> = iter::repeat_n(1, shape_diff)
            .chain(self.arg.layout().shape().iter().copied())
            .collect();

        // Find axes that were broadcasted
        let axes: Vec<_> = shape
            .into_iter()
            .zip(self.new_shape.iter().copied())
            .enumerate()
            .filter(|(_, (o, n))| o != n)
            .map(|(i, _)| i)
            .collect();

        // Sum out broadcasted axes
        let out_grad = out_grad.sum(axes, false).reshape(self.arg.layout().shape().clone());

        grads.accumulate(&self.arg, out_grad);
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
    pub fn new(arg: Tensor, axis: Vec<usize>, keep_dims: bool) -> Result<Self> {
        if let Some(&bad) = axis.iter().find(|&&i| i >= arg.layout().ndim()) {
            return Err(Error::LayoutMismatch(format!(
                "sum: axis {} out of bounds for {} dims",
                bad,
                arg.layout().ndim()
            )));
        }
        Ok(Self { arg, axis, keep_dims })
    }
}

impl TensorOp for Sum {
    fn forward(self) -> Result<Tensor> {
        let new_shape = reduce_shape(self.arg.layout().shape(), &self.axis, self.keep_dims);
        let view = reduce_view(&self.arg, &self.axis);
        let mut out_storage = self.arg.device().zeros(new_shape.size(), self.arg.dtype());
        view.storage().reduce::<ReduceSum>(view.layout(), &mut out_storage)?;
        let storage = Arc::new(RwLock::new(out_storage));
        Ok(Tensor::new(storage, Layout::from(new_shape), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape = reduce_shape(self.arg.layout().shape(), &self.axis, true);

        let out_grad = out_grad.reshape(shape).broadcast(self.arg.layout().shape().clone());
        grads.accumulate(&self.arg, out_grad);
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
    pub fn new(arg: Tensor, axis: Vec<usize>, keep_dims: bool) -> Result<Self> {
        if let Some(&bad) = axis.iter().find(|&&i| i >= arg.layout().ndim()) {
            return Err(Error::LayoutMismatch(format!(
                "max: axis {} out of bounds for {} dims",
                bad,
                arg.layout().ndim()
            )));
        }
        Ok(Self { arg, axis, keep_dims })
    }
}

impl TensorOp for Max {
    fn forward(self) -> Result<Tensor> {
        let new_shape = reduce_shape(self.arg.layout().shape(), &self.axis, self.keep_dims);
        let view = reduce_view(&self.arg, &self.axis);
        let mut out_storage = self.arg.device().zeros(new_shape.size(), self.arg.dtype());
        view.storage().reduce::<ReduceMax>(view.layout(), &mut out_storage)?;
        let storage = Arc::new(RwLock::new(out_storage));
        Ok(Tensor::new(storage, Layout::from(new_shape), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let max_keep_dims = self.arg.max(self.axis.clone(), true);
        let max_broadcast = max_keep_dims.broadcast(self.arg.layout().shape().clone());
        let grad = out_grad
            .reshape(max_keep_dims.layout().shape().clone())
            .broadcast(self.arg.layout().shape().clone());
        let arg_grad = self.arg.eq(&max_broadcast) * grad;

        grads.accumulate(&self.arg, arg_grad);
        Ok(())
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
    pub fn new(arg: Tensor, new_shape: Shape) -> Result<Self> {
        if arg.layout().size() != new_shape.size() {
            return Err(Error::LayoutMismatch(format!(
                "reshape: size {} cannot be reshaped to size {}",
                arg.layout().size(),
                new_shape.size()
            )));
        }
        Ok(Self { arg: arg.compact(), new_shape })
    }
}

impl TensorOp for Reshape {
    fn forward(self) -> Result<Tensor> {
        let storage = self.arg.storage_clone();

        Ok(Tensor::new(storage, Layout::from(self.new_shape.clone()), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let out_grad = out_grad.reshape(self.arg.layout().shape().clone());
        grads.accumulate(&self.arg, out_grad);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Narrow {
    arg: Tensor,
    dim: usize,
    start: usize,
    len: usize,
}

impl Narrow {
    pub fn new(arg: Tensor, dim: usize, start: usize, len: usize) -> Result<Self> {
        if dim >= arg.layout().ndim() {
            return Err(Error::LayoutMismatch(format!(
                "narrow: dim {} out of bounds for {} dims",
                dim,
                arg.layout().ndim()
            )));
        }
        if start + len > arg.layout().shape()[dim] {
            return Err(Error::LayoutMismatch(format!(
                "narrow: start {} + len {} exceeds dim size {}",
                start,
                len,
                arg.layout().shape()[dim]
            )));
        }
        if !arg.layout().is_contiguous() {
            return Err(Error::LayoutMismatch("narrow requires contiguous tensors".into()));
        }
        Ok(Self { arg, dim, start, len })
    }
}

impl TensorOp for Narrow {
    fn forward(self) -> Result<Tensor> {
        let mut shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        shape[self.dim] = self.len;
        let stride = self.arg.layout().strides()[self.dim] as usize;
        let offset = self.arg.layout().offset + self.start * stride;

        Ok(Tensor::new(
            self.arg.storage_clone(),
            Layout::new(shape, self.arg.layout().strides().clone(), offset),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let mut parts = Vec::with_capacity(3);
        let mut pad_shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();

        if self.start > 0 {
            pad_shape[self.dim] = self.start;
            parts.push(Tensor::zeros(pad_shape.clone(), self.arg.dtype(), self.arg.device()));
        }

        parts.push(out_grad.clone());

        let arg_dim = self.arg.layout().shape()[self.dim];
        let right_len = arg_dim - self.start - self.len;
        if right_len > 0 {
            pad_shape[self.dim] = right_len;
            parts.push(Tensor::zeros(pad_shape, self.arg.dtype(), self.arg.device()));
        }

        let arg_grad = Tensor::cat(&parts, self.dim);
        grads.accumulate(&self.arg, arg_grad);
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
        let a = arg1.layout();
        let b = arg2.layout();
        assert!(a.ndim() >= 2 && b.ndim() >= 2, "matmul requires ndim >= 2");
        assert!(
            a.ndim() == b.ndim(),
            "matmul requires same number of dimensions, got {} and {}",
            a.ndim(),
            b.ndim()
        );
        let k1 = a.shape()[a.ndim() - 1];
        let k2 = b.shape()[b.ndim() - 2];
        assert!(k1 == k2, "matmul inner dimensions must match: {} vs {}", k1, k2);
        // Batch dimensions must match
        for i in 0..a.ndim() - 2 {
            assert!(
                a.shape()[i] == b.shape()[i],
                "matmul batch dimension {} mismatch: {} vs {}",
                i,
                a.shape()[i],
                b.shape()[i]
            );
        }
        // CPU backend passes strides directly to gemm, so no need to compact.
        // MPS backend requires compact inputs.
        let (arg1, arg2) = if matches!(arg1.device(), crate::Device::Cpu) {
            (arg1, arg2)
        } else {
            (arg1.compact(), arg2.compact())
        };
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for MatMul {
    fn forward(self) -> Result<Tensor> {
        let a_shape = self.arg1.layout().shape();
        let b_shape = self.arg2.layout().shape();
        let ndim = a_shape.ndim();
        let m = a_shape[ndim - 2];
        let n = b_shape[ndim - 1];
        let mut out_dims: Vec<usize> = (0..ndim - 2).map(|i| a_shape[i]).collect();
        out_dims.push(m);
        out_dims.push(n);
        let shape: Shape = out_dims.into();
        let storage = Arc::new(RwLock::new(self.arg1.storage().matmul(
            self.arg1.layout(),
            &self.arg2.storage(),
            self.arg2.layout(),
        )?));

        Ok(Tensor::new(storage, shape.into(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        // a = [..., m, k], b = [..., k, n], out = [..., m, n]
        // da = out_grad @ b^T   -> [..., m, k]
        // db = a^T @ out_grad   -> [..., k, n]

        let arg1_grad = out_grad.matmul(&self.arg2.transpose(None));
        grads.accumulate(&self.arg1, arg1_grad);

        let arg2_grad = self.arg1.transpose(None).matmul(out_grad);
        grads.accumulate(&self.arg2, arg2_grad);

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
    pub fn new(arg: Tensor, axes: Vec<usize>) -> Result<Self> {
        Ok(Self { arg, axes })
    }
}

impl TensorOp for LogSumExp {
    fn forward(self) -> Result<Tensor> {
        // https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        let max_z = self.arg.max(self.axes.clone(), true);
        let broadcast_max_z = max_z.broadcast(self.arg.layout().shape().clone());
        let sum_z = (&self.arg - &broadcast_max_z).exp().sum(self.axes.clone(), false);
        let logsumexp = max_z.reshape(sum_z.layout().shape.clone()) + sum_z.log();

        Ok(Tensor::new(
            logsumexp.storage_clone(),
            logsumexp.layout().clone(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let max_z = self.arg.max(self.axes.clone(), true);
        let broadcast_max_z = max_z.broadcast(self.arg.layout().shape().clone());
        let exp_z = (&self.arg - broadcast_max_z).exp();
        let sum_z = exp_z.sum(self.axes.clone(), false);

        let expand_shape = reduce_shape(self.arg.layout().shape(), &self.axes, true);
        let grad_sum_z =
            (out_grad / sum_z).reshape(expand_shape).broadcast(self.arg.layout().shape().clone());
        let arg_grad = grad_sum_z * exp_z;

        grads.accumulate(&self.arg, arg_grad);
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
    pub fn new(arg: Tensor) -> Result<Self> {
        Ok(Self { arg })
    }
}

impl TensorOp for Compact {
    fn forward(self) -> Result<Tensor> {
        let mut storage = match self.arg.device() {
            crate::Device::Mps => {
                Storage::Mps(MpsStorage::empty(self.arg.layout().size(), self.arg.dtype()))
            }
            _ => self.arg.device().zeros(self.arg.layout().size(), self.arg.dtype()),
        };
        self.arg.storage().copy_compact(self.arg.layout(), &mut storage)?;
        let strides = self.arg.layout().shape().compact_strides();
        let layout = Layout::new(self.arg.layout().shape().clone(), strides, 0);
        Ok(Tensor::new(Arc::new(RwLock::new(storage)), layout, false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        grads.accumulate(&self.arg, out_grad.clone());
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

/// Gathers values along `dim` using integer indices.
///
/// The index tensor must have the same rank as the input and the same shape on
/// every non-indexed dimension. The output shape matches the index tensor.
#[derive(Debug)]
pub struct Gather {
    arg: Tensor,
    dim: usize,
    indices: Tensor,
}

impl Gather {
    pub fn new(arg: Tensor, dim: usize, indices: Tensor) -> Result<Self> {
        if arg.layout().ndim() != indices.layout().ndim() {
            return Err(Error::LayoutMismatch(format!(
                "gather: arg has {} dims but indices has {}",
                arg.layout().ndim(),
                indices.layout().ndim()
            )));
        }
        if dim >= arg.layout().ndim() {
            return Err(Error::LayoutMismatch(format!(
                "gather: dim {} out of bounds for {} dims",
                dim,
                arg.layout().ndim()
            )));
        }
        for axis in 0..arg.layout().ndim() {
            if axis != dim && arg.layout().shape()[axis] != indices.layout().shape()[axis] {
                return Err(Error::LayoutMismatch(format!(
                    "gather: shape mismatch at dim {}: {} vs {}",
                    axis,
                    arg.layout().shape()[axis],
                    indices.layout().shape()[axis]
                )));
            }
        }
        if indices.dtype() != crate::DType::I64 {
            return Err(Error::DTypeMismatch("gather indices must be i64".into()));
        }
        let indices = indices.to_device(arg.device())?;
        Ok(Self { arg: arg.compact(), dim, indices: indices.compact() })
    }
}

impl TensorOp for Gather {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(self.arg.storage().gather(
            self.arg.layout(),
            self.dim,
            &self.indices.storage(),
            self.indices.layout(),
        )?));
        Ok(Tensor::new(
            storage,
            self.indices.layout().shape().clone().into(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let out_compact = out_grad.compact();
        let grad_storage = out_compact.storage().scatter_add(
            out_compact.layout(),
            self.dim,
            &self.indices.storage(),
            self.indices.layout(),
            &arg_shape,
        )?;
        let grad_tensor = Tensor::new(
            Arc::new(RwLock::new(grad_storage)),
            Shape::from(arg_shape).into(),
            false,
            None,
        );
        grads.accumulate(&self.arg, grad_tensor);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg, &self.indices]
    }
}

/// Selects slices along `dim` using a 1-D integer index tensor.
#[derive(Debug)]
pub struct IndexSelect {
    arg: Tensor,
    dim: usize,
    indices: Tensor,
}

impl IndexSelect {
    pub fn new(arg: Tensor, dim: usize, indices: Tensor) -> Result<Self> {
        if dim >= arg.layout().ndim() {
            return Err(Error::LayoutMismatch(format!(
                "index_select: dim {} out of bounds for {} dims",
                dim,
                arg.layout().ndim()
            )));
        }
        if indices.layout().ndim() != 1 {
            return Err(Error::LayoutMismatch("index_select requires 1D indices".into()));
        }
        if indices.dtype() != crate::DType::I64 {
            return Err(Error::DTypeMismatch("index_select indices must be i64".into()));
        }
        let indices = indices.to_device(arg.device())?;
        Ok(Self { arg: arg.compact(), dim, indices: indices.compact() })
    }
}

impl TensorOp for IndexSelect {
    fn forward(self) -> Result<Tensor> {
        let storage = Arc::new(RwLock::new(self.arg.storage().index_select(
            self.arg.layout(),
            self.dim,
            &self.indices.storage(),
            self.indices.layout(),
        )?));
        let mut shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        shape[self.dim] = self.indices.layout().shape()[0];
        Ok(Tensor::new(storage, Shape::from(shape).into(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let arg_shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let out_compact = out_grad.compact();
        let grad_storage = out_compact.storage().index_add(
            out_compact.layout(),
            self.dim,
            &self.indices.storage(),
            self.indices.layout(),
            &arg_shape,
        )?;
        let grad_tensor = Tensor::new(
            Arc::new(RwLock::new(grad_storage)),
            Shape::from(arg_shape).into(),
            false,
            None,
        );
        grads.accumulate(&self.arg, grad_tensor);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg, &self.indices]
    }
}

/// Concatenates tensors along dimension 0.
/// For other dimensions, the caller transposes before/after.
#[derive(Debug)]
pub struct Cat {
    args: Vec<Tensor>,
}

impl Cat {
    pub fn new(args: Vec<Tensor>) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::LayoutMismatch("cat requires at least one tensor".into()));
        }
        let ndim = args[0].layout().ndim();
        let dtype = args[0].dtype();
        for arg in &args[1..] {
            if arg.layout().ndim() != ndim {
                return Err(Error::LayoutMismatch(format!(
                    "cat: ndim mismatch, expected {} but got {}",
                    ndim,
                    arg.layout().ndim()
                )));
            }
            if arg.dtype() != dtype {
                return Err(Error::DTypeMismatch(format!(
                    "cat: expected {:?} but got {:?}",
                    dtype,
                    arg.dtype()
                )));
            }
            for d in 1..ndim {
                if arg.layout().shape()[d] != args[0].layout().shape()[d] {
                    return Err(Error::LayoutMismatch(format!(
                        "cat: dimension {} mismatch, expected {} but got {}",
                        d,
                        args[0].layout().shape()[d],
                        arg.layout().shape()[d]
                    )));
                }
            }
        }
        let args: Vec<Tensor> = args.into_iter().map(|a| a.compact()).collect();
        Ok(Self { args })
    }
}

impl TensorOp for Cat {
    fn forward(self) -> Result<Tensor> {
        let total_dim0: usize = self.args.iter().map(|a| a.layout().shape()[0]).sum();

        let mut out_dims: Vec<usize> = self.args[0].layout().shape().iter().copied().collect();
        out_dims[0] = total_dim0;
        let out_shape: Shape = out_dims.into();

        let storage = {
            let guards: Vec<_> = self.args.iter().map(|a| a.storage()).collect();
            let parts: Vec<(&Storage, usize)> = guards
                .iter()
                .zip(self.args.iter())
                .map(|(g, a)| (&**g, a.layout().size()))
                .collect();
            Storage::cat(&parts)?
        };

        Ok(Tensor::new(
            Arc::new(RwLock::new(storage)),
            out_shape.into(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let out_grad = out_grad.compact();
        let mut offset = 0;
        for arg in &self.args {
            let size = arg.layout().shape()[0];
            let grad_slice = out_grad.narrow(0, offset, size);
            grads.accumulate(arg, grad_slice);
            offset += size;
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        self.args.iter().collect()
    }
}
