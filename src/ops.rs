//! Differentiable tensor operations and the [`TensorOp`] trait.
//!
//! Each operation implements `forward`, `backward`, and `dependencies` so the
//! autograd engine can build and traverse the computation graph.

#![allow(dead_code)]

use std::sync::{Arc, RwLock};
use std::{cmp::Ordering, fmt, iter};

use half::f16;

use crate::backprop::GradientStore;
use crate::error::{Error, Result};
use crate::layout::{Layout, Shape};
use crate::profiler;
use crate::storage::{self, BackendStorage, MpsStorage, ReduceMax, ReduceSum, Storage};
use crate::tensor::Tensor;

fn allocated_bytes(elements: usize, dtype: crate::DType) -> usize {
    elements * dtype.size_in_bytes()
}

fn profile_output(
    name: &'static str,
    inputs: &[&Tensor],
    elements: usize,
    dtype: crate::DType,
) -> Option<profiler::ProfileScope> {
    profiler::scope(name, inputs, allocated_bytes(elements, dtype))
}

fn profile_like(name: &'static str, arg: &Tensor) -> Option<profiler::ProfileScope> {
    profile_output(name, &[arg], arg.layout().size(), arg.dtype())
}

fn profile_like_binary(
    name: &'static str,
    arg1: &Tensor,
    arg2: &Tensor,
) -> Option<profiler::ProfileScope> {
    profile_output(name, &[arg1, arg2], arg1.layout().size(), arg1.dtype())
}

fn profile_view(name: &'static str, inputs: &[&Tensor]) -> Option<profiler::ProfileScope> {
    profiler::scope(name, inputs, 0)
}

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
        let _profile = profile_like("neg", &self.arg);
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
        if arg1.layout().shape() != arg2.layout().shape() {
            return Err(Error::LayoutMismatch(format!(
                "add: shape {:?} does not match {:?}",
                arg1.layout().shape(),
                arg2.layout().shape()
            )));
        }
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseAdd {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like_binary("add", &self.arg1, &self.arg2);
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
        if arg1.layout().shape() != arg2.layout().shape() {
            return Err(Error::LayoutMismatch(format!(
                "sub: shape {:?} does not match {:?}",
                arg1.layout().shape(),
                arg2.layout().shape()
            )));
        }
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseSub {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like_binary("sub", &self.arg1, &self.arg2);
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
        if arg1.layout().shape() != arg2.layout().shape() {
            return Err(Error::LayoutMismatch(format!(
                "mul: shape {:?} does not match {:?}",
                arg1.layout().shape(),
                arg2.layout().shape()
            )));
        }
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseMul {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like_binary("mul", &self.arg1, &self.arg2);
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
        if arg1.layout().shape() != arg2.layout().shape() {
            return Err(Error::LayoutMismatch(format!(
                "div: shape {:?} does not match {:?}",
                arg1.layout().shape(),
                arg2.layout().shape()
            )));
        }
        Ok(Self { arg1, arg2 })
    }
}

impl TensorOp for EWiseDiv {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like_binary("div", &self.arg1, &self.arg2);
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
        if arg.layout().shape() != e.layout().shape() {
            return Err(Error::LayoutMismatch(format!(
                "pow: shape {:?} does not match {:?}",
                arg.layout().shape(),
                e.layout().shape()
            )));
        }
        Ok(Self { arg, e })
    }
}

impl TensorOp for EWisePowf {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like_binary("pow", &self.arg, &self.e);
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
        let _profile = profile_like("log", &self.arg);
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
        let _profile = profile_like("exp", &self.arg);
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
        let _profile = profile_like("sin", &self.arg);
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
        let _profile = profile_like("cos", &self.arg);
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
        let _profile = profile_like("tanh", &self.arg);
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
        let _profile = profile_like("relu", &self.arg);
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
        let _profile = profile_like("scalar_add", &self.arg);
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
        let _profile = profile_like("scalar_mul", &self.arg);
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
        let _profile = profile_like("scalar_powf", &self.arg);
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
        let _profile = profile_view("permute", &[&self.arg]);
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
        let _profile = profile_view("broadcast", &[&self.arg]);
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
        let _profile = profile_output("sum", &[&self.arg], new_shape.size(), self.arg.dtype());
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
        let _profile = profile_output("max", &[&self.arg], new_shape.size(), self.arg.dtype());
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
        let _profile = profile_view("reshape", &[&self.arg]);
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
        let _profile = profile_view("narrow", &[&self.arg]);
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
        if a.ndim() < 2 || b.ndim() < 2 {
            return Err(Error::LayoutMismatch(format!(
                "matmul requires ndim >= 2, got {} and {}",
                a.ndim(),
                b.ndim()
            )));
        }
        if a.ndim() != b.ndim() {
            return Err(Error::LayoutMismatch(format!(
                "matmul requires same number of dimensions, got {} and {}",
                a.ndim(),
                b.ndim()
            )));
        }
        let k1 = a.shape()[a.ndim() - 1];
        let k2 = b.shape()[b.ndim() - 2];
        if k1 != k2 {
            return Err(Error::LayoutMismatch(format!(
                "matmul inner dimensions must match: {} vs {}",
                k1, k2
            )));
        }
        // Batch dimensions must match
        for i in 0..a.ndim() - 2 {
            if a.shape()[i] != b.shape()[i] {
                return Err(Error::LayoutMismatch(format!(
                    "matmul batch dimension {} mismatch: {} vs {}",
                    i,
                    a.shape()[i],
                    b.shape()[i]
                )));
            }
        }
        // CPU and CUDA backends handle non-compact layouts directly (CPU passes strides to gemm;
        // CUDA uses try_gemm_params to avoid copies for transposed inputs).
        // MPS requires compact inputs.
        let (arg1, arg2) = if matches!(arg1.device(), crate::Device::Mps) {
            (arg1.compact(), arg2.compact())
        } else {
            (arg1, arg2)
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
        let _profile =
            profile_output("matmul", &[&self.arg1, &self.arg2], shape.size(), self.arg1.dtype());
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
    /// Cached forward result for use in backward.
    lse: Option<Tensor>,
}

impl LogSumExp {
    pub fn new(arg: Tensor, axes: Vec<usize>) -> Result<Self> {
        Ok(Self { arg, axes, lse: None })
    }

    /// Whether the fused kernel path can be used: single last-axis reduce on a compact layout.
    fn can_fuse(&self) -> bool {
        self.axes.len() == 1
            && self.axes[0] == self.arg.layout().ndim() - 1
            && self.arg.layout().is_compact()
    }
}

impl TensorOp for LogSumExp {
    fn forward(mut self) -> Result<Tensor> {
        let _profile = profile_view("log_sum_exp", &[&self.arg]);

        let logsumexp = if self.can_fuse() {
            // Fused single-kernel path: reduces last axis in one pass.
            let axis = self.axes[0];
            let reduce_size = self.arg.layout().shape()[axis];
            let outer_size = self.arg.layout().size() / reduce_size;
            let out_storage =
                self.arg.storage().log_sum_exp(self.arg.layout(), outer_size, reduce_size)?;
            let out_shape = reduce_shape(self.arg.layout().shape(), &self.axes, false);
            let storage = Arc::new(RwLock::new(out_storage));
            Tensor::new(storage, Layout::from(out_shape), false, None)
        } else {
            // Decomposed fallback: max, sub, exp, sum, log, add.
            let max_z = self.arg.max(self.axes.clone(), true);
            let broadcast_max = max_z.broadcast(self.arg.layout().shape().clone());
            let exp_z = (&self.arg - &broadcast_max).exp();
            let sum_z = exp_z.sum(self.axes.clone(), false);
            &max_z.reshape(sum_z.layout().shape.clone()) + &sum_z.log()
        };

        self.lse = Some(logsumexp.clone());

        Ok(Tensor::new(
            logsumexp.storage_clone(),
            logsumexp.layout().clone(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        // d/dx logsumexp = exp(x - logsumexp) = softmax(x)
        let lse = self.lse.as_ref().expect("forward must run before backward");
        let expand_shape = reduce_shape(self.arg.layout().shape(), &self.axes, true);
        let lse_broadcast = lse.reshape(expand_shape).broadcast(self.arg.layout().shape().clone());
        let softmax = (&self.arg - &lse_broadcast).exp();
        let out_grad_broadcast = out_grad
            .reshape(reduce_shape(self.arg.layout().shape(), &self.axes, true))
            .broadcast(self.arg.layout().shape().clone());
        let arg_grad = &softmax * &out_grad_broadcast;

        grads.accumulate(&self.arg, arg_grad);
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

/// Fused log-softmax: `x[i] - log(sum_j exp(x[j]))` per row, single kernel on CUDA.
///
/// The forward saves the output for use in the backward pass to avoid recomputing softmax.
#[derive(Debug)]
pub struct FusedLogSoftmax {
    arg: Tensor,
    axis: usize,
    /// Saved output (log-softmax result) for backward.
    lsm_output: Option<Tensor>,
}

impl FusedLogSoftmax {
    pub fn new(arg: Tensor, axis: usize) -> Result<Self> {
        Ok(Self { arg, axis, lsm_output: None })
    }
}

impl TensorOp for FusedLogSoftmax {
    fn forward(mut self) -> Result<Tensor> {
        let _profile = profile_like("log_softmax", &self.arg);
        let axis = self.axis;
        let inner_size = self.arg.layout().shape()[axis];
        let outer_size = self.arg.layout().size() / inner_size;
        let compact = self.arg.compact();
        let out_storage =
            compact.storage().log_softmax_fwd(compact.layout(), outer_size, inner_size)?;
        let output = Tensor::new(
            Arc::new(RwLock::new(out_storage)),
            self.arg.layout().clone(),
            false,
            None,
        );
        self.lsm_output = Some(output.clone());
        Ok(Tensor::new(
            output.storage_clone(),
            output.layout().clone(),
            false,
            Some(Box::new(self)),
        ))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let lsm = self.lsm_output.as_ref().expect("forward must run before backward");
        let axis = self.axis;
        let inner_size = self.arg.layout().shape()[axis];
        let outer_size = self.arg.layout().size() / inner_size;
        // Compact both inputs so the fused backward kernel sees contiguous layouts.
        let grad_c = out_grad.compact();
        let lsm_c = lsm.compact();
        let grad_storage = grad_c.storage().log_softmax_bwd(
            grad_c.layout(),
            &lsm_c.storage(),
            lsm_c.layout(),
            outer_size,
            inner_size,
        )?;
        let arg_grad = Tensor::new(
            Arc::new(RwLock::new(grad_storage)),
            self.arg.layout().clone(),
            false,
            None,
        );
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
        let _profile = profile_like("compact", &self.arg);
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
        let _profile = profile_output(
            "gather",
            &[&self.arg, &self.indices],
            self.indices.layout().size(),
            self.arg.dtype(),
        );
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
        let mut shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        shape[self.dim] = self.indices.layout().shape()[0];
        let out_shape = Shape::from(shape.clone());
        let _profile = profile_output(
            "index_select",
            &[&self.arg, &self.indices],
            out_shape.size(),
            self.arg.dtype(),
        );
        let storage = Arc::new(RwLock::new(self.arg.storage().index_select(
            self.arg.layout(),
            self.dim,
            &self.indices.storage(),
            self.indices.layout(),
        )?));
        Ok(Tensor::new(storage, out_shape.into(), false, Some(Box::new(self))))
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
        let inputs: Vec<&Tensor> = self.args.iter().collect();
        let _profile = profile_output("cat", &inputs, out_shape.size(), self.args[0].dtype());

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

fn split_dim(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, size, inner)
}

fn check_select_dim(ndim: usize, dim: usize, op: &str) -> Result<()> {
    if dim >= ndim {
        return Err(Error::LayoutMismatch(format!(
            "{op}: dim {dim} out of bounds for {ndim} dims"
        )));
    }
    Ok(())
}

fn cond_mask(cond: &Tensor) -> Result<Vec<bool>> {
    match cond.dtype() {
        crate::DType::F16 => Ok(cond.to_vec::<f16>()?.iter().map(|v| v.to_f32() != 0.0).collect()),
        crate::DType::F32 => Ok(cond.to_vec::<f32>()?.iter().map(|v| *v != 0.0).collect()),
        crate::DType::I64 => Ok(cond.to_vec::<i64>()?.iter().map(|v| *v != 0).collect()),
    }
}

fn check_same_shape(a: &Tensor, b: &Tensor, op: &str) -> Result<()> {
    if a.layout().shape() != b.layout().shape() {
        return Err(Error::LayoutMismatch(format!(
            "{op}: shape {:?} does not match {:?}",
            a.layout().shape(),
            b.layout().shape()
        )));
    }
    Ok(())
}

pub fn argmax_forward(arg: &Tensor, dim: usize, keep_dims: bool) -> Result<Tensor> {
    check_select_dim(arg.layout().ndim(), dim, "argmax")?;
    let shape: Vec<usize> = arg.layout().shape().iter().copied().collect();
    let (outer, dim_size, inner) = split_dim(&shape, dim);
    assert!(dim_size > 0, "argmax requires a non-empty dimension");
    let _profile = profile_view("argmax", &[arg]);

    let mut out_shape = shape.clone();
    if keep_dims {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }
    let device = arg.device();
    let out: Tensor = match arg.dtype() {
        crate::DType::F32 => {
            let vals = arg.to_vec::<f32>()?;
            let mut idx = Vec::with_capacity(outer * inner);
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut best = 0;
                    for i in 1..dim_size {
                        if vals[base + i * inner] > vals[base + best * inner] {
                            best = i;
                        }
                    }
                    idx.push(best as i64);
                }
            }
            Tensor::from_vec(idx, out_shape, device)
        }
        crate::DType::F16 => {
            let vals = arg.to_vec::<f16>()?;
            let mut idx = Vec::with_capacity(outer * inner);
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut best = 0;
                    for i in 1..dim_size {
                        if vals[base + i * inner].to_f32() > vals[base + best * inner].to_f32() {
                            best = i;
                        }
                    }
                    idx.push(best as i64);
                }
            }
            Tensor::from_vec(idx, out_shape, device)
        }
        crate::DType::I64 => {
            let vals = arg.to_vec::<i64>()?;
            let mut idx = Vec::with_capacity(outer * inner);
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut best = 0;
                    for i in 1..dim_size {
                        if vals[base + i * inner] > vals[base + best * inner] {
                            best = i;
                        }
                    }
                    idx.push(best as i64);
                }
            }
            Tensor::from_vec(idx, out_shape, device)
        }
    };
    Ok(out)
}

fn topk_positions_f32(
    vals: &[f32],
    outer: usize,
    dim_size: usize,
    inner: usize,
    k: usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(outer * k * inner);
    for o in 0..outer {
        for j in 0..inner {
            let base = (o * dim_size) * inner + j;
            let mut order: Vec<usize> = (0..dim_size).collect();
            order.sort_by(|&a, &b| {
                vals[base + b * inner]
                    .partial_cmp(&vals[base + a * inner])
                    .unwrap_or(Ordering::Greater)
            });
            for &i in order.iter().take(k) {
                out.push(base + i * inner);
            }
        }
    }
    out
}

pub fn topk_forward(arg: &Tensor, k: usize, dim: usize) -> Result<(Tensor, Tensor)> {
    check_select_dim(arg.layout().ndim(), dim, "topk")?;
    let shape: Vec<usize> = arg.layout().shape().iter().copied().collect();
    let (outer, dim_size, inner) = split_dim(&shape, dim);
    if k == 0 || k > dim_size {
        return Err(Error::LayoutMismatch(format!(
            "topk: k {k} out of bounds for dim size {dim_size}"
        )));
    }
    let inputs: Vec<&Tensor> = vec![arg];
    let _profile = profile_output("topk", &inputs, arg.layout().size(), arg.dtype());

    let mut out_shape = shape.clone();
    out_shape[dim] = k;
    let device = arg.device();
    match arg.dtype() {
        crate::DType::F32 => {
            let vals = arg.to_vec::<f32>()?;
            let pos = topk_positions_f32(&vals, outer, dim_size, inner, k);
            let values: Vec<f32> = pos.iter().map(|&p| vals[p]).collect();
            let mut idx = Vec::with_capacity(pos.len());
            for &p in &pos {
                let in_slice = p % (dim_size * inner);
                idx.push((in_slice / inner) as i64);
            }
            Ok((
                Tensor::from_vec(values, out_shape.clone(), device),
                Tensor::from_vec(idx, out_shape, device),
            ))
        }
        crate::DType::F16 => {
            let vals = arg.to_vec::<f16>()?;
            let as_f32: Vec<f32> = vals.iter().map(|v| v.to_f32()).collect();
            let pos = topk_positions_f32(&as_f32, outer, dim_size, inner, k);
            let values: Vec<f16> = pos.iter().map(|&p| vals[p]).collect();
            let mut idx = Vec::with_capacity(pos.len());
            for &p in &pos {
                let in_slice = p % (dim_size * inner);
                idx.push((in_slice / inner) as i64);
            }
            Ok((
                Tensor::from_vec(values, out_shape.clone(), device),
                Tensor::from_vec(idx, out_shape, device),
            ))
        }
        crate::DType::I64 => {
            let vals = arg.to_vec::<i64>()?;
            let mut pos = Vec::with_capacity(outer * k * inner);
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut order: Vec<usize> = (0..dim_size).collect();
                    order.sort_by(|&a, &b| vals[base + b * inner].cmp(&vals[base + a * inner]));
                    for &i in order.iter().take(k) {
                        pos.push(base + i * inner);
                    }
                }
            }
            let values: Vec<i64> = pos.iter().map(|&p| vals[p]).collect();
            let mut idx = Vec::with_capacity(pos.len());
            for &p in &pos {
                let in_slice = p % (dim_size * inner);
                idx.push((in_slice / inner) as i64);
            }
            Ok((
                Tensor::from_vec(values, out_shape.clone(), device),
                Tensor::from_vec(idx, out_shape, device),
            ))
        }
    }
}

pub fn sort_forward(arg: &Tensor, dim: usize, descending: bool) -> Result<(Tensor, Tensor)> {
    check_select_dim(arg.layout().ndim(), dim, "sort")?;
    let shape: Vec<usize> = arg.layout().shape().iter().copied().collect();
    let (outer, dim_size, inner) = split_dim(&shape, dim);
    let inputs: Vec<&Tensor> = vec![arg];
    let _profile = profile_output("sort", &inputs, arg.layout().size(), arg.dtype());

    let device = arg.device();
    match arg.dtype() {
        crate::DType::F32 => {
            let vals = arg.to_vec::<f32>()?;
            let mut values = vec![0.0f32; vals.len()];
            let mut idx = vec![0i64; vals.len()];
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut order: Vec<usize> = (0..dim_size).collect();
                    if descending {
                        order.sort_by(|&a, &b| {
                            vals[base + b * inner]
                                .partial_cmp(&vals[base + a * inner])
                                .unwrap_or(Ordering::Greater)
                        });
                    } else {
                        order.sort_by(|&a, &b| {
                            vals[base + a * inner]
                                .partial_cmp(&vals[base + b * inner])
                                .unwrap_or(Ordering::Greater)
                        });
                    }
                    for (rank, &i) in order.iter().enumerate() {
                        values[base + rank * inner] = vals[base + i * inner];
                        idx[base + rank * inner] = i as i64;
                    }
                }
            }
            Ok((
                Tensor::from_vec(values, shape.clone(), device),
                Tensor::from_vec(idx, shape, device),
            ))
        }
        crate::DType::F16 => {
            let vals = arg.to_vec::<f16>()?;
            let as_f32: Vec<f32> = vals.iter().map(|v| v.to_f32()).collect();
            let mut values = vec![f16::from_f32(0.0); vals.len()];
            let mut idx = vec![0i64; vals.len()];
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut order: Vec<usize> = (0..dim_size).collect();
                    if descending {
                        order.sort_by(|&a, &b| {
                            as_f32[base + b * inner]
                                .partial_cmp(&as_f32[base + a * inner])
                                .unwrap_or(Ordering::Greater)
                        });
                    } else {
                        order.sort_by(|&a, &b| {
                            as_f32[base + a * inner]
                                .partial_cmp(&as_f32[base + b * inner])
                                .unwrap_or(Ordering::Greater)
                        });
                    }
                    for (rank, &i) in order.iter().enumerate() {
                        values[base + rank * inner] = vals[base + i * inner];
                        idx[base + rank * inner] = i as i64;
                    }
                }
            }
            Ok((
                Tensor::from_vec(values, shape.clone(), device),
                Tensor::from_vec(idx, shape, device),
            ))
        }
        crate::DType::I64 => {
            let vals = arg.to_vec::<i64>()?;
            let mut values = vec![0i64; vals.len()];
            let mut idx = vec![0i64; vals.len()];
            for o in 0..outer {
                for j in 0..inner {
                    let base = (o * dim_size) * inner + j;
                    let mut order: Vec<usize> = (0..dim_size).collect();
                    if descending {
                        order.sort_by(|&a, &b| vals[base + b * inner].cmp(&vals[base + a * inner]));
                    } else {
                        order.sort_by(|&a, &b| vals[base + a * inner].cmp(&vals[base + b * inner]));
                    }
                    for (rank, &i) in order.iter().enumerate() {
                        values[base + rank * inner] = vals[base + i * inner];
                        idx[base + rank * inner] = i as i64;
                    }
                }
            }
            Ok((
                Tensor::from_vec(values, shape.clone(), device),
                Tensor::from_vec(idx, shape, device),
            ))
        }
    }
}

#[derive(Debug)]
pub struct Clamp {
    arg: Tensor,
    min: f64,
    max: f64,
}

impl Clamp {
    pub fn new(arg: Tensor, min: f64, max: f64) -> Result<Self> {
        if min > max {
            return Err(Error::LayoutMismatch(format!("clamp: min {min} exceeds max {max}")));
        }
        Ok(Self { arg, min, max })
    }
}

impl TensorOp for Clamp {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like("clamp", &self.arg);
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let out = match self.arg.dtype() {
            crate::DType::F32 => {
                let vals = self.arg.to_vec::<f32>()?;
                let (lo, hi) = (self.min as f32, self.max as f32);
                Tensor::from_vec(
                    vals.iter().map(|v| v.clamp(lo, hi)).collect::<Vec<f32>>(),
                    shape,
                    device,
                )
            }
            crate::DType::F16 => {
                let vals = self.arg.to_vec::<f16>()?;
                let (lo, hi) = (self.min as f32, self.max as f32);
                Tensor::from_vec(
                    vals.iter()
                        .map(|v| f16::from_f32(v.to_f32().clamp(lo, hi)))
                        .collect::<Vec<f16>>(),
                    shape,
                    device,
                )
            }
            crate::DType::I64 => {
                let vals = self.arg.to_vec::<i64>()?;
                let (lo, hi) = (self.min as i64, self.max as i64);
                Tensor::from_vec(
                    vals.iter().map(|v| (*v).clamp(lo, hi)).collect::<Vec<i64>>(),
                    shape,
                    device,
                )
            }
        };
        Ok(Tensor::new(out.storage_clone(), out.layout().clone(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        match self.arg.dtype() {
            crate::DType::F32 => {
                let vals = self.arg.to_vec::<f32>()?;
                let go = out_grad.to_vec::<f32>()?;
                let (lo, hi) = (self.min as f32, self.max as f32);
                let grad: Vec<f32> = vals
                    .iter()
                    .zip(go.iter())
                    .map(|(v, g)| if *v >= lo && *v <= hi { *g } else { 0.0 })
                    .collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::F16 => {
                let vals = self.arg.to_vec::<f16>()?;
                let go = out_grad.to_vec::<f16>()?;
                let (lo, hi) = (self.min as f32, self.max as f32);
                let grad: Vec<f16> =
                    vals.iter()
                        .zip(go.iter())
                        .map(|(v, g)| {
                            if v.to_f32() >= lo && v.to_f32() <= hi {
                                *g
                            } else {
                                f16::from_f32(0.0)
                            }
                        })
                        .collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::I64 => {
                let vals = self.arg.to_vec::<i64>()?;
                let go = out_grad.to_vec::<i64>()?;
                let (lo, hi) = (self.min as i64, self.max as i64);
                let grad: Vec<i64> = vals
                    .iter()
                    .zip(go.iter())
                    .map(|(v, g)| if *v >= lo && *v <= hi { *g } else { 0 })
                    .collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct WhereCond {
    cond: Tensor,
    on_true: Tensor,
    on_false: Tensor,
}

impl WhereCond {
    pub fn new(cond: Tensor, on_true: Tensor, on_false: Tensor) -> Result<Self> {
        check_same_shape(&cond, &on_true, "where")?;
        check_same_shape(&cond, &on_false, "where")?;
        if on_true.dtype() != on_false.dtype() {
            return Err(Error::DTypeMismatch(format!(
                "where: {:?} vs {:?}",
                on_true.dtype(),
                on_false.dtype()
            )));
        }
        Ok(Self { cond, on_true, on_false })
    }
}

impl TensorOp for WhereCond {
    fn forward(self) -> Result<Tensor> {
        let inputs: Vec<&Tensor> = vec![&self.cond, &self.on_true, &self.on_false];
        let _profile =
            profile_output("where", &inputs, self.on_true.layout().size(), self.on_true.dtype());
        let shape: Vec<usize> = self.on_true.layout().shape().iter().copied().collect();
        let device = self.on_true.device();
        let mask = cond_mask(&self.cond)?;
        let out = match self.on_true.dtype() {
            crate::DType::F32 => {
                let t = self.on_true.to_vec::<f32>()?;
                let f = self.on_false.to_vec::<f32>()?;
                Tensor::from_vec(
                    mask.iter()
                        .enumerate()
                        .map(|(i, m)| if *m { t[i] } else { f[i] })
                        .collect::<Vec<f32>>(),
                    shape,
                    device,
                )
            }
            crate::DType::F16 => {
                let t = self.on_true.to_vec::<f16>()?;
                let f = self.on_false.to_vec::<f16>()?;
                Tensor::from_vec(
                    mask.iter()
                        .enumerate()
                        .map(|(i, m)| if *m { t[i] } else { f[i] })
                        .collect::<Vec<f16>>(),
                    shape,
                    device,
                )
            }
            crate::DType::I64 => {
                let t = self.on_true.to_vec::<i64>()?;
                let f = self.on_false.to_vec::<i64>()?;
                Tensor::from_vec(
                    mask.iter()
                        .enumerate()
                        .map(|(i, m)| if *m { t[i] } else { f[i] })
                        .collect::<Vec<i64>>(),
                    shape,
                    device,
                )
            }
        };
        Ok(Tensor::new(out.storage_clone(), out.layout().clone(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape: Vec<usize> = self.on_true.layout().shape().iter().copied().collect();
        let mask = cond_mask(&self.cond)?;
        match self.on_true.dtype() {
            crate::DType::F32 => {
                let go = out_grad.to_vec::<f32>()?;
                let zero = 0.0f32;
                let gt: Vec<f32> =
                    mask.iter().enumerate().map(|(i, m)| if *m { go[i] } else { zero }).collect();
                let gf: Vec<f32> =
                    mask.iter().enumerate().map(|(i, m)| if *m { zero } else { go[i] }).collect();
                let device = self.on_true.device();
                grads.accumulate(&self.on_true, Tensor::from_vec(gt, shape.clone(), device));
                let device = self.on_false.device();
                grads.accumulate(&self.on_false, Tensor::from_vec(gf, shape, device));
            }
            crate::DType::F16 => {
                let go = out_grad.to_vec::<f16>()?;
                let zero = f16::from_f32(0.0);
                let gt: Vec<f16> =
                    mask.iter().enumerate().map(|(i, m)| if *m { go[i] } else { zero }).collect();
                let gf: Vec<f16> =
                    mask.iter().enumerate().map(|(i, m)| if *m { zero } else { go[i] }).collect();
                let device = self.on_true.device();
                grads.accumulate(&self.on_true, Tensor::from_vec(gt, shape.clone(), device));
                let device = self.on_false.device();
                grads.accumulate(&self.on_false, Tensor::from_vec(gf, shape, device));
            }
            crate::DType::I64 => {
                let go = out_grad.to_vec::<i64>()?;
                let gt: Vec<i64> =
                    mask.iter().enumerate().map(|(i, m)| if *m { go[i] } else { 0 }).collect();
                let gf: Vec<i64> =
                    mask.iter().enumerate().map(|(i, m)| if *m { 0 } else { go[i] }).collect();
                let device = self.on_true.device();
                grads.accumulate(&self.on_true, Tensor::from_vec(gt, shape.clone(), device));
                let device = self.on_false.device();
                grads.accumulate(&self.on_false, Tensor::from_vec(gf, shape, device));
            }
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.cond, &self.on_true, &self.on_false]
    }
}

#[derive(Debug)]
pub struct MaskedFill {
    arg: Tensor,
    mask: Tensor,
    value: f64,
}

impl MaskedFill {
    pub fn new(arg: Tensor, mask: Tensor, value: f64) -> Result<Self> {
        check_same_shape(&arg, &mask, "masked_fill")?;
        Ok(Self { arg, mask, value })
    }
}

impl TensorOp for MaskedFill {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like("masked_fill", &self.arg);
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let mask = cond_mask(&self.mask)?;
        let out = match self.arg.dtype() {
            crate::DType::F32 => {
                let vals = self.arg.to_vec::<f32>()?;
                let v = self.value as f32;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, x)| if mask[i] { v } else { *x })
                        .collect::<Vec<f32>>(),
                    shape,
                    device,
                )
            }
            crate::DType::F16 => {
                let vals = self.arg.to_vec::<f16>()?;
                let v = f16::from_f32(self.value as f32);
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, x)| if mask[i] { v } else { *x })
                        .collect::<Vec<f16>>(),
                    shape,
                    device,
                )
            }
            crate::DType::I64 => {
                let vals = self.arg.to_vec::<i64>()?;
                let v = self.value as i64;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, x)| if mask[i] { v } else { *x })
                        .collect::<Vec<i64>>(),
                    shape,
                    device,
                )
            }
        };
        Ok(Tensor::new(out.storage_clone(), out.layout().clone(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let mask = cond_mask(&self.mask)?;
        match self.arg.dtype() {
            crate::DType::F32 => {
                let go = out_grad.to_vec::<f32>()?;
                let grad: Vec<f32> =
                    go.iter().enumerate().map(|(i, g)| if mask[i] { 0.0 } else { *g }).collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::F16 => {
                let go = out_grad.to_vec::<f16>()?;
                let grad: Vec<f16> = go
                    .iter()
                    .enumerate()
                    .map(|(i, g)| if mask[i] { f16::from_f32(0.0) } else { *g })
                    .collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::I64 => {
                let go = out_grad.to_vec::<i64>()?;
                let grad: Vec<i64> =
                    go.iter().enumerate().map(|(i, g)| if mask[i] { 0 } else { *g }).collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg, &self.mask]
    }
}

fn tri_keep_mask(shape: &[usize], diagonal: i32, lower: bool) -> Vec<bool> {
    let ndim = shape.len();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let batch: usize = shape[..ndim - 2].iter().product();
    let mut mask = vec![false; batch * rows * cols];
    for b in 0..batch {
        for r in 0..rows {
            for c in 0..cols {
                let keep = if lower {
                    (c as i64) <= (r as i64) + (diagonal as i64)
                } else {
                    (c as i64) >= (r as i64) + (diagonal as i64)
                };
                mask[(b * rows + r) * cols + c] = keep;
            }
        }
    }
    mask
}

#[derive(Debug)]
pub struct Tril {
    arg: Tensor,
    diagonal: i32,
}

impl Tril {
    pub fn new(arg: Tensor, diagonal: i32) -> Result<Self> {
        if arg.layout().ndim() < 2 {
            return Err(Error::LayoutMismatch("tril requires ndim >= 2".into()));
        }
        Ok(Self { arg, diagonal })
    }
}

impl TensorOp for Tril {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like("tril", &self.arg);
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let mask = tri_keep_mask(&shape, self.diagonal, true);
        let out = match self.arg.dtype() {
            crate::DType::F32 => {
                let vals = self.arg.to_vec::<f32>()?;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, v)| if mask[i] { *v } else { 0.0 })
                        .collect::<Vec<f32>>(),
                    shape,
                    device,
                )
            }
            crate::DType::F16 => {
                let vals = self.arg.to_vec::<f16>()?;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, v)| if mask[i] { *v } else { f16::from_f32(0.0) })
                        .collect::<Vec<f16>>(),
                    shape,
                    device,
                )
            }
            crate::DType::I64 => {
                let vals = self.arg.to_vec::<i64>()?;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, v)| if mask[i] { *v } else { 0 })
                        .collect::<Vec<i64>>(),
                    shape,
                    device,
                )
            }
        };
        Ok(Tensor::new(out.storage_clone(), out.layout().clone(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let mask = tri_keep_mask(&shape, self.diagonal, true);
        match self.arg.dtype() {
            crate::DType::F32 => {
                let go = out_grad.to_vec::<f32>()?;
                let grad: Vec<f32> =
                    go.iter().enumerate().map(|(i, g)| if mask[i] { *g } else { 0.0 }).collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::F16 => {
                let go = out_grad.to_vec::<f16>()?;
                let grad: Vec<f16> = go
                    .iter()
                    .enumerate()
                    .map(|(i, g)| if mask[i] { *g } else { f16::from_f32(0.0) })
                    .collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::I64 => {
                let go = out_grad.to_vec::<i64>()?;
                let grad: Vec<i64> =
                    go.iter().enumerate().map(|(i, g)| if mask[i] { *g } else { 0 }).collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}

#[derive(Debug)]
pub struct Triu {
    arg: Tensor,
    diagonal: i32,
}

impl Triu {
    pub fn new(arg: Tensor, diagonal: i32) -> Result<Self> {
        if arg.layout().ndim() < 2 {
            return Err(Error::LayoutMismatch("triu requires ndim >= 2".into()));
        }
        Ok(Self { arg, diagonal })
    }
}

impl TensorOp for Triu {
    fn forward(self) -> Result<Tensor> {
        let _profile = profile_like("triu", &self.arg);
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let mask = tri_keep_mask(&shape, self.diagonal, false);
        let out = match self.arg.dtype() {
            crate::DType::F32 => {
                let vals = self.arg.to_vec::<f32>()?;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, v)| if mask[i] { *v } else { 0.0 })
                        .collect::<Vec<f32>>(),
                    shape,
                    device,
                )
            }
            crate::DType::F16 => {
                let vals = self.arg.to_vec::<f16>()?;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, v)| if mask[i] { *v } else { f16::from_f32(0.0) })
                        .collect::<Vec<f16>>(),
                    shape,
                    device,
                )
            }
            crate::DType::I64 => {
                let vals = self.arg.to_vec::<i64>()?;
                Tensor::from_vec(
                    vals.iter()
                        .enumerate()
                        .map(|(i, v)| if mask[i] { *v } else { 0 })
                        .collect::<Vec<i64>>(),
                    shape,
                    device,
                )
            }
        };
        Ok(Tensor::new(out.storage_clone(), out.layout().clone(), false, Some(Box::new(self))))
    }

    fn backward(&self, grads: &mut GradientStore, out_grad: &Tensor) -> Result<()> {
        let shape: Vec<usize> = self.arg.layout().shape().iter().copied().collect();
        let device = self.arg.device();
        let mask = tri_keep_mask(&shape, self.diagonal, false);
        match self.arg.dtype() {
            crate::DType::F32 => {
                let go = out_grad.to_vec::<f32>()?;
                let grad: Vec<f32> =
                    go.iter().enumerate().map(|(i, g)| if mask[i] { *g } else { 0.0 }).collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::F16 => {
                let go = out_grad.to_vec::<f16>()?;
                let grad: Vec<f16> = go
                    .iter()
                    .enumerate()
                    .map(|(i, g)| if mask[i] { *g } else { f16::from_f32(0.0) })
                    .collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
            crate::DType::I64 => {
                let go = out_grad.to_vec::<i64>()?;
                let grad: Vec<i64> =
                    go.iter().enumerate().map(|(i, g)| if mask[i] { *g } else { 0 }).collect();
                grads.accumulate(&self.arg, Tensor::from_vec(grad, shape, device));
            }
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<&Tensor> {
        vec![&self.arg]
    }
}
