#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use crate::error::Result;
use crate::tensor::{Tensor, TensorId};

#[derive(Debug)]
pub enum BackpropOp {
    Neg(Tensor),
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
}

impl BackpropOp {
    fn dependencies(&self) -> Vec<&Tensor> {
        match self {
            BackpropOp::Neg(node) => vec![node],
            BackpropOp::Add(left, right)
            | BackpropOp::Sub(left, right)
            | BackpropOp::Mul(left, right) => vec![left, right],
        }
    }
}

impl Tensor {
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        let mut sorted_nodes = vec![];
        let mut queue = VecDeque::new();
        queue.push_front(self);
        sorted_nodes.push(self);

        while let Some(node) = queue.pop_back() {
            if let Some(op) = node.op() {
                for dep in op.dependencies() {
                    sorted_nodes.push(dep);
                    queue.push_front(dep);
                }
            }
        }

        sorted_nodes
    }

    pub fn backward(&self) -> Result<GradientStore> {
        let mut store = GradientStore::new();
        store.get_or_insert_with(self.id(), || self.ones_like());

        for node in self.sorted_nodes() {
            let parent_grad = store.get(node.id()).expect("Gradient must exist");

            match node.op() {
                Some(BackpropOp::Neg(node)) => {
                    let grad = store
                        .get_or_insert_with(node.id(), || self.zeros_like())
                        .add(&parent_grad.neg()?)?;
                    store.insert(node.id(), grad);
                }
                Some(BackpropOp::Add(left, right)) => {
                    let left_grad = store
                        .get_or_insert_with(left.id(), || self.zeros_like())
                        .add(&parent_grad)?;
                    store.insert(left.id(), left_grad);

                    let right_grad = store
                        .get_or_insert_with(right.id(), || self.zeros_like())
                        .add(&parent_grad)?;
                    store.insert(right.id(), right_grad);
                }
                Some(BackpropOp::Sub(left, right)) => {
                    let left_grad = store
                        .get_or_insert_with(left.id(), || self.zeros_like())
                        .add(&parent_grad)?;
                    store.insert(left.id(), left_grad);

                    let right_grad = store
                        .get_or_insert_with(right.id(), || self.zeros_like())
                        .sub(&parent_grad)?;
                    store.insert(right.id(), right_grad);
                }
                Some(BackpropOp::Mul(left, right)) => {
                    let left_grad = store
                        .get_or_insert_with(left.id(), || self.zeros_like())
                        .add(right)?;
                    store.insert(left.id(), left_grad);

                    let right_grad = store
                        .get_or_insert_with(right.id(), || self.zeros_like())
                        .add(left)?;
                    store.insert(right.id(), right_grad);
                }
                None => {
                    // No dependencies
                }
            }
        }

        Ok(store)
    }
}

pub struct GradientStore {
    store: HashMap<TensorId, Tensor>,
}

impl GradientStore {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    fn get(&self, id: TensorId) -> Option<Tensor> {
        self.store.get(&id).cloned()
    }

    fn get_or_insert_with(&mut self, id: TensorId, f: impl FnOnce() -> Tensor) -> Tensor {
        self.store.entry(id).or_insert_with(f).clone()
    }

    fn insert(&mut self, id: TensorId, tensor: Tensor) {
        self.store.insert(id, tensor);
    }
}

#[cfg(test)]
mod tests {
    use crate::{device::Device, dtype::DType};

    use super::*;

    #[test]
    fn test_sorting() {
        let a = Tensor::zeros((2, 3), DType::F32, Device::Cpu);
        let b = a.neg().unwrap();
        let c = b.neg().unwrap();
        let d = c.neg().unwrap();

        let sorted_nodes = d.sorted_nodes();

        assert_eq!(sorted_nodes.len(), 4);
    }

    #[test]
    fn test_backward_neg() {
        let a = Tensor::ones((3,), DType::F32, Device::Cpu);
        let b = a.neg().unwrap();

        let grads = b.backward().unwrap();

        let expected = Tensor::ones((3,), DType::F32, Device::Cpu).neg().unwrap();
        assert_eq!(grads.get(a.id()).unwrap(), expected);
    }

    #[test]
    fn test_backward_add() {
        let a = Tensor::ones((3,), DType::F32, Device::Cpu);
        let b = Tensor::ones((3,), DType::F32, Device::Cpu);
        let c = a.add(&b).unwrap();

        let grads = c.backward().unwrap();

        let expected = Tensor::ones((3,), DType::F32, Device::Cpu);
        assert_eq!(grads.get(a.id()).unwrap(), expected);
        assert_eq!(grads.get(b.id()).unwrap(), expected);
    }

    #[test]
    fn test_backward_sub() {
        let a = Tensor::ones((3,), DType::F32, Device::Cpu);
        let b = Tensor::ones((3,), DType::F32, Device::Cpu);
        let c = a.sub(&b).unwrap();

        let grads = c.backward().unwrap();

        let expected = Tensor::ones((3,), DType::F32, Device::Cpu);
        assert_eq!(grads.get(a.id()).unwrap(), expected);
        assert_eq!(grads.get(b.id()).unwrap(), expected.neg().unwrap());
    }

    #[test]
    fn test_mul() {
        let a = Tensor::load_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = Tensor::load_vec(vec![4.0, 5.0, 6.0], (3,), Device::Cpu);
        let c = a.mul(&b).unwrap();

        let grads = c.backward().unwrap();

        assert_eq!(grads.get(a.id()).unwrap(), b);
        assert_eq!(grads.get(b.id()).unwrap(), a);
    }
}
