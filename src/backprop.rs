#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use crate::error::Result;
use crate::tensor::{Tensor, TensorId};

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
        let mut grads = GradientStore::new();
        grads.get_or_insert_with(self.id(), || Tensor::ones_like(self));

        for node in self.sorted_nodes() {
            let out_grad = grads.get(node.id()).expect("Gradient must exist");

            if let Some(op) = node.op() {
                op.backward(&mut grads, &out_grad)?;
            }
        }

        Ok(grads)
    }
}

#[derive(Debug)]
pub struct GradientStore {
    store: HashMap<TensorId, Tensor>,
}

impl GradientStore {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn get(&self, id: TensorId) -> Option<Tensor> {
        self.store.get(&id).cloned()
    }

    pub fn get_or_insert_with(&mut self, id: TensorId, f: impl FnOnce() -> Tensor) -> Tensor {
        self.store.entry(id).or_insert_with(f).clone()
    }

    pub fn get_or_insert_zero(&mut self, tensor: &Tensor) -> &mut Tensor {
        self.store
            .entry(tensor.id())
            .or_insert_with(|| tensor.zeros_like())
    }

    pub fn insert(&mut self, id: TensorId, tensor: Tensor) {
        self.store.insert(id, tensor);
    }
}

impl Default for GradientStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::{device::Device, dtype::DType};

    use super::*;

    #[test]
    fn test_sorting() {
        let a = &Tensor::zeros((2, 3), DType::F32, Device::Cpu);
        let b = &-a;
        let c = &-b;
        let d = &-c;

        let sorted_nodes = d.sorted_nodes();

        assert_eq!(sorted_nodes, [d, c, b, a]);
    }

    #[test]
    fn test_backward_scalar_add() {
        let a = Tensor::ones((3,), DType::F32, Device::Cpu);
        let b = &a + 2.0;

        let grads = b.backward().unwrap();

        let expected = Tensor::ones((3,), DType::F32, Device::Cpu);
        assert_eq!(grads.get(a.id()).unwrap(), expected);
    }

    #[test]
    fn test_backward_scalar_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), Device::Cpu);
        let b = &a * 2.0;

        let grads = b.backward().unwrap();

        let expected = Tensor::from_vec(vec![2.0, 2.0, 2.0], (3,), Device::Cpu);
        assert_eq!(grads.get(a.id()).unwrap(), expected);
    }
}
