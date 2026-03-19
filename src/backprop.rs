#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::tensor::{Tensor, TensorId};

impl Tensor {
    /// Returns nodes in reverse topological order (output before inputs).
    /// Uses DFS post-order + reverse so each node is processed only after
    /// all its consumers have been processed during backward.
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        if !self.requires_grad() {
            return vec![];
        }

        let mut post_order = vec![];
        let mut visited = HashSet::new();

        fn dfs<'a>(
            node: &'a Tensor,
            visited: &mut HashSet<TensorId>,
            post_order: &mut Vec<&'a Tensor>,
        ) {
            if !visited.insert(node.id()) {
                return;
            }
            if let Some(op) = node.op() {
                for dep in op.dependencies() {
                    if dep.requires_grad() {
                        dfs(dep, visited, post_order);
                    }
                }
            }
            post_order.push(node);
        }

        dfs(self, &mut visited, &mut post_order);
        post_order.reverse();
        post_order
    }

    /// Computes gradients for all tensors that require grad in this tensor's
    /// computation graph via reverse-mode automatic differentiation.
    ///
    /// Returns a [`GradientStore`] mapping each tensor's [`TensorId`] to its gradient.
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

/// Stores computed gradients, keyed by [`TensorId`].
///
/// Returned by [`Tensor::backward`]. Use [`get`](GradientStore::get) with a
/// tensor's id to retrieve its gradient.
#[derive(Debug)]
pub struct GradientStore {
    store: HashMap<TensorId, Tensor>,
}

impl GradientStore {
    /// Creates an empty gradient store.
    pub fn new() -> Self {
        Self { store: HashMap::new() }
    }

    /// Returns the gradient for the given tensor id, if it exists.
    pub fn get(&self, id: TensorId) -> Option<Tensor> {
        self.store.get(&id).cloned()
    }

    pub fn get_or_insert_with(&mut self, id: TensorId, f: impl FnOnce() -> Tensor) -> Tensor {
        self.store.entry(id).or_insert_with(f).clone()
    }

    pub fn get_or_insert_zero(&mut self, tensor: &Tensor) -> &mut Tensor {
        self.store.entry(tensor.id()).or_insert_with(|| tensor.zeros_like())
    }

    pub fn insert(&mut self, id: TensorId, tensor: Tensor) {
        self.store.insert(id, tensor);
    }

    /// Returns the number of stored gradients.
    pub fn len(&self) -> usize {
        self.store.len()
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
        let a = &Tensor::zeros((2, 3), DType::F32, Device::Cpu).attach();
        let b = &-a;
        let c = &-b;
        let d = &-c;

        let sorted_nodes = d.sorted_nodes();

        assert_eq!(sorted_nodes, [d, c, b, a]);
    }

    #[test]
    fn test_sorting_with_no_grad() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu).attach();
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let c = &a + b;

        let sorted_nodes = c.sorted_nodes();

        assert_eq!(sorted_nodes, [&c, &a]);
    }

    #[test]
    fn test_backward_with_no_grad() {
        let a = Tensor::ones((2, 3), DType::F32, Device::Cpu).attach();
        let b = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let c = Tensor::ones((2, 3), DType::F32, Device::Cpu);
        let d = b + c;
        let e = &a + &d;

        let grads = e.backward().unwrap();

        assert_eq!(3, grads.len());
        assert!(grads.store.contains_key(&a.id()));
        assert!(grads.store.contains_key(&d.id()));
        assert!(grads.store.contains_key(&e.id()));
    }
}
