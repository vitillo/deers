#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::tensor::{NoGradGuard, Tensor, TensorId};

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
        let _no_grad = NoGradGuard::new();
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

    /// Returns the stored gradient for `id`, inserting `f()` if missing.
    pub fn get_or_insert_with(&mut self, id: TensorId, f: impl FnOnce() -> Tensor) -> Tensor {
        self.store.entry(id).or_insert_with(f).clone()
    }

    /// Returns a mutable gradient slot for `tensor`, inserting zeros if missing.
    pub fn get_or_insert_zero(&mut self, tensor: &Tensor) -> &mut Tensor {
        self.store.entry(tensor.id()).or_insert_with(|| tensor.zeros_like())
    }

    /// Accumulates `grad` into the gradient for `tensor`, adding element-wise.
    /// If no gradient exists yet, stores `grad` directly (avoiding a zero-init + add).
    pub fn accumulate(&mut self, tensor: &Tensor, grad: Tensor) {
        use std::collections::hash_map::Entry;
        match self.store.entry(tensor.id()) {
            Entry::Vacant(e) => {
                e.insert(grad);
            }
            Entry::Occupied(mut e) => {
                let existing = e.get_mut();
                *existing = &*existing + &grad;
            }
        }
    }

    /// Stores `tensor` as the gradient for `id`, replacing any previous value.
    pub fn insert(&mut self, id: TensorId, tensor: Tensor) {
        self.store.insert(id, tensor);
    }

    /// Returns the number of stored gradients.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns whether the store contains no gradients.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
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
    fn sorted_nodes_linear_chain() {
        // Arrange
        let a = &Tensor::zeros((2, 3), DType::F32, Device::Cpu).attach();
        let b = &-a;
        let c = &-b;
        let d = &-c;

        // Act
        let sorted = d.sorted_nodes();

        // Assert
        assert_eq!(sorted, [d, c, b, a]);
    }

    #[test]
    fn sorted_nodes_diamond() {
        // Arrange — a is used by both b and c
        let a = &Tensor::ones((2,), DType::F32, Device::Cpu).attach();
        let b = &-a;
        let c = &-a;
        let d = &(b + c);

        // Act
        let sorted = d.sorted_nodes();

        // Assert — no duplicates, shared input last
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted[0], d);
        assert_eq!(*sorted.last().unwrap(), a);
    }

    #[test]
    fn sorted_nodes_skips_no_grad() {
        // Arrange
        let a = Tensor::ones((2,), DType::F32, Device::Cpu).attach();
        let b = Tensor::ones((2,), DType::F32, Device::Cpu); // no grad
        let c = &a + b;

        // Act
        let sorted = c.sorted_nodes();

        // Assert
        assert_eq!(sorted, [&c, &a]);
    }

    #[test]
    fn backward_accumulates_diamond_grads() {
        // Arrange — a feeds two paths that recombine
        let a = Tensor::ones((2,), DType::F32, Device::Cpu).attach();
        let b = &a + &a;
        let loss = b.sum(vec![0], true);

        // Act
        let grads = loss.backward().unwrap();
        let grad_a = grads.get(a.id()).unwrap().to_vec::<f32>().unwrap();

        // Assert — grad(a) = 2 because a is used twice
        assert_eq!(grad_a, vec![2.0, 2.0]);
    }
}
