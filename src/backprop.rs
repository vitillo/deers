#![allow(dead_code)]

use std::collections::VecDeque;

use crate::error::Result;
use crate::tensor::Tensor;

#[derive(Debug)]
pub enum BackpropOp {
    Neg(Tensor),
    Add(Tensor, Tensor),
}

impl Tensor {
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        let mut sorted_nodes = vec![];
        let mut queue = VecDeque::new();
        queue.push_front(self);
        sorted_nodes.push(self);

        while let Some(node) = queue.pop_back() {
            match node.op() {
                Some(BackpropOp::Neg(node)) => {
                    sorted_nodes.push(node);
                    queue.push_front(node);
                }
                Some(BackpropOp::Add(left, right)) => {
                    sorted_nodes.push(left);
                    sorted_nodes.push(right);
                    queue.push_front(left);
                    queue.push_front(right);
                }
                None => {
                    // No dependencies
                }
            }
        }

        sorted_nodes
    }

    pub fn backward(&self) -> Result<()> {
        let ones = self.ones_like();
        self.grad().lock().unwrap().get_or_insert_with(|| ones);
        assert!(self.grad().lock().unwrap().is_some());

        for node in self.sorted_nodes() {
            let node_grad = node
                .grad()
                .lock()
                .unwrap()
                .as_ref()
                .expect("Gradient not computed")
                .clone();

            match node.op() {
                Some(BackpropOp::Neg(parent)) => {
                    let parent_grad = parent
                        .grad()
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap_or_else(|| parent.zeros_like());

                    let parent_grad = parent_grad.add(&node_grad.neg()?)?;
                    *parent.grad().lock().unwrap() = Some(parent_grad);
                }
                Some(BackpropOp::Add(_, _)) => {
                    todo!()
                }
                None => {
                    // No dependencies
                }
            }
        }

        Ok(())
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
    fn test_backward() {
        let a = Tensor::ones((3,), DType::F32, Device::Cpu);
        let b = a.neg().unwrap();

        b.backward().unwrap();

        let expected = Tensor::ones((3,), DType::F32, Device::Cpu).neg().unwrap();
        assert!(a.grad().lock().unwrap().is_some());
        assert_eq!(a.grad().lock().unwrap().take().unwrap(), expected);
    }
}
