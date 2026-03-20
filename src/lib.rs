//! Deers — a minimal PyTorch clone in Rust for learning.
//!
//! Provides tensors with automatic differentiation via a define-by-run
//! computation graph. Operations build the graph during the forward pass;
//! calling [`Tensor::backward`] traverses it in reverse to compute gradients.

mod backprop;
pub mod dataset;
mod device;
mod dtype;
mod error;
mod layout;
pub mod loss;
pub mod models;
pub mod nn;
mod ops;
pub mod optim;
mod storage;
mod tensor;
pub use device::Device;
pub use dtype::DType;
pub use tensor::Tensor;
