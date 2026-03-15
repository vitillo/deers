//! Deers — a minimal PyTorch clone in Rust for learning.
//!
//! Provides tensors with automatic differentiation via a define-by-run
//! computation graph. Operations build the graph during the forward pass;
//! calling [`Tensor::backward`] traverses it in reverse to compute gradients.

mod backprop;
mod dataset;
mod device;
mod dtype;
mod error;
mod layout;
mod ops;
mod storage;
mod tensor;
pub use device::Device;
pub use dtype::DType;
pub use tensor::Tensor;
