//! Deers — a minimal PyTorch clone in Rust for learning.
//!
//! Provides tensors with automatic differentiation via a define-by-run
//! computation graph. Operations build the graph during the forward pass;
//! calling [`Tensor::backward`] traverses it in reverse to compute gradients.
#![deny(missing_docs)]

mod backprop;
/// Checkpoint save/load helpers built on safetensors.
pub mod checkpoint;
/// Dataset loaders for training.
pub mod dataset;
mod device;
mod dtype;
mod error;
mod layout;
/// Built-in loss functions.
pub mod loss;
/// Built-in reference models.
pub mod models;
/// Neural-network layers, parameters, and functional helpers.
pub mod nn;
mod ops;
/// Optimizers and learning-rate schedules.
pub mod optim;
/// Lightweight profiling utilities.
pub mod profiler;
mod storage;
mod tensor;
/// BPE tokenizer and corpus tokenization pipelines.
pub mod tokenizer;
pub use backprop::GradientStore;
/// The supported tensor device backends.
pub use device::Device;
/// The supported tensor element types.
pub use dtype::DType;
/// Re-exported profiling types and helper entrypoint.
pub use profiler::{Profile, ProfileRow, Profiler, ProfilerConfig, profile};
/// The core tensor type.
pub use tensor::Tensor;
