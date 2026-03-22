use core::fmt;
use std::io;

/// Main library error type.
#[derive(thiserror::Error, fmt::Debug)]
pub enum Error {
    // Storage errors
    #[error("Size mismatch, expected buffer of size {expected} but got buffer of size {actual} ")]
    StorageSizeMismatch { expected: usize, actual: usize },

    #[error("Operation {op} requires tensors on the same device")]
    DeviceMismatch { op: &'static str },

    #[error("{0}")]
    NotImplemented(&'static str),

    #[error("{0}")]
    LayoutMismatch(String),

    #[error("{0}")]
    DTypeMismatch(String),

    #[error("{0}")]
    IndexOutOfBounds(String),

    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Checkpoint(String),

    #[error("cuda error: {0}")]
    Cuda(String),
}

pub type Result<T> = std::result::Result<T, Error>;
