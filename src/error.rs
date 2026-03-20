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
    ShapeMismatch(String),

    #[error(transparent)]
    DatasetParseError(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
