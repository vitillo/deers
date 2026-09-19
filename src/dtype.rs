//! Element types (`F16`, `BF16`, `F32`, `I64`) and the [`WithDType`] trait for
//! projecting Rust scalars into and out of tensor storage.
//!
//! `BF16` (brain float 16) keeps the 8-bit exponent of `F32` with a truncated
//! 7-bit mantissa. It covers nearly the full `F32` range (up to ~3.4e38) but
//! only about 3 decimal digits of precision, so large model weights load
//! unchanged while fine detail rounds to the nearest-even 7-bit mantissa.

#![allow(dead_code)]

use std::fmt;

use half::{bf16, f16};

use crate::storage::{BackendStorage, CpuStorage};

/// Supported tensor element types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE half-precision floating point.
    F16,
    /// Brain floating point: F32 range with a 7-bit mantissa.
    BF16,
    /// IEEE single-precision floating point.
    F32,
    /// 64-bit signed integer.
    I64,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}

impl DType {
    /// Returns the byte width of one element of this dtype.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F16 => std::mem::size_of::<f16>(),
            DType::BF16 => std::mem::size_of::<bf16>(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::I64 => std::mem::size_of::<i64>(),
        }
    }
}

/// Trait implemented by Rust types that can be stored in a tensor.
pub trait WithDType: Sized + Copy {
    fn to_vec(storage: &CpuStorage) -> Vec<Self>;
    fn as_slice(storage: &CpuStorage) -> &[Self];
}

impl WithDType for f16 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F16(vec) => vec.clone(),
            other => panic!("expected F16 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F16(vec) => vec.as_slice(),
            other => panic!("expected F16 storage but got {:?}", other.dtype()),
        }
    }
}

impl WithDType for bf16 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::BF16(vec) => vec.clone(),
            other => panic!("expected BF16 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::BF16(vec) => vec.as_slice(),
            other => panic!("expected BF16 storage but got {:?}", other.dtype()),
        }
    }
}

impl WithDType for f32 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::F32(vec) => vec.clone(),
            other => panic!("expected F32 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::F32(vec) => vec.as_slice(),
            other => panic!("expected F32 storage but got {:?}", other.dtype()),
        }
    }
}

impl WithDType for i64 {
    fn to_vec(storage: &CpuStorage) -> Vec<Self> {
        match storage {
            CpuStorage::I64(vec) => vec.clone(),
            other => panic!("expected I64 storage but got {:?}", other.dtype()),
        }
    }

    fn as_slice(storage: &CpuStorage) -> &[Self] {
        match storage {
            CpuStorage::I64(vec) => vec.as_slice(),
            other => panic!("expected I64 storage but got {:?}", other.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_display_and_size() {
        // Arrange
        let dtype = DType::BF16;

        // Act
        let name = dtype.to_string();
        let bytes = dtype.size_in_bytes();

        // Assert
        assert_eq!(name, "bf16");
        assert_eq!(bytes, 2);
    }

    #[test]
    fn test_bf16_exact_values_roundtrip() {
        // Arrange: values with at most 7 mantissa bits survive BF16 exactly.
        let exact = [0.0f32, 1.0, -2.0, 1.5, 100.0, 65536.0];

        // Act
        let roundtripped: Vec<f32> = exact.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();

        // Assert
        assert_eq!(roundtripped, exact);
    }

    #[test]
    fn test_bf16_rounds_to_nearest_even() {
        // Arrange: each case is (input, expected BF16 bits). Pi truncates to
        // 0x4049 (3.140625). 1 + 2^-8 sits exactly halfway between two BF16
        // values and rounds to the even mantissa (0x3F80 = 1.0), while
        // 1 + 3*2^-8 rounds up to 0x3F82 (1.015625).
        let cases = [
            (std::f32::consts::PI, 0x4049u16),
            (1.0 + 2f32.powi(-8), 0x3F80),
            (1.0 + 3.0 * 2f32.powi(-8), 0x3F82),
            (0.1, 0x3DCD),
        ];

        // Act
        let rounded: Vec<u16> = cases.iter().map(|&(v, _)| bf16::from_f32(v).to_bits()).collect();

        // Assert
        let expected: Vec<u16> = cases.iter().map(|&(_, e)| e).collect();
        assert_eq!(rounded, expected);
    }

    #[test]
    fn test_bf16_extremes() {
        // Arrange
        let max = bf16::MAX.to_f32();
        let min = bf16::MIN.to_f32();
        let min_positive = bf16::MIN_POSITIVE.to_f32();

        // Act + Assert: BF16 spans the F32 exponent range with 7 mantissa bits.
        assert_eq!(max, 3.3895314e38f32);
        assert_eq!(min, -3.3895314e38f32);
        assert_eq!(min_positive, 1.1754944e-38f32);
    }

    #[test]
    fn test_bf16_nonfinite_inputs() {
        // Arrange
        let nan = bf16::from_f32(f32::NAN);
        let pos_inf = bf16::from_f32(f32::INFINITY);
        let neg_inf = bf16::from_f32(f32::NEG_INFINITY);

        // Act + Assert
        assert!(nan.is_nan());
        assert_eq!(pos_inf, bf16::INFINITY);
        assert_eq!(neg_inf, bf16::NEG_INFINITY);
        // Overflow past BF16 range saturates to infinity.
        assert!(bf16::from_f32(bf16::MAX.to_f32() * 2.0).is_infinite());
        assert!(bf16::from_f32(f32::MAX).is_infinite());
    }
}
