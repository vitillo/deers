use std::{borrow::Borrow, iter::zip};

use crate::{
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
    storage::{BackendStorage, BinaryOp, UnaryOp},
};

use super::ReduceOp;

/// CPU-backed tensor storage, holding data as a flat `Vec` of f32 or f64.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub fn iter<'a, D: WithDType>(&'a self, layout: &'a Layout) -> Iter<'a, D> {
        Iter::new(self, layout)
    }

    pub fn len(&self) -> usize {
        match self {
            CpuStorage::F32(data) => data.len(),
            CpuStorage::F64(data) => data.len(),
        }
    }

    fn gemm_f32(left: &[f32], right: &[f32], out: &mut [f32], m: usize, n: usize, p: usize) {
        unsafe {
            gemm::gemm(
                m, p, n,
                out.as_mut_ptr(),
                1,          // dst_cs
                p as isize, // dst_rs
                false,      // read_dst
                left.as_ptr(),
                1,          // lhs_cs
                n as isize, // lhs_rs
                right.as_ptr(),
                1,          // rhs_cs
                p as isize, // rhs_rs
                0.0,        // alpha (dst = alpha*dst + beta*lhs*rhs)
                1.0,        // beta
                false, false, false,
                gemm::Parallelism::None,
            );
        }
    }

    fn gemm_f64(left: &[f64], right: &[f64], out: &mut [f64], m: usize, n: usize, p: usize) {
        unsafe {
            gemm::gemm(
                m, p, n,
                out.as_mut_ptr(),
                1,
                p as isize,
                false,
                left.as_ptr(),
                1,
                n as isize,
                right.as_ptr(),
                1,
                p as isize,
                0.0,
                1.0,
                false, false, false,
                gemm::Parallelism::None,
            );
        }
    }
}

impl From<Vec<f32>> for CpuStorage {
    fn from(v: Vec<f32>) -> Self {
        Self::F32(v)
    }
}

impl From<Vec<f64>> for CpuStorage {
    fn from(v: Vec<f64>) -> Self {
        Self::F64(v)
    }
}

impl BackendStorage for CpuStorage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        assert!(l.is_compact());

        match self {
            CpuStorage::F32(data) => {
                let array = data.iter().map(|v| v.powf(e as f32)).collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            CpuStorage::F64(data) => {
                let array = data.iter().map(|v| v.powf(e)).collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        assert!(l.is_compact());

        match self {
            CpuStorage::F32(data) => {
                let array = data.iter().map(|v| op.f32(*v)).collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            CpuStorage::F64(data) => {
                let array = data.iter().map(|v| op.f64(*v)).collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
        }
    }

    fn binary_op<O: BinaryOp>(
        &self,
        layout: &Layout,
        other: &Self,
        other_layout: &Layout,
    ) -> Result<Self> {
        assert!(layout.is_compact() && other_layout.is_compact());

        match (self, other) {
            (CpuStorage::F32(v), CpuStorage::F32(w)) => {
                let array = v
                    .iter()
                    .zip(w.iter())
                    .map(|(v, w)| O::f32(*v, *w))
                    .collect::<Vec<_>>();
                Ok(CpuStorage::F32(array))
            }
            (CpuStorage::F64(v), CpuStorage::F64(w)) => {
                let array = v
                    .iter()
                    .zip(w.iter())
                    .map(|(v, w)| O::f64(*v, *w))
                    .collect::<Vec<_>>();
                Ok(CpuStorage::F64(array))
            }
            (CpuStorage::F32(_), CpuStorage::F64(_)) => unreachable!(),
            (CpuStorage::F64(_), CpuStorage::F32(_)) => unreachable!(),
        }
    }

    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
        assert!(layout.is_compact());
        assert_eq!(0, layout.size() % dst.len());

        let reduce_size = layout.size() / dst.len();

        match (self, dst) {
            (CpuStorage::F32(src), CpuStorage::F32(dst)) => {
                for (i, chunk) in src.chunks(reduce_size).enumerate() {
                    dst[i] = chunk.iter().copied().reduce(O::f32).unwrap();
                }
            }
            (CpuStorage::F64(src), CpuStorage::F64(dst)) => {
                for (i, chunk) in src.chunks(reduce_size).enumerate() {
                    dst[i] = chunk.iter().copied().reduce(O::f64).unwrap();
                }
            }
            (CpuStorage::F32(_), CpuStorage::F64(_)) => unreachable!(),
            (CpuStorage::F64(_), CpuStorage::F32(_)) => unreachable!(),
        };

        Ok(())
    }

    fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
        }
    }

    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
        self.iter(layout.borrow()).copied().collect()
    }

    fn copy_compact(&self, src_layout: &Layout, dst: &mut CpuStorage) -> Result<()> {
        // Tensor::compact() already no-ops for contiguous tensors, so this
        // only runs on non-contiguous layouts (transposes, broadcasts, etc).
        // Direct index computation avoids the overhead of the strided iterator.
        let strides: Vec<usize> = src_layout.strides.0.iter().map(|&s| s as usize).collect();
        let shape: Vec<usize> = (0..src_layout.ndim()).map(|i| src_layout.shape[i]).collect();
        let offset = src_layout.offset;

        match (self, dst) {
            (CpuStorage::F32(src), CpuStorage::F32(dst)) => {
                copy_strided(src, dst, offset, &shape, &strides);
            }
            (CpuStorage::F64(src), CpuStorage::F64(dst)) => {
                copy_strided(src, dst, offset, &shape, &strides);
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn gather(&self, layout: &Layout, dim: usize, indices: &[usize]) -> Result<Self> {
        assert!(layout.is_compact());
        assert_eq!(dim, 1, "gather only supports dim=1 for 2D tensors");
        let rows = layout.shape[0];
        let cols = layout.shape[1];
        assert_eq!(indices.len(), rows);

        match self {
            CpuStorage::F32(data) => {
                let out: Vec<f32> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| data[i * cols + idx])
                    .collect();
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::F64(data) => {
                let out: Vec<f64> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| data[i * cols + idx])
                    .collect();
                Ok(CpuStorage::F64(out))
            }
        }
    }

    fn scatter(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &[usize],
        full_shape: &[usize],
    ) -> Result<Self> {
        assert!(layout.is_compact());
        assert_eq!(dim, 1, "scatter_add only supports dim=1 for 2D tensors");
        let rows = full_shape[0];
        let cols = full_shape[1];

        match self {
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; rows * cols];
                for (i, &idx) in indices.iter().enumerate() {
                    out[i * cols + idx] += data[i];
                }
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::F64(data) => {
                let mut out = vec![0.0f64; rows * cols];
                for (i, &idx) in indices.iter().enumerate() {
                    out[i * cols + idx] += data[i];
                }
                Ok(CpuStorage::F64(out))
            }
        }
    }

    fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
        assert!(layout.is_compact() && layout_other.is_compact());
        let n = layout.shape[layout.shape.ndim() - 1]; // cols of self, rows of other
        let m = layout.size() / n; // rows of self
        let p = layout_other.size() / n; // cols of other

        match (self, other) {
            (CpuStorage::F32(left), CpuStorage::F32(right)) => {
                let mut out = vec![0.0; m * p];
                CpuStorage::gemm_f32(left, right, &mut out, m, n, p);
                Ok(CpuStorage::F32(out))
            }
            (CpuStorage::F64(left), CpuStorage::F64(right)) => {
                let mut out = vec![0.0; m * p];
                CpuStorage::gemm_f64(left, right, &mut out, m, n, p);
                Ok(CpuStorage::F64(out))
            }
            (CpuStorage::F32(_), CpuStorage::F64(_)) => unreachable!(),
            (CpuStorage::F64(_), CpuStorage::F32(_)) => unreachable!(),
        }
    }
}

/// Copies a strided src into a contiguous dst using direct index computation.
fn copy_strided<T: Copy>(src: &[T], dst: &mut [T], offset: usize, shape: &[usize], strides: &[usize]) {
    // Recurse over dimensions, accumulating the source offset.
    copy_strided_inner(src, dst, offset, shape, strides, &mut 0);
}

fn copy_strided_inner<T: Copy>(
    src: &[T],
    dst: &mut [T],
    src_offset: usize,
    shape: &[usize],
    strides: &[usize],
    dst_offset: &mut usize,
) {
    if shape.len() == 1 {
        let size = shape[0];
        let stride = strides[0];
        for i in 0..size {
            dst[*dst_offset] = src[src_offset + i * stride];
            *dst_offset += 1;
        }
    } else {
        let size = shape[0];
        let stride = strides[0];
        for i in 0..size {
            copy_strided_inner(src, dst, src_offset + i * stride, &shape[1..], &strides[1..], dst_offset);
        }
    }
}

pub struct Iter<'a, D: WithDType> {
    array: &'a [D],
    cursor: Vec<usize>,
    layout: &'a Layout,
    index: usize,
}

impl<'a, D: WithDType> Iter<'a, D> {
    fn new(storage: &'a CpuStorage, layout: &'a Layout) -> Self {
        let array = D::as_slice(storage);
        Self {
            array,
            cursor: vec![0; layout.ndim()],
            layout,
            index: 0,
        }
    }

    fn increment(&mut self) {
        for i in (0..self.cursor.len()).rev() {
            assert!(self.cursor[i] < self.layout.shape[i]);
            if self.cursor[i] == self.layout.shape[i] - 1 {
                self.cursor[i] = 0;
            } else {
                self.cursor[i] += 1;
                assert!(self.cursor[i] < self.layout.shape[i]);
                break;
            }
        }
    }

    fn get_item(&self) -> &'a D {
        let mut offset = self.layout.offset as isize;
        for (idx, stride) in zip(&self.cursor, &self.layout.strides) {
            let idx = *idx as isize;
            offset += idx * stride;
        }
        assert!(offset >= 0);
        &self.array[offset as usize]
    }
}

impl<'a, D: WithDType> Iterator for Iter<'a, D> {
    type Item = &'a D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.layout.size() {
            return None;
        }

        let item = self.get_item();
        self.increment();
        self.index += 1;
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use crate::layout::{Shape, Strides};

    use super::*;

    #[test]
    fn test_iter() {
        let storage = CpuStorage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = Shape::from((2, 3));
        let strides = Strides(vec![3, 1]);
        let layout = Layout::new(shape, strides, 0);

        let array: Vec<f32> = storage.iter(&layout).copied().collect();

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(expected, array);
    }
}
