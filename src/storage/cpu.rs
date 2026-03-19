use std::borrow::Borrow;

use crate::{
    device::Device,
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
    storage::{BackendStorage, BinaryOp, MpsStorage, Storage, UnaryOp},
};

use super::ReduceOp;

/// CPU-backed tensor storage, holding data as a flat `Vec` of f32 or f64.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U32(Vec<u32>),
}

impl CpuStorage {
    pub fn to(self, device: Device) -> Storage {
        match device {
            Device::Cpu => Storage::Cpu(self),
            Device::Mps => Storage::Mps(MpsStorage::from_cpu_storage(self)),
            Device::Cuda => todo!(),
        }
    }

    pub fn iter<'a, D: WithDType>(&'a self, layout: &'a Layout) -> Iter<'a, D> {
        Iter::new(self, layout)
    }

    pub fn len(&self) -> usize {
        match self {
            CpuStorage::F32(data) => data.len(),
            CpuStorage::F64(data) => data.len(),
            CpuStorage::U32(data) => data.len(),
        }
    }

    /// Concatenates compact storages. Each pair is (storage, num_elements).
    pub fn cat(parts: &[(&CpuStorage, usize)]) -> CpuStorage {
        assert!(!parts.is_empty());
        let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
        match &parts[0].0 {
            CpuStorage::F32(_) => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    match storage {
                        CpuStorage::F32(v) => data.extend_from_slice(v),
                        _ => panic!("dtype mismatch in cat"),
                    }
                }
                CpuStorage::F32(data)
            }
            CpuStorage::F64(_) => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    match storage {
                        CpuStorage::F64(v) => data.extend_from_slice(v),
                        _ => panic!("dtype mismatch in cat"),
                    }
                }
                CpuStorage::F64(data)
            }
            CpuStorage::U32(_) => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    match storage {
                        CpuStorage::U32(v) => data.extend_from_slice(v),
                        _ => panic!("dtype mismatch in cat"),
                    }
                }
                CpuStorage::U32(data)
            }
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
                gemm::Parallelism::Rayon(0),
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
                gemm::Parallelism::Rayon(0),
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

impl From<Vec<u32>> for CpuStorage {
    fn from(v: Vec<u32>) -> Self {
        Self::U32(v)
    }
}

impl BackendStorage for CpuStorage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        if l.is_compact() {
            return match self {
                CpuStorage::F32(data) => {
                    let e = e as f32;
                    Ok(CpuStorage::F32(data.iter().map(|v| v.powf(e)).collect()))
                }
                CpuStorage::F64(data) => {
                    Ok(CpuStorage::F64(data.iter().map(|v| v.powf(e)).collect()))
                }
                CpuStorage::U32(_) => todo!(),
            };
        }
        let shape: Vec<usize> = l.shape().iter().copied().collect();
        let strides: Vec<isize> = l.strides().iter().copied().collect();
        match self {
            CpuStorage::F32(data) => {
                let e = e as f32;
                let mut out = vec![0.0f32; l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| v.powf(e));
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::F64(data) => {
                let mut out = vec![0.0f64; l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| v.powf(e));
                Ok(CpuStorage::F64(out))
            }
            CpuStorage::U32(_) => todo!(),
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        if l.is_compact() {
            return match self {
                CpuStorage::F32(data) => {
                    Ok(CpuStorage::F32(data.iter().map(|v| op.f32(*v)).collect()))
                }
                CpuStorage::F64(data) => {
                    Ok(CpuStorage::F64(data.iter().map(|v| op.f64(*v)).collect()))
                }
                CpuStorage::U32(_) => todo!(),
            };
        }
        let shape: Vec<usize> = l.shape().iter().copied().collect();
        let strides: Vec<isize> = l.strides().iter().copied().collect();
        match self {
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| op.f32(v));
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::F64(data) => {
                let mut out = vec![0.0f64; l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| op.f64(v));
                Ok(CpuStorage::F64(out))
            }
            CpuStorage::U32(_) => todo!(),
        }
    }

    fn binary_op<O: BinaryOp>(
        &self,
        layout: &Layout,
        other: &Self,
        other_layout: &Layout,
    ) -> Result<Self> {
        if layout.is_compact() && other_layout.is_compact() {
            return match (self, other) {
                (CpuStorage::F32(a), CpuStorage::F32(b)) => {
                    Ok(CpuStorage::F32(a.iter().zip(b.iter()).map(|(a, b)| O::f32(*a, *b)).collect()))
                }
                (CpuStorage::F64(a), CpuStorage::F64(b)) => {
                    Ok(CpuStorage::F64(a.iter().zip(b.iter()).map(|(a, b)| O::f64(*a, *b)).collect()))
                }
                _ => unreachable!(),
            };
        }
        let shape: Vec<usize> = layout.shape().iter().copied().collect();
        let a_strides: Vec<isize> = layout.strides().iter().copied().collect();
        let b_strides: Vec<isize> = other_layout.strides().iter().copied().collect();
        match (self, other) {
            (CpuStorage::F32(a), CpuStorage::F32(b)) => {
                let mut out = vec![0.0f32; layout.size()];
                strided_binary_op(
                    StridedSlice {
                        data: a,
                        offset: layout.offset,
                        strides: &a_strides,
                    },
                    StridedSlice {
                        data: b,
                        offset: other_layout.offset,
                        strides: &b_strides,
                    },
                    &mut out,
                    &shape,
                    O::f32,
                );
                Ok(CpuStorage::F32(out))
            }
            (CpuStorage::F64(a), CpuStorage::F64(b)) => {
                let mut out = vec![0.0f64; layout.size()];
                strided_binary_op(
                    StridedSlice {
                        data: a,
                        offset: layout.offset,
                        strides: &a_strides,
                    },
                    StridedSlice {
                        data: b,
                        offset: other_layout.offset,
                        strides: &b_strides,
                    },
                    &mut out,
                    &shape,
                    O::f64,
                );
                Ok(CpuStorage::F64(out))
            }
            (CpuStorage::F32(_), CpuStorage::F64(_)) => unreachable!(),
            (CpuStorage::F64(_), CpuStorage::F32(_)) => unreachable!(),
            (CpuStorage::U32(_), _) => todo!(),
            (_, CpuStorage::U32(_)) => todo!(),
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
            (CpuStorage::U32(_), _) => todo!(),
            (_, CpuStorage::U32(_)) => todo!(),
        };

        Ok(())
    }

    fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
            CpuStorage::U32(_) => DType::U32,
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
            (CpuStorage::U32(src), CpuStorage::U32(dst)) => {
                copy_strided(src, dst, offset, &shape, &strides);
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn gather(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &CpuStorage,
        indices_layout: &Layout,
    ) -> Result<Self> {
        assert!(layout.is_compact());
        assert_eq!(dim, 1, "gather only supports dim=1 for 2D tensors");
        let rows = layout.shape[0];
        let cols = layout.shape[1];
        assert!(indices_layout.is_compact());
        assert_eq!(indices_layout.ndim(), 1, "gather indices must be 1D");
        assert_eq!(indices_layout.shape()[0], rows);

        let indices: Vec<usize> = match indices {
            CpuStorage::U32(_) => indices
                .to_vec::<u32>(indices_layout)
                .into_iter()
                .map(|v| v as usize)
                .collect(),
            _ => todo!(),
        };
        assert_eq!(indices.len(), rows);

        for idx in &indices {
            assert!(*idx < cols, "gather index out of bounds");
        }

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
            CpuStorage::U32(data) => {
                let out: Vec<u32> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| data[i * cols + idx])
                    .collect();
                Ok(CpuStorage::U32(out))
            }
        }
    }

    fn scatter(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &CpuStorage,
        indices_layout: &Layout,
        full_shape: &[usize],
    ) -> Result<Self> {
        assert!(layout.is_compact());
        assert_eq!(dim, 1, "scatter_add only supports dim=1 for 2D tensors");
        assert!(indices_layout.is_compact());
        assert_eq!(indices_layout.ndim(), 1, "scatter indices must be 1D");
        let rows = full_shape[0];
        let cols = full_shape[1];
        let indices: Vec<usize> = match indices {
            CpuStorage::U32(_) => indices
                .to_vec::<u32>(indices_layout)
                .into_iter()
                .map(|v| v as usize)
                .collect(),
            _ => todo!(),
        };

        for idx in &indices {
            assert!(*idx < cols, "scatter index out of bounds");
        }

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
            CpuStorage::U32(data) => {
                let mut out = vec![0u32; rows * cols];
                for (i, &idx) in indices.iter().enumerate() {
                    out[i * cols + idx] += data[i];
                }
                Ok(CpuStorage::U32(out))
            }
        }
    }

    fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
        assert!(layout.is_compact() && layout_other.is_compact());
        let ndim = layout.shape.ndim();
        let k = layout.shape[ndim - 1];
        let m = layout.shape[ndim - 2];
        let n = layout_other.shape[ndim - 1];
        let batch: usize = (0..ndim - 2).map(|i| layout.shape[i]).product();
        let a_skip = m * k;
        let b_skip = k * n;
        let c_skip = m * n;

        match (self, other) {
            (CpuStorage::F32(left), CpuStorage::F32(right)) => {
                let mut out = vec![0.0f32; batch * m * n];
                for i in 0..batch {
                    CpuStorage::gemm_f32(
                        &left[i * a_skip..],
                        &right[i * b_skip..],
                        &mut out[i * c_skip..],
                        m, k, n,
                    );
                }
                Ok(CpuStorage::F32(out))
            }
            (CpuStorage::F64(left), CpuStorage::F64(right)) => {
                let mut out = vec![0.0f64; batch * m * n];
                for i in 0..batch {
                    CpuStorage::gemm_f64(
                        &left[i * a_skip..],
                        &right[i * b_skip..],
                        &mut out[i * c_skip..],
                        m, k, n,
                    );
                }
                Ok(CpuStorage::F64(out))
            }
            (CpuStorage::F32(_), CpuStorage::F64(_)) => unreachable!(),
            (CpuStorage::F64(_), CpuStorage::F32(_)) => unreachable!(),
            (CpuStorage::U32(_), _) => todo!(),
            (_, CpuStorage::U32(_)) => todo!(),
        }
    }
}

struct StridedSlice<'a, T> {
    data: &'a [T],
    offset: usize,
    strides: &'a [isize],
}

/// Applies a binary function to two strided arrays, writing to a compact output.
fn strided_binary_op<T: Copy, F: Fn(T, T) -> T>(
    a: StridedSlice<'_, T>,
    b: StridedSlice<'_, T>,
    out: &mut [T],
    shape: &[usize],
    f: F,
) {
    strided_binary_op_inner(a, b, out, &mut 0, shape, &f);
}

fn strided_binary_op_inner<T: Copy, F: Fn(T, T) -> T>(
    a: StridedSlice<'_, T>,
    b: StridedSlice<'_, T>,
    out: &mut [T],
    out_off: &mut usize,
    shape: &[usize],
    f: &F,
) {
    if shape.len() == 1 {
        let (sa, sb) = (a.strides[0] as usize, b.strides[0] as usize);
        for i in 0..shape[0] {
            out[*out_off] = f(a.data[a.offset + i * sa], b.data[b.offset + i * sb]);
            *out_off += 1;
        }
    } else {
        let (sa, sb) = (a.strides[0] as usize, b.strides[0] as usize);
        for i in 0..shape[0] {
            strided_binary_op_inner(
                StridedSlice {
                    data: a.data,
                    offset: a.offset + i * sa,
                    strides: &a.strides[1..],
                },
                StridedSlice {
                    data: b.data,
                    offset: b.offset + i * sb,
                    strides: &b.strides[1..],
                },
                out,
                out_off,
                &shape[1..],
                f,
            );
        }
    }
}

/// Applies a unary function to a strided array, writing to a compact output.
fn strided_unary_op<T: Copy, F: Fn(T) -> T>(
    a: &[T], a_off: usize, a_strides: &[isize],
    out: &mut [T], shape: &[usize], f: F,
) {
    strided_unary_op_inner(a, a_off, a_strides, out, &mut 0, shape, &f);
}

fn strided_unary_op_inner<T: Copy, F: Fn(T) -> T>(
    a: &[T], a_off: usize, a_strides: &[isize],
    out: &mut [T], out_off: &mut usize,
    shape: &[usize], f: &F,
) {
    if shape.len() == 1 {
        let sa = a_strides[0] as usize;
        for i in 0..shape[0] {
            out[*out_off] = f(a[a_off + i * sa]);
            *out_off += 1;
        }
    } else {
        let sa = a_strides[0] as usize;
        for i in 0..shape[0] {
            strided_unary_op_inner(
                a, a_off + i * sa, &a_strides[1..],
                out, out_off, &shape[1..], f,
            );
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
        for (idx, stride) in self.cursor.iter().zip(&self.layout.strides) {
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
