use std::borrow::Borrow;

use half::f16;

use crate::{
    device::Device,
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
    storage::{BackendStorage, BinaryOp, MpsStorage, Storage, UnaryOp},
};

use super::ReduceOp;

/// CPU-backed tensor storage.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F16(Vec<f16>),
    F32(Vec<f32>),
    I64(Vec<i64>),
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
            CpuStorage::F16(data) => data.len(),
            CpuStorage::F32(data) => data.len(),
            CpuStorage::I64(data) => data.len(),
        }
    }

    /// Concatenates compact storages. Each pair is (storage, num_elements).
    pub fn cat(parts: &[(&CpuStorage, usize)]) -> CpuStorage {
        assert!(!parts.is_empty());
        let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
        match &parts[0].0 {
            CpuStorage::F16(_) => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    match storage {
                        CpuStorage::F16(v) => data.extend_from_slice(v),
                        _ => panic!("dtype mismatch in cat"),
                    }
                }
                CpuStorage::F16(data)
            }
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
            CpuStorage::I64(_) => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    match storage {
                        CpuStorage::I64(v) => data.extend_from_slice(v),
                        _ => panic!("dtype mismatch in cat"),
                    }
                }
                CpuStorage::I64(data)
            }
        }
    }

    fn gemm_f16(left: &[f16], right: &[f16], out: &mut [f16], m: usize, n: usize, p: usize) {
        unsafe {
            gemm::gemm(
                m,
                p,
                n,
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
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }

    fn gemm_f32(left: &[f32], right: &[f32], out: &mut [f32], m: usize, n: usize, p: usize) {
        unsafe {
            gemm::gemm(
                m,
                p,
                n,
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
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }

    fn compact_indices(indices: &CpuStorage, layout: &Layout) -> Vec<usize> {
        match indices {
            CpuStorage::I64(_) => indices
                .to_vec::<i64>(layout)
                .into_iter()
                .map(|v| usize::try_from(v).expect("indices must be non-negative"))
                .collect(),
            _ => panic!("index tensor must be i64"),
        }
    }
}

impl From<Vec<f16>> for CpuStorage {
    fn from(v: Vec<f16>) -> Self {
        Self::F16(v)
    }
}

impl From<Vec<f32>> for CpuStorage {
    fn from(v: Vec<f32>) -> Self {
        Self::F32(v)
    }
}

impl From<Vec<i64>> for CpuStorage {
    fn from(v: Vec<i64>) -> Self {
        Self::I64(v)
    }
}

impl BackendStorage for CpuStorage {
    fn ewise_powf(&self, e: f64, l: &Layout) -> Result<Self> {
        if l.is_compact() {
            return match self {
                CpuStorage::F16(data) => {
                    Ok(CpuStorage::F16(
                        data.iter()
                            .map(|v| f16::from_f32(v.to_f32().powf(e as f32)))
                            .collect(),
                    ))
                }
                CpuStorage::F32(data) => {
                    let e = e as f32;
                    Ok(CpuStorage::F32(data.iter().map(|v| v.powf(e)).collect()))
                }
                CpuStorage::I64(_) => todo!(),
            };
        }

        let shape: Vec<usize> = l.shape().iter().copied().collect();
        let strides: Vec<isize> = l.strides().iter().copied().collect();
        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| {
                    f16::from_f32(v.to_f32().powf(e as f32))
                });
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let e = e as f32;
                let mut out = vec![0.0f32; l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| v.powf(e));
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::I64(_) => todo!(),
        }
    }

    fn unary_op<O: UnaryOp>(&self, op: O, l: &Layout) -> Result<Self> {
        if l.is_compact() {
            return match self {
                CpuStorage::F16(data) => {
                    Ok(CpuStorage::F16(data.iter().map(|v| op.f16(*v)).collect()))
                }
                CpuStorage::F32(data) => {
                    Ok(CpuStorage::F32(data.iter().map(|v| op.f32(*v)).collect()))
                }
                CpuStorage::I64(_) => todo!(),
            };
        }

        let shape: Vec<usize> = l.shape().iter().copied().collect();
        let strides: Vec<isize> = l.strides().iter().copied().collect();
        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| op.f16(v));
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; l.size()];
                strided_unary_op(data, l.offset, &strides, &mut out, &shape, |v| op.f32(v));
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::I64(_) => todo!(),
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
                (CpuStorage::F16(a), CpuStorage::F16(b)) => Ok(CpuStorage::F16(
                    a.iter().zip(b.iter()).map(|(a, b)| O::f16(*a, *b)).collect(),
                )),
                (CpuStorage::F32(a), CpuStorage::F32(b)) => Ok(CpuStorage::F32(
                    a.iter().zip(b.iter()).map(|(a, b)| O::f32(*a, *b)).collect(),
                )),
                _ => unreachable!(),
            };
        }

        let shape: Vec<usize> = layout.shape().iter().copied().collect();
        let a_strides: Vec<isize> = layout.strides().iter().copied().collect();
        let b_strides: Vec<isize> = other_layout.strides().iter().copied().collect();
        match (self, other) {
            (CpuStorage::F16(a), CpuStorage::F16(b)) => {
                let mut out = vec![f16::from_f32(0.0); layout.size()];
                strided_binary_op(
                    StridedSlice { data: a, offset: layout.offset, strides: &a_strides },
                    StridedSlice { data: b, offset: other_layout.offset, strides: &b_strides },
                    &mut out,
                    &shape,
                    O::f16,
                );
                Ok(CpuStorage::F16(out))
            }
            (CpuStorage::F32(a), CpuStorage::F32(b)) => {
                let mut out = vec![0.0f32; layout.size()];
                strided_binary_op(
                    StridedSlice { data: a, offset: layout.offset, strides: &a_strides },
                    StridedSlice { data: b, offset: other_layout.offset, strides: &b_strides },
                    &mut out,
                    &shape,
                    O::f32,
                );
                Ok(CpuStorage::F32(out))
            }
            _ => unreachable!(),
        }
    }

    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
        assert!(layout.is_compact());
        assert_eq!(0, layout.size() % dst.len());

        let reduce_size = layout.size() / dst.len();
        match (self, dst) {
            (CpuStorage::F16(src), CpuStorage::F16(dst)) => {
                for (i, chunk) in src.chunks(reduce_size).enumerate() {
                    dst[i] = chunk.iter().copied().reduce(O::f16).unwrap();
                }
            }
            (CpuStorage::F32(src), CpuStorage::F32(dst)) => {
                for (i, chunk) in src.chunks(reduce_size).enumerate() {
                    dst[i] = chunk.iter().copied().reduce(O::f32).unwrap();
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn dtype(&self) -> DType {
        match self {
            CpuStorage::F16(_) => DType::F16,
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::I64(_) => DType::I64,
        }
    }

    fn to_vec<D: WithDType>(&self, layout: impl Borrow<Layout>) -> Vec<D> {
        self.iter(layout.borrow()).copied().collect()
    }

    fn copy_compact(&self, src_layout: &Layout, dst: &mut CpuStorage) -> Result<()> {
        let strides: Vec<usize> = src_layout.strides.0.iter().map(|&s| s as usize).collect();
        let shape: Vec<usize> = (0..src_layout.ndim()).map(|i| src_layout.shape[i]).collect();
        let offset = src_layout.offset;

        match (self, dst) {
            (CpuStorage::F16(src), CpuStorage::F16(dst)) => copy_strided(src, dst, offset, &shape, &strides),
            (CpuStorage::F32(src), CpuStorage::F32(dst)) => copy_strided(src, dst, offset, &shape, &strides),
            (CpuStorage::I64(src), CpuStorage::I64(dst)) => copy_strided(src, dst, offset, &shape, &strides),
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
        assert!(indices_layout.is_compact());

        let rows = layout.shape[0];
        let cols = layout.shape[1];
        assert_eq!(indices_layout.ndim(), 1, "gather indices must be 1D");
        assert_eq!(indices_layout.shape()[0], rows);

        let indices = Self::compact_indices(indices, indices_layout);
        for idx in &indices {
            assert!(*idx < cols, "gather index out of bounds");
        }

        match self {
            CpuStorage::F16(data) => Ok(CpuStorage::F16(
                indices.iter().enumerate().map(|(i, &idx)| data[i * cols + idx]).collect(),
            )),
            CpuStorage::F32(data) => Ok(CpuStorage::F32(
                indices.iter().enumerate().map(|(i, &idx)| data[i * cols + idx]).collect(),
            )),
            CpuStorage::I64(data) => Ok(CpuStorage::I64(
                indices.iter().enumerate().map(|(i, &idx)| data[i * cols + idx]).collect(),
            )),
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
        let indices = Self::compact_indices(indices, indices_layout);
        for idx in &indices {
            assert!(*idx < cols, "scatter index out of bounds");
        }

        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); rows * cols];
                for (i, &idx) in indices.iter().enumerate() {
                    out[i * cols + idx] += data[i];
                }
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; rows * cols];
                for (i, &idx) in indices.iter().enumerate() {
                    out[i * cols + idx] += data[i];
                }
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::I64(data) => {
                let mut out = vec![0i64; rows * cols];
                for (i, &idx) in indices.iter().enumerate() {
                    out[i * cols + idx] += data[i];
                }
                Ok(CpuStorage::I64(out))
            }
        }
    }

    fn index_select(&self, layout: &Layout, indices: &Self, indices_layout: &Layout) -> Result<Self> {
        assert!(layout.is_compact());
        assert_eq!(layout.ndim(), 2, "index_select requires 2D input");
        assert!(indices_layout.is_compact());

        let rows = layout.shape[0];
        let cols = layout.shape[1];
        let indices = Self::compact_indices(indices, indices_layout);
        for idx in &indices {
            assert!(*idx < rows, "index_select index {} out of bounds ({})", idx, rows);
        }

        match self {
            CpuStorage::F16(data) => {
                let mut out = Vec::with_capacity(indices.len() * cols);
                for &idx in &indices {
                    out.extend_from_slice(&data[idx * cols..(idx + 1) * cols]);
                }
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let mut out = Vec::with_capacity(indices.len() * cols);
                for &idx in &indices {
                    out.extend_from_slice(&data[idx * cols..(idx + 1) * cols]);
                }
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::I64(data) => {
                let mut out = Vec::with_capacity(indices.len() * cols);
                for &idx in &indices {
                    out.extend_from_slice(&data[idx * cols..(idx + 1) * cols]);
                }
                Ok(CpuStorage::I64(out))
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
            (CpuStorage::F16(left), CpuStorage::F16(right)) => {
                let mut out = vec![f16::from_f32(0.0); batch * m * n];
                for i in 0..batch {
                    CpuStorage::gemm_f16(
                        &left[i * a_skip..],
                        &right[i * b_skip..],
                        &mut out[i * c_skip..],
                        m,
                        k,
                        n,
                    );
                }
                Ok(CpuStorage::F16(out))
            }
            (CpuStorage::F32(left), CpuStorage::F32(right)) => {
                let mut out = vec![0.0f32; batch * m * n];
                for i in 0..batch {
                    CpuStorage::gemm_f32(
                        &left[i * a_skip..],
                        &right[i * b_skip..],
                        &mut out[i * c_skip..],
                        m,
                        k,
                        n,
                    );
                }
                Ok(CpuStorage::F32(out))
            }
            _ => todo!("matmul only supports floating tensors"),
        }
    }
}

struct StridedSlice<'a, T> {
    data: &'a [T],
    offset: usize,
    strides: &'a [isize],
}

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
                StridedSlice { data: a.data, offset: a.offset + i * sa, strides: &a.strides[1..] },
                StridedSlice { data: b.data, offset: b.offset + i * sb, strides: &b.strides[1..] },
                out,
                out_off,
                &shape[1..],
                f,
            );
        }
    }
}

fn strided_unary_op<T: Copy, F: Fn(T) -> T>(
    a: &[T],
    a_off: usize,
    a_strides: &[isize],
    out: &mut [T],
    shape: &[usize],
    f: F,
) {
    strided_unary_op_inner(a, a_off, a_strides, out, &mut 0, shape, &f);
}

fn strided_unary_op_inner<T: Copy, F: Fn(T) -> T>(
    a: &[T],
    a_off: usize,
    a_strides: &[isize],
    out: &mut [T],
    out_off: &mut usize,
    shape: &[usize],
    f: &F,
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
            strided_unary_op_inner(a, a_off + i * sa, &a_strides[1..], out, out_off, &shape[1..], f);
        }
    }
}

fn copy_strided<T: Copy>(
    src: &[T],
    dst: &mut [T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) {
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
        Self { array, cursor: vec![0; layout.ndim()], layout, index: 0 }
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
            offset += *idx as isize * stride;
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
