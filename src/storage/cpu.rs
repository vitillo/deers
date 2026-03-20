use std::borrow::Borrow;

use half::f16;

use crate::{
    device::Device,
    dtype::{DType, WithDType},
    error::{Error, Result},
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
    pub fn cat(parts: &[(&CpuStorage, usize)]) -> Result<CpuStorage> {
        if parts.is_empty() {
            return Err(Error::LayoutMismatch("cat: empty parts".into()));
        }
        let expected = parts[0].0.dtype();
        let total_len: usize = parts.iter().map(|(_, len)| *len).sum();
        for (storage, _) in &parts[1..] {
            if storage.dtype() != expected {
                return Err(Error::DTypeMismatch(format!(
                    "cat: expected {:?} but got {:?}",
                    expected,
                    storage.dtype()
                )));
            }
        }
        match expected {
            DType::F16 => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    if let CpuStorage::F16(v) = storage {
                        data.extend_from_slice(v);
                    }
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    if let CpuStorage::F32(v) = storage {
                        data.extend_from_slice(v);
                    }
                }
                Ok(CpuStorage::F32(data))
            }
            DType::I64 => {
                let mut data = Vec::with_capacity(total_len);
                for (storage, _) in parts {
                    if let CpuStorage::I64(v) = storage {
                        data.extend_from_slice(v);
                    }
                }
                Ok(CpuStorage::I64(data))
            }
        }
    }

    fn gemm<T: 'static>(
        left: &[T],
        right: &[T],
        out: &mut [T],
        m: usize,
        n: usize,
        p: usize,
        alpha: T,
        beta: T,
    ) {
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
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }

    fn compact_indices(indices: &CpuStorage, layout: &Layout) -> Result<Vec<usize>> {
        match indices {
            CpuStorage::I64(_) => {
                let mut out = Vec::with_capacity(layout.size());
                for v in indices.to_vec::<i64>(layout) {
                    let idx = usize::try_from(v)
                        .map_err(|_| Error::IndexOutOfBounds(format!("index {} is negative", v)))?;
                    out.push(idx);
                }
                Ok(out)
            }
            _ => Err(Error::DTypeMismatch(format!(
                "index tensor must be i64, got {:?}",
                indices.dtype()
            ))),
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
                CpuStorage::F16(data) => Ok(CpuStorage::F16(
                    data.iter().map(|v| f16::from_f32(v.to_f32().powf(e as f32))).collect(),
                )),
                CpuStorage::F32(data) => {
                    let e = e as f32;
                    Ok(CpuStorage::F32(data.iter().map(|v| v.powf(e)).collect()))
                }
                CpuStorage::I64(_) => todo!(),
            };
        }

        let shape = l.shape().as_slice();
        let strides = &l.strides.0;
        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); l.size()];
                strided_unary_op(data, l.offset, strides, &mut out, shape, |v| {
                    f16::from_f32(v.to_f32().powf(e as f32))
                });
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let e = e as f32;
                let mut out = vec![0.0f32; l.size()];
                strided_unary_op(data, l.offset, strides, &mut out, shape, |v| v.powf(e));
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

        let shape = l.shape().as_slice();
        let strides = &l.strides.0;
        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); l.size()];
                strided_unary_op(data, l.offset, strides, &mut out, shape, |v| op.f16(v));
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; l.size()];
                strided_unary_op(data, l.offset, strides, &mut out, shape, |v| op.f32(v));
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
                (a, b) => Err(Error::DTypeMismatch(format!(
                    "binary_op: {:?} vs {:?}",
                    a.dtype(),
                    b.dtype()
                ))),
            };
        }

        let shape = layout.shape().as_slice();
        let a_strides = &layout.strides.0;
        let b_strides = &other_layout.strides.0;
        match (self, other) {
            (CpuStorage::F16(a), CpuStorage::F16(b)) => {
                let mut out = vec![f16::from_f32(0.0); layout.size()];
                strided_binary_op(
                    StridedSlice { data: a, offset: layout.offset, strides: a_strides },
                    StridedSlice { data: b, offset: other_layout.offset, strides: b_strides },
                    &mut out,
                    shape,
                    O::f16,
                );
                Ok(CpuStorage::F16(out))
            }
            (CpuStorage::F32(a), CpuStorage::F32(b)) => {
                let mut out = vec![0.0f32; layout.size()];
                strided_binary_op(
                    StridedSlice { data: a, offset: layout.offset, strides: a_strides },
                    StridedSlice { data: b, offset: other_layout.offset, strides: b_strides },
                    &mut out,
                    shape,
                    O::f32,
                );
                Ok(CpuStorage::F32(out))
            }
            (a, b) => Err(Error::LayoutMismatch(format!(
                "binary_op: dtype mismatch, {:?} vs {:?}",
                a.dtype(),
                b.dtype()
            ))),
        }
    }

    fn reduce<O: ReduceOp>(&self, layout: &Layout, dst: &mut Self) -> Result<()> {
        if !layout.is_compact() {
            return Err(Error::LayoutMismatch("reduce: layout must be compact".into()));
        }
        if !layout.size().is_multiple_of(dst.len()) {
            return Err(Error::LayoutMismatch(format!(
                "reduce: source size {} not divisible by dst size {}",
                layout.size(),
                dst.len()
            )));
        }

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
            (src, dst) => {
                return Err(Error::DTypeMismatch(format!(
                    "reduce: {:?} vs {:?}",
                    src.dtype(),
                    dst.dtype()
                )));
            }
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
            (CpuStorage::F16(src), CpuStorage::F16(dst)) => {
                copy_strided(src, dst, offset, &shape, &strides)
            }
            (CpuStorage::F32(src), CpuStorage::F32(dst)) => {
                copy_strided(src, dst, offset, &shape, &strides)
            }
            (CpuStorage::I64(src), CpuStorage::I64(dst)) => {
                copy_strided(src, dst, offset, &shape, &strides)
            }
            (src, dst) => {
                return Err(Error::DTypeMismatch(format!(
                    "copy_compact: {:?} vs {:?}",
                    src.dtype(),
                    dst.dtype()
                )));
            }
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
        if !layout.is_compact() {
            return Err(Error::LayoutMismatch("gather: source layout must be compact".into()));
        }
        if !indices_layout.is_compact() {
            return Err(Error::LayoutMismatch("gather: indices layout must be compact".into()));
        }
        if layout.ndim() != indices_layout.ndim() {
            return Err(Error::LayoutMismatch(format!(
                "gather: rank mismatch, source has {} dims but indices has {}",
                layout.ndim(),
                indices_layout.ndim()
            )));
        }
        if dim >= layout.ndim() {
            return Err(Error::LayoutMismatch(format!(
                "gather: dim {} out of bounds for {} dims",
                dim,
                layout.ndim()
            )));
        }

        let left_len: usize = indices_layout.shape().iter().take(dim).product();
        let index_len = indices_layout.shape()[dim];
        let right_len: usize = indices_layout.shape().iter().skip(dim + 1).product();
        let src_dim = layout.shape()[dim];
        for axis in 0..layout.ndim() {
            if axis != dim && layout.shape()[axis] != indices_layout.shape()[axis] {
                return Err(Error::LayoutMismatch(format!(
                    "gather: shape mismatch at dim {}: {} vs {}",
                    axis,
                    layout.shape()[axis],
                    indices_layout.shape()[axis]
                )));
            }
        }

        let indices = Self::compact_indices(indices, indices_layout)?;
        for &idx in &indices {
            if idx >= src_dim {
                return Err(Error::IndexOutOfBounds(format!(
                    "gather: index {} out of bounds for dim size {}",
                    idx, src_dim
                )));
            }
        }

        match self {
            CpuStorage::F16(data) => Ok(CpuStorage::F16(gather_into(
                data, left_len, src_dim, index_len, right_len, &indices,
            ))),
            CpuStorage::F32(data) => Ok(CpuStorage::F32(gather_into(
                data, left_len, src_dim, index_len, right_len, &indices,
            ))),
            CpuStorage::I64(data) => Ok(CpuStorage::I64(gather_into(
                data, left_len, src_dim, index_len, right_len, &indices,
            ))),
        }
    }

    fn scatter_add(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &CpuStorage,
        indices_layout: &Layout,
        dst_shape: &[usize],
    ) -> Result<Self> {
        if !layout.is_compact() {
            return Err(Error::LayoutMismatch("scatter_add: source layout must be compact".into()));
        }
        if !indices_layout.is_compact() {
            return Err(Error::LayoutMismatch(
                "scatter_add: indices layout must be compact".into(),
            ));
        }
        if layout.ndim() != indices_layout.ndim() {
            return Err(Error::LayoutMismatch(format!(
                "scatter_add: rank mismatch, source has {} dims but indices has {}",
                layout.ndim(),
                indices_layout.ndim()
            )));
        }
        if layout.ndim() != dst_shape.len() {
            return Err(Error::LayoutMismatch(format!(
                "scatter_add: rank mismatch, source has {} dims but destination has {}",
                layout.ndim(),
                dst_shape.len()
            )));
        }
        if dim >= dst_shape.len() {
            return Err(Error::LayoutMismatch(format!(
                "scatter_add: dim {} out of bounds for {} dims",
                dim,
                dst_shape.len()
            )));
        }

        let left_len: usize = layout.shape().iter().take(dim).product();
        let index_len = layout.shape()[dim];
        let right_len: usize = layout.shape().iter().skip(dim + 1).product();
        let dst_dim = dst_shape[dim];
        let indices = Self::compact_indices(indices, indices_layout)?;
        for (axis, &dst_axis) in dst_shape.iter().enumerate().take(layout.ndim()) {
            if axis != dim {
                if layout.shape()[axis] != dst_axis {
                    return Err(Error::LayoutMismatch(format!(
                        "scatter_add: shape mismatch at dim {}: source {} vs destination {}",
                        axis,
                        layout.shape()[axis],
                        dst_axis
                    )));
                }
                if layout.shape()[axis] != indices_layout.shape()[axis] {
                    return Err(Error::LayoutMismatch(format!(
                        "scatter_add: shape mismatch at dim {}: source {} vs indices {}",
                        axis,
                        layout.shape()[axis],
                        indices_layout.shape()[axis]
                    )));
                }
            }
        }
        for &idx in &indices {
            if idx >= dst_dim {
                return Err(Error::IndexOutOfBounds(format!(
                    "scatter_add: index {} out of bounds for dim size {}",
                    idx, dst_dim
                )));
            }
        }

        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); dst_shape.iter().product()];
                scatter_add_into(&mut out, data, left_len, dst_dim, index_len, right_len, &indices);
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; dst_shape.iter().product()];
                scatter_add_into(&mut out, data, left_len, dst_dim, index_len, right_len, &indices);
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::I64(data) => {
                let mut out = vec![0i64; dst_shape.iter().product()];
                scatter_add_into(&mut out, data, left_len, dst_dim, index_len, right_len, &indices);
                Ok(CpuStorage::I64(out))
            }
        }
    }

    fn index_select(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
    ) -> Result<Self> {
        if !layout.is_compact() {
            return Err(Error::LayoutMismatch(
                "index_select: source layout must be compact".into(),
            ));
        }
        if !indices_layout.is_compact() {
            return Err(Error::LayoutMismatch(
                "index_select: indices layout must be compact".into(),
            ));
        }
        if indices_layout.ndim() != 1 {
            return Err(Error::LayoutMismatch(format!(
                "index_select: indices must be 1D, got {} dims",
                indices_layout.ndim()
            )));
        }
        if dim >= layout.ndim() {
            return Err(Error::LayoutMismatch(format!(
                "index_select: dim {} out of bounds for {} dims",
                dim,
                layout.ndim()
            )));
        }

        let indices = Self::compact_indices(indices, indices_layout)?;
        let left_len: usize = layout.shape().iter().take(dim).product();
        let src_dim = layout.shape()[dim];
        let right_len: usize = layout.shape().iter().skip(dim + 1).product();
        for &idx in &indices {
            if idx >= src_dim {
                return Err(Error::IndexOutOfBounds(format!(
                    "index_select: index {} out of bounds for dim size {}",
                    idx, src_dim
                )));
            }
        }

        match self {
            CpuStorage::F16(data) => {
                Ok(CpuStorage::F16(index_select_into(data, left_len, src_dim, right_len, &indices)))
            }
            CpuStorage::F32(data) => {
                Ok(CpuStorage::F32(index_select_into(data, left_len, src_dim, right_len, &indices)))
            }
            CpuStorage::I64(data) => {
                Ok(CpuStorage::I64(index_select_into(data, left_len, src_dim, right_len, &indices)))
            }
        }
    }

    fn index_add(
        &self,
        layout: &Layout,
        dim: usize,
        indices: &Self,
        indices_layout: &Layout,
        dst_shape: &[usize],
    ) -> Result<Self> {
        if !layout.is_compact() {
            return Err(Error::LayoutMismatch("index_add: source layout must be compact".into()));
        }
        if !indices_layout.is_compact() {
            return Err(Error::LayoutMismatch("index_add: indices layout must be compact".into()));
        }
        if indices_layout.ndim() != 1 {
            return Err(Error::LayoutMismatch(format!(
                "index_add: indices must be 1D, got {} dims",
                indices_layout.ndim()
            )));
        }
        if dim >= dst_shape.len() {
            return Err(Error::LayoutMismatch(format!(
                "index_add: dim {} out of bounds for {} dims",
                dim,
                dst_shape.len()
            )));
        }

        let indices = Self::compact_indices(indices, indices_layout)?;
        let left_len: usize = dst_shape[..dim].iter().product();
        let dst_dim = dst_shape[dim];
        let right_len: usize = dst_shape[(dim + 1)..].iter().product();
        let src_dim = layout.shape()[dim];

        if src_dim != indices.len() {
            return Err(Error::LayoutMismatch(format!(
                "index_add: source length along dim is {} but indices length is {}",
                src_dim,
                indices.len()
            )));
        }
        if layout.size() != left_len * src_dim * right_len {
            return Err(Error::LayoutMismatch(format!(
                "index_add: source size {} doesn't match destination shape outside indexed dim ({})",
                layout.size(),
                left_len * src_dim * right_len
            )));
        }
        for &idx in &indices {
            if idx >= dst_dim {
                return Err(Error::IndexOutOfBounds(format!(
                    "index_add: index {} out of bounds for dim size {}",
                    idx, dst_dim
                )));
            }
        }

        match self {
            CpuStorage::F16(data) => {
                let mut out = vec![f16::from_f32(0.0); dst_shape.iter().product()];
                index_add_into(&mut out, data, left_len, dst_dim, right_len, &indices);
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::F32(data) => {
                let mut out = vec![0.0f32; dst_shape.iter().product()];
                index_add_into(&mut out, data, left_len, dst_dim, right_len, &indices);
                Ok(CpuStorage::F32(out))
            }
            CpuStorage::I64(data) => {
                let mut out = vec![0i64; dst_shape.iter().product()];
                index_add_into(&mut out, data, left_len, dst_dim, right_len, &indices);
                Ok(CpuStorage::I64(out))
            }
        }
    }

    fn matmul(&self, layout: &Layout, other: &Self, layout_other: &Layout) -> Result<Self> {
        if !layout.is_compact() {
            return Err(Error::LayoutMismatch("matmul: left layout must be compact".into()));
        }
        if !layout_other.is_compact() {
            return Err(Error::LayoutMismatch("matmul: right layout must be compact".into()));
        }
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
                    CpuStorage::gemm(
                        &left[i * a_skip..],
                        &right[i * b_skip..],
                        &mut out[i * c_skip..],
                        m,
                        k,
                        n,
                        f16::from_f32(0.0),
                        f16::from_f32(1.0),
                    );
                }
                Ok(CpuStorage::F16(out))
            }
            (CpuStorage::F32(left), CpuStorage::F32(right)) => {
                let mut out = vec![0.0f32; batch * m * n];
                for i in 0..batch {
                    CpuStorage::gemm(
                        &left[i * a_skip..],
                        &right[i * b_skip..],
                        &mut out[i * c_skip..],
                        m,
                        k,
                        n,
                        0.0f32,
                        1.0f32,
                    );
                }
                Ok(CpuStorage::F32(out))
            }
            (a, b) => {
                Err(Error::DTypeMismatch(format!("matmul: {:?} vs {:?}", a.dtype(), b.dtype())))
            }
        }
    }
}

fn gather_into<T: Copy>(
    data: &[T],
    left_len: usize,
    src_dim: usize,
    index_len: usize,
    right_len: usize,
    indices: &[usize],
) -> Vec<T> {
    let mut out = Vec::with_capacity(left_len * index_len * right_len);
    for left in 0..left_len {
        let src_base = left * src_dim * right_len;
        let ids_base = left * index_len * right_len;
        for index_i in 0..index_len {
            let ids_offset = ids_base + index_i * right_len;
            for right in 0..right_len {
                let src_i = indices[ids_offset + right];
                out.push(data[src_base + src_i * right_len + right]);
            }
        }
    }
    out
}

fn scatter_add_into<T: Copy + std::ops::AddAssign>(
    out: &mut [T],
    data: &[T],
    left_len: usize,
    dst_dim: usize,
    index_len: usize,
    right_len: usize,
    indices: &[usize],
) {
    for left in 0..left_len {
        let src_base = left * index_len * right_len;
        let dst_base = left * dst_dim * right_len;
        for index_i in 0..index_len {
            let ids_offset = src_base + index_i * right_len;
            for right in 0..right_len {
                let dst_i = indices[ids_offset + right];
                out[dst_base + dst_i * right_len + right] += data[ids_offset + right];
            }
        }
    }
}

fn index_select_into<T: Copy>(
    data: &[T],
    left_len: usize,
    src_dim: usize,
    right_len: usize,
    indices: &[usize],
) -> Vec<T> {
    let mut out = Vec::with_capacity(left_len * indices.len() * right_len);
    for left in 0..left_len {
        let src_base = left * src_dim * right_len;
        for &idx in indices {
            let start = src_base + idx * right_len;
            out.extend_from_slice(&data[start..start + right_len]);
        }
    }
    out
}

fn index_add_into<T: Copy + std::ops::AddAssign>(
    out: &mut [T],
    data: &[T],
    left_len: usize,
    dst_dim: usize,
    right_len: usize,
    indices: &[usize],
) {
    let src_dim = indices.len();
    for left in 0..left_len {
        let src_base = left * src_dim * right_len;
        let dst_base = left * dst_dim * right_len;
        for (src_i, &dst_i) in indices.iter().enumerate() {
            let src_offset = src_base + src_i * right_len;
            let dst_offset = dst_base + dst_i * right_len;
            for right in 0..right_len {
                out[dst_offset + right] += data[src_offset + right];
            }
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
            strided_unary_op_inner(
                a,
                a_off + i * sa,
                &a_strides[1..],
                out,
                out_off,
                &shape[1..],
                f,
            );
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
            copy_strided_inner(
                src,
                dst,
                src_offset + i * stride,
                &shape[1..],
                &strides[1..],
                dst_offset,
            );
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
