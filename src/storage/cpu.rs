use std::{borrow::Borrow, iter::zip};

use crate::{
    dtype::{DType, WithDType},
    error::Result,
    layout::Layout,
    storage::{BackendStorage, BinaryOp, UnaryOp},
};

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
        assert!(
            src_layout.size() <= dst.len(),
            "Destination storage has insufficient space"
        );

        match dst {
            CpuStorage::F32(dst) => {
                for (i, v) in self.iter::<f32>(src_layout).enumerate() {
                    dst[i] = *v;
                }
            }
            CpuStorage::F64(dst) => {
                for (i, v) in self.iter::<f64>(src_layout).enumerate() {
                    dst[i] = *v;
                }
            }
        }

        Ok(())
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
