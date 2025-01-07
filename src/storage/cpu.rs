use crate::{
    dtype::{DType, WithDType},
    error::Result,
    storage::{BackendStorage, BinaryOp, UnaryOp},
};

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
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
    fn ewise_powf(&self, e: f64) -> Result<Self> {
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

    fn unary_op<O: UnaryOp>(&self, op: O) -> Result<Self> {
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

    fn binary_op<O: BinaryOp>(&self, other: &Self) -> Result<Self> {
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

    fn to_vec<D: WithDType>(&self) -> Vec<D> {
        D::to_vec(self)
    }
}
