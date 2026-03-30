//! Safetensors-based checkpoint serialization for model and optimizer state.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use half::f16;
use safetensors::tensor::View;
use safetensors::{Dtype as SafeDtype, SafeTensors, serialize_to_file};

use crate::error::{Error, Result};
use crate::{DType, Device, Tensor};

struct TensorBlob {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &TensorBlob {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Saves a named tensor map to a safetensors file at `path`.
pub fn save_tensors(path: &Path, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    let mut blobs = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        blobs.push((name.clone(), tensor_blob(tensor)?));
    }
    let views = blobs.iter().map(|(name, tensor)| (name.as_str(), tensor)).collect::<Vec<_>>();
    serialize_to_file(views, None, path)?;
    Ok(())
}

/// Loads a named tensor map from a safetensors file onto `device`.
pub fn load_tensors(path: &Path, device: Device) -> Result<BTreeMap<String, Tensor>> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut loaded = BTreeMap::new();

    for name in tensors.names() {
        let tensor = read_tensor(&tensors, name, device)?;
        loaded.insert(name.to_owned(), tensor);
    }

    Ok(loaded)
}

fn read_tensor(tensors: &SafeTensors<'_>, name: &str, device: Device) -> Result<Tensor> {
    let view = tensors.tensor(name)?;
    let shape = view.shape().to_vec();
    let tensor = match view.dtype() {
        SafeDtype::F16 => {
            let values = view
                .data()
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        SafeDtype::F32 => {
            let values = view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        SafeDtype::I64 => {
            let values = view
                .data()
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)
        }
        other => {
            return Err(Error::Checkpoint(format!(
                "unsupported safetensors dtype in checkpoint: {other:?}"
            )));
        }
    };
    Ok(tensor)
}

fn tensor_blob(tensor: &Tensor) -> Result<TensorBlob> {
    let shape = tensor.layout().shape().iter().copied().collect::<Vec<_>>();
    let blob = match tensor.dtype() {
        DType::F16 => {
            let mut data = Vec::with_capacity(tensor.layout().size() * 2);
            for value in tensor.to_vec::<f16>()? {
                data.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            TensorBlob { dtype: SafeDtype::F16, shape, data }
        }
        DType::F32 => {
            let mut data = Vec::with_capacity(tensor.layout().size() * 4);
            for value in tensor.to_vec::<f32>()? {
                data.extend_from_slice(&value.to_le_bytes());
            }
            TensorBlob { dtype: SafeDtype::F32, shape, data }
        }
        DType::I64 => {
            let mut data = Vec::with_capacity(tensor.layout().size() * 8);
            for value in tensor.to_vec::<i64>()? {
                data.extend_from_slice(&value.to_le_bytes());
            }
            TensorBlob { dtype: SafeDtype::I64, shape, data }
        }
    };
    Ok(blob)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use super::{load_tensors, save_tensors};
    use crate::{Device, Tensor};

    #[test]
    fn test_save_and_load_tensors_roundtrip() {
        // Arrange
        let path = std::env::temp_dir().join(format!(
            "deers-checkpoint-roundtrip-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "linear.weight".to_owned(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), Device::Cpu),
        );
        tensors.insert("tokens".to_owned(), Tensor::from_vec(vec![1i64, 2, 3], (3,), Device::Cpu));

        // Act
        save_tensors(&path, &tensors).unwrap();
        let loaded = load_tensors(&path, Device::Cpu).unwrap();

        // Assert
        assert_eq!(loaded["linear.weight"].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["tokens"].to_vec::<i64>().unwrap(), vec![1, 2, 3]);

        let _ = fs::remove_file(path);
    }
}
