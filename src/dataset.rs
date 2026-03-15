#![allow(dead_code)]

use std::io::Read;
use std::{fs::File, path::Path};

use crate::error::Result;

use crate::{Device, Tensor};

/// The MNIST handwritten digit dataset (60k train + 10k test images).
///
/// Images are `(N, 28, 28)` tensors with pixel values normalized to `[0, 1]`.
/// Labels are `(N,)` tensors with integer class values `0..=9` stored as f32.
pub struct MNISTDataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub num_classes: usize,
}

impl MNISTDataset {
    fn parse_images(image_path: &Path) -> Result<Tensor> {
        let mut file = File::open(image_path).unwrap();
        let magic = read_u32(&mut file)?;
        assert_eq!(magic, 2051);

        let num_images = read_u32(&mut file)? as usize;
        let num_rows = read_u32(&mut file)? as usize;
        let num_cols = read_u32(&mut file)? as usize;
        assert_eq!(28, num_rows);
        assert_eq!(28, num_cols);

        let mut images = vec![0u8; num_images * num_rows * num_cols];
        file.read_exact(&mut images)?;
        let images = images
            .into_iter()
            .map(|v| v as f32 / 255.0)
            .collect::<Vec<f32>>();

        Ok(Tensor::from_vec(
            images,
            (num_images, num_rows, num_cols),
            Device::Cpu,
        ))
    }

    fn parse_labels(label_path: &Path) -> Result<Tensor> {
        let mut file = File::open(label_path)?;
        let magic = read_u32(&mut file)?;
        assert_eq!(magic, 2049);

        let num_labels = read_u32(&mut file)? as usize;

        let mut labels = vec![0u8; num_labels];
        file.read_exact(&mut labels)?;
        let labels = labels.into_iter().map(|v| v as f32).collect::<Vec<f32>>();

        Ok(Tensor::from_vec(labels, (num_labels,), Device::Cpu))
    }

    /// Loads MNIST from IDX files in `data/mnist/`.
    pub fn load() -> Result<Self> {
        let train_images = Self::parse_images(Path::new("data/mnist/train-images-idx3-ubyte"))?;
        let train_labels = Self::parse_labels(Path::new("data/mnist/train-labels-idx1-ubyte"))?;
        let test_images = Self::parse_images(Path::new("data/mnist/t10k-images-idx3-ubyte"))?;
        let test_labels = Self::parse_labels(Path::new("data/mnist/t10k-labels-idx1-ubyte"))?;
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_classes: 10,
        })
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Shape;

    #[test]
    fn test_mnist_dataset() {
        // See https://github.com/huggingface/candle/blob/17cbbe4286f25934197db79a244fd0694259c899/candle-examples/examples/mnist-training/main.rs#L251
        // See https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist?tabs=azure-storage

        let dataset = MNISTDataset::load().unwrap();

        assert_eq!(
            *dataset.train_images.layout().shape(),
            Shape::from((60000, 28, 28))
        );
        assert_eq!(
            *dataset.train_labels.layout().shape(),
            Shape::from((60000,))
        );
        assert_eq!(
            *dataset.test_images.layout().shape(),
            Shape::from((10000, 28, 28))
        );
        assert_eq!(*dataset.test_labels.layout().shape(), Shape::from((10000,)));
    }
}
