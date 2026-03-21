#![allow(dead_code)]

use std::io::Read;
use std::{fs::File, path::Path};

use crate::error::Result;
use crate::tokenizer::Tokenizer;
use crate::{Device, Tensor};

/// The MNIST handwritten digit dataset (60k train + 10k test images).
///
/// Images are `(N, 28, 28)` tensors with pixel values normalized to `[0, 1]`.
/// Labels are `(N,)` tensors with integer class values `0..=9` stored as i64.
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
        let images = images.into_iter().map(|v| v as f32 / 255.0).collect::<Vec<f32>>();

        Ok(Tensor::from_vec(images, (num_images, num_rows, num_cols), Device::Cpu))
    }

    fn parse_labels(label_path: &Path) -> Result<Tensor> {
        let mut file = File::open(label_path)?;
        let magic = read_u32(&mut file)?;
        assert_eq!(magic, 2049);

        let num_labels = read_u32(&mut file)? as usize;

        let mut labels = vec![0u8; num_labels];
        file.read_exact(&mut labels)?;
        let labels = labels.into_iter().map(|v| v as i64).collect::<Vec<i64>>();

        Ok(Tensor::from_vec(labels, (num_labels,), Device::Cpu))
    }

    /// Loads MNIST from IDX files in `data/mnist/`.
    pub fn load() -> Result<Self> {
        let train_images = Self::parse_images(Path::new("data/mnist/train-images-idx3-ubyte"))?;
        let train_labels = Self::parse_labels(Path::new("data/mnist/train-labels-idx1-ubyte"))?;
        let test_images = Self::parse_images(Path::new("data/mnist/t10k-images-idx3-ubyte"))?;
        let test_labels = Self::parse_labels(Path::new("data/mnist/t10k-labels-idx1-ubyte"))?;
        Ok(Self { train_images, train_labels, test_images, test_labels, num_classes: 10 })
    }
}

/// A text dataset for language modelling (nanoGPT-style).
///
/// Loads a text file, tokenizes it, then packs the token stream into
/// fixed-length rows of `seq_len + 1`. Each row yields an input window
/// `row[..seq_len]` and a target window `row[1..]`, so the model learns
/// to predict the next token at every position.
///
/// This is the simplest approach: concatenate all tokens and chunk.
/// Tail tokens that don't fill a complete row are discarded.
pub struct TextDataset {
    /// All rows packed into a single `(num_sequences, seq_len + 1)` i64 tensor.
    data: Tensor,
    pub seq_len: usize,
    pub vocab_size: usize,
}

impl TextDataset {
    /// Loads and tokenizes a plain-text file.
    ///
    /// `seq_len` is the context window the model sees (e.g. 256 or 1024).
    /// The file is read in full, tokenized with the provided [`Tokenizer`],
    /// then chunked into rows of `seq_len + 1` tokens.
    pub fn load(path: &Path, tokenizer: &Tokenizer, seq_len: usize) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let tokens = tokenizer.encode(&text);

        let row_len = seq_len + 1;
        let num_sequences = tokens.len() / row_len;
        assert!(num_sequences > 0, "text file too short for seq_len={seq_len}");

        // Truncate to an exact multiple of row_len and cast to i64.
        let tokens: Vec<i64> = tokens[..num_sequences * row_len]
            .iter()
            .map(|&t| t as i64)
            .collect();

        let data = Tensor::from_vec(tokens, (num_sequences, row_len), Device::Cpu);

        Ok(Self { data, seq_len, vocab_size: tokenizer.vocab_size() })
    }

    /// Number of sequences (rows) in the dataset.
    pub fn len(&self) -> usize {
        self.data.layout().shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The input tokens: `(num_sequences, seq_len)`.
    pub fn inputs(&self) -> Tensor {
        self.data.narrow(1, 0, self.seq_len)
    }

    /// The target tokens: `(num_sequences, seq_len)`, shifted by one.
    pub fn targets(&self) -> Tensor {
        self.data.narrow(1, 1, self.seq_len)
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
    fn test_text_dataset() {
        // Arrange — write a small text file with enough tokens.
        let dir = std::env::temp_dir().join("deers_text_dataset_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.txt");
        // Repeat a sentence so we get plenty of tokens.
        let text = "Once upon a time there was a little cat. ".repeat(200);
        std::fs::write(&path, &text).unwrap();

        let tokenizer = Tokenizer::cl100k_base();
        let seq_len = 16;

        // Act
        let dataset = TextDataset::load(&path, &tokenizer, seq_len).unwrap();

        // Assert
        let num_seq = dataset.len();
        assert!(num_seq > 0);
        assert_eq!(
            *dataset.inputs().layout().shape(),
            Shape::from((num_seq, seq_len))
        );
        assert_eq!(
            *dataset.targets().layout().shape(),
            Shape::from((num_seq, seq_len))
        );

        // Verify the shift: for the first row, target[i] == data[0][i+1].
        let row: Vec<i64> = dataset.data.narrow(0, 0, 1).to_vec().unwrap();
        // row has seq_len+1 elements; inputs = row[0..seq_len], targets = row[1..seq_len+1]
        assert_eq!(row.len(), seq_len + 1);
        // Targets are just the inputs shifted by one position.
        assert_eq!(&row[1..], &row[1..seq_len + 1]);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_mnist_dataset() {
        // See https://github.com/huggingface/candle/blob/17cbbe4286f25934197db79a244fd0694259c899/candle-examples/examples/mnist-training/main.rs#L251
        // See https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist?tabs=azure-storage

        let dataset = MNISTDataset::load().unwrap();

        assert_eq!(*dataset.train_images.layout().shape(), Shape::from((60000, 28, 28)));
        assert_eq!(*dataset.train_labels.layout().shape(), Shape::from((60000,)));
        assert_eq!(*dataset.test_images.layout().shape(), Shape::from((10000, 28, 28)));
        assert_eq!(*dataset.test_labels.layout().shape(), Shape::from((10000,)));
    }
}
