//! Dataset loaders for training and evaluation.
//!
//! Includes MNIST (auto-downloaded IDX files), a text dataset for
//! next-token prediction, and a flat token-bin format for large corpora.

#![allow(dead_code)]

use rand::RngExt;
use std::io::Read;
use std::path::Path;
use std::{fs, fs::File};

use crate::error::Result;
use crate::{Device, Tensor};

/// The MNIST handwritten digit dataset (60k train + 10k test images).
///
/// Images are `(N, 28, 28)` tensors with pixel values normalized to `[0, 1]`.
/// Labels are `(N,)` tensors with integer class values `0..=9` stored as i64.
#[derive(Debug)]
pub struct MNISTDataset {
    /// Training images shaped `(60000, 28, 28)`.
    pub train_images: Tensor,
    /// Training labels shaped `(60000,)`.
    pub train_labels: Tensor,
    /// Test images shaped `(10000, 28, 28)`.
    pub test_images: Tensor,
    /// Test labels shaped `(10000,)`.
    pub test_labels: Tensor,
    /// Number of digit classes in the dataset.
    pub num_classes: usize,
}

impl MNISTDataset {
    fn parse_images(image_path: &Path) -> Result<Tensor> {
        let mut file = File::open(image_path)?;
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

    /// Loads MNIST from IDX files in `data/mnist/`, downloading if needed.
    pub fn load() -> Result<Self> {
        let base_url = "https://ossci-datasets.s3.amazonaws.com/mnist";
        let dir = Path::new("data/mnist");
        let files = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ];
        for file in &files {
            let url = format!("{base_url}/{file}.gz");
            let gz_path = dir.join(format!("{file}.gz"));
            let final_path = dir.join(file);
            if !final_path.exists() {
                download_if_missing(&url, &gz_path);
                decompress_gz(&gz_path, &final_path);
                fs::remove_file(&gz_path).ok();
            }
        }

        let train_images = Self::parse_images(&dir.join("train-images-idx3-ubyte"))?;
        let train_labels = Self::parse_labels(&dir.join("train-labels-idx1-ubyte"))?;
        let test_images = Self::parse_images(&dir.join("t10k-images-idx3-ubyte"))?;
        let test_labels = Self::parse_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
        Ok(Self { train_images, train_labels, test_images, test_labels, num_classes: 10 })
    }
}

/// A text dataset for language modelling (nanoGPT-style).
///
/// Packs a pre-tokenized stream into fixed-length rows of `seq_len + 1`.
/// Each row yields an input window `row[..seq_len]` and a target window
/// `row[1..]`, so the model learns to predict the next token at every
/// position.
///
/// This is the simplest approach: concatenate all tokens and chunk.
/// Tail tokens that don't fill a complete row are discarded.
#[derive(Debug)]
pub struct TextDataset {
    /// All rows packed into a single `(num_sequences, seq_len + 1)` i64 tensor.
    pub data: Tensor,
    /// Context length used to form input/target windows.
    pub seq_len: usize,
    /// Vocabulary size reported by the tokenizer that produced these tokens.
    pub vocab_size: usize,
}

impl TextDataset {
    /// Creates a dataset from a pre-tokenized stream.
    ///
    /// `tokens` is the flat token stream (e.g. from [`Tokenizer::encode`]).
    /// `vocab_size` should match the tokenizer that produced the tokens.
    /// `seq_len` is the context window the model sees (e.g. 256 or 1024).
    ///
    /// The tokens are packed into rows of `seq_len + 1`; any remainder is
    /// discarded.
    pub fn from_tokens(tokens: &[u32], vocab_size: usize, seq_len: usize) -> Self {
        let row_len = seq_len + 1;
        let num_sequences = tokens.len() / row_len;
        assert!(num_sequences > 0, "token stream too short for seq_len={seq_len}");

        let tokens: Vec<i64> =
            tokens[..num_sequences * row_len].iter().map(|&t| t as i64).collect();
        let data = Tensor::from_vec(tokens, (num_sequences, row_len), Device::Cpu);

        Self { data, seq_len, vocab_size }
    }

    /// Number of sequences (rows) in the dataset.
    pub fn len(&self) -> usize {
        self.data.layout().shape()[0]
    }

    /// Returns whether the dataset contains no complete sequences.
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

/// A flat token dataset stored as raw little-endian `u16` token ids.
///
/// This keeps the full TinyStories token stream compact on disk and in RAM,
/// then samples random contiguous windows at training time instead of
/// materializing the entire corpus as a 2D `i64` tensor.
pub struct TokenBinDataset {
    tokens: Vec<u16>,
    seq_len: usize,
}

impl std::fmt::Debug for TokenBinDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBinDataset")
            .field("num_tokens", &self.tokens.len())
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

impl TokenBinDataset {
    /// Loads a flat `u16` token bin from disk.
    pub fn load(path: &Path, seq_len: usize) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        assert!(bytes.len().is_multiple_of(2), "token bin must contain u16 values");

        let tokens = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();

        assert!(tokens.len() > seq_len, "token bin too short for seq_len={seq_len}");
        Ok(Self { tokens, seq_len })
    }

    /// Returns the number of tokens in the underlying flat stream.
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Samples a random batch of contiguous token windows.
    pub fn sample_batch(&self, batch_size: usize, device: Device) -> (Tensor, Tensor) {
        let mut rng = rand::rng();
        let max_start = self.tokens.len() - (self.seq_len + 1);
        let starts = (0..batch_size).map(|_| rng.random_range(0..=max_start)).collect::<Vec<_>>();
        self.batch_from_starts(&starts, device)
    }

    /// Builds a batch from explicit start offsets into the token stream.
    pub fn batch_from_starts(&self, starts: &[usize], device: Device) -> (Tensor, Tensor) {
        let mut inputs = Vec::with_capacity(starts.len() * self.seq_len);
        let mut targets = Vec::with_capacity(starts.len() * self.seq_len);

        for &start in starts {
            let row = &self.tokens[start..start + self.seq_len + 1];
            inputs.extend(row[..self.seq_len].iter().map(|&token| i64::from(token)));
            targets.extend(row[1..].iter().map(|&token| i64::from(token)));
        }

        let shape = (starts.len(), self.seq_len);
        (Tensor::from_vec(inputs, shape, device), Tensor::from_vec(targets, shape, device))
    }
}

/// Downloads `url` to `path` if the file doesn't already exist.
///
/// Creates parent directories as needed. Uses a `.part` suffix during
/// download so a partial file won't be mistaken for a cached copy.
fn download_if_missing(url: &str, path: &Path) {
    if path.exists() {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("failed to create download directory");
    }

    let part_path = path.with_extension("part");
    println!("Downloading {url}");
    let response = ureq::get(url).call().expect("download failed");
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&part_path).expect("failed to create download file");
    std::io::copy(&mut reader, &mut file).expect("failed to write downloaded data");
    fs::rename(&part_path, path).expect("failed to rename downloaded file");
    println!("Saved to {}", path.display());
}

/// Decompresses a `.gz` file to the given output path.
fn decompress_gz(gz_path: &Path, out_path: &Path) {
    let gz_file = File::open(gz_path).expect("failed to open gzip file");
    let mut decoder = flate2::read::GzDecoder::new(gz_file);
    let mut out_file = File::create(out_path).expect("failed to create decompressed output file");
    std::io::copy(&mut decoder, &mut out_file).expect("failed to decompress gzip data");
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
    fn test_text_dataset_from_tokens() {
        // Arrange — a synthetic token stream.
        let seq_len = 16;
        let vocab_size = 100;
        let tokens: Vec<u32> = (0..200).map(|i| i % vocab_size).collect();

        // Act
        let dataset = TextDataset::from_tokens(&tokens, vocab_size as usize, seq_len);

        // Assert
        let num_seq = dataset.len();
        assert!(num_seq > 0);
        assert_eq!(*dataset.inputs().layout().shape(), Shape::from((num_seq, seq_len)));
        assert_eq!(*dataset.targets().layout().shape(), Shape::from((num_seq, seq_len)));

        // Verify the shift: for the first row, target[i] == data[0][i+1].
        let row: Vec<i64> = dataset.data.narrow(0, 0, 1).to_vec().unwrap();
        assert_eq!(row.len(), seq_len + 1);
        assert_eq!(&row[1..], &row[1..seq_len + 1]);
    }

    #[test]
    fn test_token_bin_dataset_batch_from_starts() {
        // Arrange
        let dir = std::env::temp_dir().join("deers_token_bin_dataset_test");
        std::fs::create_dir_all(&dir).unwrap();
        let bin_path = dir.join("train.bin");
        let tokens = (0..32u16).collect::<Vec<_>>();
        let bytes = tokens.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>();
        std::fs::write(&bin_path, bytes).unwrap();
        let dataset = TokenBinDataset::load(&bin_path, 4).unwrap();

        // Act
        let (inputs, targets) = dataset.batch_from_starts(&[0, 5], Device::Cpu);

        // Assert
        assert_eq!(inputs.to_vec::<i64>().unwrap(), vec![0, 1, 2, 3, 5, 6, 7, 8]);
        assert_eq!(targets.to_vec::<i64>().unwrap(), vec![1, 2, 3, 4, 6, 7, 8, 9]);

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
