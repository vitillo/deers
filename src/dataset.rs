#![allow(dead_code)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::{fs, fs::File};

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
/// Loads a text file, tokenizes it, then packs the token stream into
/// fixed-length rows of `seq_len + 1`. Each row yields an input window
/// `row[..seq_len]` and a target window `row[1..]`, so the model learns
/// to predict the next token at every position.
///
/// This is the simplest approach: concatenate all tokens and chunk.
/// Tail tokens that don't fill a complete row are discarded.
pub struct TextDataset {
    /// All rows packed into a single `(num_sequences, seq_len + 1)` i64 tensor.
    pub data: Tensor,
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

        // Tokenize in ~1MB chunks instead of the whole file at once.
        // tiktoken is very slow on multi-GB strings; chunking avoids that.
        // We split on newline boundaries so BPE results are identical.
        const CHUNK_SIZE: usize = 1 << 20; // 1 MB
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut start = 0;
        while start < bytes.len() {
            let mut end = (start + CHUNK_SIZE).min(bytes.len());
            // Extend to the next newline to avoid splitting mid-line.
            if end < bytes.len() {
                if let Some(nl) = bytes[end..].iter().position(|&b| b == b'\n') {
                    end += nl + 1;
                } else {
                    end = bytes.len();
                }
            }
            tokens.extend(tokenizer.encode(&text[start..end]));
            start = end;
        }

        let row_len = seq_len + 1;
        let num_sequences = tokens.len() / row_len;
        assert!(num_sequences > 0, "text file too short for seq_len={seq_len}");

        // Truncate to an exact multiple of row_len and cast to i64.
        let tokens: Vec<i64> =
            tokens[..num_sequences * row_len].iter().map(|&t| t as i64).collect();

        let data = Tensor::from_vec(tokens, (num_sequences, row_len), Device::Cpu);

        Ok(Self { data, seq_len, vocab_size: tokenizer.vocab_size() })
    }

    /// Downloads a text file from `url` (caching in `cache_dir`), then loads it.
    ///
    /// The filename is derived from the URL. If the file already exists in
    /// `cache_dir` it is reused without re-downloading.
    pub fn from_url(
        url: &str,
        cache_dir: &Path,
        tokenizer: &Tokenizer,
        seq_len: usize,
    ) -> Result<Self> {
        let filename = url.rsplit('/').next().expect("url has no path component");
        let path: PathBuf = cache_dir.join(filename);
        download_if_missing(url, &path);
        Self::load(&path, tokenizer, seq_len)
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

/// Downloads `url` to `path` if the file doesn't already exist.
///
/// Creates parent directories as needed. Uses a `.part` suffix during
/// download so a partial file won't be mistaken for a cached copy.
fn download_if_missing(url: &str, path: &Path) {
    if path.exists() {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let part_path = path.with_extension("part");
    println!("Downloading {url}");
    let response = ureq::get(url).call().expect("download failed");
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&part_path).unwrap();
    std::io::copy(&mut reader, &mut file).unwrap();
    fs::rename(&part_path, path).unwrap();
    println!("Saved to {}", path.display());
}

/// Decompresses a `.gz` file to the given output path.
fn decompress_gz(gz_path: &Path, out_path: &Path) {
    let gz_file = File::open(gz_path).unwrap();
    let mut decoder = flate2::read::GzDecoder::new(gz_file);
    let mut out_file = File::create(out_path).unwrap();
    std::io::copy(&mut decoder, &mut out_file).unwrap();
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
        assert_eq!(*dataset.inputs().layout().shape(), Shape::from((num_seq, seq_len)));
        assert_eq!(*dataset.targets().layout().shape(), Shape::from((num_seq, seq_len)));

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
