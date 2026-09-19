//! One tokenizer type per encoding, joined by a shared contract.
//!
//! Each encoding owns its module and its type: GPT-family tokenizers live in
//! [`gpt`], the Qwen tokenizer and its chat template in [`qwen3`], and the
//! Qwen3.5 encoding in [`qwen3_5`]. Dataset
//! plumbing that works with any encoding lives in [`bins`] and stays generic
//! over the [`Tokenizer`] trait, so it never names an encoding.

mod bins;
mod gpt;
mod qwen3;
mod qwen3_5;

pub use bins::{TokenBinPaths, prepare_text_token_bins};
pub use gpt::{Cl100kTokenizer, Gpt2Tokenizer};
pub use qwen3::{ChatMessage, Qwen3Tokenizer};
pub use qwen3_5::{ChatMessage as Qwen3_5ChatMessage, Qwen3_5Tokenizer};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::Result;

/// Text to token ids and back, with a vocabulary size.
///
/// Every encoding honors this contract, so dataset code takes `&impl
/// Tokenizer` and stays encoding-independent.
pub trait Tokenizer {
    /// Encodes text into token ids, including special tokens.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decodes token ids back into text.
    fn decode(&self, tokens: &[u32]) -> String;

    /// Decodes token ids back into text, replacing invalid UTF-8 with the
    /// standard replacement character instead of panicking.
    fn decode_lossy(&self, tokens: &[u32]) -> String;

    /// Returns the tokenizer vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Tokenizes a text file, reading line by line.
    ///
    /// Returns the full token stream. This is useful for building a
    /// [`TextDataset`](crate::dataset::TextDataset) via
    /// [`TextDataset::from_tokens`](crate::dataset::TextDataset::from_tokens).
    fn tokenize_file(&self, path: &Path) -> Result<Vec<u32>> {
        let input = File::open(path)?;
        let mut reader = BufReader::new(input);
        let mut tokens = Vec::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }
            tokens.extend(self.encode(&line));
        }

        Ok(tokens)
    }
}
