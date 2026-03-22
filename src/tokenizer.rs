//! BPE tokenizer wrapping tiktoken-rs.
//!
//! Provides encode/decode for text ↔ token ids, plus helpers for
//! tokenizing text files into flat binary token bins for training.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::{fs, fs::File};

use tiktoken_rs::CoreBPE;

use crate::error::Result;

/// Thin wrapper around a `tiktoken-rs` BPE tokenizer.
pub struct Tokenizer {
    bpe: CoreBPE,
    vocab_size: usize,
}

impl Tokenizer {
    /// Creates a tokenizer using GPT-2's r50k_base encoding (50257 vocab).
    pub fn gpt2() -> Self {
        let bpe = tiktoken_rs::r50k_base().expect("failed to load r50k_base");
        // r50k_base: 50256 base tokens + 1 special token (<|endoftext|>)
        Self { bpe, vocab_size: 50257 }
    }

    /// Creates a tokenizer using OpenAI's cl100k_base encoding (100K vocab).
    pub fn cl100k_base() -> Self {
        let bpe = tiktoken_rs::cl100k_base().expect("failed to load cl100k_base");
        // cl100k_base: 100256 base tokens + 5 special tokens
        // (<|endoftext|>, <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>, <|endofprompt|>)
        Self { bpe, vocab_size: 100261 }
    }

    /// Returns the tokenizer vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encodes text into token ids, including special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    /// Decodes token ids back into text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.bpe.decode(tokens.to_vec()).expect("failed to decode tokens")
    }

    /// Decodes token ids back into text, replacing invalid UTF-8 with the
    /// standard replacement character instead of panicking.
    pub fn decode_lossy(&self, tokens: &[u32]) -> String {
        let bytes =
            self.bpe._decode_native_and_split(tokens.to_vec()).flatten().collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Tokenizes a text file, reading line by line.
    ///
    /// Returns the full token stream. This is useful for building a
    /// [`TextDataset`](crate::dataset::TextDataset) via
    /// [`TextDataset::from_tokens`](crate::dataset::TextDataset::from_tokens).
    pub fn tokenize_file(&self, path: &Path) -> Result<Vec<u32>> {
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

/// Paths for a prepared token-bin dataset.
pub struct TokenBinPaths {
    /// Path to the training token bin.
    pub train: PathBuf,
    /// Path to the validation token bin.
    pub val: PathBuf,
}

/// Tokenizes a text corpus into flat binary token bins.
///
/// The full token stream is first written to a temporary `all.bin`, then split
/// contiguously into `train.bin` and `val.bin`, preserving the "last chunk is
/// validation" behavior common in language-model examples.
pub fn prepare_text_token_bins(
    text_path: &Path,
    tokenizer: &Tokenizer,
    out_dir: &Path,
    val_ratio: f32,
) -> Result<TokenBinPaths> {
    assert!((0.0..1.0).contains(&val_ratio), "val_ratio must be in [0, 1)");

    fs::create_dir_all(out_dir)?;

    let all_path = out_dir.join("all.bin");
    let train_path = out_dir.join("train.bin");
    let val_path = out_dir.join("val.bin");

    let total_tokens = tokenize_text_file_to_bin(text_path, tokenizer, &all_path)?;
    let val_tokens = ((total_tokens as f32) * val_ratio).round() as usize;
    let val_tokens = val_tokens.max(1).min(total_tokens.saturating_sub(1));
    let train_tokens = total_tokens - val_tokens;

    println!(
        "Splitting token bins: train={} tokens, val={} tokens ({:.1}%)",
        train_tokens,
        val_tokens,
        val_ratio * 100.0
    );
    split_token_bin(&all_path, &train_path, train_tokens, &val_path)?;
    fs::remove_file(&all_path)?;

    Ok(TokenBinPaths { train: train_path, val: val_path })
}

fn tokenize_text_file_to_bin(path: &Path, tokenizer: &Tokenizer, out_path: &Path) -> Result<usize> {
    let input = File::open(path)?;
    let mut reader = BufReader::new(input);
    let output = File::create(out_path)?;
    let mut writer = BufWriter::new(output);
    let total_bytes = std::fs::metadata(path)?.len() as usize;
    let report_bytes = (64 * 1024 * 1024).min(total_bytes.max(1));
    let mut next_report = report_bytes;
    let mut processed_bytes = 0usize;
    let mut total_tokens = 0usize;
    let mut line = String::new();
    let mut printed_progress = false;

    println!(
        "Tokenizing {} ({:.1} MiB) into {}...",
        path.display(),
        format_mib(total_bytes),
        out_path.display()
    );

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        processed_bytes += bytes_read;

        for token in tokenizer.encode(&line) {
            let token = u16::try_from(token).expect("token id must fit in u16");
            writer.write_all(&token.to_le_bytes())?;
            total_tokens += 1;
        }

        if processed_bytes >= next_report || processed_bytes == total_bytes {
            print!(
                "\r  prepared {:.1}% | {:.1}/{:.1} MiB | {} tokens",
                progress_pct(processed_bytes, total_bytes),
                format_mib(processed_bytes),
                format_mib(total_bytes),
                total_tokens
            );
            std::io::stdout().flush().unwrap();
            printed_progress = true;
            next_report = next_report.saturating_add(report_bytes);
        }
    }

    writer.flush()?;
    if printed_progress {
        println!();
    }
    println!("Finished tokenizing: {} tokens written to {}", total_tokens, out_path.display());
    Ok(total_tokens)
}

fn format_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn progress_pct(processed_bytes: usize, total_bytes: usize) -> f64 {
    100.0 * processed_bytes as f64 / total_bytes.max(1) as f64
}

fn split_token_bin(
    all_path: &Path,
    train_path: &Path,
    train_tokens: usize,
    val_path: &Path,
) -> Result<()> {
    let bytes = std::fs::read(all_path)?;
    let split_at = train_tokens * 2;
    std::fs::write(train_path, &bytes[..split_at])?;
    std::fs::write(val_path, &bytes[split_at..])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        // Arrange
        let tok = Tokenizer::cl100k_base();
        let text = "Hello, world!";

        // Act
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);

        // Assert
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_produces_tokens() {
        // Arrange
        let tok = Tokenizer::cl100k_base();

        // Act
        let tokens = tok.encode("The quick brown fox");

        // Assert
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|&t| (t as usize) < tok.vocab_size()));
    }

    #[test]
    fn test_vocab_size() {
        // Arrange / Act
        let tok = Tokenizer::cl100k_base();

        // Assert
        assert_eq!(tok.vocab_size(), 100261);
    }

    #[test]
    fn test_special_tokens_within_vocab() {
        // Arrange
        let tok = Tokenizer::cl100k_base();

        // Act — encode text containing <|endoftext|>
        let tokens = tok.encode("hello <|endoftext|> world");

        // Assert — all token ids must be < vocab_size
        assert!(tokens.iter().all(|&t| (t as usize) < tok.vocab_size()));
    }

    #[test]
    fn test_prepare_text_token_bins_roundtrips() {
        use crate::dataset::TokenBinDataset;

        // Arrange
        let dir = std::env::temp_dir().join("deers_prepare_token_bins_test");
        std::fs::create_dir_all(&dir).unwrap();
        let text_path = dir.join("tiny.txt");
        std::fs::write(&text_path, "Once upon a time.\nThere was a cat.\n".repeat(32)).unwrap();
        let tokenizer = Tokenizer::gpt2();

        // Act
        let paths = prepare_text_token_bins(&text_path, &tokenizer, &dir, 0.25).unwrap();
        let train = TokenBinDataset::load(&paths.train, 8).unwrap();
        let val = TokenBinDataset::load(&paths.val, 8).unwrap();

        // Assert
        assert!(paths.train.exists());
        assert!(paths.val.exists());
        assert!(train.num_tokens() > val.num_tokens());
        assert!(val.num_tokens() > 0);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }
}
