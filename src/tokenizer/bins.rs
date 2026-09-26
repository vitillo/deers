//! Encoding-independent helpers that turn a text corpus into token bins.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::{fs, fs::File};

use super::Tokenizer;
use crate::error::Result;

/// Magic prefix identifying the versioned token-bin format.
///
/// Versioned bins store raw little-endian `u32` ids after this header, so a
/// headerless bin from the old `u16` format is rejected instead of misread.
pub const TOKEN_BIN_MAGIC: &[u8; 8] = b"DEERSTB\x01";
/// Bytes per token id in the versioned format.
const TOKEN_BIN_ITEM_BYTES: usize = 4;
/// Largest token id the versioned token-bin format can store.
///
/// This covers Qwen3 (151,936 ids) and Qwen3.5 (248,320 ids) vocabularies.
const MAX_TOKEN_BIN_ID: u32 = u32::MAX;

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
    tokenizer: &impl Tokenizer,
    out_dir: &Path,
    val_ratio: f32,
) -> Result<TokenBinPaths> {
    assert!((0.0..1.0).contains(&val_ratio), "val_ratio must be in [0, 1)");
    let vocab_size = tokenizer.vocab_size();
    assert!(
        vocab_size as u64 <= u64::from(MAX_TOKEN_BIN_ID) + 1,
        "tokenizer vocab size {vocab_size} exceeds supported token-bin id range 0..={MAX_TOKEN_BIN_ID}"
    );

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

fn tokenize_text_file_to_bin(
    path: &Path,
    tokenizer: &impl Tokenizer,
    out_path: &Path,
) -> Result<usize> {
    let input = File::open(path)?;
    let mut reader = BufReader::new(input);
    let output = File::create(out_path)?;
    let mut writer = BufWriter::new(output);
    writer.write_all(TOKEN_BIN_MAGIC)?;
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
            std::io::stdout().flush().expect("failed to flush stdout");
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
    assert!(
        bytes.starts_with(TOKEN_BIN_MAGIC),
        "token bin is missing its version header"
    );
    let payload = &bytes[TOKEN_BIN_MAGIC.len()..];
    assert!(
        payload.len().is_multiple_of(TOKEN_BIN_ITEM_BYTES),
        "token bin payload must contain u32 values"
    );
    let split_at = TOKEN_BIN_MAGIC.len() + train_tokens * TOKEN_BIN_ITEM_BYTES;
    assert!(split_at <= bytes.len(), "train split runs past the token bin");
    let mut train = File::create(train_path)?;
    train.write_all(TOKEN_BIN_MAGIC)?;
    train.write_all(&bytes[TOKEN_BIN_MAGIC.len()..split_at])?;
    let mut val = File::create(val_path)?;
    val.write_all(TOKEN_BIN_MAGIC)?;
    val.write_all(&bytes[split_at..])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Gpt2Tokenizer;

    /// Fixed-output tokenizer so tests control exact ids, including ones the
    /// old `u16` bins could not store.
    struct FixedIdsTokenizer {
        ids: Vec<u32>,
        vocab_size: usize,
    }

    impl crate::tokenizer::Tokenizer for FixedIdsTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            self.ids.clone()
        }

        fn decode(&self, _tokens: &[u32]) -> String {
            String::new()
        }

        fn decode_lossy(&self, _tokens: &[u32]) -> String {
            String::new()
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[test]
    fn test_prepare_text_token_bins_roundtrips() {
        use crate::dataset::TokenBinDataset;

        // Arrange
        let dir = std::env::temp_dir().join("deers_prepare_token_bins_test");
        std::fs::create_dir_all(&dir).unwrap();
        let text_path = dir.join("tiny.txt");
        std::fs::write(&text_path, "Once upon a time.\nThere was a cat.\n".repeat(32)).unwrap();
        let tokenizer = Gpt2Tokenizer::new();

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

    #[test]
    fn test_prepare_text_token_bins_roundtrips_ids_above_u16() {
        use crate::dataset::TokenBinDataset;
        use crate::Device;

        // Arrange
        let dir = std::env::temp_dir().join("deers_prepare_token_bins_large_ids_test");
        std::fs::create_dir_all(&dir).unwrap();
        let text_path = dir.join("tiny.txt");
        std::fs::write(&text_path, "a\nb\n").unwrap();
        let per_line = vec![0u32, 1, 65_535, 65_536, 151_935, 248_319];
        let tokenizer = FixedIdsTokenizer { ids: per_line.clone(), vocab_size: 248_320 };
        let stream: Vec<i64> =
            per_line.iter().cycle().take(per_line.len() * 2).map(|&id| id as i64).collect();

        // Act
        let paths = prepare_text_token_bins(&text_path, &tokenizer, &dir, 0.25).unwrap();
        let raw_train = std::fs::read(&paths.train).unwrap();
        let raw_val = std::fs::read(&paths.val).unwrap();
        let train = TokenBinDataset::load(&paths.train, 4).unwrap();
        let val = TokenBinDataset::load(&paths.val, 1).unwrap();
        let (train_inputs, train_targets) = train.batch_from_starts(&[0, 4], Device::Cpu);
        let (val_inputs, val_targets) = val.batch_from_starts(&[0, 1], Device::Cpu);

        // Assert
        assert!(raw_train.starts_with(TOKEN_BIN_MAGIC));
        assert!(raw_val.starts_with(TOKEN_BIN_MAGIC));
        assert_eq!(train.num_tokens(), 9);
        assert_eq!(val.num_tokens(), 3);
        let mut train_tokens: Vec<i64> = train_inputs.to_vec().unwrap();
        train_tokens.push(train_targets.to_vec::<i64>().unwrap()[7]);
        assert_eq!(train_tokens, stream[..9]);
        let mut val_tokens: Vec<i64> = val_inputs.to_vec().unwrap();
        val_tokens.push(val_targets.to_vec::<i64>().unwrap()[1]);
        assert_eq!(val_tokens, stream[9..]);

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "exceeds supported token-bin id range 0..=4294967295")]
    fn test_prepare_text_token_bins_rejects_vocab_beyond_bin_range() {
        // Arrange
        let dir = std::env::temp_dir().join("deers_prepare_token_bins_vocab_range_test");
        let text_path = dir.join("tiny.txt");
        let vocab_size = usize::try_from(u64::from(u32::MAX) + 2).unwrap();
        let tokenizer = FixedIdsTokenizer { ids: vec![0], vocab_size };

        // Act
        let _ = prepare_text_token_bins(&text_path, &tokenizer, &dir, 0.25);
    }
}
