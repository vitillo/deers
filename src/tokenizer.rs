//! BPE tokenizer wrapping tiktoken-rs.
//!
//! Provides encode/decode for text ↔ token ids. Ships with a stock
//! cl100k_base tokenizer for development; swap in a nanochat-trained
//! tokenizer later via [`Tokenizer::from_file`].

use tiktoken_rs::CoreBPE;

pub struct Tokenizer {
    bpe: CoreBPE,
    vocab_size: usize,
}

impl Tokenizer {
    /// Creates a tokenizer using OpenAI's cl100k_base encoding (100K vocab).
    pub fn cl100k_base() -> Self {
        let bpe = tiktoken_rs::cl100k_base().expect("failed to load cl100k_base");
        // cl100k_base has 100256 tokens (100K base + 256 special)
        Self { bpe, vocab_size: 100256 }
    }

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
        assert_eq!(tok.vocab_size(), 100256);
    }
}
