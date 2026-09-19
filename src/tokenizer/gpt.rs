//! GPT-family encodings backed by tiktoken-rs built-ins.

use tiktoken_rs::CoreBPE;

use super::Tokenizer;

/// GPT-2's r50k_base encoding.
pub struct Gpt2Tokenizer {
    bpe: CoreBPE,
}

impl Gpt2Tokenizer {
    /// Creates a tokenizer using GPT-2's r50k_base encoding (50257 vocab).
    pub fn new() -> Self {
        let bpe = tiktoken_rs::r50k_base().expect("failed to load r50k_base");
        Self { bpe }
    }
}

impl Default for Gpt2Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.bpe.decode(tokens.to_vec()).expect("failed to decode tokens")
    }

    fn decode_lossy(&self, tokens: &[u32]) -> String {
        let bytes =
            self.bpe._decode_native_and_split(tokens.to_vec()).flatten().collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        // r50k_base: 50256 base tokens + 1 special token (<|endoftext|>)
        50257
    }
}

/// OpenAI's cl100k_base encoding.
pub struct Cl100kTokenizer {
    bpe: CoreBPE,
}

impl Cl100kTokenizer {
    /// Creates a tokenizer using OpenAI's cl100k_base encoding (100K vocab).
    pub fn new() -> Self {
        let bpe = tiktoken_rs::cl100k_base().expect("failed to load cl100k_base");
        Self { bpe }
    }
}

impl Default for Cl100kTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for Cl100kTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.bpe.decode(tokens.to_vec()).expect("failed to decode tokens")
    }

    fn decode_lossy(&self, tokens: &[u32]) -> String {
        let bytes =
            self.bpe._decode_native_and_split(tokens.to_vec()).flatten().collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        // cl100k_base: 100256 base tokens + 5 special tokens
        // (<|endoftext|>, <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>, <|endofprompt|>)
        100261
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_reports_r50k_vocab_size() {
        // Arrange / Act
        let tok = Gpt2Tokenizer::new();

        // Assert
        assert_eq!(tok.vocab_size(), 50257);
    }

    #[test]
    fn cl100k_roundtrips_text() {
        // Arrange
        let tok = Cl100kTokenizer::new();
        let text = "Hello, world!";

        // Act
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);

        // Assert
        assert_eq!(decoded, text);
    }

    #[test]
    fn cl100k_encode_produces_tokens() {
        // Arrange
        let tok = Cl100kTokenizer::new();

        // Act
        let tokens = tok.encode("The quick brown fox");

        // Assert
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|&t| (t as usize) < tok.vocab_size()));
    }

    #[test]
    fn cl100k_vocab_size() {
        // Arrange / Act
        let tok = Cl100kTokenizer::new();

        // Assert
        assert_eq!(tok.vocab_size(), 100261);
    }

    #[test]
    fn cl100k_special_tokens_within_vocab() {
        // Arrange
        let tok = Cl100kTokenizer::new();

        // Act — encode text containing <|endoftext|>
        let tokens = tok.encode("hello <|endoftext|> world");

        // Assert — all token ids must be < vocab_size
        assert!(tokens.iter().all(|&t| (t as usize) < tok.vocab_size()));
    }
}
