//! Qwen3 byte-pair encoding with its instruction delimiters and chat template.

use std::collections::HashMap;
use std::io::Read;

use flate2::read::GzDecoder;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const QWEN3_VOCAB_SIZE: usize = 151_936;
const QWEN3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN3_VOCAB: &[u8] = include_bytes!("../../assets/qwen3-vocab.json.gz");
const QWEN3_SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("<|endoftext|>", 151643),
    ("<|im_start|>", 151644),
    ("<|im_end|>", 151645),
    ("<|object_ref_start|>", 151646),
    ("<|object_ref_end|>", 151647),
    ("<|box_start|>", 151648),
    ("<|box_end|>", 151649),
    ("<|quad_start|>", 151650),
    ("<|quad_end|>", 151651),
    ("<|vision_start|>", 151652),
    ("<|vision_end|>", 151653),
    ("<|vision_pad|>", 151654),
    ("<|image_pad|>", 151655),
    ("<|video_pad|>", 151656),
    ("<tool_call>", 151657),
    ("</tool_call>", 151658),
    ("<|fim_prefix|>", 151659),
    ("<|fim_middle|>", 151660),
    ("<|fim_suffix|>", 151661),
    ("<|fim_pad|>", 151662),
    ("<|repo_name|>", 151663),
    ("<|file_sep|>", 151664),
    ("<tool_response>", 151665),
    ("</tool_response>", 151666),
    ("<think>", 151667),
    ("</think>", 151668),
];

/// Qwen3 byte-pair encoding.
pub struct Qwen3Tokenizer {
    bpe: CoreBPE,
}

impl Qwen3Tokenizer {
    /// Creates the Qwen3 byte-pair encoding with its instruction delimiters.
    ///
    /// BPE starts with bytes and repeatedly joins the adjacent pair with the
    /// best merge rank. Special tokens bypass those merges so control strings
    /// such as `<|im_start|>` remain one token.
    pub fn new() -> Self {
        let mut json = String::new();
        GzDecoder::new(QWEN3_VOCAB)
            .read_to_string(&mut json)
            .expect("failed to decompress Qwen3 vocabulary");
        let vocabulary: HashMap<String, u32> =
            serde_json::from_str(&json).expect("failed to parse Qwen3 vocabulary");
        let byte_decoder = qwen_byte_decoder();
        let encoder = vocabulary
            .into_iter()
            .map(|(token, rank)| {
                let bytes =
                    token.chars().map(|character| byte_decoder[&character]).collect::<Vec<_>>();
                (bytes, rank)
            })
            .collect();
        let special_tokens =
            QWEN3_SPECIAL_TOKENS.iter().map(|&(token, rank)| (token.to_owned(), rank)).collect();
        let bpe = CoreBPE::new(encoder, special_tokens, QWEN3_PATTERN)
            .expect("failed to build Qwen3 tokenizer");

        Self { bpe }
    }

    /// Formats Qwen chat messages, optionally ending at the assistant prompt.
    pub fn apply_chat_template(
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
    ) -> String {
        let mut prompt = String::new();
        for message in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(message.role);
            prompt.push('\n');
            prompt.push_str(message.content);
            prompt.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }
        prompt
    }
}

impl Default for Qwen3Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for Qwen3Tokenizer {
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
        QWEN3_VOCAB_SIZE
    }
}

/// One role and content segment in a Qwen chat prompt.
pub struct ChatMessage<'a> {
    /// The speaker name, such as `system`, `user`, or `assistant`.
    pub role: &'a str,
    /// The speaker's message text.
    pub content: &'a str,
}

fn qwen_byte_decoder() -> HashMap<char, u8> {
    let mut bytes = (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect::<Vec<_>>();
    let mut code_points = bytes.iter().map(|&byte| u32::from(byte)).collect::<Vec<_>>();
    let mut extra = 0u32;

    for byte in 0u8..=u8::MAX {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            code_points.push(256 + extra);
            extra += 1;
        }
    }

    code_points
        .into_iter()
        .zip(bytes)
        .map(|(code_point, byte)| {
            (char::from_u32(code_point).expect("byte encoding must be valid Unicode"), byte)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_encodes_published_vocabulary_ids() {
        // Arrange
        let tokenizer = Qwen3Tokenizer::new();

        // Act
        let english = tokenizer.encode("Hello, world!");
        let chinese = tokenizer.encode("你好，世界！");

        // Assert
        assert_eq!(english, [9707, 11, 1879, 0]);
        assert_eq!(chinese, [108386, 3837, 99489, 6313]);
        assert_eq!(tokenizer.decode(&english), "Hello, world!");
        assert_eq!(tokenizer.decode(&chinese), "你好，世界！");
        assert_eq!(tokenizer.vocab_size(), 151936);
    }

    #[test]
    fn qwen3_preserves_instruction_delimiters() {
        // Arrange
        let tokenizer = Qwen3Tokenizer::new();
        let text = "<|im_start|>user\nHello<|im_end|>\n";

        // Act
        let tokens = tokenizer.encode(text);

        // Assert
        assert_eq!(tokens, [151644, 872, 198, 9707, 151645, 198]);
        assert_eq!(tokenizer.decode(&tokens), text);
    }

    #[test]
    fn qwen3_lossy_decode_replaces_invalid_utf8() {
        // Arrange
        let tokenizer = Qwen3Tokenizer::new();

        // Act
        let decoded = tokenizer.decode_lossy(&[94]);

        // Assert
        assert_eq!(decoded, "�");
    }

    #[test]
    fn qwen3_formats_and_encodes_a_multi_turn_chat() {
        // Arrange
        let tokenizer = Qwen3Tokenizer::new();
        let messages = [
            ChatMessage { role: "system", content: "Be concise." },
            ChatMessage { role: "user", content: "What is 2 + 2?" },
            ChatMessage { role: "assistant", content: "4" },
            ChatMessage { role: "user", content: "Say it again." },
        ];

        // Act
        let prompt = Qwen3Tokenizer::apply_chat_template(&messages, true);
        let tokens = tokenizer.encode(&prompt);

        // Assert
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n<|im_start|>user\nSay it again.<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            tokens,
            [
                151644, 8948, 198, 3430, 63594, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220,
                17, 488, 220, 17, 30, 151645, 198, 151644, 77091, 198, 19, 151645, 198, 151644,
                872, 198, 45764, 432, 1549, 13, 151645, 198, 151644, 77091, 198,
            ]
        );
        assert_eq!(tokenizer.decode(&tokens), prompt);
    }
}
