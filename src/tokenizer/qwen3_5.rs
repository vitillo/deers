//! Qwen3.5 byte-pair encoding with its instruction delimiters and chat template.

use std::collections::HashMap;
use std::io::Read;

use flate2::read::GzDecoder;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const QWEN3_5_VOCAB_SIZE: usize = 248_320;
const QWEN3_5_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN3_5_TOKENIZER_JSON: &[u8] = include_bytes!("../../assets/qwen3_5-tokenizer.json.gz");
const QWEN3_5_SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("<|endoftext|>", 248044),
    ("<|im_start|>", 248045),
    ("<|im_end|>", 248046),
    ("<|object_ref_start|>", 248047),
    ("<|object_ref_end|>", 248048),
    ("<|box_start|>", 248049),
    ("<|box_end|>", 248050),
    ("<|quad_start|>", 248051),
    ("<|quad_end|>", 248052),
    ("<|vision_start|>", 248053),
    ("<|vision_end|>", 248054),
    ("<|vision_pad|>", 248055),
    ("<|image_pad|>", 248056),
    ("<|video_pad|>", 248057),
    ("<tool_call>", 248058),
    ("</tool_call>", 248059),
    ("<|fim_prefix|>", 248060),
    ("<|fim_middle|>", 248061),
    ("<|fim_suffix|>", 248062),
    ("<|fim_pad|>", 248063),
    ("<|repo_name|>", 248064),
    ("<|file_sep|>", 248065),
];

/// Qwen3.5 byte-pair encoding.
pub struct Qwen3_5Tokenizer {
    bpe: CoreBPE,
}

impl Qwen3_5Tokenizer {
    /// Creates the Qwen3.5 byte-pair encoding with its instruction delimiters.
    ///
    /// The vocabulary comes from the published `tokenizer.json`: its
    /// `model.vocab` table holds ids 0..248043, and the 22 added tokens above
    /// bypass merges so control strings such as `<|im_start|>` remain one
    /// token. The split pattern carries `\p{M}` so combining marks stay with
    /// their base letter, unlike the Qwen3 pattern.
    pub fn new() -> Self {
        let mut json = String::new();
        GzDecoder::new(QWEN3_5_TOKENIZER_JSON)
            .read_to_string(&mut json)
            .expect("failed to decompress Qwen3.5 tokenizer");
        let published: PublishedTokenizer =
            serde_json::from_str(&json).expect("failed to parse Qwen3.5 tokenizer");
        let byte_decoder = qwen_byte_decoder();
        let encoder = published
            .model
            .vocab
            .into_iter()
            .map(|(token, rank)| {
                let bytes =
                    token.chars().map(|character| byte_decoder[&character]).collect::<Vec<_>>();
                (bytes, rank)
            })
            .collect();
        let special_tokens = QWEN3_5_SPECIAL_TOKENS
            .iter()
            .map(|&(token, rank)| (token.to_owned(), rank))
            .collect();
        let bpe = CoreBPE::new(encoder, special_tokens, QWEN3_5_PATTERN)
            .expect("failed to build Qwen3.5 tokenizer");

        Self { bpe }
    }

    /// Formats Qwen3.5 chat messages, optionally ending at the assistant prompt.
    ///
    /// This is the text-only reduction of the published chat template: plain
    /// messages render inside `<|im_start|>` delimiters, an assistant message
    /// after the last user query carries its reasoning in a `<think>` block,
    /// and the generation prompt opens that block for the model to fill in.
    pub fn apply_chat_template(
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
    ) -> String {
        let last_query = messages
            .iter()
            .rposition(|message| message.role == "user" && !is_tool_response(message.content))
            .unwrap_or(usize::MAX);
        let mut prompt = String::new();
        for (index, message) in messages.iter().enumerate() {
            let content = message.content.trim();
            if message.role == "assistant" && index > last_query {
                let (reasoning, answer) = split_reasoning(content);
                prompt.push_str("<|im_start|>assistant\n<think>\n");
                prompt.push_str(&reasoning);
                prompt.push_str("\n</think>\n\n");
                prompt.push_str(&answer);
            } else {
                prompt.push_str("<|im_start|>");
                prompt.push_str(message.role);
                prompt.push('\n');
                prompt.push_str(content);
            }
            prompt.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
        prompt
    }
}

impl Default for Qwen3_5Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for Qwen3_5Tokenizer {
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
        QWEN3_5_VOCAB_SIZE
    }
}

/// One role and content segment in a Qwen3.5 chat prompt.
pub struct ChatMessage<'a> {
    /// The speaker name, such as `system`, `user`, or `assistant`.
    pub role: &'a str,
    /// The speaker's message text.
    pub content: &'a str,
}

/// The published `tokenizer.json` envelope; only the BPE table is used.
#[derive(serde::Deserialize)]
struct PublishedTokenizer {
    model: PublishedModel,
}

/// The `model` section of the published `tokenizer.json`.
#[derive(serde::Deserialize)]
struct PublishedModel {
    vocab: HashMap<String, u32>,
}

/// A user message counts as a query unless it wraps a tool response.
fn is_tool_response(content: &str) -> bool {
    let content = content.trim();
    content.starts_with("<tool_response>") && content.ends_with("</tool_response>")
}

/// Splits trailing-assistant content into its reasoning and answer halves.
fn split_reasoning(content: &str) -> (String, String) {
    match content.split_once("</think>") {
        Some((head, tail)) => {
            let reasoning = head.rsplit("<think>").next().unwrap_or("").trim().to_owned();
            (reasoning, tail.trim_start_matches('\n').to_owned())
        }
        None => (String::new(), content.to_owned()),
    }
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
    fn qwen3_5_encodes_published_vocabulary_ids() {
        // Arrange
        let tokenizer = Qwen3_5Tokenizer::new();

        // Act
        let english = tokenizer.encode("Hello, world!");
        let chinese = tokenizer.encode("你好，世界！");

        // Assert
        assert_eq!(english, [9419, 11, 1814, 0]);
        assert_eq!(chinese, [109266, 3709, 96748, 6115]);
        assert_eq!(tokenizer.decode(&english), "Hello, world!");
        assert_eq!(tokenizer.decode(&chinese), "你好，世界！");
        assert_eq!(tokenizer.vocab_size(), 248320);
    }

    #[test]
    fn qwen3_5_preserves_instruction_delimiters() {
        // Arrange
        let tokenizer = Qwen3_5Tokenizer::new();
        let text = "<|im_start|>user\nHello<|im_end|>\n";

        // Act
        let tokens = tokenizer.encode(text);

        // Assert
        assert_eq!(tokens, [248045, 846, 198, 9419, 248046, 198]);
        assert_eq!(tokenizer.decode(&tokens), text);
    }

    #[test]
    fn qwen3_5_formats_and_encodes_a_multi_turn_chat() {
        // Arrange
        let tokenizer = Qwen3_5Tokenizer::new();
        let messages = [
            ChatMessage { role: "system", content: "Be concise." },
            ChatMessage { role: "user", content: "What is 2 + 2?" },
            ChatMessage { role: "assistant", content: "4" },
            ChatMessage { role: "user", content: "Say it again." },
        ];

        // Act
        let prompt = Qwen3_5Tokenizer::apply_chat_template(&messages, true);
        let tokens = tokenizer.encode(&prompt);

        // Assert
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n\
             <|im_start|>user\nWhat is 2 + 2?<|im_end|>\n\
             <|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\nSay it again.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            tokens,
            [
                248045, 8678, 198, 3320, 61446, 13, 248046, 198, 248045, 846, 198, 3710, 369,
                220, 17, 478, 220, 17, 30, 248046, 198, 248045, 74455, 198, 19, 248046, 198,
                248045, 846, 198, 44240, 424, 1495, 13, 248046, 198, 248045, 74455, 198, 13314,
                741, 29, 271, 510, 26003, 29, 271,
            ]
        );
        assert_eq!(tokenizer.decode(&tokens), prompt);
    }

    #[test]
    fn qwen3_5_matches_the_published_tokenizer() {
        // Arrange
        let tokenizer = Qwen3_5Tokenizer::new();
        let mut json = String::new();
        GzDecoder::new(QWEN3_5_TOKENIZER_JSON)
            .read_to_string(&mut json)
            .expect("failed to decompress Qwen3.5 tokenizer");
        let reference_path = std::env::temp_dir().join("deers_qwen3_5_reference.json");
        std::fs::write(&reference_path, &json).expect("failed to stage reference tokenizer");
        let reference = tokenizers::Tokenizer::from_file(&reference_path)
            .expect("failed to load published tokenizer");
        std::fs::remove_file(&reference_path).ok();
        let messages = [
            ChatMessage { role: "system", content: "Be concise." },
            ChatMessage { role: "user", content: "What is 2 + 2?" },
            ChatMessage { role: "assistant", content: "4" },
            ChatMessage { role: "user", content: "Say it again." },
        ];
        let prompt = Qwen3_5Tokenizer::apply_chat_template(&messages, true);
        let cases = [
            "Hello, world!",
            "你好，世界！",
            "Guten Morgen, Welt!",
            "مرحبا بالعالم",
            "caf\u{e9}",
            "<|im_start|>user\nHello<|im_end|>\n",
            "a<|vision_start|>b",
            "<tool_call>\n<function=x>\n</function>\n</tool_call>",
            prompt.as_str(),
        ];

        // Act and Assert: every case must agree token-for-token.
        for text in cases {
            let expected =
                reference.encode(text, true).expect("reference encode must succeed");
            assert_eq!(tokenizer.encode(text), expected.get_ids(), "mismatch for {text:?}");
        }
    }
}
