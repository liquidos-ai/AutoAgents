use tokenizers::Tokenizer as BaseTokenizer;

use super::Tokenizer;

const BOS_TOKEN_ID: u32 = 1;
const EOS_TOKEN_ID: u32 = 2;

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    bpe: BaseTokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl SentencePieceTokenizer {
    /// Load the tokenizer from bytes (for WASM targets)
    #[cfg(target_arch = "wasm32")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let bpe = BaseTokenizer::from_bytes(bytes).map_err(|e| e.to_string())?;

        Ok(Self {
            bpe,
            bos_token_id: BOS_TOKEN_ID,
            eos_token_id: EOS_TOKEN_ID,
        })
    }
}

impl Tokenizer for SentencePieceTokenizer {
    /// Load the [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    fn new(tokenizer_path: &str) -> Result<Self, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let bpe = BaseTokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
            Ok(Self {
                bpe,
                bos_token_id: BOS_TOKEN_ID,
                eos_token_id: EOS_TOKEN_ID,
            })
        }

        #[cfg(target_arch = "wasm32")]
        {
            // For WASM, we expect the tokenizer_path to contain base64-encoded bytes
            // or we need to load it differently. For now, return an error suggesting
            // to use from_bytes instead
            Err("For WASM targets, use SentencePieceTokenizer::from_bytes() instead".to_string())
        }
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let bos_token = if bos { vec![self.bos_token_id] } else { vec![] };
        let eos_token = if eos { vec![self.eos_token_id] } else { vec![] };

        let tokens = self.bpe.encode(text, false).unwrap().get_ids().to_vec();

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.bpe.decode(tokens, false).unwrap()
    }

    fn bos_id(&self) -> u32 {
        self.bos_token_id
    }

    fn eos_id(&self) -> u32 {
        self.eos_token_id
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![self.eos_id()]
    }

    fn streaming_context_size(&self) -> usize {
        // SentencePiece tokens represent subwords with special markers (e.g., _ suffix for spaces),
        // requiring a short token buffer for correct incremental decoding.
        4 // should be good enough for spacing + utf-8 decoding
    }
}
