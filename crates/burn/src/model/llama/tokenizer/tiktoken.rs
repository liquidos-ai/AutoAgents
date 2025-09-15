use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "<|end_of_text|>";
const EOT_TOKEN: &str = "<|eot_id|>";
const EOM_TOKEN: &str = "<|eom_id|>";

const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;
const SPECIAL_TOKENS: [&str; 11] = [
    BOS_TOKEN,
    EOS_TOKEN,
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    EOM_TOKEN, // end of message
    EOT_TOKEN, // end of turn
    "<|python_tag|>",
];
const PATTERN: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

#[derive(Clone)]
pub struct Tiktoken {
    bpe: CoreBPE,
    bos_token_id: usize,
    eos_token_id: usize,
    eot_token_id: usize,
    eom_token_id: usize,
}

impl std::fmt::Debug for Tiktoken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tiktoken")
            .field("bos_token_id", &self.bos_token_id)
            .field("eos_token_id", &self.eos_token_id)
            .field("eot_token_id", &self.eot_token_id)
            .field("eom_token_id", &self.eom_token_id)
            .finish()
    }
}

impl Tiktoken {
    /// Load the tokenizer from bytes (for WASM targets)
    #[cfg(target_arch = "wasm32")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use std::io::Cursor;

        let mut mergeable_ranks: HashMap<Vec<u8>, u32, BuildHasherDefault<FxHasher>> =
            HashMap::with_hasher(BuildHasherDefault::<FxHasher>::default());

        let cursor = Cursor::new(bytes);
        for line in BufReader::new(cursor).lines() {
            let line = match line {
                Ok(val) => val,
                Err(err) => return Err(err.to_string()),
            };
            let mut parts = line.split(' ');
            let token = STANDARD
                .decode(parts.next().ok_or("Missing token")?)
                .map_err(|e| e.to_string())?;
            let rank = parts
                .next()
                .ok_or("Missing rank")?
                .parse::<u32>()
                .map_err(|e| e.to_string())?;

            mergeable_ranks.insert(token, rank);
        }
        let num_base_tokens = mergeable_ranks.len();

        let special_tokens = [
            SPECIAL_TOKENS
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            (0..NUM_RESERVED_SPECIAL_TOKENS - SPECIAL_TOKENS.len())
                .map(|i| format!("<|reserved_special_token_{}|>", i + 2))
                .collect::<Vec<_>>(),
        ]
        .concat();
        let special_tokens = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, (i + num_base_tokens) as u32))
            .collect::<HashMap<String, u32, BuildHasherDefault<FxHasher>>>();

        let bos_token_id = special_tokens[BOS_TOKEN] as usize;
        let eos_token_id = special_tokens[EOS_TOKEN] as usize;
        let eot_token_id = special_tokens[EOT_TOKEN] as usize;
        let eom_token_id = special_tokens[EOM_TOKEN] as usize;

        let bpe =
            CoreBPE::new(mergeable_ranks, special_tokens, PATTERN).map_err(|e| e.to_string())?;
        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            eom_token_id,
        })
    }
}

impl Tokenizer for Tiktoken {
    /// Load the [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    fn new(tiktoken_bpe_file: &str) -> Result<Self, String> {
        let file = File::open(tiktoken_bpe_file).map_err(|e| e.to_string())?;
        let mut mergeable_ranks: HashMap<Vec<u8>, u32, BuildHasherDefault<FxHasher>> =
            HashMap::with_hasher(BuildHasherDefault::<FxHasher>::default());

        for line in BufReader::new(file).lines() {
            let line = match line {
                Ok(val) => val,
                Err(err) => return Err(err.to_string()),
            };
            let mut parts = line.split(' ');
            let token = STANDARD
                .decode(parts.next().ok_or("Missing token")?)
                .map_err(|e| e.to_string())?;
            let rank = parts
                .next()
                .ok_or("Missing rank")?
                .parse::<u32>()
                .map_err(|e| e.to_string())?;

            mergeable_ranks.insert(token, rank);
        }
        let num_base_tokens = mergeable_ranks.len();

        let special_tokens = [
            SPECIAL_TOKENS
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            (0..NUM_RESERVED_SPECIAL_TOKENS - SPECIAL_TOKENS.len())
                .map(|i| format!("<|reserved_special_token_{}|>", i + 2))
                .collect::<Vec<_>>(),
        ]
        .concat();
        let special_tokens = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, (i + num_base_tokens) as u32))
            .collect::<HashMap<String, u32, BuildHasherDefault<FxHasher>>>();

        let bos_token_id = special_tokens[BOS_TOKEN] as usize;
        let eos_token_id = special_tokens[EOS_TOKEN] as usize;
        let eot_token_id = special_tokens[EOT_TOKEN] as usize;
        let eom_token_id = special_tokens[EOM_TOKEN] as usize;

        let bpe =
            CoreBPE::new(mergeable_ranks, special_tokens, PATTERN).map_err(|e| e.to_string())?;
        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            eom_token_id,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let bos_token: Vec<u32> = if bos {
            vec![self.bos_token_id as u32]
        } else {
            vec![]
        };
        let eos_token: Vec<u32> = if eos {
            vec![self.eos_token_id as u32]
        } else {
            vec![]
        };

        let tokens: Vec<u32> = self.bpe.encode_with_special_tokens(text);

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.bpe
            .decode(tokens.to_vec())
            .expect("Should decode tokens")
    }

    fn bos_id(&self) -> u32 {
        self.bos_token_id as u32
    }

    fn eos_id(&self) -> u32 {
        self.eos_token_id as u32
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![
            self.eos_id(),
            self.eom_token_id as u32,
            self.eot_token_id as u32,
        ]
    }
}
