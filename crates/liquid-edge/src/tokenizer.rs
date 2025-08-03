//! Tokenizer module for liquid-edge
//!
//! This module provides tokenization capabilities using HuggingFace tokenizers
//! with support for various model formats and special token handling.

use crate::{
    config::{SpecialTokensMap, TokenizerConfig},
    error::{EdgeError, EdgeResult},
    traits::{SpecialTokenIds, TokenizedOutput, TokenizerTrait},
};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// Tokenizer wrapper that implements the TokenizerTrait
pub struct Tokenizer {
    inner: HfTokenizer,
    config: TokenizerConfig,
    special_tokens: SpecialTokensMap,
    special_token_ids: SpecialTokenIds,
}

impl Tokenizer {
    /// Create a new tokenizer from a file
    pub fn from_file<P: AsRef<Path>>(tokenizer_path: P) -> EdgeResult<Self> {
        let inner = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| EdgeError::tokenization(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            inner,
            config: TokenizerConfig::default(),
            special_tokens: SpecialTokensMap::default(),
            special_token_ids: SpecialTokenIds::default(),
        })
    }

    /// Create a new tokenizer with configuration
    pub fn from_file_with_config<P: AsRef<Path>>(
        tokenizer_path: P,
        config: TokenizerConfig,
        special_tokens: SpecialTokensMap,
    ) -> EdgeResult<Self> {
        let inner = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| EdgeError::tokenization(format!("Failed to load tokenizer: {e}")))?;

        let special_token_ids = Self::extract_special_token_ids(&inner, &special_tokens)?;

        Ok(Self {
            inner,
            config,
            special_tokens,
            special_token_ids,
        })
    }

    /// Load complete tokenizer configuration from directory
    #[cfg(feature = "serde")]
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> EdgeResult<Self> {
        let dir = dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        let config_path = dir.join("tokenizer_config.json");
        let special_tokens_path = dir.join("special_tokens_map.json");

        // Load tokenizer
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| EdgeError::tokenization(format!("Failed to load tokenizer: {e}")))?;

        // Load configuration
        let config = if config_path.exists() {
            TokenizerConfig::from_file(&config_path)?
        } else {
            TokenizerConfig::default()
        };

        // Load special tokens
        let special_tokens = if special_tokens_path.exists() {
            SpecialTokensMap::from_file(&special_tokens_path)?
        } else {
            SpecialTokensMap::default()
        };

        let special_token_ids = Self::extract_special_token_ids(&inner, &special_tokens)?;

        Ok(Self {
            inner,
            config,
            special_tokens,
            special_token_ids,
        })
    }

    /// Extract special token IDs from the tokenizer and special tokens map
    fn extract_special_token_ids(
        tokenizer: &HfTokenizer,
        special_tokens: &SpecialTokensMap,
    ) -> EdgeResult<SpecialTokenIds> {
        let mut special_token_ids = SpecialTokenIds::default();

        // Extract BOS token ID
        if let Some(bos_token) = special_tokens.get_bos_token() {
            if let Ok(encoding) = tokenizer.encode(bos_token.as_str(), false) {
                if let Some(&token_id) = encoding.get_ids().first() {
                    special_token_ids.bos_token_id = Some(token_id);
                }
            }
        }

        // Extract EOS token ID
        if let Some(eos_token) = special_tokens.get_eos_token() {
            if let Ok(encoding) = tokenizer.encode(eos_token.as_str(), false) {
                if let Some(&token_id) = encoding.get_ids().first() {
                    special_token_ids.eos_token_id = Some(token_id);
                }
            }
        }

        // Extract PAD token ID
        if let Some(pad_token) = special_tokens.get_pad_token() {
            if let Ok(encoding) = tokenizer.encode(pad_token.as_str(), false) {
                if let Some(&token_id) = encoding.get_ids().first() {
                    special_token_ids.pad_token_id = Some(token_id);
                }
            }
        }

        // Extract UNK token ID
        if let Some(unk_token) = special_tokens.get_unk_token() {
            if let Ok(encoding) = tokenizer.encode(unk_token.as_str(), false) {
                if let Some(&token_id) = encoding.get_ids().first() {
                    special_token_ids.unk_token_id = Some(token_id);
                }
            }
        }

        Ok(special_token_ids)
    }

    /// Get the underlying HuggingFace tokenizer
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }

    /// Get tokenizer configuration
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Get special tokens map
    pub fn special_tokens_map(&self) -> &SpecialTokensMap {
        &self.special_tokens
    }

    /// Encode multiple texts in batch
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> EdgeResult<Vec<Vec<u32>>> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| EdgeError::tokenization(format!("Batch encoding failed: {e}")))?;

        Ok(encodings
            .into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect())
    }

    /// Decode multiple token sequences in batch
    pub fn decode_batch(
        &self,
        token_sequences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> EdgeResult<Vec<String>> {
        let mut results = Vec::with_capacity(token_sequences.len());

        for tokens in token_sequences {
            let decoded = self
                .inner
                .decode(tokens, skip_special_tokens)
                .map_err(|e| EdgeError::tokenization(format!("Batch decoding failed: {e}")))?;
            results.push(decoded);
        }

        Ok(results)
    }

    /// Get token at position
    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.inner.id_to_token(token_id)
    }

    /// Get token ID for a string token
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Check if a token ID is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        if let Some(bos_id) = self.special_token_ids.bos_token_id {
            if token_id == bos_id {
                return true;
            }
        }
        if let Some(eos_id) = self.special_token_ids.eos_token_id {
            if token_id == eos_id {
                return true;
            }
        }
        if let Some(pad_id) = self.special_token_ids.pad_token_id {
            if token_id == pad_id {
                return true;
            }
        }
        if let Some(unk_id) = self.special_token_ids.unk_token_id {
            if token_id == unk_id {
                return true;
            }
        }
        false
    }

    /// Truncate tokens to maximum length
    pub fn truncate_tokens(&self, mut tokens: Vec<u32>, max_length: usize) -> Vec<u32> {
        if tokens.len() > max_length {
            tokens.truncate(max_length);

            // Add EOS token if truncated and config specifies it
            if self.config.add_eos_token {
                if let Some(eos_id) = self.special_token_ids.eos_token_id {
                    if tokens.last() != Some(&eos_id) {
                        if tokens.len() == max_length {
                            tokens[max_length - 1] = eos_id;
                        } else {
                            tokens.push(eos_id);
                        }
                    }
                }
            }
        }
        tokens
    }

    /// Pad tokens to specified length
    pub fn pad_tokens(&self, mut tokens: Vec<u32>, target_length: usize) -> Vec<u32> {
        if tokens.len() < target_length {
            let pad_id = self.special_token_ids.pad_token_id.unwrap_or(0);
            tokens.resize(target_length, pad_id);
        }
        tokens
    }

    /// Create attention mask for tokens (1 for real tokens, 0 for padding)
    pub fn create_attention_mask(&self, tokens: &[u32]) -> Vec<u32> {
        let pad_id = self.special_token_ids.pad_token_id.unwrap_or(0);
        tokens
            .iter()
            .map(|&token| if token == pad_id { 0 } else { 1 })
            .collect()
    }
}

impl TokenizerTrait for Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> EdgeResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| EdgeError::tokenization(format!("Encoding failed: {e}")))?;

        let mut tokens = encoding.get_ids().to_vec();

        // Apply tokenizer config
        if add_special_tokens {
            // Add BOS token if configured and not already present
            if self.config.add_bos_token {
                if let Some(bos_id) = self.special_token_ids.bos_token_id {
                    if tokens.first() != Some(&bos_id) {
                        tokens.insert(0, bos_id);
                    }
                }
            }

            // Add EOS token if configured and not already present
            if self.config.add_eos_token {
                if let Some(eos_id) = self.special_token_ids.eos_token_id {
                    if tokens.last() != Some(&eos_id) {
                        tokens.push(eos_id);
                    }
                }
            }
        }

        // Truncate if exceeds model max length
        if tokens.len() > self.config.model_max_length {
            tokens = self.truncate_tokens(tokens, self.config.model_max_length);
        }

        Ok(tokens)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> EdgeResult<String> {
        let decoded = self
            .inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| EdgeError::tokenization(format!("Decoding failed: {e}")))?;

        // Clean up tokenization spaces if configured
        if self.config.clean_up_tokenization_spaces {
            Ok(decoded.trim().to_string())
        } else {
            Ok(decoded)
        }
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn special_tokens(&self) -> SpecialTokenIds {
        self.special_token_ids.clone()
    }

    fn tokenize_detailed(&self, text: &str) -> EdgeResult<TokenizedOutput> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| EdgeError::tokenization(format!("Detailed tokenization failed: {e}")))?;

        let token_ids = encoding.get_ids().to_vec();
        let tokens = encoding.get_tokens().to_vec();
        let attention_mask = self.create_attention_mask(&token_ids);
        let special_tokens_mask = token_ids
            .iter()
            .map(|&id| self.is_special_token(id))
            .collect();
        let offsets = encoding.get_offsets().to_vec();

        Ok(TokenizedOutput {
            token_ids,
            tokens,
            attention_mask,
            special_tokens_mask,
            offsets,
        })
    }
}

/// Builder for creating tokenizers with specific configuration
pub struct TokenizerBuilder {
    config: TokenizerConfig,
    special_tokens: SpecialTokensMap,
}

impl TokenizerBuilder {
    /// Create a new tokenizer builder
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
            special_tokens: SpecialTokensMap::default(),
        }
    }

    /// Set tokenizer configuration
    pub fn with_config(mut self, config: TokenizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set special tokens map
    pub fn with_special_tokens(mut self, special_tokens: SpecialTokensMap) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Set model max length
    pub fn with_model_max_length(mut self, max_length: usize) -> Self {
        self.config.model_max_length = max_length;
        self
    }

    /// Enable BOS token addition
    pub fn with_bos_token(mut self, add_bos: bool) -> Self {
        self.config.add_bos_token = add_bos;
        self
    }

    /// Enable EOS token addition
    pub fn with_eos_token(mut self, add_eos: bool) -> Self {
        self.config.add_eos_token = add_eos;
        self
    }

    /// Build tokenizer from file
    pub fn build_from_file<P: AsRef<Path>>(self, tokenizer_path: P) -> EdgeResult<Tokenizer> {
        Tokenizer::from_file_with_config(tokenizer_path, self.config, self.special_tokens)
    }
}

impl Default for TokenizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for tokenization
pub mod utils {
    use super::*;

    /// Validate chunk parameters
    pub fn chunk_text_validation(max_tokens: usize, overlap: usize) -> EdgeResult<()> {
        if overlap >= max_tokens {
            return Err(EdgeError::invalid_input(
                "Overlap must be less than max_tokens",
            ));
        }
        Ok(())
    }

    /// Split text into chunks that fit within token limits
    pub fn chunk_text(
        tokenizer: &Tokenizer,
        text: &str,
        max_tokens: usize,
        overlap: usize,
    ) -> EdgeResult<Vec<String>> {
        chunk_text_validation(max_tokens, overlap)?;

        let tokens = tokenizer.encode(text, false)?;
        if tokens.len() <= max_tokens {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < tokens.len() {
            let end = (start + max_tokens).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let chunk_text = tokenizer.decode(chunk_tokens, true)?;
            chunks.push(chunk_text);

            if end == tokens.len() {
                break;
            }

            start = end - overlap;
        }

        Ok(chunks)
    }

    /// Count tokens in text without full tokenization
    pub fn count_tokens(tokenizer: &Tokenizer, text: &str) -> EdgeResult<usize> {
        let tokens = tokenizer.encode(text, true)?;
        Ok(tokens.len())
    }

    /// Estimate tokens in text using character-based heuristic
    pub fn estimate_tokens(text: &str) -> usize {
        // Rough estimate: 1 token per 4 characters for English text
        (text.len() as f32 / 4.0).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock tests that don't require actual tokenizer files
    #[test]
    fn test_tokenizer_builder() {
        let builder = TokenizerBuilder::new()
            .with_model_max_length(512)
            .with_bos_token(true)
            .with_eos_token(true);

        assert_eq!(builder.config.model_max_length, 512);
        assert!(builder.config.add_bos_token);
        assert!(builder.config.add_eos_token);
    }

    #[test]
    fn test_special_token_ids_default() {
        let special_tokens = SpecialTokenIds::default();
        assert!(special_tokens.bos_token_id.is_none());
        assert!(special_tokens.eos_token_id.is_none());
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "Hello, world!";
        let estimated = utils::estimate_tokens(text);
        assert!(estimated > 0);
        assert!(estimated <= text.len());
    }

    #[test]
    fn test_chunk_text_overlap_validation() {
        // This test validates input parameters without needing a real tokenizer
        // We test the validation logic directly instead of using an invalid mock
        let result = utils::chunk_text_validation(10, 15); // overlap > max_tokens
        assert!(result.is_err());
    }
}
