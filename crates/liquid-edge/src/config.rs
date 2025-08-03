//! Configuration types and utilities for liquid-edge runtime
//!
//! This module provides configuration structures for models, tokenizers,
//! and runtime settings with validation and file loading capabilities.

use crate::error::{EdgeError, EdgeResult};
use std::path::Path;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Model configuration containing model-specific parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct ModelConfig {
    /// End-of-sequence token ID
    #[cfg_attr(feature = "serde", serde(default = "default_eos_token_id"))]
    pub eos_token_id: u32,

    /// Beginning-of-sequence token ID
    #[cfg_attr(feature = "serde", serde(default = "default_bos_token_id"))]
    pub bos_token_id: u32,

    /// Maximum position embeddings (context length)
    #[cfg_attr(feature = "serde", serde(default = "default_max_position_embeddings"))]
    pub max_position_embeddings: usize,

    /// Number of positions (GPT-2 style naming)
    #[cfg_attr(feature = "serde", serde(default = "default_max_position_embeddings"))]
    pub n_positions: usize,

    /// Vocabulary size
    #[cfg_attr(feature = "serde", serde(default = "default_vocab_size"))]
    pub vocab_size: usize,

    /// Model name or identifier
    pub model_name: String,

    /// Model type (e.g., "gpt2", "llama", "mistral")
    pub model_type: Option<String>,

    /// Model architecture version
    pub architecture_version: Option<String>,
}

/// Tokenizer configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct TokenizerConfig {
    /// Whether to add beginning-of-sequence token
    #[cfg_attr(feature = "serde", serde(default))]
    pub add_bos_token: bool,

    /// Whether to add end-of-sequence token
    #[cfg_attr(feature = "serde", serde(default))]
    pub add_eos_token: bool,

    /// Beginning-of-sequence token string
    #[cfg_attr(feature = "serde", serde(default))]
    pub bos_token: Option<String>,

    /// End-of-sequence token string
    #[cfg_attr(feature = "serde", serde(default))]
    pub eos_token: Option<String>,

    /// Maximum model length for tokenization
    #[cfg_attr(feature = "serde", serde(default = "default_model_max_length"))]
    pub model_max_length: usize,

    /// Padding token string
    #[cfg_attr(feature = "serde", serde(default))]
    pub pad_token: Option<String>,

    /// Unknown token string
    #[cfg_attr(feature = "serde", serde(default))]
    pub unk_token: Option<String>,

    /// Whether to clean up text before tokenization
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub clean_up_tokenization_spaces: bool,
}

/// Special token representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct SpecialToken {
    /// Token content/string
    pub content: String,

    /// Whether to strip left whitespace
    #[cfg_attr(feature = "serde", serde(default))]
    pub lstrip: bool,

    /// Whether the token is normalized
    #[cfg_attr(feature = "serde", serde(default))]
    pub normalized: bool,

    /// Whether to strip right whitespace
    #[cfg_attr(feature = "serde", serde(default))]
    pub rstrip: bool,

    /// Whether this is a single word token
    #[cfg_attr(feature = "serde", serde(default))]
    pub single_word: bool,
}

/// Map of special tokens
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Default)]
pub struct SpecialTokensMap {
    /// Beginning-of-sequence token
    #[cfg_attr(feature = "serde", serde(default))]
    pub bos_token: Option<serde_json::Value>,

    /// End-of-sequence token
    #[cfg_attr(feature = "serde", serde(default))]
    pub eos_token: Option<serde_json::Value>,

    /// Unknown token
    #[cfg_attr(feature = "serde", serde(default))]
    pub unk_token: Option<serde_json::Value>,

    /// Padding token
    #[cfg_attr(feature = "serde", serde(default))]
    pub pad_token: Option<serde_json::Value>,

    /// Mask token
    #[cfg_attr(feature = "serde", serde(default))]
    pub mask_token: Option<serde_json::Value>,

    /// Additional special tokens
    #[cfg_attr(feature = "serde", serde(default))]
    pub additional_special_tokens: Vec<serde_json::Value>,
}

/// Runtime configuration for inference settings
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RuntimeConfig {
    /// Maximum number of tokens to generate
    #[cfg_attr(feature = "serde", serde(default = "default_max_new_tokens"))]
    pub max_new_tokens: usize,

    /// Temperature for sampling (0.0 = deterministic)
    #[cfg_attr(feature = "serde", serde(default = "default_temperature"))]
    pub temperature: f32,

    /// Top-p (nucleus) sampling parameter
    #[cfg_attr(feature = "serde", serde(default = "default_top_p"))]
    pub top_p: f32,

    /// Top-k sampling parameter (0 = disabled)
    #[cfg_attr(feature = "serde", serde(default))]
    pub top_k: usize,

    /// Repetition penalty
    #[cfg_attr(feature = "serde", serde(default = "default_repetition_penalty"))]
    pub repetition_penalty: f32,

    /// Number of threads for inference (0 = auto)
    #[cfg_attr(feature = "serde", serde(default))]
    pub num_threads: usize,

    /// Batch size for inference
    #[cfg_attr(feature = "serde", serde(default = "default_batch_size"))]
    pub batch_size: usize,

    /// Memory optimization level (0 = none, 1 = basic, 2 = aggressive)
    #[cfg_attr(feature = "serde", serde(default))]
    pub memory_optimization: u8,

    /// Enable KV cache optimization
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub enable_kv_cache: bool,
}

/// Complete edge runtime configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct EdgeConfig {
    /// Model configuration
    pub model: ModelConfig,

    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,

    /// Special tokens mapping
    pub special_tokens: SpecialTokensMap,

    /// Runtime settings
    pub runtime: RuntimeConfig,

    /// Chat template (Jinja2 format)
    #[cfg(feature = "chat")]
    pub chat_template: Option<String>,
}

// Default value functions
fn default_eos_token_id() -> u32 {
    2
}

fn default_bos_token_id() -> u32 {
    1
}

fn default_max_position_embeddings() -> usize {
    2048
}

fn default_vocab_size() -> usize {
    32000
}

fn default_model_max_length() -> usize {
    2048
}

fn default_max_new_tokens() -> usize {
    100
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_batch_size() -> usize {
    1
}

fn default_true() -> bool {
    true
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            eos_token_id: default_eos_token_id(),
            bos_token_id: default_bos_token_id(),
            max_position_embeddings: default_max_position_embeddings(),
            n_positions: default_max_position_embeddings(),
            vocab_size: default_vocab_size(),
            model_name: "unknown".to_string(),
            model_type: None,
            architecture_version: None,
        }
    }
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos_token: false,
            add_eos_token: false,
            bos_token: None,
            eos_token: None,
            model_max_length: default_model_max_length(),
            pad_token: None,
            unk_token: None,
            clean_up_tokenization_spaces: true,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: 0,
            repetition_penalty: default_repetition_penalty(),
            num_threads: 0,
            batch_size: default_batch_size(),
            memory_optimization: 0,
            enable_kv_cache: true,
        }
    }
}

impl SpecialToken {
    /// Create a special token from a JSON value
    pub fn from_value(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::String(s) => Some(SpecialToken {
                content: s.clone(),
                lstrip: false,
                normalized: false,
                rstrip: false,
                single_word: false,
            }),
            serde_json::Value::Object(obj) => {
                let content = obj.get("content")?.as_str()?.to_string();
                Some(SpecialToken {
                    content,
                    lstrip: obj.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                    normalized: obj
                        .get("normalized")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    rstrip: obj.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                    single_word: obj
                        .get("single_word")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                })
            }
            _ => None,
        }
    }

    /// Get the token content as string
    pub fn as_str(&self) -> &str {
        &self.content
    }
}

impl ModelConfig {
    /// Load model config from a JSON file
    #[cfg(feature = "serde")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> EdgeResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| EdgeError::configuration(format!("Failed to read config file: {e}")))?;

        let mut config: ModelConfig = serde_json::from_str(&content)
            .map_err(|e| EdgeError::configuration(format!("Failed to parse config: {e}")))?;

        config.validate()?;
        config.normalize();

        Ok(config)
    }

    /// Validate the model configuration
    pub fn validate(&self) -> EdgeResult<()> {
        if self.vocab_size == 0 {
            return Err(EdgeError::configuration(
                "vocab_size must be greater than 0",
            ));
        }

        if self.max_position_embeddings == 0 {
            return Err(EdgeError::configuration(
                "max_position_embeddings must be greater than 0",
            ));
        }

        if self.model_name.is_empty() {
            return Err(EdgeError::configuration("model_name cannot be empty"));
        }

        Ok(())
    }

    /// Normalize configuration values (handle different naming conventions)
    pub fn normalize(&mut self) {
        // Handle GPT-2 style naming where n_positions might be used instead
        if self.max_position_embeddings == default_max_position_embeddings()
            && self.n_positions != default_max_position_embeddings()
        {
            self.max_position_embeddings = self.n_positions;
        }
    }

    /// Get the effective context length
    pub fn context_length(&self) -> usize {
        self.max_position_embeddings.max(self.n_positions)
    }
}

impl TokenizerConfig {
    /// Load tokenizer config from a JSON file
    #[cfg(feature = "serde")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> EdgeResult<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            EdgeError::configuration(format!("Failed to read tokenizer config: {e}"))
        })?;

        let config: TokenizerConfig = serde_json::from_str(&content).map_err(|e| {
            EdgeError::configuration(format!("Failed to parse tokenizer config: {e}"))
        })?;

        config.validate()?;
        Ok(config)
    }

    /// Validate the tokenizer configuration
    pub fn validate(&self) -> EdgeResult<()> {
        if self.model_max_length == 0 {
            return Err(EdgeError::configuration(
                "model_max_length must be greater than 0",
            ));
        }
        Ok(())
    }
}

impl SpecialTokensMap {
    /// Load special tokens mapping from a JSON file
    #[cfg(feature = "serde")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> EdgeResult<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            EdgeError::configuration(format!("Failed to read special tokens map: {e}"))
        })?;

        let map: SpecialTokensMap = serde_json::from_str(&content).map_err(|e| {
            EdgeError::configuration(format!("Failed to parse special tokens map: {e}"))
        })?;

        Ok(map)
    }

    /// Get BOS token as SpecialToken
    pub fn get_bos_token(&self) -> Option<SpecialToken> {
        self.bos_token.as_ref().and_then(SpecialToken::from_value)
    }

    /// Get EOS token as SpecialToken
    pub fn get_eos_token(&self) -> Option<SpecialToken> {
        self.eos_token.as_ref().and_then(SpecialToken::from_value)
    }

    /// Get UNK token as SpecialToken
    pub fn get_unk_token(&self) -> Option<SpecialToken> {
        self.unk_token.as_ref().and_then(SpecialToken::from_value)
    }

    /// Get PAD token as SpecialToken
    pub fn get_pad_token(&self) -> Option<SpecialToken> {
        self.pad_token.as_ref().and_then(SpecialToken::from_value)
    }
}

impl RuntimeConfig {
    /// Validate runtime configuration
    pub fn validate(&self) -> EdgeResult<()> {
        if self.temperature < 0.0 {
            return Err(EdgeError::configuration("temperature must be non-negative"));
        }

        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(EdgeError::configuration(
                "top_p must be between 0.0 and 1.0",
            ));
        }

        if self.repetition_penalty <= 0.0 {
            return Err(EdgeError::configuration(
                "repetition_penalty must be positive",
            ));
        }

        if self.memory_optimization > 2 {
            return Err(EdgeError::configuration(
                "memory_optimization must be 0, 1, or 2",
            ));
        }

        Ok(())
    }

    /// Check if sampling is deterministic
    pub fn is_deterministic(&self) -> bool {
        self.temperature == 0.0
    }
}

impl EdgeConfig {
    /// Create a new EdgeConfig with the given model name
    pub fn new(model_name: impl Into<String>) -> Self {
        let model_config = ModelConfig {
            model_name: model_name.into(),
            ..Default::default()
        };

        Self {
            model: model_config,
            tokenizer: TokenizerConfig::default(),
            special_tokens: SpecialTokensMap::default(),
            runtime: RuntimeConfig::default(),
            #[cfg(feature = "chat")]
            chat_template: None,
        }
    }

    /// Load complete configuration from a directory
    #[cfg(feature = "serde")]
    pub fn from_directory<P: AsRef<Path>>(
        dir: P,
        model_name: impl Into<String>,
    ) -> EdgeResult<Self> {
        let dir = dir.as_ref();

        let model_config =
            ModelConfig::from_file(dir.join("config.json")).unwrap_or_else(|_| ModelConfig {
                model_name: model_name.into(),
                ..Default::default()
            });

        let tokenizer_config =
            TokenizerConfig::from_file(dir.join("tokenizer_config.json")).unwrap_or_default();

        let special_tokens =
            SpecialTokensMap::from_file(dir.join("special_tokens_map.json")).unwrap_or_default();

        #[cfg(feature = "chat")]
        let chat_template = std::fs::read_to_string(dir.join("chat_template.jinja")).ok();

        let config = EdgeConfig {
            model: model_config,
            tokenizer: tokenizer_config,
            special_tokens,
            runtime: RuntimeConfig::default(),
            #[cfg(feature = "chat")]
            chat_template,
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate the complete configuration
    pub fn validate(&self) -> EdgeResult<()> {
        self.model.validate()?;
        self.tokenizer.validate()?;
        self.runtime.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.max_position_embeddings, 2048);
    }

    #[test]
    fn test_model_config_validation() {
        let mut config = ModelConfig::default();
        config.vocab_size = 0;
        assert!(config.validate().is_err());

        config.vocab_size = 1000;
        config.model_name = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_special_token_from_string() {
        let value = serde_json::Value::String("<s>".to_string());
        let token = SpecialToken::from_value(&value).unwrap();
        assert_eq!(token.content, "<s>");
        assert!(!token.lstrip);
    }

    #[test]
    fn test_runtime_config_validation() {
        let mut config = RuntimeConfig::default();
        config.temperature = -1.0;
        assert!(config.validate().is_err());

        config.temperature = 0.7;
        config.top_p = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_edge_config_creation() {
        let config = EdgeConfig::new("test-model");
        assert_eq!(config.model.model_name, "test-model");
        assert!(config.validate().is_ok());
    }
}
