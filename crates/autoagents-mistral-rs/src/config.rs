//! Configuration structures for mistral.rs provider

use crate::models::ModelSource;
use mistralrs::IsqType;
use serde::{Deserialize, Serialize};

/// Complete configuration for MistralRsProvider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsConfig {
    /// Model source (HuggingFace or GGUF)
    pub model_source: ModelSource,

    /// In-Situ Quantization type (for HuggingFace models)
    #[serde(skip)]
    pub isq_type: Option<IsqType>,

    /// Enable paged attention for memory efficiency
    pub paged_attention: bool,

    /// Enable logging during model operations
    pub logging: bool,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0)
    pub temperature: Option<f32>,

    /// Top-p sampling parameter
    pub top_p: Option<f32>,

    /// Top-k sampling parameter
    pub top_k: Option<u32>,

    /// Optional system prompt
    pub system_prompt: Option<String>,
}

impl Default for MistralRsConfig {
    fn default() -> Self {
        Self {
            model_source: ModelSource::HuggingFace {
                repo_id: "microsoft/Phi-3.5-mini-instruct".to_string(),
                revision: None,
                model_type: crate::models::ModelType::Auto,
            },
            isq_type: None,
            paged_attention: false,
            logging: false,
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            system_prompt: None,
        }
    }
}

/// Builder for MistralRsConfig
#[derive(Debug)]
pub struct MistralRsConfigBuilder {
    config: MistralRsConfig,
}

impl MistralRsConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: MistralRsConfig::default(),
        }
    }

    /// Set the model source
    pub fn model_source(mut self, source: ModelSource) -> Self {
        self.config.model_source = source;
        self
    }

    /// Set the ISQ type for in-situ quantization (HuggingFace models only)
    pub fn with_isq(mut self, isq: IsqType) -> Self {
        self.config.isq_type = Some(isq);
        self
    }

    /// Enable paged attention
    pub fn with_paged_attention(mut self) -> Self {
        self.config.paged_attention = true;
        self
    }

    /// Enable logging
    pub fn with_logging(mut self) -> Self {
        self.config.logging = true;
        self
    }

    /// Set maximum tokens to generate
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set sampling temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    /// Set top-p sampling parameter
    pub fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = Some(p);
        self
    }

    /// Set top-k sampling parameter
    pub fn top_k(mut self, k: u32) -> Self {
        self.config.top_k = Some(k);
        self
    }

    /// Set system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> MistralRsConfig {
        self.config
    }
}

impl Default for MistralRsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelSource;

    #[test]
    fn test_default_config() {
        let config = MistralRsConfig::default();
        assert!(!config.paged_attention);
        assert!(!config.logging);
        assert_eq!(config.max_tokens, Some(512));
        assert_eq!(config.temperature, Some(0.7));
    }

    #[test]
    fn test_config_builder_basic() {
        let config = MistralRsConfigBuilder::default()
            .max_tokens(1024)
            .temperature(0.8)
            .build();

        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.temperature, Some(0.8));
    }

    #[test]
    fn test_config_builder_full() {
        let source = ModelSource::HuggingFace {
            repo_id: "test/model".to_string(),
            revision: None,
            model_type: crate::models::ModelType::Auto,
        };

        let config = MistralRsConfigBuilder::default()
            .model_source(source.clone())
            .with_paged_attention()
            .with_logging()
            .max_tokens(2048)
            .temperature(0.9)
            .top_p(0.95)
            .top_k(50)
            .system_prompt("You are a helpful assistant")
            .build();

        assert_eq!(config.model_source, source);
        assert!(config.paged_attention);
        assert!(config.logging);
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.temperature, Some(0.9));
        assert_eq!(config.top_p, Some(0.95));
        assert_eq!(config.top_k, Some(50));
        assert_eq!(
            config.system_prompt,
            Some("You are a helpful assistant".to_string())
        );
    }

    #[test]
    fn test_config_builder_gguf_source() {
        let source = ModelSource::Gguf {
            model_dir: "/models".to_string(),
            files: vec!["model.gguf".to_string()],
            tokenizer: None,
            chat_template: None,
        };

        let config = MistralRsConfigBuilder::default()
            .model_source(source.clone())
            .build();

        assert_eq!(config.model_source, source);
    }

    #[test]
    fn test_config_builder_isq() {
        let config = MistralRsConfigBuilder::default()
            .with_isq(IsqType::Q8_0)
            .build();

        assert!(config.isq_type.is_some());
    }

    #[test]
    fn test_config_clone() {
        let config = MistralRsConfig::default();
        let cloned = config.clone();

        assert_eq!(config.paged_attention, cloned.paged_attention);
        assert_eq!(config.logging, cloned.logging);
        assert_eq!(config.max_tokens, cloned.max_tokens);
    }

    #[test]
    fn test_config_builder_default() {
        let builder1 = MistralRsConfigBuilder::default();
        let builder2 = MistralRsConfigBuilder::default();

        let config1 = builder1.build();
        let config2 = builder2.build();

        assert_eq!(config1.max_tokens, config2.max_tokens);
        assert_eq!(config1.temperature, config2.temperature);
    }
}
