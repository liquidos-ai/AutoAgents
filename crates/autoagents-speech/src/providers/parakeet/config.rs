//! Configuration for Parakeet STT provider

use super::model::ModelVariant;
use serde::{Deserialize, Serialize};

/// Configuration for Parakeet STT provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParakeetConfig {
    /// Model variant to use (TDT or Nemotron)
    #[serde(default)]
    pub model_variant: ModelVariant,

    /// Model directory path (contains ONNX files and tokenizer)
    pub model_path: String,

    /// Optional execution provider (defaults to CPU)
    /// Options: "cpu", "cuda", "tensorrt", "directml", etc.
    #[serde(default)]
    pub execution_provider: Option<String>,

    /// Optional language hint for multilingual models (TDT)
    /// If None, language will be auto-detected
    #[serde(default)]
    pub language: Option<String>,
}

impl ParakeetConfig {
    /// Create a new configuration
    pub fn new(model_variant: ModelVariant, model_path: impl Into<String>) -> Self {
        Self {
            model_variant,
            model_path: model_path.into(),
            execution_provider: None,
            language: None,
        }
    }

    /// Set execution provider (e.g., "cuda", "tensorrt")
    pub fn with_execution_provider(mut self, provider: impl Into<String>) -> Self {
        self.execution_provider = Some(provider.into());
        self
    }

    /// Set language hint
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            model_variant: ModelVariant::default(),
            model_path: ".".to_string(),
            execution_provider: None,
            language: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParakeetConfig::default();
        assert_eq!(config.model_variant, ModelVariant::TDT);
        assert_eq!(config.model_path, ".");
        assert!(config.execution_provider.is_none());
        assert!(config.language.is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let config = ParakeetConfig::new(ModelVariant::Nemotron, "./models/nemotron")
            .with_execution_provider("cuda")
            .with_language("en");

        assert_eq!(config.model_variant, ModelVariant::Nemotron);
        assert_eq!(config.model_path, "./models/nemotron");
        assert_eq!(config.execution_provider, Some("cuda".to_string()));
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_serialization() {
        let config = ParakeetConfig::new(ModelVariant::TDT, "./models/tdt");
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ParakeetConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_variant, config.model_variant);
        assert_eq!(deserialized.model_path, config.model_path);
    }
}
