//! Error handling and conversions for llama.cpp backend.

use autoagents_llm::error::LLMError;
use std::fmt;

/// Internal error type for llama.cpp operations.
#[derive(Debug)]
pub enum LlamaCppProviderError {
    /// Model loading failed.
    ModelLoad(String),
    /// Context creation failed.
    ContextLoad(String),
    /// Tokenization or detokenization failed.
    Tokenization(String),
    /// Inference failed.
    Inference(String),
    /// Configuration error.
    Config(String),
    /// Prompt or template error.
    Template(String),
    /// Embedding error.
    Embedding(String),
    /// Unsupported feature.
    Unsupported(String),
    /// Generic error.
    Other(String),
}

impl fmt::Display for LlamaCppProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlamaCppProviderError::ModelLoad(e) => write!(f, "Model Load Error: {}", e),
            LlamaCppProviderError::ContextLoad(e) => write!(f, "Context Load Error: {}", e),
            LlamaCppProviderError::Tokenization(e) => write!(f, "Tokenization Error: {}", e),
            LlamaCppProviderError::Inference(e) => write!(f, "Inference Error: {}", e),
            LlamaCppProviderError::Config(e) => write!(f, "Configuration Error: {}", e),
            LlamaCppProviderError::Template(e) => write!(f, "Template Error: {}", e),
            LlamaCppProviderError::Embedding(e) => write!(f, "Embedding Error: {}", e),
            LlamaCppProviderError::Unsupported(e) => write!(f, "Unsupported: {}", e),
            LlamaCppProviderError::Other(e) => write!(f, "llama.cpp Error: {}", e),
        }
    }
}

impl std::error::Error for LlamaCppProviderError {}

impl From<LlamaCppProviderError> for LLMError {
    fn from(err: LlamaCppProviderError) -> Self {
        match err {
            LlamaCppProviderError::ModelLoad(e) => {
                LLMError::ProviderError(format!("Failed to load model: {}", e))
            }
            LlamaCppProviderError::ContextLoad(e) => {
                LLMError::ProviderError(format!("Failed to create context: {}", e))
            }
            LlamaCppProviderError::Tokenization(e) => {
                LLMError::ProviderError(format!("Tokenization failed: {}", e))
            }
            LlamaCppProviderError::Inference(e) => {
                LLMError::ProviderError(format!("Inference failed: {}", e))
            }
            LlamaCppProviderError::Config(e) => {
                LLMError::InvalidRequest(format!("Invalid configuration: {}", e))
            }
            LlamaCppProviderError::Template(e) => {
                LLMError::InvalidRequest(format!("Template error: {}", e))
            }
            LlamaCppProviderError::Embedding(e) => {
                LLMError::ProviderError(format!("Embedding failed: {}", e))
            }
            LlamaCppProviderError::Unsupported(e) => LLMError::NoToolSupport(e),
            LlamaCppProviderError::Other(e) => LLMError::ProviderError(format!(
                "llama.cpp error: {}",
                e
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LlamaCppProviderError::ModelLoad("missing file".to_string());
        assert_eq!(err.to_string(), "Model Load Error: missing file");
    }

    #[test]
    fn test_error_to_llm_error() {
        let err = LlamaCppProviderError::Config("bad config".to_string());
        let llm_err: LLMError = err.into();
        assert!(llm_err.to_string().contains("Invalid configuration"));
    }
}
