//! Error handling and conversions for mistral.rs backend

use autoagents_llm::error::LLMError;
use std::fmt;

/// Internal error type for mistral.rs operations
#[derive(Debug)]
pub enum MistralRsError {
    /// Model loading failed
    ModelLoadError(String),
    /// Inference failed
    InferenceError(String),
    /// Configuration error
    ConfigError(String),
    /// Generic error
    Other(String),
}

impl fmt::Display for MistralRsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MistralRsError::ModelLoadError(e) => write!(f, "Model Load Error: {}", e),
            MistralRsError::InferenceError(e) => write!(f, "Inference Error: {}", e),
            MistralRsError::ConfigError(e) => write!(f, "Configuration Error: {}", e),
            MistralRsError::Other(e) => write!(f, "Mistral.rs Error: {}", e),
        }
    }
}

impl std::error::Error for MistralRsError {}

/// Convert MistralRsError to LLMError
impl From<MistralRsError> for LLMError {
    fn from(err: MistralRsError) -> Self {
        match err {
            MistralRsError::ModelLoadError(e) => {
                LLMError::ProviderError(format!("Failed to load model: {}", e))
            }
            MistralRsError::InferenceError(e) => {
                LLMError::ProviderError(format!("Inference failed: {}", e))
            }
            MistralRsError::ConfigError(e) => {
                LLMError::InvalidRequest(format!("Invalid configuration: {}", e))
            }
            MistralRsError::Other(e) => LLMError::ProviderError(format!("Mistral.rs error: {}", e)),
        }
    }
}

/// Convert anyhow::Error to LLMError
pub(crate) fn convert_anyhow_error(err: anyhow::Error) -> LLMError {
    LLMError::ProviderError(format!("Mistral.rs error: {}", err))
}

/// Convert generic errors to LLMError
#[allow(dead_code)]
pub(crate) fn convert_error(err: impl std::error::Error) -> LLMError {
    LLMError::ProviderError(format!("Mistral.rs error: {}", err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistralrs_error_display() {
        let err = MistralRsError::ModelLoadError("model not found".to_string());
        assert_eq!(err.to_string(), "Model Load Error: model not found");

        let err = MistralRsError::InferenceError("out of memory".to_string());
        assert_eq!(err.to_string(), "Inference Error: out of memory");

        let err = MistralRsError::ConfigError("invalid parameter".to_string());
        assert_eq!(err.to_string(), "Configuration Error: invalid parameter");

        let err = MistralRsError::Other("unknown error".to_string());
        assert_eq!(err.to_string(), "Mistral.rs Error: unknown error");
    }

    #[test]
    fn test_mistralrs_error_to_llm_error() {
        let err = MistralRsError::ModelLoadError("test".to_string());
        let llm_err: LLMError = err.into();
        assert!(llm_err.to_string().contains("Failed to load model"));

        let err = MistralRsError::InferenceError("test".to_string());
        let llm_err: LLMError = err.into();
        assert!(llm_err.to_string().contains("Inference failed"));

        let err = MistralRsError::ConfigError("test".to_string());
        let llm_err: LLMError = err.into();
        assert!(llm_err.to_string().contains("Invalid configuration"));
    }

    #[test]
    fn test_convert_anyhow_error() {
        let err = anyhow::anyhow!("test error");
        let llm_err = convert_anyhow_error(err);
        assert!(llm_err.to_string().contains("Mistral.rs error"));
        assert!(llm_err.to_string().contains("test error"));
    }
}
