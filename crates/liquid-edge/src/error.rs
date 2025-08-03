//! Error types for liquid-edge runtime
//!
//! This module provides comprehensive error handling for all operations
//! in the edge inference runtime.

use ort::OrtError;
use thiserror::Error;

/// Main error type for liquid-edge operations
#[derive(Error, Debug)]
pub enum EdgeError {
    /// Model loading or initialization errors
    #[error("Model error: {message}")]
    Model { message: String },

    /// Inference execution errors
    #[error("Inference error: {message}")]
    Inference { message: String },

    /// Tokenization errors
    #[error("Tokenization error: {message}")]
    Tokenization { message: String },

    /// Configuration loading or parsing errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Runtime backend errors
    #[error("Runtime error: {message}")]
    Runtime { message: String },

    /// I/O operations errors
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// JSON serialization/deserialization errors
    #[cfg(feature = "serde")]
    #[error("JSON error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },

    /// Template rendering errors
    #[cfg(feature = "jinja-templates")]
    #[error("Template error: {message}")]
    Template { message: String },

    /// Chat-specific errors
    #[cfg(feature = "chat")]
    #[error("Chat error: {message}")]
    Chat { message: String },

    /// ONNX Runtime specific errors
    #[cfg(feature = "onnx-runtime")]
    #[error("ONNX Runtime error: {source}")]
    OnnxRuntime {
        #[from]
        source: OrtError,
    },

    /// Tokenizer library errors
    #[error("HuggingFace tokenizer error: {source}")]
    HfTokenizer {
        #[from]
        source: tokenizers::Error,
    },

    /// Invalid input parameters
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Feature not available
    #[error("Feature '{feature}' is not available. Enable the corresponding feature flag.")]
    FeatureNotAvailable { feature: String },

    /// Resource not found
    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    /// Operation timeout
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// Memory allocation or limit errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Concurrency or threading errors
    #[error("Async error: {message}")]
    Async { message: String },
}

/// Specialized result type for liquid-edge operations
pub type EdgeResult<T> = Result<T, EdgeError>;

impl EdgeError {
    /// Create a new model error
    pub fn model<S: Into<String>>(message: S) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    /// Create a new inference error
    pub fn inference<S: Into<String>>(message: S) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }

    /// Create a new tokenization error
    pub fn tokenization<S: Into<String>>(message: S) -> Self {
        Self::Tokenization {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new runtime error
    pub fn runtime<S: Into<String>>(message: S) -> Self {
        Self::Runtime {
            message: message.into(),
        }
    }

    /// Create a new template error
    #[cfg(feature = "jinja-templates")]
    pub fn template<S: Into<String>>(message: S) -> Self {
        Self::Template {
            message: message.into(),
        }
    }

    /// Create a new chat error
    #[cfg(feature = "chat")]
    pub fn chat<S: Into<String>>(message: S) -> Self {
        Self::Chat {
            message: message.into(),
        }
    }

    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a new feature not available error
    pub fn feature_not_available<S: Into<String>>(feature: S) -> Self {
        Self::FeatureNotAvailable {
            feature: feature.into(),
        }
    }

    /// Create a new not found error
    pub fn not_found<S: Into<String>>(resource: S) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Create a new timeout error
    pub fn timeout(duration_ms: u64) -> Self {
        Self::Timeout { duration_ms }
    }

    /// Create a new memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a new async error
    pub fn async_error<S: Into<String>>(message: S) -> Self {
        Self::Async {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            EdgeError::Timeout { .. } => true,
            EdgeError::Memory { .. } => false,
            EdgeError::Model { .. } => false,
            EdgeError::Configuration { .. } => false,
            EdgeError::NotFound { .. } => false,
            EdgeError::FeatureNotAvailable { .. } => false,
            EdgeError::Inference { .. } => true,
            EdgeError::Tokenization { .. } => true,
            EdgeError::Runtime { .. } => true,
            EdgeError::Io { .. } => true,
            #[cfg(feature = "serde")]
            EdgeError::Json { .. } => false,
            #[cfg(feature = "jinja-templates")]
            EdgeError::Template { .. } => true,
            #[cfg(feature = "chat")]
            EdgeError::Chat { .. } => true,
            #[cfg(feature = "onnx-runtime")]
            EdgeError::OnnxRuntime { .. } => true,
            EdgeError::HfTokenizer { .. } => true,
            EdgeError::InvalidInput { .. } => false,
            EdgeError::Async { .. } => true,
        }
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            EdgeError::Model { .. } => "model",
            EdgeError::Inference { .. } => "inference",
            EdgeError::Tokenization { .. } => "tokenization",
            EdgeError::Configuration { .. } => "configuration",
            EdgeError::Runtime { .. } => "runtime",
            EdgeError::Io { .. } => "io",
            #[cfg(feature = "serde")]
            EdgeError::Json { .. } => "serialization",
            #[cfg(feature = "jinja-templates")]
            EdgeError::Template { .. } => "template",
            #[cfg(feature = "chat")]
            EdgeError::Chat { .. } => "chat",
            #[cfg(feature = "onnx-runtime")]
            EdgeError::OnnxRuntime { .. } => "onnx_runtime",
            EdgeError::HfTokenizer { .. } => "tokenizer",
            EdgeError::InvalidInput { .. } => "input",
            EdgeError::FeatureNotAvailable { .. } => "feature",
            EdgeError::NotFound { .. } => "not_found",
            EdgeError::Timeout { .. } => "timeout",
            EdgeError::Memory { .. } => "memory",
            EdgeError::Async { .. } => "async",
        }
    }
}

/// Convert anyhow::Error to EdgeError
impl From<anyhow::Error> for EdgeError {
    fn from(err: anyhow::Error) -> Self {
        EdgeError::runtime(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = EdgeError::model("Test model error");
        assert_eq!(err.category(), "model");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let edge_err = EdgeError::Io { source: io_err };
        assert_eq!(edge_err.category(), "io");
        assert!(edge_err.is_recoverable());
    }

    #[test]
    fn test_timeout_error() {
        let err = EdgeError::timeout(5000);
        assert!(err.is_recoverable());
        assert_eq!(err.category(), "timeout");
    }
}
