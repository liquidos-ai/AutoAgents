//! Error types for Parakeet STT provider

use thiserror::Error;

/// Parakeet-specific errors
#[derive(Error, Debug)]
pub enum ParakeetError {
    /// Parakeet library error
    #[error("Parakeet error: {0}")]
    ParakeetError(#[from] parakeet_rs::Error),

    /// Model loading failed
    #[error("Failed to load model from {path}: {reason}")]
    ModelLoadError { path: String, reason: String },

    /// Transcription failed
    #[error("Transcription failed: {0}")]
    TranscriptionError(String),

    /// Invalid audio format
    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other errors
    #[error("Parakeet provider error: {0}")]
    Other(String),
}

/// Result type for Parakeet operations
pub type Result<T> = std::result::Result<T, ParakeetError>;

/// Convert Parakeet errors to STT errors
impl From<ParakeetError> for crate::error::STTError {
    fn from(err: ParakeetError) -> Self {
        match err {
            ParakeetError::ParakeetError(e) => {
                crate::error::STTError::ProviderError(e.to_string(), "Parakeet".to_string())
            }
            ParakeetError::ModelLoadError { path, reason } => {
                crate::error::STTError::ModelNotFound(reason, path)
            }
            ParakeetError::TranscriptionError(msg) => {
                crate::error::STTError::TranscriptionFailed(msg, 0.0, 16000)
            }
            ParakeetError::InvalidAudioFormat(msg) => {
                crate::error::STTError::InvalidAudioFormat(msg, 16000, 0, 1, 0)
            }
            ParakeetError::ConfigError(msg) => crate::error::STTError::InvalidConfiguration(
                msg,
                "config".to_string(),
                "".to_string(),
            ),
            ParakeetError::IoError(e) => {
                crate::error::STTError::IoError(e, "file operation".to_string(), "".to_string())
            }
            ParakeetError::Other(msg) => {
                crate::error::STTError::Other(msg, "Parakeet provider".to_string())
            }
        }
    }
}
