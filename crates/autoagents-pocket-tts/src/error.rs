//! Error handling for Pocket-TTS provider

use autoagents_tts::TTSError;
use thiserror::Error;

/// Pocket-TTS specific errors
#[derive(Debug, Error)]
pub enum PocketTTSError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Voice loading failed: {0}")]
    VoiceLoadError(String),

    #[error("Audio generation failed: {0}")]
    GenerationError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Voice not found: {0}")]
    VoiceNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(feature = "server")]
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Other error: {0}")]
    Other(String),
}

/// Convert anyhow::Error to PocketTTSError
impl From<anyhow::Error> for PocketTTSError {
    fn from(err: anyhow::Error) -> Self {
        PocketTTSError::Other(err.to_string())
    }
}

/// Convert PocketTTSError to TTSError
impl From<PocketTTSError> for TTSError {
    fn from(err: PocketTTSError) -> Self {
        match err {
            PocketTTSError::ModelLoadError(msg) => TTSError::ModelNotFound(msg),
            PocketTTSError::VoiceLoadError(msg) => TTSError::InvalidVoiceData(msg),
            PocketTTSError::GenerationError(msg) => TTSError::GenerationFailed(msg),
            PocketTTSError::InvalidConfig(msg) => TTSError::InvalidConfiguration(msg),
            PocketTTSError::VoiceNotFound(msg) => TTSError::VoiceNotFound(msg),
            PocketTTSError::IoError(e) => TTSError::IoError(e),
            PocketTTSError::JsonError(e) => TTSError::SerializationError(e.to_string()),
            #[cfg(feature = "server")]
            PocketTTSError::HttpError(e) => TTSError::Other(e.to_string()),
            PocketTTSError::Other(msg) => TTSError::Other(msg),
        }
    }
}

/// Result type for Pocket-TTS operations
pub type Result<T> = std::result::Result<T, PocketTTSError>;
