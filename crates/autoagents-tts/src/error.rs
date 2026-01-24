use thiserror::Error;

/// TTS-related errors
#[derive(Error, Debug)]
pub enum TTSError {
    /// Provider-specific error
    #[error("TTS provider error: {0}")]
    ProviderError(String),

    /// Voice not found
    #[error("Voice not found: {0}")]
    VoiceNotFound(String),

    /// Invalid voice data
    #[error("Invalid voice data: {0}")]
    InvalidVoiceData(String),

    /// Audio generation failed
    #[error("Audio generation failed: {0}")]
    GenerationFailed(String),

    /// Streaming not supported
    #[error("Streaming not supported by this provider")]
    StreamingNotSupported,

    /// Format not supported
    #[error("Audio format not supported: {0:?}")]
    FormatNotSupported(crate::types::AudioFormat),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Voice state persistence error
    #[error("Voice state persistence error: {0}")]
    PersistenceError(String),

    /// Other errors
    #[error("TTS error: {0}")]
    Other(String),
}

/// Result type for TTS operations
pub type TTSResult<T> = Result<T, TTSError>;
