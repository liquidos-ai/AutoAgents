use thiserror::Error;

/// TTS-related errors
#[derive(Error, Debug)]
pub enum TTSError {
    /// Provider-specific error
    #[error(
        "TTS provider error: {0}\nProvider: {1}\nDetails: This error originated from the TTS provider implementation"
    )]
    ProviderError(String, String),

    /// Voice not found
    #[error(
        "Voice not found: '{0}'\nAvailable voices: {1}\nSuggestion: Use list_predefined_voices() to see all available voices"
    )]
    VoiceNotFound(String, String),

    /// Invalid voice data
    #[error(
        "Invalid voice data: {0}\nContext: {1}\nSuggestion: Ensure voice embeddings are properly downloaded from HuggingFace"
    )]
    InvalidVoiceData(String, String),

    /// Audio generation failed
    #[error(
        "Audio generation failed: {0}\nInput text length: {1} characters\nVoice: {2}\nSuggestion: Try shorter text or check model initialization"
    )]
    GenerationFailed(String, usize, String),

    /// Streaming not supported
    #[error(
        "Streaming not supported by this provider\nProvider: {0}\nSuggestion: Use generate_speech() instead of generate_speech_stream()"
    )]
    StreamingNotSupported(String),

    /// IO error
    #[error(
        "IO error during TTS operation: {0}\nOperation: {1}\nPath: {2}\nSuggestion: Check file permissions and disk space"
    )]
    IoError(std::io::Error, String, String),

    /// Model not found
    #[error(
        "Model not found: '{0}'\nModel path: {1}\nSuggestion: Ensure model is downloaded from HuggingFace. Check HUGGINGFACE_TOKEN environment variable"
    )]
    ModelNotFound(String, String),

    /// Other errors
    #[error("TTS error: {0}\nContext: {1}")]
    Other(String, String),
}

/// Result type for TTS operations
pub type TTSResult<T> = Result<T, TTSError>;
