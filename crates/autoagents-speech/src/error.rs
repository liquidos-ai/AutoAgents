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

    /// Format not supported
    #[error(
        "Audio format not supported: {0:?}\nProvider: {1}\nSupported formats: {2}\nSuggestion: Use one of the supported formats"
    )]
    FormatNotSupported(crate::types::AudioFormat, String, String),

    /// IO error
    #[error(
        "IO error during TTS operation: {0}\nOperation: {1}\nPath: {2}\nSuggestion: Check file permissions and disk space"
    )]
    IoError(std::io::Error, String, String),

    /// Serialization error
    #[error(
        "Serialization error: {0}\nData type: {1}\nSuggestion: Check that data structure is serializable"
    )]
    SerializationError(String, String),

    /// Model not found
    #[error(
        "Model not found: '{0}'\nModel path: {1}\nSuggestion: Ensure model is downloaded from HuggingFace. Check HUGGINGFACE_TOKEN environment variable"
    )]
    ModelNotFound(String, String),

    /// Invalid configuration
    #[error(
        "Invalid configuration: {0}\nParameter: {1}\nValid range: {2}\nSuggestion: Review configuration documentation"
    )]
    InvalidConfiguration(String, String, String),

    /// Provider not ready
    #[error(
        "Provider not ready: {0}\nProvider: {1}\nInitialization state: {2}\nSuggestion: Wait for provider initialization or check logs for errors"
    )]
    ProviderNotReady(String, String, String),

    /// Other errors
    #[error("TTS error: {0}\nContext: {1}")]
    Other(String, String),
}

/// Result type for TTS operations
pub type TTSResult<T> = Result<T, TTSError>;

/// STT-related errors
#[derive(Error, Debug)]
pub enum STTError {
    /// Provider-specific error
    #[error(
        "STT provider error: {0}\nProvider: {1}\nDetails: This error originated from the STT provider implementation"
    )]
    ProviderError(String, String),

    /// Transcription failed
    #[error(
        "Transcription failed: {0}\nAudio duration: {1}s\nSample rate: {2}Hz\nSuggestion: Check audio quality or try a different model"
    )]
    TranscriptionFailed(String, f32, u32),

    /// Streaming not supported
    #[error(
        "Streaming not supported by this provider\nProvider: {0}\nSuggestion: Use transcribe() instead of transcribe_stream()"
    )]
    StreamingNotSupported(String),

    /// Invalid audio format
    #[error(
        "Invalid audio format: {0}\nExpected sample rate: {1}Hz, Got: {2}Hz\nExpected channels: {3}, Got: {4}\nSuggestion: Resample audio to the correct format"
    )]
    InvalidAudioFormat(String, u32, u32, u16, u16),

    /// IO error
    #[error(
        "IO error during STT operation: {0}\nOperation: {1}\nPath: {2}\nSuggestion: Check file permissions and disk space"
    )]
    IoError(std::io::Error, String, String),

    /// Model not found
    #[error(
        "Model not found: '{0}'\nModel path: {1}\nSuggestion: Ensure model is downloaded from HuggingFace"
    )]
    ModelNotFound(String, String),

    /// Invalid configuration
    #[error(
        "Invalid configuration: {0}\nParameter: {1}\nValid range: {2}\nSuggestion: Review configuration documentation"
    )]
    InvalidConfiguration(String, String, String),

    /// Provider not ready
    #[error(
        "Provider not ready: {0}\nProvider: {1}\nInitialization state: {2}\nSuggestion: Wait for provider initialization or check logs for errors"
    )]
    ProviderNotReady(String, String, String),

    /// Language not supported
    #[error(
        "Language not supported: '{0}'\nProvider: {1}\nSupported languages: {2}\nSuggestion: Use a supported language or switch to a multilingual model"
    )]
    LanguageNotSupported(String, String, String),

    /// Other errors
    #[error("STT error: {0}\nContext: {1}")]
    Other(String, String),
}

/// Result type for STT operations
pub type STTResult<T> = Result<T, STTError>;
