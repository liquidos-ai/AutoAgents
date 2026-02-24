//! Error types for Parakeet STT provider

use thiserror::Error;

/// Parakeet-specific errors
#[derive(Error, Debug)]
pub enum ParakeetError {
    /// Model initialization error
    #[error(
        "Model initialization failed: {0}\nModel variant: {1}\nModel path: {2}\nDevice: {3}\nSuggestion: Ensure model files exist at the specified path and have correct format (ONNX)"
    )]
    ModelInitError(String, String, String, String),

    /// Model loading failed
    #[error(
        "Failed to load model from {path}: {reason}\nModel variant: {variant}\nSuggestion: Verify model path exists and files are not corrupted. Check model was exported correctly."
    )]
    ModelLoadError {
        path: String,
        reason: String,
        variant: String,
    },

    /// Transcription error
    #[error(
        "Transcription failed: {0}\nStage: {1}\nDetails: {2}\nSuggestion: Check audio format matches expected 16kHz mono PCM, and model state is valid"
    )]
    TranscriptionError(String, String, String),

    /// Audio processing error
    #[error(
        "Audio processing failed: {0}\nExpected: {1}\nActual: {2}\nSuggestion: Verify audio is in correct format (16kHz, mono, f32 samples)"
    )]
    AudioProcessingError(String, String, String),

    /// Invalid audio format
    #[error(
        "Invalid audio format: {0}\nExpected sample rate: {1} Hz\nExpected channels: {2}\nActual sample rate: {3} Hz\nActual channels: {4}\nSuggestion: Resample audio to 16kHz mono before processing"
    )]
    InvalidAudioFormat(String, u32, u16, u32, u16),

    /// Chunk processing error (for streaming models)
    #[error(
        "Chunk processing failed: {0}\nChunk size: {1} samples\nExpected: {2} samples\nModel: {3}\nSuggestion: Use recommended chunk size from model.chunk_size() method"
    )]
    ChunkProcessingError(String, usize, usize, String),

    /// Streaming error
    #[error("Streaming operation failed: {0}\nModel variant: {1}\nOperation: {2}\nSuggestion: {3}")]
    StreamingError(String, String, String, String),

    /// Configuration error
    #[error(
        "Configuration error: {0}\nParameter: {1}\nValue: {2}\nSuggestion: Check configuration parameters match model requirements"
    )]
    ConfigError(String, String, String),

    /// Execution provider error
    #[error(
        "Execution provider error: {0}\nProvider: {1}\nSuggestion: Try using 'cpu' provider or ensure CUDA/CoreML is properly installed"
    )]
    ExecutionProviderError(String, String),

    /// Feature extraction error
    #[error(
        "Feature extraction failed: {0}\nAudio length: {1} samples\nSuggestion: Ensure audio contains valid speech and is not corrupted"
    )]
    FeatureExtractionError(String, usize),

    /// Language not supported
    #[error(
        "Language not supported: {0}\nModel variant: {1}\nSupported languages: {2}\nSuggestion: Use TDT model for multilingual support or specify a supported language"
    )]
    LanguageNotSupported(String, String, String),

    /// Parakeet library error
    #[error("Parakeet library error: {0}\nOperation: {1}\nContext: {2}")]
    ParakeetLibraryError(String, String, String),

    /// IO error
    #[error("IO error: {0}\nFile path: {1}\nOperation: {2}")]
    IoError(std::io::Error, String, String),

    /// Other errors
    #[error("Parakeet provider error: {0}\nContext: {1}")]
    Other(String, String),
}

impl ParakeetError {
    /// Create a simple transcription error with context
    pub fn transcription_error(msg: impl Into<String>) -> Self {
        Self::TranscriptionError(
            msg.into(),
            "unknown stage".to_string(),
            "no additional details".to_string(),
        )
    }

    /// Create a transcription error with detailed context
    pub fn transcription_error_detailed(
        msg: impl Into<String>,
        stage: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self::TranscriptionError(msg.into(), stage.into(), details.into())
    }

    /// Create a model initialization error
    pub fn model_init_error(
        msg: impl Into<String>,
        variant: impl Into<String>,
        path: impl Into<String>,
        device: impl Into<String>,
    ) -> Self {
        Self::ModelInitError(msg.into(), variant.into(), path.into(), device.into())
    }

    /// Create a model load error
    pub fn model_load_error(
        path: impl Into<String>,
        reason: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self::ModelLoadError {
            path: path.into(),
            reason: reason.into(),
            variant: variant.into(),
        }
    }

    /// Create an audio processing error
    pub fn audio_processing_error(
        msg: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::AudioProcessingError(msg.into(), expected.into(), actual.into())
    }

    /// Create an invalid audio format error
    pub fn invalid_audio_format(
        msg: impl Into<String>,
        expected_rate: u32,
        expected_channels: u16,
        actual_rate: u32,
        actual_channels: u16,
    ) -> Self {
        Self::InvalidAudioFormat(
            msg.into(),
            expected_rate,
            expected_channels,
            actual_rate,
            actual_channels,
        )
    }

    /// Create a chunk processing error
    pub fn chunk_processing_error(
        msg: impl Into<String>,
        chunk_size: usize,
        expected_size: usize,
        model: impl Into<String>,
    ) -> Self {
        Self::ChunkProcessingError(msg.into(), chunk_size, expected_size, model.into())
    }

    /// Create a streaming error
    pub fn streaming_error(
        msg: impl Into<String>,
        variant: impl Into<String>,
        operation: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::StreamingError(
            msg.into(),
            variant.into(),
            operation.into(),
            suggestion.into(),
        )
    }

    /// Create a configuration error
    pub fn config_error(
        msg: impl Into<String>,
        parameter: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::ConfigError(msg.into(), parameter.into(), value.into())
    }

    /// Create an execution provider error
    pub fn execution_provider_error(msg: impl Into<String>, provider: impl Into<String>) -> Self {
        Self::ExecutionProviderError(msg.into(), provider.into())
    }

    /// Create a feature extraction error
    pub fn feature_extraction_error(msg: impl Into<String>, audio_length: usize) -> Self {
        Self::FeatureExtractionError(msg.into(), audio_length)
    }

    /// Create a language not supported error
    pub fn language_not_supported(
        lang: impl Into<String>,
        variant: impl Into<String>,
        supported: impl Into<String>,
    ) -> Self {
        Self::LanguageNotSupported(lang.into(), variant.into(), supported.into())
    }

    /// Create a parakeet library error
    pub fn library_error(
        msg: impl Into<String>,
        operation: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::ParakeetLibraryError(msg.into(), operation.into(), context.into())
    }
}

/// Result type for Parakeet operations
pub type Result<T> = std::result::Result<T, ParakeetError>;

/// Convert Parakeet errors to STT errors
impl From<ParakeetError> for crate::error::STTError {
    fn from(err: ParakeetError) -> Self {
        match err {
            ParakeetError::ModelInitError(msg, variant, path, device) => {
                crate::error::STTError::ProviderError(
                    format!(
                        "{} (variant: {}, path: {}, device: {})",
                        msg, variant, path, device
                    ),
                    "Parakeet".to_string(),
                )
            }
            ParakeetError::ModelLoadError {
                path,
                reason,
                variant,
            } => crate::error::STTError::ModelNotFound(
                format!("{} (variant: {})", reason, variant),
                path,
            ),
            ParakeetError::TranscriptionError(msg, stage, details) => {
                crate::error::STTError::TranscriptionFailed(
                    format!("{} (stage: {}, details: {})", msg, stage, details),
                    0.0,
                    16000,
                )
            }
            ParakeetError::AudioProcessingError(msg, expected, actual) => {
                crate::error::STTError::Other(
                    format!(
                        "Audio processing failed: {} (expected: {}, actual: {})",
                        msg, expected, actual
                    ),
                    "Parakeet provider".to_string(),
                )
            }
            ParakeetError::InvalidAudioFormat(msg, exp_rate, exp_ch, _act_rate, act_ch) => {
                crate::error::STTError::InvalidAudioFormat(msg, exp_rate, 0, exp_ch, act_ch)
            }
            ParakeetError::ChunkProcessingError(msg, chunk_size, expected, model) => {
                crate::error::STTError::Other(
                    format!(
                        "Chunk processing error: {} (chunk: {} samples, expected: {} samples, model: {})",
                        msg, chunk_size, expected, model
                    ),
                    "Parakeet provider".to_string(),
                )
            }
            ParakeetError::StreamingError(msg, variant, operation, suggestion) => {
                crate::error::STTError::StreamingNotSupported(format!(
                    "{} (variant: {}, operation: {}, suggestion: {})",
                    msg, variant, operation, suggestion
                ))
            }
            ParakeetError::ConfigError(msg, parameter, value) => {
                crate::error::STTError::InvalidConfiguration(msg, parameter, value)
            }
            ParakeetError::ExecutionProviderError(msg, provider) => {
                crate::error::STTError::ProviderError(
                    format!("{} (provider: {})", msg, provider),
                    "Parakeet".to_string(),
                )
            }
            ParakeetError::FeatureExtractionError(msg, audio_length) => {
                crate::error::STTError::Other(
                    format!(
                        "Feature extraction error: {} (audio: {} samples)",
                        msg, audio_length
                    ),
                    "Parakeet provider".to_string(),
                )
            }
            ParakeetError::LanguageNotSupported(lang, variant, supported) => {
                crate::error::STTError::Other(
                    format!(
                        "Language '{}' not supported by {} (supported: {})",
                        lang, variant, supported
                    ),
                    "Parakeet provider".to_string(),
                )
            }
            ParakeetError::ParakeetLibraryError(msg, operation, context) => {
                crate::error::STTError::ProviderError(
                    format!("{} (operation: {}, context: {})", msg, operation, context),
                    "Parakeet".to_string(),
                )
            }
            ParakeetError::IoError(e, path, operation) => {
                crate::error::STTError::IoError(e, operation, path)
            }
            ParakeetError::Other(msg, context) => {
                crate::error::STTError::Other(msg, format!("Parakeet: {}", context))
            }
        }
    }
}

/// Convert parakeet-rs errors to ParakeetError
impl From<parakeet_rs::Error> for ParakeetError {
    fn from(err: parakeet_rs::Error) -> Self {
        ParakeetError::ParakeetLibraryError(
            err.to_string(),
            "parakeet-rs operation".to_string(),
            "wrapped error from parakeet-rs library".to_string(),
        )
    }
}

/// Convert IO errors to ParakeetError
impl From<std::io::Error> for ParakeetError {
    fn from(err: std::io::Error) -> Self {
        ParakeetError::IoError(err, "unknown path".to_string(), "IO operation".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_error_helpers() {
        let err = ParakeetError::transcription_error("failed");
        match err {
            ParakeetError::TranscriptionError(msg, stage, details) => {
                assert_eq!(msg, "failed");
                assert_eq!(stage, "unknown stage");
                assert_eq!(details, "no additional details");
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let detailed =
            ParakeetError::transcription_error_detailed("bad", "decode", "buffer overflow");
        match detailed {
            ParakeetError::TranscriptionError(msg, stage, details) => {
                assert_eq!(msg, "bad");
                assert_eq!(stage, "decode");
                assert_eq!(details, "buffer overflow");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_model_init_error_helper() {
        let err = ParakeetError::model_init_error("init failed", "EOU", "/path/to/model", "cpu");
        match err {
            ParakeetError::ModelInitError(msg, variant, path, device) => {
                assert_eq!(msg, "init failed");
                assert_eq!(variant, "EOU");
                assert_eq!(path, "/path/to/model");
                assert_eq!(device, "cpu");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_audio_processing_error_helper() {
        let err =
            ParakeetError::audio_processing_error("wrong format", "16kHz mono", "48kHz stereo");
        match err {
            ParakeetError::AudioProcessingError(msg, expected, actual) => {
                assert_eq!(msg, "wrong format");
                assert_eq!(expected, "16kHz mono");
                assert_eq!(actual, "48kHz stereo");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_chunk_processing_error_helper() {
        let err = ParakeetError::chunk_processing_error("size mismatch", 1280, 2560, "EOU");
        match err {
            ParakeetError::ChunkProcessingError(msg, chunk_size, expected, model) => {
                assert_eq!(msg, "size mismatch");
                assert_eq!(chunk_size, 1280);
                assert_eq!(expected, 2560);
                assert_eq!(model, "EOU");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_parakeet_error_conversion() {
        let err = ParakeetError::config_error("bad value", "language", "xyz");
        let converted: crate::error::STTError = err.into();
        match converted {
            crate::error::STTError::InvalidConfiguration(msg, param, value) => {
                assert_eq!(msg, "bad value");
                assert_eq!(param, "language");
                assert_eq!(value, "xyz");
            }
            other => panic!("Unexpected conversion: {other:?}"),
        }
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let parakeet_err: ParakeetError = io_err.into();
        match parakeet_err {
            ParakeetError::IoError(_, path, op) => {
                assert_eq!(path, "unknown path");
                assert_eq!(op, "IO operation");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }
}
