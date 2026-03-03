//! Error types for Parakeet STT provider

use thiserror::Error;

/// Parakeet-specific errors
#[derive(Error, Debug)]
pub enum ParakeetError {
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
}

impl ParakeetError {
    /// Create a transcription error with detailed context
    pub fn transcription_error_detailed(
        msg: impl Into<String>,
        stage: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self::TranscriptionError(msg.into(), stage.into(), details.into())
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

    /// Create a language not supported error
    pub fn language_not_supported(
        lang: impl Into<String>,
        variant: impl Into<String>,
        supported: impl Into<String>,
    ) -> Self {
        Self::LanguageNotSupported(lang.into(), variant.into(), supported.into())
    }
}

/// Result type for Parakeet operations
pub type Result<T> = std::result::Result<T, ParakeetError>;

/// Convert Parakeet errors to STT errors
impl From<ParakeetError> for crate::error::STTError {
    fn from(err: ParakeetError) -> Self {
        match err {
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
            ParakeetError::InvalidAudioFormat(msg, exp_rate, exp_ch, act_rate, act_ch) => {
                crate::error::STTError::InvalidAudioFormat(msg, exp_rate, act_rate, exp_ch, act_ch)
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
    fn test_transcription_error_detailed_helper() {
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
