//! Error types for Pocket-TTS provider

use thiserror::Error;

/// Pocket-TTS specific errors
#[derive(Error, Debug)]
pub enum PocketTTSError {
    /// Voice processing error
    #[error(
        "Voice processing failed: {0}\nVoice name: {1}\nOperation: {2}\nSuggestion: Check voice name spelling and ensure voice embeddings are downloaded"
    )]
    VoiceError(String, String, String),

    /// Audio generation error
    #[error(
        "Audio generation failed: {0}\nStage: {1}\nDetails: {2}\nSuggestion: Check input text encoding and model state"
    )]
    GenerationError(String, String, String),

    /// IO error
    #[error("IO error: {0}\nFile path: {1}\nOperation: {2}")]
    IoError(std::io::Error, String, String),

    /// Pocket-TTS library error
    #[error("Pocket-TTS library error: {0}\nOperation: {1}\nContext: {2}")]
    PocketTTSLibraryError(String, String, String),

    /// Cache error
    #[error(
        "Cache operation failed: {0}\nCache type: {1}\nSuggestion: This may indicate a concurrency issue or corrupted cache"
    )]
    CacheError(String, String),

    /// Tensor processing error
    #[error(
        "Tensor processing failed: {0}\nExpected shape: {1}\nActual shape: {2}\nSuggestion: This indicates a model output format mismatch"
    )]
    TensorError(String, String, String),

    /// Download error
    #[error(
        "Download failed: {0}\nResource: {1}\nURL: {2}\nSuggestion: Check internet connection and HuggingFace access token"
    )]
    DownloadError(String, String, String),
}

impl PocketTTSError {
    /// Create a simple generation error with context
    pub fn generation_error(msg: impl Into<String>) -> Self {
        Self::GenerationError(
            msg.into(),
            "unknown stage".to_string(),
            "no additional details".to_string(),
        )
    }

    /// Create a generation error with detailed context
    pub fn generation_error_detailed(
        msg: impl Into<String>,
        stage: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self::GenerationError(msg.into(), stage.into(), details.into())
    }

    /// Create a voice error with context
    pub fn voice_error(msg: impl Into<String>, voice_name: impl Into<String>) -> Self {
        Self::VoiceError(
            msg.into(),
            voice_name.into(),
            "voice resolution".to_string(),
        )
    }

    /// Create a voice error with detailed context
    pub fn voice_error_detailed(
        msg: impl Into<String>,
        voice_name: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self::VoiceError(msg.into(), voice_name.into(), operation.into())
    }

    /// Create a cache error
    pub fn cache_error(msg: impl Into<String>, cache_type: impl Into<String>) -> Self {
        Self::CacheError(msg.into(), cache_type.into())
    }

    /// Create a tensor error with shape information
    pub fn tensor_error(
        msg: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::TensorError(msg.into(), expected.into(), actual.into())
    }

    /// Create a download error
    pub fn download_error(
        msg: impl Into<String>,
        resource: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self::DownloadError(msg.into(), resource.into(), url.into())
    }
}

/// Result type for Pocket-TTS operations
pub type Result<T> = std::result::Result<T, PocketTTSError>;

// Conversion to parent crate's TTSError
impl From<PocketTTSError> for crate::TTSError {
    fn from(err: PocketTTSError) -> Self {
        match err {
            PocketTTSError::VoiceError(msg, voice, op) => crate::TTSError::InvalidVoiceData(
                format!("{} (voice: {}, operation: {})", msg, voice, op),
                "pocket-tts".to_string(),
            ),
            PocketTTSError::GenerationError(msg, stage, details) => {
                crate::TTSError::GenerationFailed(
                    format!("{} (stage: {}, details: {})", msg, stage, details),
                    0, // text length unknown at this point
                    "unknown".to_string(),
                )
            }
            PocketTTSError::IoError(e, path, op) => crate::TTSError::IoError(e, op, path),
            PocketTTSError::PocketTTSLibraryError(msg, op, ctx) => crate::TTSError::ProviderError(
                format!("{} (operation: {}, context: {})", msg, op, ctx),
                "pocket-tts".to_string(),
            ),
            PocketTTSError::CacheError(msg, cache_type) => crate::TTSError::Other(
                format!("Cache error: {} (type: {})", msg, cache_type),
                "pocket-tts".to_string(),
            ),
            PocketTTSError::TensorError(msg, expected, actual) => crate::TTSError::Other(
                format!(
                    "Tensor error: {} (expected: {}, actual: {})",
                    msg, expected, actual
                ),
                "pocket-tts".to_string(),
            ),
            PocketTTSError::DownloadError(msg, resource, url) => {
                crate::TTSError::ModelNotFound(format!("{} (resource: {})", msg, resource), url)
            }
        }
    }
}

// Conversion from pocket-tts errors
impl From<anyhow::Error> for PocketTTSError {
    fn from(err: anyhow::Error) -> Self {
        PocketTTSError::PocketTTSLibraryError(
            err.to_string(),
            "anyhow conversion".to_string(),
            "wrapped error from pocket-tts library".to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_error_helpers() {
        let err = PocketTTSError::generation_error("boom");
        match err {
            PocketTTSError::GenerationError(msg, stage, details) => {
                assert_eq!(msg, "boom");
                assert_eq!(stage, "unknown stage");
                assert_eq!(details, "no additional details");
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let detailed = PocketTTSError::generation_error_detailed("bad", "decode", "oom");
        match detailed {
            PocketTTSError::GenerationError(msg, stage, details) => {
                assert_eq!(msg, "bad");
                assert_eq!(stage, "decode");
                assert_eq!(details, "oom");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_voice_error_helpers() {
        let err = PocketTTSError::voice_error("invalid", "alice");
        match err {
            PocketTTSError::VoiceError(msg, voice, op) => {
                assert_eq!(msg, "invalid");
                assert_eq!(voice, "alice");
                assert_eq!(op, "voice resolution");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_pocket_tts_error_conversion() {
        let err = PocketTTSError::voice_error("bad voice", "voice_x");
        let converted: crate::TTSError = err.into();
        match converted {
            crate::TTSError::InvalidVoiceData(msg, provider) => {
                assert!(msg.contains("voice_x"));
                assert_eq!(provider, "pocket-tts");
            }
            other => panic!("Unexpected conversion: {other:?}"),
        }
    }

    #[test]
    fn test_anyhow_conversion() {
        let err = anyhow::anyhow!("wrapped");
        let pocket: PocketTTSError = err.into();
        match pocket {
            PocketTTSError::PocketTTSLibraryError(msg, op, ctx) => {
                assert!(msg.contains("wrapped"));
                assert_eq!(op, "anyhow conversion");
                assert_eq!(ctx, "wrapped error from pocket-tts library");
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }
}
