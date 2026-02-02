use crate::{AudioChunk, AudioFormat, ModelInfo, SpeechRequest, SpeechResponse, TTSResult};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// Marker Trait for TTS providers
///
/// This trait combines all TTS capabilities into a single provider interface.
/// Providers should implement this marker trait along with the specific capability traits.
#[async_trait]
pub trait TTSProvider: TTSSpeechProvider + TTSModelsProvider + Send + Sync {}

/// Trait for TTS speech generation capabilities
#[async_trait]
pub trait TTSSpeechProvider: Send + Sync {
    /// Generate speech from text (required)
    ///
    /// # Arguments
    /// * `request` - Speech generation request with text, voice, and format
    ///
    /// # Returns
    /// Speech response with audio data and metadata
    async fn generate_speech(&self, request: SpeechRequest) -> TTSResult<SpeechResponse>;

    /// Generate speech as a stream (optional)
    ///
    /// # Arguments
    /// * `request` - Speech generation request
    ///
    /// # Returns
    /// Stream of audio chunks
    async fn generate_speech_stream<'a>(
        &'a self,
        _request: SpeechRequest,
    ) -> TTSResult<Pin<Box<dyn Stream<Item = TTSResult<AudioChunk>> + Send + 'a>>> {
        // Default implementation: not supported
        Err(crate::error::TTSError::StreamingNotSupported(
            "Not Supported".to_string(),
        ))
    }

    /// Check if streaming is supported (default: false)
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Get supported audio formats (default: WAV only)
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![AudioFormat::Wav]
    }

    /// Get default sample rate
    fn default_sample_rate(&self) -> u32 {
        24000
    }
}

/// Trait for TTS model management capabilities
#[async_trait]
pub trait TTSModelsProvider: Send + Sync {
    /// List available models (optional)
    ///
    /// # Returns
    /// List of available model information
    async fn list_models(&self) -> TTSResult<Vec<ModelInfo>> {
        Ok(vec![])
    }

    /// Get current model information (required)
    ///
    /// # Returns
    /// Current model information
    fn get_current_model(&self) -> ModelInfo;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<String> {
        vec!["en".to_string()]
    }
}
