use crate::error::TTSResult;
use crate::types::{AudioChunk, AudioFormat, SpeechRequest, SpeechResponse};
use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;

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
    async fn generate_speech_stream(
        &self,
        request: SpeechRequest,
    ) -> TTSResult<Pin<Box<dyn Stream<Item = TTSResult<AudioChunk>> + Send>>> {
        // Default implementation: not supported
        let _ = request;
        Err(crate::error::TTSError::StreamingNotSupported)
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
