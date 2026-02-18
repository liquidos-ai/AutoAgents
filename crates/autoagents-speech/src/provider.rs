use crate::{
    AudioChunk, AudioFormat, ModelInfo, STTResult, SpeechRequest, SpeechResponse, TTSResult,
    TextChunk, TranscriptionRequest, TranscriptionResponse,
};
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

//
// STT Provider Traits
//

/// Marker trait for STT providers
///
/// This trait combines all STT capabilities into a single provider interface.
/// Providers should implement this marker trait along with the specific capability traits.
#[async_trait]
pub trait STTProvider: STTSpeechProvider + STTModelsProvider + Send + Sync {}

/// Trait for STT transcription capabilities
#[async_trait]
pub trait STTSpeechProvider: Send + Sync {
    /// Transcribe audio to text (required)
    ///
    /// # Arguments
    /// * `request` - Transcription request with audio and options
    ///
    /// # Returns
    /// Transcription response with text and optional timestamps
    async fn transcribe(&self, request: TranscriptionRequest) -> STTResult<TranscriptionResponse>;

    /// Transcribe audio as a stream (optional)
    ///
    /// # Arguments
    /// * `request` - Transcription request
    ///
    /// # Returns
    /// Stream of text chunks
    async fn transcribe_stream<'a>(
        &'a self,
        _request: TranscriptionRequest,
    ) -> STTResult<Pin<Box<dyn Stream<Item = STTResult<TextChunk>> + Send + 'a>>> {
        Err(crate::error::STTError::StreamingNotSupported(
            "Not Supported".to_string(),
        ))
    }

    /// Check if streaming is supported (default: false)
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Get supported sample rate (default: 16000Hz)
    fn supported_sample_rate(&self) -> u32 {
        16000
    }

    /// Get supported number of channels (default: 1 for mono)
    fn supported_channels(&self) -> u16 {
        1
    }

    /// Check if timestamps are supported (default: false)
    fn supports_timestamps(&self) -> bool {
        false
    }
}

/// Trait for STT model management capabilities
#[async_trait]
pub trait STTModelsProvider: Send + Sync {
    /// List available models (optional)
    ///
    /// # Returns
    /// List of available model information
    async fn list_models(&self) -> STTResult<Vec<ModelInfo>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AudioData, AudioFormat, ModelInfo, SpeechRequest, SpeechResponse, TranscriptionRequest,
        TranscriptionResponse, VoiceIdentifier,
    };
    use async_trait::async_trait;

    //
    // TTS Tests
    //

    #[derive(Debug)]
    struct DummyProvider;

    #[async_trait]
    impl TTSSpeechProvider for DummyProvider {
        async fn generate_speech(&self, request: SpeechRequest) -> TTSResult<SpeechResponse> {
            Ok(SpeechResponse {
                audio: AudioData {
                    samples: vec![0.0],
                    channels: 1,
                    sample_rate: request.sample_rate.unwrap_or(24000),
                },
                text: request.text,
                duration_ms: 0,
            })
        }
    }

    #[async_trait]
    impl TTSModelsProvider for DummyProvider {
        fn get_current_model(&self) -> ModelInfo {
            ModelInfo {
                id: "dummy".to_string(),
                name: "Dummy".to_string(),
                description: None,
                languages: vec!["en".to_string()],
            }
        }
    }

    impl TTSProvider for DummyProvider {}

    #[tokio::test]
    async fn test_default_streaming_not_supported() {
        let provider = DummyProvider;
        let request = SpeechRequest {
            text: "hello".to_string(),
            voice: VoiceIdentifier::new("test"),
            format: AudioFormat::Wav,
            sample_rate: None,
        };

        let err = match provider.generate_speech_stream(request).await {
            Ok(_) => panic!("expected streaming not supported"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            crate::error::TTSError::StreamingNotSupported(_)
        ));
        assert!(!provider.supports_streaming());
    }

    #[test]
    fn test_default_provider_formats_and_languages() {
        let provider = DummyProvider;
        assert_eq!(provider.supported_formats(), vec![AudioFormat::Wav]);
        assert_eq!(provider.default_sample_rate(), 24000);
        assert_eq!(provider.supported_languages(), vec!["en".to_string()]);
    }

    //
    // STT Tests
    //

    #[derive(Debug)]
    struct DummySTTProvider;

    #[async_trait]
    impl STTSpeechProvider for DummySTTProvider {
        async fn transcribe(
            &self,
            request: TranscriptionRequest,
        ) -> STTResult<TranscriptionResponse> {
            Ok(TranscriptionResponse {
                text: format!("Transcribed {} samples", request.audio.samples.len()),
                timestamps: None,
                duration_ms: 0,
            })
        }
    }

    #[async_trait]
    impl STTModelsProvider for DummySTTProvider {
        fn get_current_model(&self) -> ModelInfo {
            ModelInfo {
                id: "dummy".to_string(),
                name: "Dummy STT".to_string(),
                description: None,
                languages: vec!["en".to_string()],
            }
        }
    }

    impl STTProvider for DummySTTProvider {}

    #[tokio::test]
    async fn test_stt_default_streaming_not_supported() {
        let provider = DummySTTProvider;
        let request = TranscriptionRequest {
            audio: AudioData {
                samples: vec![0.0; 16000],
                sample_rate: 16000,
                channels: 1,
            },
            language: None,
            include_timestamps: false,
        };

        let err = match provider.transcribe_stream(request).await {
            Ok(_) => panic!("expected streaming not supported"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            crate::error::STTError::StreamingNotSupported(_)
        ));
        assert!(!provider.supports_streaming());
    }

    #[test]
    fn test_stt_default_provider_settings() {
        let provider = DummySTTProvider;
        assert_eq!(provider.supported_sample_rate(), 16000);
        assert_eq!(provider.supported_channels(), 1);
        assert_eq!(provider.supported_languages(), vec!["en".to_string()]);
        assert!(!provider.supports_timestamps());
    }
}
