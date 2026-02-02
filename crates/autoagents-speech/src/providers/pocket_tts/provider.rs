//! Pocket-TTS provider implementation
//!
//! Implements TTSProvider traits for library backend

use super::config::PocketTTSConfig;
use super::error::Result;
use super::tts::PocketTTSBackend;
use crate::{
    AudioChunk, ModelInfo, SpeechRequest, SpeechResponse, TTSError, TTSModelsProvider, TTSProvider,
    TTSResult, TTSSpeechProvider,
};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// Pocket-TTS provider
pub struct PocketTTS {
    config: PocketTTSConfig,
    backend: PocketTTSBackend,
}

impl PocketTTS {
    /// Create a new Pocket-TTS provider
    pub fn new(config: Option<PocketTTSConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let backend = PocketTTSBackend::new(
            config.model_variant,
            config.temperature,
            config.lsd_decode_steps,
            config.eos_threshold,
            config.noise_clamp,
        )?;

        Ok(Self { config, backend })
    }

    /// Get the configuration
    pub fn config(&self) -> &PocketTTSConfig {
        &self.config
    }

    /// List all available predefined voices
    pub fn list_predefined_voices(&self) -> Vec<String> {
        use super::voices::PredefinedVoice;
        PredefinedVoice::all()
            .iter()
            .map(|v| v.identifier().to_string())
            .collect()
    }

    /// Get default voice name
    pub fn default_voice(&self) -> String {
        self.config
            .default_voice
            .as_ref()
            .map(|v| v.identifier().to_string())
            .unwrap_or_else(|| "alba".to_string())
    }
}

// Implement the marker trait
impl TTSProvider for PocketTTS {}

#[async_trait]
impl TTSSpeechProvider for PocketTTS {
    async fn generate_speech(&self, request: SpeechRequest) -> TTSResult<SpeechResponse> {
        self.backend.generate(request).await.map_err(TTSError::from)
    }

    async fn generate_speech_stream<'a>(
        &'a self,
        request: SpeechRequest,
    ) -> TTSResult<Pin<Box<dyn Stream<Item = TTSResult<AudioChunk>> + Send + 'a>>> {
        let stream = self
            .backend
            .generate_stream(request)
            .await
            .map_err(TTSError::from)?;
        let audio_stream = futures::stream::StreamExt::map(stream, |result| {
            result.map_err(TTSError::from).map(|response| AudioChunk {
                samples: response.audio.samples,
                is_final: false, // In streaming, we don't know when it's final, TODO: Can we improve this
            })
        });
        Ok(Box::pin(audio_stream))
    }
}

#[async_trait]
impl TTSModelsProvider for PocketTTS {
    async fn list_models(&self) -> TTSResult<Vec<ModelInfo>> {
        // For now, we only support one model variant
        Ok(vec![self.get_current_model()])
    }

    fn get_current_model(&self) -> ModelInfo {
        let variant = &self.config.model_variant;
        ModelInfo {
            id: variant.to_string(),
            name: variant.to_string(),
            description: Some(variant.description().to_string()),
            languages: vec!["en".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires HuggingFace model download"]
    fn test_provider_creation() {
        let result = PocketTTS::new(None);
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires HuggingFace model download"]
    async fn test_list_models() {
        let provider = PocketTTS::new(None).unwrap();
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
