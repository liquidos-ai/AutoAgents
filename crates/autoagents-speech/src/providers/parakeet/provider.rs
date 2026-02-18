//! Parakeet STT provider implementation

use super::config::ParakeetConfig;
use super::error::Result;
use super::stt::ParakeetBackend;
use crate::{
    ModelInfo, STTModelsProvider, STTProvider, STTResult, STTSpeechProvider, TextChunk,
    TranscriptionRequest, TranscriptionResponse,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Parakeet STT provider
pub struct Parakeet {
    config: ParakeetConfig,
    backend: Arc<Mutex<ParakeetBackend>>,
}

impl Parakeet {
    /// Create a new Parakeet STT provider
    pub fn new(config: ParakeetConfig) -> Result<Self> {
        let backend = ParakeetBackend::new(&config)?;
        Ok(Self {
            config,
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &ParakeetConfig {
        &self.config
    }

    /// Reset streaming state (for Nemotron and EOU)
    pub async fn reset(&self) {
        let backend = self.backend.lock().await;
        backend.reset();
    }

    /// Process audio chunk in streaming mode (Nemotron and EOU)
    pub async fn process_chunk(&self, audio_chunk: Vec<f32>) -> STTResult<TextChunk> {
        let backend = self.backend.lock().await;
        backend
            .transcribe_chunk(audio_chunk)
            .await
            .map_err(Into::into)
    }
}

// Implement the marker trait
impl STTProvider for Parakeet {}

#[async_trait]
impl STTSpeechProvider for Parakeet {
    async fn transcribe(&self, request: TranscriptionRequest) -> STTResult<TranscriptionResponse> {
        let backend = self.backend.lock().await;
        backend.transcribe(request).await.map_err(Into::into)
    }

    async fn transcribe_stream<'a>(
        &'a self,
        request: TranscriptionRequest,
    ) -> STTResult<Pin<Box<dyn Stream<Item = STTResult<TextChunk>> + Send + 'a>>> {
        // Only Nemotron supports streaming
        if !self.config.model_variant.supports_streaming() {
            return Err(crate::error::STTError::StreamingNotSupported(format!(
                "{} does not support streaming",
                self.config.model_variant
            )));
        }

        let audio = request.audio;
        let backend = self.backend.clone();

        // Reset state before streaming
        {
            let b = backend.lock().await;
            b.reset();
        }

        // Create stream that processes chunks
        // For Nemotron, recommended chunk size is 8960 samples (560ms at 16kHz)
        // For EOU, recommended chunk size is 2560 samples (160ms at 16kHz)
        let chunk_size = match self.config.model_variant {
            super::model::ModelVariant::EOU => 2560, // 160ms
            _ => 8960,                               // 560ms for Nemotron
        };

        let chunks: Vec<Vec<f32>> = audio
            .samples
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let stream = futures::stream::iter(chunks).then(move |chunk| {
            let backend = backend.clone();
            async move {
                let b = backend.lock().await;
                b.transcribe_chunk(chunk).await.map_err(Into::into)
            }
        });

        Ok(Box::pin(stream))
    }

    fn supports_streaming(&self) -> bool {
        self.config.model_variant.supports_streaming()
    }

    fn supported_sample_rate(&self) -> u32 {
        16000
    }

    fn supported_channels(&self) -> u16 {
        1
    }

    fn supports_timestamps(&self) -> bool {
        self.config.model_variant.supports_timestamps()
    }
}

#[async_trait]
impl STTModelsProvider for Parakeet {
    async fn list_models(&self) -> STTResult<Vec<ModelInfo>> {
        Ok(vec![self.get_current_model()])
    }

    fn get_current_model(&self) -> ModelInfo {
        let variant = &self.config.model_variant;
        ModelInfo {
            id: variant.id().to_string(),
            name: variant.to_string(),
            description: Some(variant.description().to_string()),
            languages: variant.supported_languages(),
        }
    }

    fn supported_languages(&self) -> Vec<String> {
        self.config.model_variant.supported_languages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::parakeet::ModelVariant;

    #[test]
    fn test_provider_settings() {
        let config = ParakeetConfig::new(ModelVariant::TDT, "./models/tdt");
        let provider = Parakeet::new(config);

        // Note: This will fail without actual model files, so we skip if error
        if provider.is_err() {
            return;
        }

        let provider = provider.unwrap();
        assert_eq!(provider.supported_sample_rate(), 16000);
        assert_eq!(provider.supported_channels(), 1);
        assert!(provider.supports_timestamps()); // TDT supports timestamps
        assert!(!provider.supports_streaming()); // TDT doesn't support streaming
    }

    #[test]
    fn test_nemotron_streaming_support() {
        let config = ParakeetConfig::new(ModelVariant::Nemotron, "./models/nemotron");
        let provider = Parakeet::new(config);

        if provider.is_err() {
            return;
        }

        let provider = provider.unwrap();
        assert!(provider.supports_streaming()); // Nemotron supports streaming
        assert!(!provider.supports_timestamps()); // Nemotron doesn't support timestamps
    }

    #[test]
    fn test_eou_streaming_and_detection_support() {
        let config = ParakeetConfig::new(ModelVariant::EOU, "./models/eou");
        let provider = Parakeet::new(config);

        if provider.is_err() {
            return;
        }

        let provider = provider.unwrap();
        assert!(provider.supports_streaming()); // EOU supports streaming
        assert!(!provider.supports_timestamps()); // EOU doesn't support timestamps

        // Verify EOU-specific properties from model variant
        let model_variant = &provider.config.model_variant;
        assert!(model_variant.supports_eou_detection()); // EOU has end-of-utterance detection
        assert_eq!(model_variant.supported_languages(), vec!["en"]); // English only
    }
}
