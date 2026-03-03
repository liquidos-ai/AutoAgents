//! Parakeet STT provider implementation

use super::config::ParakeetConfig;
use super::error::Result;
use super::stt::{ParakeetBackend, validate_language};
use crate::{
    ModelInfo, STTModelsProvider, STTProvider, STTResult, STTSpeechProvider, TextChunk,
    TranscriptionRequest, TranscriptionResponse,
};
use async_trait::async_trait;
use futures::Stream;
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

    /// Reset streaming state
    pub async fn reset(&self) {
        let mut backend = self.backend.lock().await;
        backend.reset();
    }

    /// Process a single audio chunk in streaming mode.
    ///
    /// Callers must call `reset()` before starting a new streaming session and must
    /// provide chunks of exactly `config.model_variant.chunk_size()` samples.
    /// For pre-recorded audio, prefer `transcribe_stream()` which handles chunking
    /// and padding automatically.
    pub async fn process_chunk(&self, audio_chunk: Vec<f32>) -> STTResult<TextChunk> {
        let mut backend = self.backend.lock().await;
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
        validate_language(request.language.as_deref(), &self.config.model_variant)
            .map_err(crate::error::STTError::from)?;
        let mut backend = self.backend.lock().await;
        backend.transcribe(request).await.map_err(Into::into)
    }

    async fn transcribe_stream<'a>(
        &'a self,
        request: TranscriptionRequest,
    ) -> STTResult<Pin<Box<dyn Stream<Item = STTResult<TextChunk>> + Send + 'a>>> {
        if !self.config.model_variant.supports_streaming() {
            return Err(crate::error::STTError::StreamingNotSupported(format!(
                "{} does not support streaming",
                self.config.model_variant
            )));
        }

        // request.audio is already Arc<AudioData>; clone the Arc (cheap) for use in the stream.
        let audio = request.audio;
        let backend = self.backend.clone();

        // Reset state before streaming
        {
            let mut b = backend.lock().await;
            b.reset();
        }

        let chunk_size = self.config.model_variant.chunk_size();

        // Lazily produce chunks via unfold to avoid a full upfront copy of all audio.
        // The Tokio mutex is acquired and released for each chunk so that other tasks
        // are not starved between inferences.
        let stream = futures::stream::unfold(0usize, move |offset| {
            let audio = audio.clone(); // cheap Arc clone
            let backend = backend.clone();
            async move {
                let samples = &audio.samples;
                if offset >= samples.len() {
                    return None;
                }

                let end = (offset + chunk_size).min(samples.len());
                let chunk = if end - offset == chunk_size {
                    samples[offset..end].to_vec()
                } else {
                    let mut padded = Vec::with_capacity(chunk_size);
                    padded.extend_from_slice(&samples[offset..end]);
                    padded.resize(chunk_size, 0.0);
                    padded
                };

                let next_offset = offset + chunk_size;
                let mut b = backend.lock().await;
                let result = b.transcribe_chunk(chunk).await.map_err(Into::into);
                // Mutex is released here when `b` is dropped, before the caller resumes.
                Some((result, next_offset))
            }
        });

        Ok(Box::pin(stream))
    }

    fn supports_streaming(&self) -> bool {
        self.config.model_variant.supports_streaming()
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
