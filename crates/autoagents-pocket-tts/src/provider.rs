//! Pocket-TTS provider implementation
//!
//! Implements TTSProvider traits for both library and server backends

use crate::config::{BackendConfig, PocketTTSConfig};
use crate::error::Result;
use autoagents_tts::{
    AudioChunk, ModelInfo, SpeechRequest, SpeechResponse, TTSModelsProvider, TTSProvider, TTSSpeechProvider,
    TTSVoiceProvider, VoiceIdentifier, VoiceState, TTSResult,
};
use async_trait::async_trait;
use futures::Stream;
use std::path::Path;
use std::pin::Pin;

/// Pocket-TTS provider
pub struct PocketTTSProvider {
    config: PocketTTSConfig,
    backend: Backend,
}

/// Internal backend enum
enum Backend {
    #[cfg(feature = "library")]
    Library(crate::library::LibraryBackend),
    #[cfg(feature = "server")]
    Server(crate::server::ServerBackend),
}

impl PocketTTSProvider {
    /// Create a new Pocket-TTS provider
    pub fn new(config: PocketTTSConfig) -> Result<Self> {
        let backend = match &config.backend {
            #[cfg(feature = "library")]
            BackendConfig::Library(lib_config) => {
                let backend = crate::library::LibraryBackend::new(
                    config.model_variant,
                    lib_config.clone(),
                    config.temperature,
                    config.lsd_decode_steps,
                    config.eos_threshold,
                    config.noise_clamp,
                )?;
                Backend::Library(backend)
            }
            #[cfg(feature = "server")]
            BackendConfig::Server(server_config) => {
                let backend = crate::server::ServerBackend::new(server_config.clone())?;
                Backend::Server(backend)
            }
        };

        Ok(Self { config, backend })
    }

    /// Get the configuration
    pub fn config(&self) -> &PocketTTSConfig {
        &self.config
    }
}

// Implement the marker trait
impl TTSProvider for PocketTTSProvider {
    fn provider_name(&self) -> &str {
        "pocket-tts"
    }
    
    fn provider_version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }
}

#[async_trait]
impl TTSSpeechProvider for PocketTTSProvider {
    async fn generate_speech(&self, request: SpeechRequest) -> TTSResult<SpeechResponse> {
        match &self.backend {
            #[cfg(feature = "library")]
            Backend::Library(lib) => lib.generate(request).await.map_err(Into::into),
            #[cfg(feature = "server")]
            Backend::Server(srv) => srv.generate(request).await.map_err(Into::into),
        }
    }

    async fn generate_speech_stream(
        &self,
        request: SpeechRequest,
    ) -> TTSResult<Pin<Box<dyn Stream<Item = TTSResult<AudioChunk>> + Send>>> {
        match &self.backend {
            #[cfg(feature = "library")]
            Backend::Library(lib) => {
                let stream = lib.generate_stream(request).await.map_err(|e| autoagents_tts::TTSError::from(e))?;
                let audio_stream = futures::stream::StreamExt::map(stream, |result| {
                    result
                        .map_err(|e| autoagents_tts::TTSError::from(e))
                        .map(|response| AudioChunk {
                            samples: response.audio.samples,
                            is_final: false, // In streaming, we don't know when it's final
                        })
                });
                Ok(Box::pin(audio_stream))
            }
            #[cfg(feature = "server")]
            Backend::Server(srv) => {
                let stream = srv.generate_stream(request).await.map_err(|e| autoagents_tts::TTSError::from(e))?;
                let audio_stream = futures::stream::StreamExt::map(stream, |result| {
                    result
                        .map_err(|e| autoagents_tts::TTSError::from(e))
                        .map(|response| AudioChunk {
                            samples: response.audio.samples,
                            is_final: false,
                        })
                });
                Ok(Box::pin(audio_stream))
            }
        }
    }
}

#[async_trait]
impl TTSVoiceProvider for PocketTTSProvider {
    async fn create_voice_from_file(
        &self,
        audio_path: &Path,
    ) -> TTSResult<VoiceState> {
        // Read audio file
        let audio_bytes = tokio::fs::read(audio_path)
            .await
            .map_err(|e| autoagents_tts::TTSError::IoError(e))?;

        self.create_voice_from_bytes(&audio_bytes).await
    }

    async fn create_voice_from_bytes(
        &self,
        audio_bytes: &[u8],
    ) -> TTSResult<VoiceState> {
        // Generate a unique ID for this voice
        let voice_id = format!("voice_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());
        
        let voice_data = match &self.backend {
            #[cfg(feature = "library")]
            Backend::Library(lib) => lib
                .create_voice(voice_id.clone(), audio_bytes.to_vec())
                .await
                .map_err(|e: crate::error::PocketTTSError| autoagents_tts::TTSError::from(e))?,
            #[cfg(feature = "server")]
            Backend::Server(srv) => srv
                .create_voice(voice_id.clone(), audio_bytes.to_vec())
                .await
                .map_err(|e: crate::error::PocketTTSError| autoagents_tts::TTSError::from(e))?,
        };

        Ok(VoiceState {
            id: voice_id.clone(),
            name: Some(voice_id),
            data: voice_data,
        })
    }

    async fn load_voice_state(&self, path: &Path) -> TTSResult<VoiceState> {
        // For library backend, load from safetensors file
        match &self.backend {
            #[cfg(feature = "library")]
            Backend::Library(lib) => {
                // Use pocket-tts to load the voice state
                let voice_state = lib.model
                    .get_voice_state_from_prompt_file(path)
                    .map_err(|e| autoagents_tts::TTSError::InvalidVoiceData(e.to_string()))?;
                
                let voice_data = crate::conversion::model_state_to_voice_state_data(&voice_state)
                    .map_err(|e| autoagents_tts::TTSError::from(e))?;
                
                let voice_id = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                Ok(VoiceState {
                    id: voice_id.clone(),
                    name: Some(voice_id),
                    data: voice_data,
                })
            }
            #[cfg(feature = "server")]
            Backend::Server(_) => {
                Err(autoagents_tts::TTSError::Other(
                    "Server backend does not support loading voice state from disk".to_string(),
                ))
            }
        }
    }

    fn get_predefined_voice(&self, name: &str) -> TTSResult<VoiceIdentifier> {
        // Validate that it's a known predefined voice
        use crate::voices::PredefinedVoice;
        let _voice: PredefinedVoice = name
            .parse()
            .map_err(|e: String| autoagents_tts::TTSError::VoiceNotFound(e))?;
        
        Ok(VoiceIdentifier::Predefined(name.to_string()))
    }

    fn list_predefined_voices(&self) -> Vec<String> {
        use crate::voices::PredefinedVoice;
        PredefinedVoice::all()
            .iter()
            .map(|v| v.identifier().to_string())
            .collect()
    }

    fn default_voice(&self) -> String {
        self.config.default_voice
            .as_ref()
            .map(|v| v.identifier().to_string())
            .unwrap_or_else(|| "alba".to_string())
    }
}

#[async_trait]
impl TTSModelsProvider for PocketTTSProvider {
    async fn list_models(&self) -> TTSResult<Vec<ModelInfo>> {
        // For now, we only support one model variant
        Ok(vec![self.get_current_model()])
    }

    fn get_current_model(&self) -> ModelInfo {
        let variant = &self.config.model_variant;
        ModelInfo {
            id: variant.to_string(),
            name: variant.to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
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
        let config = PocketTTSConfig::default();
        let result = PocketTTSProvider::new(config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires HuggingFace model download"]
    async fn test_list_models() {
        let config = PocketTTSConfig::default();
        let provider = PocketTTSProvider::new(config).unwrap();
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
