//! Library backend - runs Pocket-TTS model locally

#[cfg(feature = "library")]
use crate::config::LibraryConfig;
use crate::conversion::{create_speech_response, model_state_to_voice_state_data, voice_state_data_to_model_state};
use crate::error::{PocketTTSError, Result};
use crate::models::ModelVariant;
use crate::voices::PredefinedVoice;
use autoagents_tts::{SpeechRequest, SpeechResponse, VoiceIdentifier, VoiceStateData};
use pocket_tts::TTSModel;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Library backend that runs Pocket-TTS locally
#[cfg(feature = "library")]
pub struct LibraryBackend {
    /// Loaded TTS model
    pub(crate) model: TTSModel,
    /// Configuration
    config: LibraryConfig,
    /// Temperature for generation (reserved for future use)
    #[allow(dead_code)]
    temperature: f32,
    /// LSD decode steps (reserved for future use)
    #[allow(dead_code)]
    lsd_decode_steps: usize,
    /// EOS threshold (reserved for future use)
    #[allow(dead_code)]
    eos_threshold: f32,
    /// Noise clamp (reserved for future use)
    #[allow(dead_code)]
    noise_clamp: Option<f32>,
    /// Cached voice states (in-memory)
    voice_cache: Arc<RwLock<HashMap<String, pocket_tts::ModelState>>>,
}

#[cfg(feature = "library")]
impl LibraryBackend {
    /// Create a new library backend
    pub fn new(
        model_variant: ModelVariant,
        config: LibraryConfig,
        temperature: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
    ) -> Result<Self> {
        // Load the model
        let device = if config.use_metal {
            #[cfg(feature = "metal")]
            {
                candle_core::Device::Metal(candle_core::MetalDevice::new(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(PocketTTSError::InvalidConfig(
                    "Metal support not compiled in. Rebuild with --features metal".to_string(),
                ));
            }
        } else {
            candle_core::Device::Cpu
        };

        let model = TTSModel::load_with_params_device(
            model_variant.hf_id(),
            temperature,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            &device,
        )?;

        // Ensure cache directory exists
        std::fs::create_dir_all(&config.cache_dir)
            .map_err(|e| PocketTTSError::IoError(e))?;

        Ok(Self {
            model,
            config,
            temperature,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            voice_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate speech from text with a voice identifier
    pub async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse> {
        // Resolve voice synchronously (blocking I/O doesn't benefit from async)
        let voice_state = self.resolve_voice(&request.voice)?;
        
        // Clone data needed for blocking task
        let model = self.model.clone();
        let text = request.text.clone();
        let sample_rate = self.model.sample_rate as u32;
        
        // Run CPU-intensive generation in blocking thread to avoid blocking async runtime
        let audio_tensor = tokio::task::spawn_blocking(move || {
            model
                .generate(&text, &voice_state)
                .map_err(|e| PocketTTSError::GenerationError(e.to_string()))
        })
        .await
        .map_err(|e| PocketTTSError::GenerationError(format!("Task join error: {}", e)))??;

        // Convert to response (fast, stays on async thread)
        create_speech_response(
            request.text,
            &audio_tensor,
            sample_rate,
        )
    }

    /// Generate streaming audio chunks
    pub async fn generate_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<impl futures::Stream<Item = Result<SpeechResponse>> + Send> {
        let voice_state = self.resolve_voice(&request.voice)?;
        let sample_rate = self.model.sample_rate as u32;
        let text = request.text.clone();
        
        // Clone the model for use in the blocking task
        let model = self.model.clone();
        
        // Create a channel to stream chunks from blocking task to async stream
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Spawn blocking task that sends chunks as they're generated
        tokio::task::spawn_blocking(move || {
            let stream_iter = model.generate_stream(&text, &voice_state);
            
            for (idx, result) in stream_iter.enumerate() {
                let response = result
                    .map_err(|e| PocketTTSError::GenerationError(e.to_string()))
                    .and_then(|tensor| {
                        create_speech_response(
                            format!("{}_{}", text, idx),
                            &tensor,
                            sample_rate,
                        )
                    });
                
                // Send chunk immediately - if receiver is dropped, stop generating
                if tx.send(response).is_err() {
                    break;
                }
            }
            // Channel is automatically closed when tx is dropped
        });
        
        // Convert the receiver to a stream
        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    /// Resolve a voice identifier to a ModelState
    fn resolve_voice(&self, voice_id: &VoiceIdentifier) -> Result<pocket_tts::ModelState> {
        match voice_id {
            VoiceIdentifier::Predefined(name) => {
                self.load_predefined_voice(name)
            }
            VoiceIdentifier::Custom(voice_state) => {
                // Extract the ModelState from VoiceState
                voice_state_data_to_model_state(&voice_state.data)
            }
            VoiceIdentifier::File(path) => {
                self.load_voice_from_file(path)
            }
            VoiceIdentifier::Bytes(bytes) => {
                self.load_voice_from_bytes(bytes)
            }
        }
    }

    /// Load a predefined voice
    fn load_predefined_voice(&self, name: &str) -> Result<pocket_tts::ModelState> {
        // Check cache first
        {
            let cache = self.voice_cache.read()
                .map_err(|e| PocketTTSError::VoiceLoadError(format!("Cache lock poisoned: {}", e)))?;
            if let Some(state) = cache.get(name) {
                return Ok(state.clone());
            }
        }

        // Parse the predefined voice
        let voice: PredefinedVoice = name
            .parse()
            .map_err(|e: String| PocketTTSError::VoiceNotFound(e))?;

        // Download and load embeddings from HuggingFace
        let hf_path = voice.hf_path();
        let local_path = pocket_tts::weights::download_if_necessary(&hf_path)
            .map_err(|e| PocketTTSError::VoiceLoadError(format!("Failed to download {}: {}", name, e)))?;

        let state = self
            .model
            .get_voice_state_from_prompt_file(&local_path)
            .map_err(|e| PocketTTSError::VoiceLoadError(e.to_string()))?;

        // Cache it
        {
            let mut cache = self.voice_cache.write()
                .map_err(|e| PocketTTSError::VoiceLoadError(format!("Cache lock poisoned: {}", e)))?;
            cache.insert(name.to_string(), state.clone());
        }

        Ok(state)
    }

    /// Load a custom voice from cache (reserved for future use)
    #[allow(dead_code)]
    async fn load_custom_voice(&self, name: &str) -> Result<pocket_tts::ModelState> {
        // Check in-memory cache first
        {
            let cache = self.voice_cache.read()
                .map_err(|e| PocketTTSError::VoiceLoadError(format!("Cache lock poisoned: {}", e)))?;
            if let Some(state) = cache.get(name) {
                return Ok(state.clone());
            }
        }

        // Try to load from disk cache
        let cache_path = self.get_cache_path(name);
        if cache_path.exists() {
            let state = self
                .model
                .get_voice_state_from_prompt_file(&cache_path)
                .map_err(|e| PocketTTSError::VoiceLoadError(e.to_string()))?;

            // Cache in memory
            {
                let mut cache = self.voice_cache.write()
                    .map_err(|e| PocketTTSError::VoiceLoadError(format!("Cache lock poisoned: {}", e)))?;
                cache.insert(name.to_string(), state.clone());
            }

            return Ok(state);
        }

        Err(PocketTTSError::VoiceNotFound(format!(
            "Custom voice '{}' not found in cache",
            name
        )))
    }

    /// Load voice from file path
    fn load_voice_from_file(&self, path: &Path) -> Result<pocket_tts::ModelState> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "safetensors" => {
                // Pre-computed embeddings
                self.model
                    .get_voice_state_from_prompt_file(path)
                    .map_err(|e| PocketTTSError::VoiceLoadError(e.to_string()))
            }
            "wav" | "wave" => {
                // Raw audio - encode through Mimi
                self.model
                    .get_voice_state(path)
                    .map_err(|e| PocketTTSError::VoiceLoadError(e.to_string()))
            }
            _ => Err(PocketTTSError::VoiceLoadError(format!(
                "Unsupported file extension '{}'. Expected .wav or .safetensors",
                ext
            ))),
        }
    }

    /// Load voice from audio bytes
    fn load_voice_from_bytes(&self, bytes: &[u8]) -> Result<pocket_tts::ModelState> {
        self.model
            .get_voice_state_from_bytes(bytes)
            .map_err(|e| PocketTTSError::VoiceLoadError(e.to_string()))
    }

    /// Create a voice from audio bytes and save it
    pub async fn create_voice(
        &self,
        name: String,
        audio_bytes: Vec<u8>,
    ) -> Result<VoiceStateData> {
        // Generate voice state from audio
        let voice_state = self.load_voice_from_bytes(&audio_bytes)?;

        // Save to disk cache
        let cache_path = self.get_cache_path(&name);
        self.save_voice_state(&cache_path, &voice_state)?;

        // Cache in memory
        {
            let mut cache = self.voice_cache.write()
                .map_err(|e| PocketTTSError::VoiceLoadError(format!("Cache lock poisoned: {}", e)))?;
            cache.insert(name, voice_state.clone());
        }

        // Convert to VoiceStateData
        model_state_to_voice_state_data(&voice_state)
    }

    /// Save voice state to a file
    fn save_voice_state(
        &self,
        path: &Path,
        state: &pocket_tts::ModelState,
    ) -> Result<()> {
        // Flatten ModelState for safetensors
        let mut flat_tensors = std::collections::HashMap::new();
        for (module_name, module_state) in state.iter() {
            for (key, tensor) in module_state.iter() {
                let flat_key = format!("{}_{}", module_name, key);
                flat_tensors.insert(flat_key, tensor.clone());
            }
        }

        // Save to safetensors file
        candle_core::safetensors::save(&flat_tensors, path)
            .map_err(|e| PocketTTSError::VoiceLoadError(format!("Failed to save voice state: {}", e)))?;

        Ok(())
    }

    /// Load voice state from VoiceStateData (reserved for future use)
    #[allow(dead_code)]
    pub async fn load_voice_state(&self, data: &VoiceStateData) -> Result<()> {
        let _state = voice_state_data_to_model_state(data)?;
        
        // For now, we don't have a name for this voice, so we don't cache it
        // This is mainly for restoring voices from history
        // TODO: Consider adding a name parameter
        
        Ok(())
    }

    /// List available predefined voices (reserved for future use)
    #[allow(dead_code)]
    pub fn list_predefined_voices(&self) -> Vec<String> {
        PredefinedVoice::all()
            .iter()
            .map(|v| v.identifier().to_string())
            .collect()
    }

    /// List cached custom voices (reserved for future use)
    #[allow(dead_code)]
    pub fn list_custom_voices(&self) -> Result<Vec<String>> {
        let mut voices = Vec::new();
        
        if let Ok(entries) = std::fs::read_dir(&self.config.cache_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.path().file_stem() {
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors") {
                        voices.push(name.to_string_lossy().to_string());
                    }
                }
            }
        }
        
        Ok(voices)
    }

    /// Get cache path for a voice
    fn get_cache_path(&self, name: &str) -> PathBuf {
        self.config.cache_dir.join(format!("{}.safetensors", name))
    }
}

#[cfg(test)]
#[cfg(feature = "library")]
mod tests {
    use super::*;
    use crate::config::LibraryConfig;
    use tempfile::TempDir;

    fn create_test_config() -> (LibraryConfig, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = LibraryConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            use_metal: false,
        };
        (config, temp_dir)
    }

    #[test]
    #[ignore = "requires HuggingFace model download"]
    fn test_cache_path_generation() {
        let (config, _temp) = create_test_config();
        let backend = LibraryBackend::new(
            ModelVariant::default(),
            config,
            0.7,
            1,
            -4.0,
            None,
        )
        .unwrap();
        
        let path = backend.get_cache_path("test_voice");
        assert!(path.to_string_lossy().contains("test_voice.safetensors"));
    }

    #[test]
    #[ignore = "requires HuggingFace model download"]
    fn test_list_predefined_voices() {
        let (config, _temp) = create_test_config();
        let backend = LibraryBackend::new(
            ModelVariant::default(),
            config,
            0.7,
            1,
            -4.0,
            None,
        )
        .unwrap();
        
        let voices = backend.list_predefined_voices();
        assert!(voices.contains(&"alba".to_string()));
        assert!(voices.contains(&"marius".to_string()));
        assert_eq!(voices.len(), 8);
    }
}
