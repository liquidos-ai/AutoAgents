//! Library backend - runs Pocket-TTS model locally

use super::conversion::samples_to_audio_data;
use super::error::{PocketTTSError, Result};
use super::models::ModelVariant;
use super::voices::PredefinedVoice;
use crate::{SpeechRequest, SpeechResponse, VoiceIdentifier};
use pocket_tts::TTSModel;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Library backend that runs Pocket-TTS locally
pub struct LibraryBackend {
    /// Loaded TTS model
    pub(crate) model: TTSModel,
    /// Temperature for generation
    #[allow(dead_code)]
    temperature: f32,
    /// LSD decode steps
    #[allow(dead_code)]
    lsd_decode_steps: usize,
    /// EOS threshold
    #[allow(dead_code)]
    eos_threshold: f32,
    /// Noise clamp
    #[allow(dead_code)]
    noise_clamp: Option<f32>,
    /// Cached voice states (in-memory)
    voice_cache: Arc<RwLock<HashMap<String, pocket_tts::ModelState>>>,
}

impl LibraryBackend {
    /// Create a new library backend
    pub fn new(
        model_variant: ModelVariant,
        temperature: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
    ) -> Result<Self> {
        // Load the model on CPU (Metal support can be added via feature flags later)
        let device = candle_core::Device::Cpu;

        let model = TTSModel::load_with_params_device(
            model_variant.hf_id(),
            temperature,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            &device,
        )?;

        Ok(Self {
            model,
            temperature,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            voice_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate speech from text with a voice identifier
    pub async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse> {
        // Resolve voice synchronously
        let voice_state = self.resolve_voice(&request.voice)?;

        // Clone data needed for blocking task
        let model = self.model.clone();
        let text = request.text.clone();
        let sample_rate = self.model.sample_rate as u32;

        // Run CPU-intensive generation in blocking thread
        let result = tokio::task::spawn_blocking(move || {
            model.generate(&text, &voice_state).map_err(|e| {
                PocketTTSError::generation_error_detailed(
                    e.to_string(),
                    "model generation",
                    format!("text length: {}", text.len()),
                )
            })
        })
        .await
        .map_err(|e| {
            PocketTTSError::generation_error_detailed(
                format!("Task join error: {}", e),
                "spawn_blocking",
                "generation task failed to join",
            )
        })??;

        // Convert tensor to audio samples
        // The result may have shape [samples] or [1, samples], so squeeze if needed
        let tensor = if result.dims().len() > 1 {
            result.squeeze(0).map_err(|e| {
                PocketTTSError::tensor_error(
                    format!("Failed to squeeze tensor: {}", e),
                    "1D or 2D",
                    format!("{:?}", result.dims()),
                )
            })?
        } else {
            result
        };

        let samples = tensor.to_vec1::<f32>().map_err(|e| {
            PocketTTSError::generation_error_detailed(
                format!("Failed to extract samples: {}", e),
                "tensor conversion",
                format!("tensor shape: {:?}", tensor.dims()),
            )
        })?;

        let audio_data = samples_to_audio_data(samples, sample_rate);

        // Calculate duration
        let duration_ms =
            (audio_data.samples.len() as f64 / audio_data.sample_rate as f64 * 1000.0) as u64;

        Ok(SpeechResponse {
            text: request.text,
            audio: audio_data,
            duration_ms,
        })
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

        // Create a channel to stream chunks
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn blocking task that sends chunks as they're generated
        tokio::task::spawn_blocking(move || {
            let stream_iter = model.generate_stream(&text, &voice_state);

            for (idx, result) in stream_iter.enumerate() {
                let response = result
                    .map_err(|e| {
                        PocketTTSError::generation_error_detailed(
                            e.to_string(),
                            "streaming generation",
                            format!("chunk index: {}", idx),
                        )
                    })
                    .and_then(|tensor| {
                        // Streaming returns tensors with shape [batch, 1, samples]
                        // We need to squeeze to get [samples]
                        let tensor = tensor.squeeze(0).and_then(|t| t.squeeze(0)).map_err(|e| {
                            PocketTTSError::tensor_error(
                                format!("Failed to squeeze tensor: {}", e),
                                "[samples]",
                                format!("{:?}", tensor.dims()),
                            )
                        })?;

                        let samples = tensor.to_vec1::<f32>().map_err(|e| {
                            PocketTTSError::generation_error_detailed(
                                format!("Failed to extract samples: {}", e),
                                "streaming tensor conversion",
                                format!("chunk {}, tensor shape: {:?}", idx, tensor.dims()),
                            )
                        })?;

                        let audio_data = samples_to_audio_data(samples, sample_rate);
                        let duration_ms = (audio_data.samples.len() as f64
                            / audio_data.sample_rate as f64
                            * 1000.0) as u64;

                        Ok(SpeechResponse {
                            text: format!("{}_{}", text, idx),
                            audio: audio_data,
                            duration_ms,
                        })
                    });

                // Send chunk immediately
                if tx.send(response).is_err() {
                    break;
                }
            }
        });

        // Convert the receiver to a stream
        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    /// Resolve a voice identifier to a ModelState
    fn resolve_voice(&self, voice_id: &VoiceIdentifier) -> Result<pocket_tts::ModelState> {
        self.load_predefined_voice(&voice_id.name)
    }

    /// Load a predefined voice
    fn load_predefined_voice(&self, name: &str) -> Result<pocket_tts::ModelState> {
        // Check cache first
        {
            let cache = self.voice_cache.read().map_err(|e| {
                PocketTTSError::cache_error(
                    format!("Cache lock poisoned: {}", e),
                    "voice cache read",
                )
            })?;
            if let Some(state) = cache.get(name) {
                return Ok(state.clone());
            }
        }

        // Parse the predefined voice
        let voice: PredefinedVoice = name.parse().map_err(|e: String| {
            PocketTTSError::voice_error_detailed(e, name.to_string(), "parsing voice name")
        })?;

        // Download and load embeddings from HuggingFace
        let hf_path = voice.hf_path();
        let local_path = pocket_tts::weights::download_if_necessary(&hf_path).map_err(|e| {
            PocketTTSError::download_error(
                format!("Failed to download voice embeddings: {}", e),
                name.to_string(),
                hf_path.clone(),
            )
        })?;

        let state = self
            .model
            .get_voice_state_from_prompt_file(&local_path)
            .map_err(|e| {
                PocketTTSError::voice_error_detailed(
                    e.to_string(),
                    name.to_string(),
                    "loading voice embeddings from file",
                )
            })?;

        // Cache it
        {
            let mut cache = self.voice_cache.write().map_err(|e| {
                PocketTTSError::cache_error(
                    format!("Cache lock poisoned: {}", e),
                    "voice cache write",
                )
            })?;
            cache.insert(name.to_string(), state.clone());
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires HuggingFace model download"]
    fn test_backend_creation() {
        let result = LibraryBackend::new(ModelVariant::default(), 0.7, 1, -4.0, None);
        assert!(result.is_ok());
    }
}
