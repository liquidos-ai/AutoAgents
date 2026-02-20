//! Parakeet STT backend implementation

#![allow(clippy::upper_case_acronyms)]

use super::config::ParakeetConfig;
use super::error::{ParakeetError, Result};
use super::model::ModelVariant;
use crate::{TextChunk, TokenTimestamp, TranscriptionRequest, TranscriptionResponse};
use parakeet_rs::{
    ExecutionConfig, ExecutionProvider, Nemotron, ParakeetEOU, ParakeetTDT, TimedToken,
    TimestampMode, Transcriber,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Backend implementation for Parakeet STT
pub enum ParakeetBackend {
    TDT(Arc<Mutex<ParakeetTDT>>),
    Nemotron(Arc<Mutex<Nemotron>>),
    EOU(Arc<Mutex<ParakeetEOU>>),
}

impl ParakeetBackend {
    /// Create a new backend from configuration
    pub fn new(config: &ParakeetConfig) -> Result<Self> {
        let exec_config = Self::build_execution_config(config.execution_provider.as_deref());

        match config.model_variant {
            ModelVariant::TDT => {
                let model = ParakeetTDT::from_pretrained(&config.model_path, Some(exec_config))
                    .map_err(|e| ParakeetError::ModelLoadError {
                        path: config.model_path.clone(),
                        reason: e.to_string(),
                        variant: "TDT".to_string(),
                    })?;
                Ok(ParakeetBackend::TDT(Arc::new(Mutex::new(model))))
            }
            ModelVariant::Nemotron => {
                let model = Nemotron::from_pretrained(&config.model_path, Some(exec_config))
                    .map_err(|e| ParakeetError::ModelLoadError {
                        path: config.model_path.clone(),
                        reason: e.to_string(),
                        variant: "Nemotron".to_string(),
                    })?;
                Ok(ParakeetBackend::Nemotron(Arc::new(Mutex::new(model))))
            }
            ModelVariant::EOU => {
                let model = ParakeetEOU::from_pretrained(&config.model_path, Some(exec_config))
                    .map_err(|e| ParakeetError::ModelLoadError {
                        path: config.model_path.clone(),
                        reason: e.to_string(),
                        variant: "EOU".to_string(),
                    })?;
                Ok(ParakeetBackend::EOU(Arc::new(Mutex::new(model))))
            }
        }
    }

    /// Build execution config from provider string
    fn build_execution_config(provider: Option<&str>) -> ExecutionConfig {
        let mut config = ExecutionConfig::new();

        if let Some(provider_str) = provider {
            let exec_provider = match provider_str.to_lowercase().as_str() {
                #[cfg(feature = "cuda")]
                "cuda" => ExecutionProvider::Cuda,
                #[cfg(feature = "tensorrt")]
                "tensorrt" => ExecutionProvider::TensorRT,
                #[cfg(feature = "directml")]
                "directml" => ExecutionProvider::DirectML,
                #[cfg(feature = "coreml")]
                "coreml" => ExecutionProvider::CoreML,
                _ => ExecutionProvider::Cpu,
            };
            config = config.with_execution_provider(exec_provider);
        }

        config
    }

    /// Transcribe audio to text
    pub async fn transcribe(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        let start = Instant::now();
        let audio = request.audio;

        // Validate audio format
        if audio.sample_rate != 16000 {
            return Err(ParakeetError::invalid_audio_format(
                "Sample rate mismatch",
                16000,
                1,
                audio.sample_rate,
                audio.channels as u16,
            ));
        }

        if audio.channels != 1 {
            return Err(ParakeetError::invalid_audio_format(
                "Channel count mismatch",
                16000,
                1,
                audio.sample_rate,
                audio.channels as u16,
            ));
        }

        let result = match self {
            ParakeetBackend::TDT(model) => {
                let model = model.clone();
                let timestamp_mode = if request.include_timestamps {
                    Some(TimestampMode::Words)
                } else {
                    None
                };

                let parakeet_result = tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().unwrap();
                    model.transcribe_samples(
                        audio.samples,
                        audio.sample_rate,
                        audio.channels as u16,
                        timestamp_mode,
                    )
                })
                .await
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "spawn_blocking",
                    "async task execution"
                ))?
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "TDT transcription",
                    "model processing"
                ))?;

                InternalTranscriptionResult::from(parakeet_result)
            }
            ParakeetBackend::Nemotron(model) => {
                let model = model.clone();
                let text = tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().unwrap();
                    model.reset();
                    model.transcribe_audio(&audio.samples)
                })
                .await
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "spawn_blocking",
                    "async task execution"
                ))?
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "Nemotron transcription",
                    "streaming model processing"
                ))?;

                InternalTranscriptionResult::Text(text)
            }
            ParakeetBackend::EOU(model) => {
                let model = model.clone();
                // For EOU, process entire audio in chunks
                const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz

                let text = tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().unwrap();
                    let mut result_text = String::new();

                    for chunk in audio.samples.chunks(CHUNK_SIZE) {
                        match model.transcribe(chunk, false) {
                            Ok(text) => {
                                if !text.is_empty() {
                                    result_text.push_str(&text);
                                }
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    Ok(result_text)
                })
                .await
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "spawn_blocking",
                    "async task execution"
                ))?
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "EOU transcription",
                    "chunk processing"
                ))?;

                InternalTranscriptionResult::Text(text)
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        let response = match result {
            InternalTranscriptionResult::Text(text) => TranscriptionResponse {
                text,
                timestamps: None,
                duration_ms,
            },
            InternalTranscriptionResult::WithTimestamps { text, tokens } => {
                let timestamps = tokens
                    .into_iter()
                    .map(|t| TokenTimestamp {
                        text: t.text,
                        start: t.start,
                        end: t.end,
                    })
                    .collect();

                TranscriptionResponse {
                    text,
                    timestamps: Some(timestamps),
                    duration_ms,
                }
            }
        };

        Ok(response)
    }

    /// Transcribe audio chunk (streaming mode)
    pub async fn transcribe_chunk(&self, audio: Vec<f32>) -> Result<TextChunk> {
        let audio_len = audio.len(); // Capture length before moving
        
        match self {
            ParakeetBackend::Nemotron(model) => {
                let model = model.clone();
                let text = tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().unwrap();
                    model.transcribe_chunk(&audio)
                })
                .await
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "spawn_blocking",
                    "async chunk processing"
                ))?
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "Nemotron chunk transcription",
                    format!("{} samples", audio_len)
                ))?;

                Ok(TextChunk {
                    text,
                    is_final: false,
                })
            }
            ParakeetBackend::EOU(model) => {
                let model = model.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().unwrap();
                    model.transcribe(&audio, false) // reset_on_eou = false (matches parakeet-rs examples)
                })
                .await
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "spawn_blocking",
                    "async chunk processing"
                ))?
                .map_err(|e| ParakeetError::transcription_error_detailed(
                    e.to_string(),
                    "EOU chunk transcription",
                    format!("{} samples", audio_len)
                ))?;

                // Check if EOU was detected
                let is_final = result.ends_with(" [EOU]");
                let text = if is_final {
                    result.trim_end_matches(" [EOU]").to_string()
                } else {
                    result
                };

                Ok(TextChunk { text, is_final })
            }
            ParakeetBackend::TDT(_) => Err(ParakeetError::streaming_error(
                "TDT model does not support streaming",
                "TDT",
                "chunk processing",
                "Use Nemotron or EOU models for streaming transcription"
            )),
        }
    }

    /// Reset streaming state
    pub fn reset(&self) {
        match self {
            ParakeetBackend::Nemotron(model) => {
                let mut model = model.lock().unwrap();
                model.reset();
            }
            ParakeetBackend::EOU(_model) => {
                // EOU has soft reset built into transcribe() when reset_on_eou=true
                // No explicit reset needed
            }
            ParakeetBackend::TDT(_) => {
                // TDT doesn't have state to reset
            }
        }
    }
}

/// Enum to handle both text-only and timestamped results
enum InternalTranscriptionResult {
    Text(String),
    WithTimestamps {
        text: String,
        tokens: Vec<TimedToken>,
    },
}

impl From<parakeet_rs::TranscriptionResult> for InternalTranscriptionResult {
    fn from(result: parakeet_rs::TranscriptionResult) -> Self {
        if result.tokens.is_empty() {
            InternalTranscriptionResult::Text(result.text)
        } else {
            InternalTranscriptionResult::WithTimestamps {
                text: result.text,
                tokens: result.tokens,
            }
        }
    }
}
