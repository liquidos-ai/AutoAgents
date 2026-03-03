//! Parakeet STT backend implementation

use super::config::ParakeetConfig;
use super::error::{ParakeetError, Result};
use super::model::ModelVariant;
use crate::{AudioData, TextChunk, TokenTimestamp, TranscriptionRequest, TranscriptionResponse};
use parakeet_rs::Transcriber;
use parakeet_rs::{
    ExecutionConfig, ExecutionProvider, Nemotron, ParakeetEOU, ParakeetTDT, TimedToken,
    TimestampMode,
};
use std::time::Instant;

fn validate_audio(audio: &AudioData) -> Result<()> {
    if audio.sample_rate != 16000 || audio.channels != 1 {
        return Err(ParakeetError::invalid_audio_format(
            "Audio must be 16 kHz mono",
            16000,
            1,
            audio.sample_rate,
            audio.channels as u16,
        ));
    }
    Ok(())
}

/// Backend implementation for Parakeet STT
///
/// Each variant is boxed to reduce stack size, as the model structs are large.
#[allow(clippy::upper_case_acronyms)]
pub enum ParakeetBackend {
    TDT(Box<ParakeetTDT>),
    Nemotron(Box<Nemotron>),
    EOU(Box<ParakeetEOU>),
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
                Ok(ParakeetBackend::TDT(Box::new(model)))
            }
            ModelVariant::Nemotron => {
                let model = Nemotron::from_pretrained(&config.model_path, Some(exec_config))
                    .map_err(|e| ParakeetError::ModelLoadError {
                        path: config.model_path.clone(),
                        reason: e.to_string(),
                        variant: "Nemotron".to_string(),
                    })?;
                Ok(ParakeetBackend::Nemotron(Box::new(model)))
            }
            ModelVariant::EOU => {
                let model = ParakeetEOU::from_pretrained(&config.model_path, Some(exec_config))
                    .map_err(|e| ParakeetError::ModelLoadError {
                        path: config.model_path.clone(),
                        reason: e.to_string(),
                        variant: "EOU".to_string(),
                    })?;
                Ok(ParakeetBackend::EOU(Box::new(model)))
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
    pub async fn transcribe(
        &mut self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse> {
        let start = Instant::now();
        let audio = request.audio; // Arc<AudioData>

        validate_audio(&audio)?;

        let result = match self {
            ParakeetBackend::TDT(model) => {
                let timestamp_mode = if request.include_timestamps {
                    Some(TimestampMode::Words)
                } else {
                    None
                };
                // TDT needs an owned Vec<f32>; clone the samples out of the Arc.
                let samples = audio.samples.clone();
                let sample_rate = audio.sample_rate;
                let channels = audio.channels as u16;

                let parakeet_result: parakeet_rs::TranscriptionResult =
                    tokio::task::block_in_place(|| {
                        model.transcribe_samples(samples, sample_rate, channels, timestamp_mode)
                    })
                    .map_err(|e| {
                        ParakeetError::transcription_error_detailed(
                            e.to_string(),
                            "TDT transcription",
                            "model processing",
                        )
                    })?;

                InternalTranscriptionResult::from(parakeet_result)
            }
            ParakeetBackend::Nemotron(model) => {
                let text = tokio::task::block_in_place(|| {
                    model.reset();
                    model.transcribe_audio(&audio.samples)
                })
                .map_err(|e| {
                    ParakeetError::transcription_error_detailed(
                        e.to_string(),
                        "Nemotron transcription",
                        "streaming model processing",
                    )
                })?;

                InternalTranscriptionResult::Text(text)
            }
            ParakeetBackend::EOU(model) => {
                let chunk_size = ModelVariant::EOU.chunk_size();

                let text = tokio::task::block_in_place(|| {
                    let mut result_text = String::default();

                    for chunk in audio.samples.chunks(chunk_size) {
                        match model.transcribe(chunk, true) {
                            Ok(text) => {
                                result_text.push_str(&text);
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    Ok(result_text)
                })
                .map_err(|e| {
                    ParakeetError::transcription_error_detailed(
                        e.to_string(),
                        "EOU transcription",
                        "chunk processing",
                    )
                })?;

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
    pub async fn transcribe_chunk(&mut self, audio: Vec<f32>) -> Result<TextChunk> {
        let audio_len = audio.len();

        match self {
            ParakeetBackend::Nemotron(model) => {
                let expected = ModelVariant::Nemotron.chunk_size();
                if audio_len != expected {
                    return Err(ParakeetError::chunk_processing_error(
                        "unexpected chunk size",
                        audio_len,
                        expected,
                        "Nemotron",
                    ));
                }

                let text = tokio::task::block_in_place(|| model.transcribe_chunk(&audio)).map_err(
                    |e| {
                        ParakeetError::transcription_error_detailed(
                            e.to_string(),
                            "Nemotron chunk transcription",
                            format!("{} samples", audio_len),
                        )
                    },
                )?;

                Ok(TextChunk {
                    text,
                    is_final: false,
                })
            }
            ParakeetBackend::EOU(model) => {
                let expected = ModelVariant::EOU.chunk_size();
                if audio_len != expected {
                    return Err(ParakeetError::chunk_processing_error(
                        "unexpected chunk size",
                        audio_len,
                        expected,
                        "EOU",
                    ));
                }

                // reset_on_eou=true: model auto-resets its state when EOU is detected,
                // which is the correct behaviour for streaming utterance boundaries.
                let result = tokio::task::block_in_place(|| model.transcribe(&audio, true))
                    .map_err(|e| {
                        ParakeetError::transcription_error_detailed(
                            e.to_string(),
                            "EOU chunk transcription",
                            format!("{} samples", audio_len),
                        )
                    })?;

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
                "Use Nemotron or EOU models for streaming transcription",
            )),
        }
    }

    /// Reset streaming state
    pub fn reset(&mut self) {
        match self {
            ParakeetBackend::Nemotron(model) => {
                model.reset();
            }
            ParakeetBackend::EOU(_) => {
                // ParakeetEOU has no public reset(); state is reset automatically on each EOU
                // detection via reset_on_eou=true in transcribe_chunk(), and cleared on the
                // next session start via the first transcribe() call.
            }
            ParakeetBackend::TDT(_) => {
                // TDT doesn't have state to reset
            }
        }
    }
}

/// Validate that the requested language is supported by the given model variant
pub(super) fn validate_language(language: Option<&str>, variant: &ModelVariant) -> Result<()> {
    if let Some(lang) = language {
        let supported = variant.supported_languages();
        if !supported.iter().any(|s| s == lang) {
            return Err(ParakeetError::language_not_supported(
                lang,
                variant.to_string(),
                supported.join(", "),
            ));
        }
    }
    Ok(())
}

/// Enum to handle both text-only and timestamped results
#[allow(clippy::upper_case_acronyms)]
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
