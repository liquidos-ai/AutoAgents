//! # AutoAgents Speech
//!
//! Speech (TTS/STT) provider abstractions for the AutoAgents framework.
//!
//! This crate provides trait-based abstraction layers for speech providers, allowing
//! different backends to be used interchangeably within the AutoAgents ecosystem.
//!
//! ## Features
//!
//! ### TTS (Text-to-Speech)
//! - **Speech Generation**: Generate audio from text
//! - **Voice Management**: Use predefined voices
//! - **Streaming Support**: Optional streaming for real-time audio generation
//! - **Model Management**: Support for multiple models and languages
//!
//! ### STT (Speech-to-Text)
//! - **Transcription**: Convert audio to text
//! - **Streaming Support**: Real-time audio transcription
//! - **Timestamp Support**: Token-level timestamps for transcriptions
//! - **Multilingual**: Support for multiple languages with auto-detection
//!
//! ## Architecture
//!
//! The crate follows a trait-based design with provider implementations in the `providers` module:
//!
//! ### TTS Traits
//! - `TTSProvider`: Marker trait combining all TTS capabilities
//! - `TTSSpeechProvider`: Speech generation capabilities
//! - `TTSModelsProvider`: Model and language support
//!
//! ### STT Traits
//! - `STTProvider`: Marker trait combining all STT capabilities
//! - `STTSpeechProvider`: Transcription capabilities
//! - `STTModelsProvider`: Model and language support
//!
//! ## Providers
//!
//! Enable providers using feature flags:
//! - `pocket-tts`: Pocket-TTS model support (TTS)
//! - `parakeet`: Parakeet (NVIDIA) model support (STT)
//!
//! ## Example - TTS
//!
//! ```rust
//! use autoagents_speech::{TTSProvider, SpeechRequest, VoiceIdentifier, AudioFormat};
//!
//! async fn generate_speech(provider: &dyn TTSProvider, text: &str) {
//!     let request = SpeechRequest {
//!         text: text.to_string(),
//!         voice: VoiceIdentifier::new("alba"),
//!         format: AudioFormat::Wav,
//!         sample_rate: Some(24000),
//!     };
//!
//!     let response = provider.generate_speech(request).await.unwrap();
//!     println!("Generated {} samples", response.audio.samples.len());
//! }
//! ```
//!
//! ## Example - STT
//!
//! ```rust
//! use autoagents_speech::{STTProvider, TranscriptionRequest, AudioData};
//!
//! async fn transcribe_audio(provider: &dyn STTProvider, audio: Vec<f32>) {
//!     let request = TranscriptionRequest {
//!         audio: AudioData {
//!             samples: audio,
//!             sample_rate: 16000,
//!             channels: 1,
//!         },
//!         language: None,
//!         include_timestamps: false,
//!     };
//!
//!     let response = provider.transcribe(request).await.unwrap();
//!     println!("Transcription: {}", response.text);
//! }
//! ```

pub mod error;
mod provider;
pub mod types;

// Provider implementations
pub mod providers;

// Re-export main TTS types
pub use error::{TTSError, TTSResult};
pub use provider::{TTSModelsProvider, TTSProvider, TTSSpeechProvider};
pub use types::{
    AudioChunk, AudioData, AudioFormat, ModelInfo, SharedAudioData, SpeechRequest, SpeechResponse,
    VoiceIdentifier,
};

// Re-export main STT types
pub use error::{STTError, STTResult};
pub use provider::{STTModelsProvider, STTProvider, STTSpeechProvider};
pub use types::{TextChunk, TokenTimestamp, TranscriptionRequest, TranscriptionResponse};

#[cfg(feature = "playback")]
pub mod playback;
