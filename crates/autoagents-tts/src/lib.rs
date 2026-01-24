//! # AutoAgents TTS
//!
//! TTS (Text-to-Speech) provider abstractions for the AutoAgents framework.
//!
//! This crate provides a trait-based abstraction layer for TTS providers, allowing
//! different TTS backends to be used interchangeably within the AutoAgents ecosystem.
//!
//! ## Features
//!
//! - **Speech Generation**: Generate audio from text
//! - **Voice Management**: Clone voices, use predefined voices, and persist voice states
//! - **Streaming Support**: Optional streaming for real-time audio generation
//! - **Model Management**: Support for multiple models and languages
//! - **Memory Efficient**: Configurable storage policies for audio data
//!
//! ## Architecture
//!
//! The crate follows a trait-based design similar to the LLM provider system:
//!
//! - `TTSProvider`: Marker trait combining all capabilities
//! - `TTSSpeechProvider`: Speech generation capabilities
//! - `TTSVoiceProvider`: Voice management and cloning
//! - `TTSModelsProvider`: Model and language support
//!
//! ## Example
//!
//! ```rust,ignore
//! use autoagents_tts::{TTSProvider, SpeechRequest, VoiceIdentifier, AudioFormat};
//!
//! async fn generate_speech(provider: &dyn TTSProvider, text: &str) {
//!     let request = SpeechRequest {
//!         text: text.to_string(),
//!         voice: VoiceIdentifier::Predefined("alba".to_string()),
//!         format: AudioFormat::Wav,
//!         sample_rate: Some(24000),
//!     };
//!     
//!     let response = provider.generate_speech(request).await.unwrap();
//!     println!("Generated {} samples", response.audio.samples.len());
//! }
//! ```

pub mod builder;
pub mod error;
pub mod models;
pub mod provider;
pub mod speech;
pub mod types;
pub mod voice;

// Re-export main types
pub use builder::TTSBuilder;
pub use error::{TTSError, TTSResult};
pub use models::{ModelInfo, TTSModelsProvider};
pub use provider::TTSProvider;
pub use speech::TTSSpeechProvider;
pub use types::{
    AudioChunk, AudioData, AudioFormat, AudioStoragePolicy, SharedAudioData, SpeechRequest,
    SpeechResponse, TTSMode, VoiceIdentifier, VoiceState, VoiceStateData,
};
pub use voice::TTSVoiceProvider;
