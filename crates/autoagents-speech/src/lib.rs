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
//! ### STT (Speech-to-Text) - Coming Soon
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
//! ## Providers
//!
//! Enable providers using feature flags:
//! - `pocket-tts`: Pocket-TTS model support
//!
//! ## Example
//!
//! ```rust,ignore
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

pub mod error;
pub mod models;
pub mod provider;
pub mod speech;
pub mod types;

// Provider implementations (when features are enabled)
#[cfg(feature = "pocket-tts")]
pub mod providers;

// Re-export main types
pub use error::{TTSError, TTSResult};
pub use models::{ModelInfo, TTSModelsProvider};
pub use provider::TTSProvider;
pub use speech::TTSSpeechProvider;
pub use types::{
    AudioChunk, AudioData, AudioFormat, SharedAudioData, SpeechRequest, SpeechResponse,
    VoiceIdentifier,
};

// Re-export provider types when enabled
#[cfg(feature = "pocket-tts")]
pub use providers::pocket_tts::{
    ModelVariant, PocketTTSConfig, PocketTTSProvider, PredefinedVoice,
};
