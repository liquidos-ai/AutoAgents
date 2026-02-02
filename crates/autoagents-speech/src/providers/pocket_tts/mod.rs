//! Pocket-TTS provider for AutoAgents Speech framework
//!
//! This module provides a Pocket-TTS implementation for the AutoAgents TTS trait system.
//!
//! # Examples
//!
//! ```no_run
//! use autoagents_speech::providers::pocket_tts::{PocketTTS, PocketTTSConfig};
//! use autoagents_speech::{TTSSpeechProvider, SpeechRequest, VoiceIdentifier, AudioFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create provider with default configuration
//!     let provider = PocketTTS::new(Some(PocketTTSConfig::default()))?;
//!
//!     // Generate speech
//!     let request = SpeechRequest {
//!         text: "Hello, world!".to_string(),
//!         voice: VoiceIdentifier::new("alba"),
//!         format: AudioFormat::Wav,
//!         sample_rate: Some(24000),
//!     };
//!
//!     let response = provider.generate_speech(request).await?;
//!
//!     // Use response.audio...
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod error;
pub mod model;
pub mod voices;

mod provider;
mod tts;

// Re-exports
pub use config::PocketTTSConfig;
pub use error::{PocketTTSError, Result};
pub use model::ModelVariant;
pub use provider::PocketTTS;
pub use voices::PredefinedVoice;
