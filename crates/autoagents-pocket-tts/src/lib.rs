//! Pocket-TTS provider for AutoAgents TTS framework
//!
//! This crate provides a Pocket-TTS implementation for the AutoAgents TTS trait system.
//!
//! # Features
//!
//! - `library` (default): Local model execution using pocket-tts library
//! - `server`: Remote server backend via HTTP API
//!
//! # Examples
//!
//! ## Library Backend (Default)
//!
//! ```no_run
//! use autoagents_pocket_tts::{PocketTTSProvider, PocketTTSConfig};
//! use autoagents_tts::{TTSSpeechProvider, SpeechRequest, VoiceIdentifier, AudioFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create provider with default configuration
//!     let provider = PocketTTSProvider::new(PocketTTSConfig::default())?;
//!     
//!     // Generate speech
//!     let request = SpeechRequest {
//!         text: "Hello, world!".to_string(),
//!         voice: VoiceIdentifier::Predefined("alba".to_string()),
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
//!
//! ## Server Backend
//!
//! ```no_run
//! # #[cfg(feature = "server")]
//! # {
//! use autoagents_pocket_tts::{PocketTTSProvider, PocketTTSConfig};
//! use autoagents_pocket_tts::config::{BackendConfig, ServerConfig};
//! use autoagents_tts::TTSSpeechProvider;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = PocketTTSConfig {
//!         backend: BackendConfig::Server(ServerConfig {
//!             base_url: "http://localhost:8000".to_string(),
//!             api_key: None,
//!             timeout_secs: 120,
//!         }),
//!         ..Default::default()
//!     };
//!     
//!     let provider = PocketTTSProvider::new(config)?;
//!     // Use provider...
//!     Ok(())
//! }
//! # }
//! ```

pub mod config;
pub mod error;
pub mod models;
pub mod voices;

mod conversion;
#[cfg(feature = "library")]
mod library;
#[cfg(feature = "server")]
mod server;
mod provider;

// Re-exports
pub use config::{PocketTTSConfig, BackendConfig};
#[cfg(feature = "library")]
pub use config::LibraryConfig;
#[cfg(feature = "server")]
pub use config::ServerConfig;

pub use error::{PocketTTSError, Result};
pub use models::ModelVariant;
pub use voices::PredefinedVoice;
pub use provider::PocketTTSProvider;

// Re-export autoagents-tts traits for convenience
pub use autoagents_tts::{
    TTSProvider, TTSSpeechProvider, TTSVoiceProvider, TTSModelsProvider,
    SpeechRequest, SpeechResponse, VoiceIdentifier, VoiceState,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_imports() {
        // Ensure all public types are accessible
        let _config: PocketTTSConfig = PocketTTSConfig::default();
        let _variant: ModelVariant = ModelVariant::default();
        let _voice: PredefinedVoice = PredefinedVoice::default();
    }
}
