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
//! - `vad`: Silero VAD support (speech segmentation)
//!

pub mod error;
pub mod model_source;
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
pub use model_source::ModelSource;
pub use provider::{STTModelsProvider, STTProvider, STTSpeechProvider};
pub use types::{TextChunk, TokenTimestamp, TranscriptionRequest, TranscriptionResponse};

#[cfg(feature = "playback")]
pub mod playback;

#[cfg(feature = "audio-capture")]
pub mod audio_capture;

#[cfg(feature = "vad")]
pub mod vad;
