//! Speech provider implementations
//!
//! This module contains concrete implementations of the TTS/STT providers.
//! Each provider is feature-gated and can be enabled individually.

#[cfg(feature = "pocket-tts")]
pub mod pocket_tts;

#[cfg(feature = "parakeet")]
pub mod parakeet;
