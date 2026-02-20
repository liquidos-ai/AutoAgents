//! Parakeet model variants
//!
//! This module defines the different Parakeet model variants and their characteristics.
//! Each variant has specific requirements for chunk size, streaming support, and language support.
//!
//! ## Model Variants:
//! - **TDT**: Batch processing with 25 language support and timestamps
//! - **Nemotron**: Streaming ASR with 560ms chunks (8960 samples at 16kHz)
//! - **EOU**: Real-time streaming with 160ms chunks (2560 samples at 16kHz) and end-of-utterance detection

use serde::{Deserialize, Serialize};
use std::fmt;

/// Available Parakeet model variants
///
/// Each variant has different capabilities and optimal chunk sizes:
/// - Use `chunk_size()` to get the recommended chunk size for streaming
/// - Use `chunk_duration_ms()` to get the chunk duration in milliseconds
/// - Use `supports_streaming()` to check if the model supports real-time processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelVariant {
    /// TDT (Multilingual) - Supports 25 languages with auto-detection
    #[default]
    TDT,
    /// Nemotron (Streaming) - Cache-aware streaming ASR with punctuation (English only)
    Nemotron,
    /// EOU (End-of-Utterance) - Real-time streaming with end-of-utterance detection (English only)
    EOU,
}

impl ModelVariant {
    /// Get model description
    pub fn description(&self) -> &str {
        match self {
            ModelVariant::TDT => "Parakeet TDT 0.6B - Multilingual ASR with 25 language support",
            ModelVariant::Nemotron => "Nemotron 0.6B - Streaming ASR with punctuation (English)",
            ModelVariant::EOU => {
                "Parakeet EOU 120M - Real-time streaming with end-of-utterance detection (English)"
            }
        }
    }

    /// Get model identifier
    pub fn id(&self) -> &str {
        match self {
            ModelVariant::TDT => "parakeet-tdt-0.6b",
            ModelVariant::Nemotron => "nemotron-speech-streaming-0.6b",
            ModelVariant::EOU => "parakeet-eou-120m",
        }
    }

    /// Check if model supports streaming
    pub fn supports_streaming(&self) -> bool {
        match self {
            ModelVariant::TDT => false,
            ModelVariant::Nemotron => true,
            ModelVariant::EOU => true,
        }
    }

    /// Check if model supports timestamps
    pub fn supports_timestamps(&self) -> bool {
        match self {
            ModelVariant::TDT => true,
            ModelVariant::Nemotron => false,
            ModelVariant::EOU => false,
        }
    }

    /// Check if model supports end-of-utterance detection
    pub fn supports_eou_detection(&self) -> bool {
        match self {
            ModelVariant::TDT => false,
            ModelVariant::Nemotron => false,
            ModelVariant::EOU => true,
        }
    }

    /// Get recommended chunk size in samples (at 16kHz)
    ///
    /// Returns the optimal chunk size for streaming processing:
    /// - EOU: 2560 samples (160ms at 16kHz)
    /// - Nemotron: 8960 samples (560ms at 16kHz)
    /// - TDT: Not applicable for streaming (returns 0)
    pub fn chunk_size(&self) -> usize {
        match self {
            ModelVariant::TDT => 0,         // TDT doesn't support streaming
            ModelVariant::Nemotron => 8960, // 560ms at 16kHz
            ModelVariant::EOU => 2560,      // 160ms at 16kHz
        }
    }

    /// Get chunk duration in milliseconds
    pub fn chunk_duration_ms(&self) -> u32 {
        match self {
            ModelVariant::TDT => 0,
            ModelVariant::Nemotron => 560,
            ModelVariant::EOU => 160,
        }
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> Vec<String> {
        match self {
            ModelVariant::TDT => vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "zh".to_string(),
                "ja".to_string(),
                "ko".to_string(),
                "pt".to_string(),
                "ru".to_string(),
                "it".to_string(),
                "nl".to_string(),
                "pl".to_string(),
                "tr".to_string(),
                "ar".to_string(),
                "hi".to_string(),
                "th".to_string(),
                "vi".to_string(),
                "id".to_string(),
                "uk".to_string(),
                "cs".to_string(),
                "ro".to_string(),
                "sv".to_string(),
                "da".to_string(),
                "fi".to_string(),
                "no".to_string(),
            ],
            ModelVariant::Nemotron => vec!["en".to_string()],
            ModelVariant::EOU => vec!["en".to_string()],
        }
    }
}

impl fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_variant_properties() {
        let tdt = ModelVariant::TDT;
        assert_eq!(tdt.id(), "parakeet-tdt-0.6b");
        assert!(!tdt.supports_streaming());
        assert!(tdt.supports_timestamps());
        assert!(!tdt.supports_eou_detection());
        assert!(tdt.supported_languages().len() > 1);
        assert_eq!(tdt.chunk_size(), 0);
        assert_eq!(tdt.chunk_duration_ms(), 0);

        let nemotron = ModelVariant::Nemotron;
        assert_eq!(nemotron.id(), "nemotron-speech-streaming-0.6b");
        assert!(nemotron.supports_streaming());
        assert!(!nemotron.supports_timestamps());
        assert!(!nemotron.supports_eou_detection());
        assert_eq!(nemotron.supported_languages(), vec!["en".to_string()]);
        assert_eq!(nemotron.chunk_size(), 8960);
        assert_eq!(nemotron.chunk_duration_ms(), 560);

        let eou = ModelVariant::EOU;
        assert_eq!(eou.id(), "parakeet-eou-120m");
        assert!(eou.supports_streaming());
        assert!(!eou.supports_timestamps());
        assert!(eou.supports_eou_detection());
        assert_eq!(eou.supported_languages(), vec!["en".to_string()]);
        assert_eq!(eou.chunk_size(), 2560);
        assert_eq!(eou.chunk_duration_ms(), 160);
    }

    #[test]
    fn test_default_variant() {
        assert_eq!(ModelVariant::default(), ModelVariant::TDT);
    }

    #[test]
    fn test_serialization() {
        let variant = ModelVariant::Nemotron;
        let json = serde_json::to_string(&variant).unwrap();
        let deserialized: ModelVariant = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, variant);
    }
}
