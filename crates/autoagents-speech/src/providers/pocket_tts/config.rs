//! Configuration for Pocket-TTS provider

use super::model::ModelVariant;
use super::voices::PredefinedVoice;
use serde::{Deserialize, Serialize};

/// Configuration for Pocket-TTS provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketTTSConfig {
    /// Model variant to use
    #[serde(default)]
    pub model_variant: ModelVariant,

    /// Temperature for generation (0.0 - 1.0, default: 0.7)
    pub temperature: f32,

    /// Number of LSD decode steps (default: 1)
    pub lsd_decode_steps: usize,

    /// End-of-sequence threshold (default: -4.0)
    pub eos_threshold: f32,

    /// Optional noise clamping value
    #[serde(default)]
    pub noise_clamp: Option<f32>,

    /// Default predefined voice (if not specified in requests)
    #[serde(default)]
    pub default_voice: Option<PredefinedVoice>,
}

impl Default for PocketTTSConfig {
    fn default() -> Self {
        Self {
            model_variant: ModelVariant::default(),
            temperature: 0.7,
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
            noise_clamp: None,
            default_voice: Some(PredefinedVoice::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PocketTTSConfig::default();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.lsd_decode_steps, 1);
        assert_eq!(config.eos_threshold, -4.0);
        assert_eq!(config.model_variant, ModelVariant::default());
    }

    #[test]
    fn test_config_serialization() {
        let config = PocketTTSConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PocketTTSConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.temperature, config.temperature);
    }
}
