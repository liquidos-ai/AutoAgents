//! Configuration for Pocket-TTS provider

use crate::models::ModelVariant;
use crate::voices::PredefinedVoice;
use serde::{Deserialize, Serialize};

/// Configuration for Pocket-TTS provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketTTSConfig {
    /// Model variant to use
    #[serde(default)]
    pub model_variant: ModelVariant,

    /// Temperature for generation (0.0 - 1.0, default: 0.7)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Number of LSD decode steps (default: 1)
    #[serde(default = "default_lsd_steps")]
    pub lsd_decode_steps: usize,

    /// End-of-sequence threshold (default: -4.0)
    #[serde(default = "default_eos_threshold")]
    pub eos_threshold: f32,

    /// Optional noise clamping value
    #[serde(default)]
    pub noise_clamp: Option<f32>,

    /// Default predefined voice (if not specified in requests)
    #[serde(default)]
    pub default_voice: Option<PredefinedVoice>,

    /// Backend-specific configuration
    #[serde(flatten)]
    pub backend: BackendConfig,
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "lowercase")]
pub enum BackendConfig {
    /// Library backend - runs model locally
    #[cfg(feature = "library")]
    Library(LibraryConfig),

    /// Server backend - connects to remote TTS server
    #[cfg(feature = "server")]
    Server(ServerConfig),
}

/// Configuration for library backend
#[cfg(feature = "library")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryConfig {
    /// Directory to cache voice state files (.safetensors)
    #[serde(default = "default_cache_dir")]
    pub cache_dir: std::path::PathBuf,

    /// Use Metal acceleration on macOS (requires metal feature)
    #[serde(default)]
    pub use_metal: bool,
}

/// Configuration for server backend
#[cfg(feature = "server")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server base URL (e.g., "http://localhost:8000")
    pub base_url: String,

    /// Optional API key for authentication
    #[serde(default)]
    pub api_key: Option<String>,

    /// Request timeout in seconds (default: 120)
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

// Default value functions
fn default_temperature() -> f32 {
    0.7
}

fn default_lsd_steps() -> usize {
    1
}

fn default_eos_threshold() -> f32 {
    -4.0
}

#[cfg(feature = "library")]
fn default_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("autoagents")
        .join("tts")
        .join("voice_states")
}

#[cfg(feature = "server")]
fn default_timeout() -> u64 {
    120
}

impl Default for PocketTTSConfig {
    fn default() -> Self {
        Self {
            model_variant: ModelVariant::default(),
            temperature: default_temperature(),
            lsd_decode_steps: default_lsd_steps(),
            eos_threshold: default_eos_threshold(),
            noise_clamp: None,
            default_voice: Some(PredefinedVoice::default()),
            backend: BackendConfig::default(),
        }
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        // Default to library backend if available, otherwise server
        #[cfg(feature = "library")]
        {
            BackendConfig::Library(LibraryConfig::default())
        }
        #[cfg(all(feature = "server", not(feature = "library")))]
        {
            BackendConfig::Server(ServerConfig::default())
        }
        #[cfg(not(any(feature = "library", feature = "server")))]
        {
            compile_error!("At least one backend feature (library or server) must be enabled");
        }
    }
}

#[cfg(feature = "library")]
impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            use_metal: false,
        }
    }
}

#[cfg(feature = "server")]
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000".to_string(),
            api_key: None,
            timeout_secs: default_timeout(),
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
