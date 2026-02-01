use crate::error::TTSResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Model information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Supported languages
    pub languages: Vec<String>,
}

/// Trait for TTS model management capabilities
#[async_trait]
pub trait TTSModelsProvider: Send + Sync {
    /// List available models (optional)
    ///
    /// # Returns
    /// List of available model information
    async fn list_models(&self) -> TTSResult<Vec<ModelInfo>> {
        Ok(vec![])
    }

    /// Get current model information (required)
    ///
    /// # Returns
    /// Current model information
    fn get_current_model(&self) -> ModelInfo;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<String> {
        vec!["en".to_string()]
    }
}
