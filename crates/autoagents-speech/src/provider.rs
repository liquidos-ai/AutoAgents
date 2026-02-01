use crate::models::TTSModelsProvider;
use crate::speech::TTSSpeechProvider;
use async_trait::async_trait;

/// Marker trait for TTS providers
///
/// This trait combines all TTS capabilities into a single provider interface.
/// Providers should implement this marker trait along with the specific capability traits.
#[async_trait]
pub trait TTSProvider: TTSSpeechProvider + TTSModelsProvider + Send + Sync {
    /// Get the provider name
    fn provider_name(&self) -> &str;

    /// Get the provider version
    fn provider_version(&self) -> &str {
        "unknown"
    }

    /// Check if the provider is initialized and ready
    async fn is_ready(&self) -> bool {
        false
    }
}
