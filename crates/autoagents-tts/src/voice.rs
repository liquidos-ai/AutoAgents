use crate::error::TTSResult;
use crate::types::{VoiceIdentifier, VoiceState};
use async_trait::async_trait;
use std::path::Path;

/// Trait for TTS voice management capabilities
#[async_trait]
pub trait TTSVoiceProvider: Send + Sync {
    /// Create a voice from an audio file (required)
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, MP3, etc.)
    ///
    /// # Returns
    /// Voice state that can be used for generation
    async fn create_voice_from_file(
        &self,
        path: &Path,
    ) -> TTSResult<VoiceState>;

    /// Create a voice from audio bytes (required)
    ///
    /// # Arguments
    /// * `audio_bytes` - Raw audio data
    ///
    /// # Returns
    /// Voice state that can be used for generation
    async fn create_voice_from_bytes(&self, audio_bytes: &[u8]) -> TTSResult<VoiceState>;

    /// Save voice state to disk (optional - implement for persistence)
    ///
    /// # Arguments
    /// * `voice_state` - Voice state to save
    /// * `path` - Path to save the voice state (.safetensors file)
    async fn save_voice_state(
        &self,
        voice_state: &VoiceState,
        path: &Path,
    ) -> TTSResult<()> {
        let _ = (voice_state, path);
        Err(crate::error::TTSError::Other(
            "Voice state persistence not implemented for this provider".to_string(),
        ))
    }

    /// Load voice state from disk (optional - implement for persistence)
    ///
    /// # Arguments
    /// * `path` - Path to the saved voice state (.safetensors file)
    ///
    /// # Returns
    /// Loaded voice state
    async fn load_voice_state(&self, path: &Path) -> TTSResult<VoiceState> {
        let _ = path;
        Err(crate::error::TTSError::Other(
            "Voice state persistence not implemented for this provider".to_string(),
        ))
    }

    /// Get a predefined voice by name (required)
    ///
    /// # Arguments
    /// * `name` - Name of the predefined voice
    ///
    /// # Returns
    /// Voice identifier for the predefined voice
    fn get_predefined_voice(&self, name: &str) -> TTSResult<VoiceIdentifier>;

    /// List all available predefined voices (required)
    ///
    /// # Returns
    /// List of predefined voice names
    fn list_predefined_voices(&self) -> Vec<String>;

    /// Get default voice name
    fn default_voice(&self) -> String {
        "default".to_string()
    }
}
