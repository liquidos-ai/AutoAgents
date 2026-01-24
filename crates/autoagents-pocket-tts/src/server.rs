//! Server backend - connects to remote Pocket-TTS server

#[cfg(feature = "server")]
use crate::config::ServerConfig;
use crate::error::{PocketTTSError, Result};
use autoagents_tts::{SpeechRequest, SpeechResponse, VoiceIdentifier, VoiceStateData};
use std::time::Duration;

/// Server backend that connects to a remote Pocket-TTS server
#[cfg(feature = "server")]
pub struct ServerBackend {
    /// HTTP client
    client: reqwest::Client,
    /// Configuration
    config: ServerConfig,
}

#[cfg(feature = "server")]
impl ServerBackend {
    /// Create a new server backend
    pub fn new(config: ServerConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| PocketTTSError::HttpError(e))?;

        Ok(Self { client, config })
    }

    /// Generate speech from text
    pub async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse> {
        let url = format!("{}/v1/audio/speech", self.config.base_url);

        // Build request body
        let mut req_builder = self.client.post(&url).json(&serde_json::json!({
            "text": request.text,
            "voice": voice_identifier_to_string(&request.voice),
        }));

        // Add API key if configured
        if let Some(api_key) = &self.config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        // Send request
        let response = req_builder
            .send()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            return Err(PocketTTSError::GenerationError(format!(
                "Server returned error {}: {}",
                status, body
            )));
        }

        // Parse response
        let speech_response: SpeechResponse = response
            .json()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        Ok(speech_response)
    }

    /// Generate streaming audio
    pub async fn generate_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<impl futures::Stream<Item = Result<SpeechResponse>>> {
        let url = format!("{}/v1/audio/speech/stream", self.config.base_url);

        let mut req_builder = self.client.post(&url).json(&serde_json::json!({
            "text": request.text,
            "voice": voice_identifier_to_string(&request.voice),
        }));

        if let Some(api_key) = &self.config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            return Err(PocketTTSError::GenerationError(format!(
                "Server returned error {}: {}",
                status, body
            )));
        }

        // Parse streaming response (newline-delimited JSON)
        let stream = response.bytes_stream();
        let json_stream = stream.map(|result| {
            result
                .map_err(|e| PocketTTSError::HttpError(e))
                .and_then(|bytes| {
                    serde_json::from_slice::<SpeechResponse>(&bytes)
                        .map_err(|e| PocketTTSError::JsonError(e))
                })
        });

        Ok(json_stream)
    }

    /// Create a voice from audio bytes
    pub async fn create_voice(
        &self,
        name: String,
        audio_bytes: Vec<u8>,
    ) -> Result<VoiceStateData> {
        let url = format!("{}/v1/voices", self.config.base_url);

        let mut req_builder = self.client.post(&url).json(&serde_json::json!({
            "name": name,
            "audio": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &audio_bytes),
        }));

        if let Some(api_key) = &self.config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            return Err(PocketTTSError::VoiceLoadError(format!(
                "Server returned error {}: {}",
                status, body
            )));
        }

        let voice_data: VoiceStateData = response
            .json()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        Ok(voice_data)
    }

    /// List available voices
    pub async fn list_voices(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/voices", self.config.base_url);

        let mut req_builder = self.client.get(&url);

        if let Some(api_key) = &self.config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(PocketTTSError::Other(format!(
                "Server returned error {}",
                status
            )));
        }

        let voices: Vec<String> = response
            .json()
            .await
            .map_err(|e| PocketTTSError::HttpError(e))?;

        Ok(voices)
    }
}

/// Convert VoiceIdentifier to string for API requests
#[cfg(feature = "server")]
fn voice_identifier_to_string(voice: &VoiceIdentifier) -> String {
    match voice {
        VoiceIdentifier::Predefined(name) => name.clone(),
        VoiceIdentifier::Custom(name) => name.clone(),
        VoiceIdentifier::File(path) => path.to_string_lossy().to_string(),
        VoiceIdentifier::Bytes(_) => "inline_audio".to_string(),
    }
}

#[cfg(test)]
#[cfg(feature = "server")]
mod tests {
    use super::*;

    #[test]
    fn test_voice_identifier_to_string() {
        assert_eq!(
            voice_identifier_to_string(&VoiceIdentifier::Predefined("alba".to_string())),
            "alba"
        );
        assert_eq!(
            voice_identifier_to_string(&VoiceIdentifier::Custom("my_voice".to_string())),
            "my_voice"
        );
    }
}
