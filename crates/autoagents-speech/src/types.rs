use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

/// Audio data with normalized samples
#[derive(Clone, Debug)]
pub struct AudioData {
    /// Audio samples normalized to [-1.0, 1.0]
    pub samples: Vec<f32>,
    /// Number of audio channels (typically 1 for mono)
    pub channels: usize,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl Serialize for AudioData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("AudioData", 3)?;

        // Serialize samples as base64
        let bytes: Vec<u8> = self.samples.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base64_samples =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);

        state.serialize_field("samples", &base64_samples)?;
        state.serialize_field("channels", &self.channels)?;
        state.serialize_field("sample_rate", &self.sample_rate)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for AudioData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct AudioDataHelper {
            samples: String,
            channels: usize,
            sample_rate: u32,
        }

        let helper = AudioDataHelper::deserialize(deserializer)?;

        // Deserialize samples from base64
        let bytes =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &helper.samples)
                .map_err(serde::de::Error::custom)?;

        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr)
            })
            .collect();

        Ok(AudioData {
            samples,
            channels: helper.channels,
            sample_rate: helper.sample_rate,
        })
    }
}

/// Shared reference to audio data for memory efficiency
pub type SharedAudioData = Arc<AudioData>;

/// Audio format for output
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AudioFormat {
    #[default]
    Wav,
    Mp3,
    Flac,
    Ogg,
}

/// Voice identifier for TTS generation (predefined voices only)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceIdentifier {
    /// Predefined voice name (e.g., "alba", "marius")
    pub name: String,
}

impl VoiceIdentifier {
    /// Create a voice identifier from a predefined voice name
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Get the voice name
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl From<String> for VoiceIdentifier {
    fn from(name: String) -> Self {
        Self::new(name)
    }
}

impl From<&str> for VoiceIdentifier {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

/// Speech generation request
#[derive(Clone, Debug)]
pub struct SpeechRequest {
    pub text: String,
    pub voice: VoiceIdentifier,
    pub format: AudioFormat,
    pub sample_rate: Option<u32>,
}

/// Speech generation response
#[derive(Clone, Debug)]
pub struct SpeechResponse {
    pub audio: AudioData,
    pub text: String,
    pub duration_ms: u64,
}

/// Audio chunk for streaming
#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub is_final: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_data_serialization() {
        let audio = AudioData {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            channels: 1,
            sample_rate: 24000,
        };

        let json = serde_json::to_string(&audio).unwrap();
        let deserialized: AudioData = serde_json::from_str(&json).unwrap();

        assert_eq!(audio.samples.len(), deserialized.samples.len());
        assert_eq!(audio.channels, deserialized.channels);
        assert_eq!(audio.sample_rate, deserialized.sample_rate);

        for (a, b) in audio.samples.iter().zip(deserialized.samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_voice_identifier_serialization() {
        let voice = VoiceIdentifier::new("alba");
        let json = serde_json::to_string(&voice).unwrap();
        let deserialized: VoiceIdentifier = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "alba");
    }

    #[test]
    fn test_voice_identifier_from_string() {
        let voice: VoiceIdentifier = "marius".into();
        assert_eq!(voice.name(), "marius");
    }
}
