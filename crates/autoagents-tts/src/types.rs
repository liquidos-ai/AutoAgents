use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::Any;
use std::path::PathBuf;
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

/// Policy for storing audio data
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioStoragePolicy {
    /// No storage - stream only, discard after use
    None,
    /// Store in output struct only (default)
    OutputOnly,
    /// Store in context history only
    HistoryOnly,
    /// Store in both output and history (Arc-shared for efficiency)
    Full,
}

impl Default for AudioStoragePolicy {
    fn default() -> Self {
        Self::OutputOnly
    }
}

/// TTS operational mode
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TTSMode {
    /// TTS is disabled (default)
    Disabled,
    /// Generate both text and audio
    TextAndAudio,
    /// Generate audio only
    AudioOnly,
    /// Manual TTS via ctx.speak() calls
    OnDemand,
}

impl Default for TTSMode {
    fn default() -> Self {
        Self::Disabled
    }
}

/// Audio format for output
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::Wav
    }
}

/// Voice state data (provider-specific)
#[derive(Clone)]
pub enum VoiceStateData {
    /// Pocket-TTS voice state
    PocketTTS(Arc<dyn Any + Send + Sync>),
    /// Other provider-specific states
    Other(Arc<dyn Any + Send + Sync>),
}

impl std::fmt::Debug for VoiceStateData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoiceStateData::PocketTTS(_) => write!(f, "PocketTTS(<opaque>)"),
            VoiceStateData::Other(_) => write!(f, "Other(<opaque>)"),
        }
    }
}

/// Voice state with metadata
#[derive(Clone, Debug)]
pub struct VoiceState {
    pub id: String,
    pub name: Option<String>,
    pub data: VoiceStateData,
}

/// Voice identifier for TTS generation
#[derive(Clone, Debug)]
pub enum VoiceIdentifier {
    /// Predefined voice by name (e.g., "alba", "marius")
    Predefined(String),
    /// Custom cloned voice state
    Custom(Arc<VoiceState>),
    /// Path to audio file for voice cloning
    File(PathBuf),
    /// Audio bytes for voice cloning (not serializable)
    Bytes(Arc<Vec<u8>>),
}

impl Serialize for VoiceIdentifier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        match self {
            VoiceIdentifier::Predefined(name) => {
                let mut state = serializer.serialize_struct("VoiceIdentifier", 2)?;
                state.serialize_field("type", "predefined")?;
                state.serialize_field("name", name)?;
                state.end()
            }
            VoiceIdentifier::File(path) => {
                let mut state = serializer.serialize_struct("VoiceIdentifier", 2)?;
                state.serialize_field("type", "file")?;
                state.serialize_field("path", &path.to_string_lossy())?;
                state.end()
            }
            VoiceIdentifier::Custom(_) => {
                let mut state = serializer.serialize_struct("VoiceIdentifier", 1)?;
                state.serialize_field("type", "custom")?;
                state.end()
            }
            VoiceIdentifier::Bytes(_) => Err(serde::ser::Error::custom(
                "Cannot serialize VoiceIdentifier::Bytes",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for VoiceIdentifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct VoiceIdentifierHelper {
            r#type: String,
            name: Option<String>,
            path: Option<String>,
        }

        let helper = VoiceIdentifierHelper::deserialize(deserializer)?;

        match helper.r#type.as_str() {
            "predefined" => Ok(VoiceIdentifier::Predefined(
                helper
                    .name
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?,
            )),
            "file" => Ok(VoiceIdentifier::File(PathBuf::from(
                helper
                    .path
                    .ok_or_else(|| serde::de::Error::missing_field("path"))?,
            ))),
            "custom" => Err(serde::de::Error::custom(
                "Cannot deserialize VoiceIdentifier::Custom",
            )),
            _ => Err(serde::de::Error::unknown_variant(
                &helper.r#type,
                &["predefined", "file", "custom"],
            )),
        }
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
    fn test_voice_identifier_predefined_serialization() {
        let voice = VoiceIdentifier::Predefined("alba".to_string());
        let json = serde_json::to_string(&voice).unwrap();
        let deserialized: VoiceIdentifier = serde_json::from_str(&json).unwrap();

        match deserialized {
            VoiceIdentifier::Predefined(name) => assert_eq!(name, "alba"),
            _ => panic!("Expected Predefined variant"),
        }
    }

    #[test]
    fn test_audio_storage_policy_default() {
        assert_eq!(
            AudioStoragePolicy::default(),
            AudioStoragePolicy::OutputOnly
        );
    }

    #[test]
    fn test_tts_mode_default() {
        assert_eq!(TTSMode::default(), TTSMode::Disabled);
    }
}
