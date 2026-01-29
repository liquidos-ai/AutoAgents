//! Predefined voices catalog for Pocket-TTS

use serde::{Deserialize, Serialize};

/// Predefined voice identifiers
///
/// These voices are available from the kyutai/pocket-tts-without-voice-cloning
/// HuggingFace repository as pre-computed .safetensors embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PredefinedVoice {
    /// Alba - default voice
    Alba,
    /// Marius
    Marius,
    /// Javert
    Javert,
    /// Jean
    Jean,
    /// Fantine
    Fantine,
    /// Cosette
    Cosette,
    /// Eponine
    Eponine,
    /// Azelma
    Azelma,
}

impl PredefinedVoice {
    /// Get the HuggingFace path for this voice's embeddings
    pub fn hf_path(&self) -> String {
        format!(
            "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{}.safetensors",
            self.identifier()
        )
    }

    /// Get the string identifier for this voice
    pub fn identifier(&self) -> &'static str {
        match self {
            PredefinedVoice::Alba => "alba",
            PredefinedVoice::Marius => "marius",
            PredefinedVoice::Javert => "javert",
            PredefinedVoice::Jean => "jean",
            PredefinedVoice::Fantine => "fantine",
            PredefinedVoice::Cosette => "cosette",
            PredefinedVoice::Eponine => "eponine",
            PredefinedVoice::Azelma => "azelma",
        }
    }

    /// Get all available predefined voices
    pub fn all() -> &'static [PredefinedVoice] {
        &[
            PredefinedVoice::Alba,
            PredefinedVoice::Marius,
            PredefinedVoice::Javert,
            PredefinedVoice::Jean,
            PredefinedVoice::Fantine,
            PredefinedVoice::Cosette,
            PredefinedVoice::Eponine,
            PredefinedVoice::Azelma,
        ]
    }
}

impl Default for PredefinedVoice {
    fn default() -> Self {
        PredefinedVoice::Alba
    }
}

impl std::fmt::Display for PredefinedVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.identifier())
    }
}

impl std::str::FromStr for PredefinedVoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "alba" => Ok(PredefinedVoice::Alba),
            "marius" => Ok(PredefinedVoice::Marius),
            "javert" => Ok(PredefinedVoice::Javert),
            "jean" => Ok(PredefinedVoice::Jean),
            "fantine" => Ok(PredefinedVoice::Fantine),
            "cosette" => Ok(PredefinedVoice::Cosette),
            "eponine" => Ok(PredefinedVoice::Eponine),
            "azelma" => Ok(PredefinedVoice::Azelma),
            _ => Err(format!("Unknown predefined voice: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predefined_voice_count() {
        assert_eq!(PredefinedVoice::all().len(), 8);
    }

    #[test]
    fn test_predefined_voice_default() {
        assert_eq!(PredefinedVoice::default(), PredefinedVoice::Alba);
    }

    #[test]
    fn test_predefined_voice_from_str() {
        assert_eq!(
            "alba".parse::<PredefinedVoice>().unwrap(),
            PredefinedVoice::Alba
        );
        assert_eq!(
            "MARIUS".parse::<PredefinedVoice>().unwrap(),
            PredefinedVoice::Marius
        );
        assert!("unknown".parse::<PredefinedVoice>().is_err());
    }

    #[test]
    fn test_predefined_voice_hf_path() {
        assert_eq!(
            PredefinedVoice::Alba.hf_path(),
            "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors"
        );
    }

    #[test]
    fn test_predefined_voice_display() {
        assert_eq!(PredefinedVoice::Marius.to_string(), "marius");
    }
}
