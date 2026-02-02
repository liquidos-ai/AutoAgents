//! Model variant configurations for Pocket-TTS

use serde::{Deserialize, Serialize};

/// Available Pocket-TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelVariant {
    /// Default model variant (b6369a24)
    #[default]
    #[serde(rename = "b6369a24")]
    B6369a24,
}

impl ModelVariant {
    /// Get the HuggingFace identifier for this model variant
    pub fn hf_id(&self) -> &'static str {
        match self {
            ModelVariant::B6369a24 => "b6369a24",
        }
    }

    /// Get a human-readable description of this model variant
    pub fn description(&self) -> &'static str {
        match self {
            ModelVariant::B6369a24 => "Default Pocket-TTS model (24kHz, 6 layers)",
        }
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hf_id())
    }
}

impl std::str::FromStr for ModelVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "b6369a24" | "default" => Ok(ModelVariant::B6369a24),
            _ => Err(format!("Unknown model variant: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_variant_default() {
        assert_eq!(ModelVariant::default(), ModelVariant::B6369a24);
    }

    #[test]
    fn test_model_variant_from_str() {
        assert_eq!(
            "b6369a24".parse::<ModelVariant>().unwrap(),
            ModelVariant::B6369a24
        );
        assert_eq!(
            "default".parse::<ModelVariant>().unwrap(),
            ModelVariant::B6369a24
        );
        assert!("invalid".parse::<ModelVariant>().is_err());
    }

    #[test]
    fn test_model_variant_display() {
        assert_eq!(ModelVariant::B6369a24.to_string(), "b6369a24");
    }
}
