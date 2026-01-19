//! Model source definitions for llama.cpp backend.

use serde::{Deserialize, Serialize};

/// Source type for loading models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    /// Local GGUF file path.
    Gguf {
        /// Path to a GGUF model file.
        model_path: String,
    },
    /// HuggingFace repository ID.
    HuggingFace {
        /// HuggingFace repo ID (e.g. "org/model").
        repo_id: String,
        /// Optional GGUF filename override.
        filename: Option<String>,
    },
}

impl ModelSource {
    /// Convenience constructor for a local GGUF model.
    pub fn gguf(model_path: impl Into<String>) -> Self {
        Self::Gguf {
            model_path: model_path.into(),
        }
    }

    /// Convenience constructor for a HuggingFace model repository.
    pub fn huggingface(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            filename: None,
        }
    }

    /// Convenience constructor for a HuggingFace repo with a GGUF filename override.
    pub fn huggingface_with_filename(
        repo_id: impl Into<String>,
        filename: impl Into<String>,
    ) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            filename: Some(filename.into()),
        }
    }

    /// Return the model path for this source.
    pub fn model_path(&self) -> Option<&str> {
        match self {
            ModelSource::Gguf { model_path } => Some(model_path),
            ModelSource::HuggingFace { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_source_path() {
        let source = ModelSource::gguf("test.gguf");
        assert_eq!(source.model_path(), Some("test.gguf"));
    }

    #[test]
    fn test_model_source_hf() {
        let source = ModelSource::huggingface("org/model");
        assert!(source.model_path().is_none());
        assert_eq!(
            source,
            ModelSource::HuggingFace {
                repo_id: "org/model".to_string(),
                filename: None,
            }
        );
    }
}
