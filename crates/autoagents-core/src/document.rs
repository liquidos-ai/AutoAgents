use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::embeddings::{Embed, EmbedError, TextEmbedder};

/// Represents a piece of content along with optional metadata.
/// Mirrors the LangChain document abstraction (`page_content` + `metadata`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Document {
    pub page_content: String,
    #[serde(default)]
    pub metadata: Value,
}

impl Document {
    pub fn new(page_content: impl Into<String>) -> Self {
        Self {
            page_content: page_content.into(),
            metadata: Value::Object(Default::default()),
        }
    }

    pub fn with_metadata(page_content: impl Into<String>, metadata: Value) -> Self {
        Self {
            page_content: page_content.into(),
            metadata,
        }
    }
}

impl Embed for Document {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.page_content.clone());
        Ok(())
    }
}
