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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn document_new_initializes_empty_metadata() {
        let document = Document::new("hello");

        assert_eq!(document.page_content, "hello");
        assert_eq!(document.metadata, json!({}));
    }

    #[test]
    fn document_with_metadata_preserves_metadata() {
        let metadata = json!({
            "source": "unit-test",
            "page": 2
        });
        let document = Document::with_metadata("hello", metadata.clone());

        assert_eq!(document.page_content, "hello");
        assert_eq!(document.metadata, metadata);
    }

    #[test]
    fn document_embed_pushes_page_content_into_embedder() {
        let document = Document::new("embedded text");
        let mut embedder = TextEmbedder::new();

        document.embed(&mut embedder).expect("document embeds");

        assert_eq!(embedder.parts(), &["embedded text"]);
    }
}
