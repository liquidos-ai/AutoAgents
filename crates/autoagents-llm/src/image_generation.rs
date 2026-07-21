use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{builder::LLMBackend, error::LLMError};

#[async_trait]
pub trait ImageGenerationProvider {
    async fn generate_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LLMError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the image to generate (or the edit to apply).
    pub prompt: String,
    /// Optional model override; falls back to the provider's configured model.
    pub model: Option<String>,
    /// Optional input images for image-editing requests.
    pub input_images: Option<Vec<ImageInput>>,
    /// Optional provider-specific request options. When this is a JSON object,
    /// its top-level keys are merged into the outgoing request body (overriding
    /// defaults), letting callers set fields such as a provider's image count or
    /// generation config. Non-object values are ignored.
    pub metadata: Option<Value>,
}

/// Merges caller-provided [`ImageGenerationRequest::metadata`] into a request body.
///
/// When both `body` and `metadata` are JSON objects, each top-level metadata key
/// is inserted into `body`, overriding any existing key. Anything else is a no-op.
pub(crate) fn merge_metadata(body: &mut Value, metadata: Option<&Value>) {
    if let (Some(obj), Some(Value::Object(extra))) = (body.as_object_mut(), metadata) {
        for (key, value) in extra {
            obj.insert(key.clone(), value.clone());
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    pub mime_type: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub images: Vec<GeneratedImage>,
    pub backend: LLMBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub mime_type: String,
    pub data: Vec<u8>,
    pub metadata: Value,
}

#[cfg(test)]
mod tests {
    use super::merge_metadata;
    use serde_json::json;

    #[test]
    fn test_merge_metadata_inserts_and_overrides_keys() {
        let mut body = json!({ "model": "m", "keep": 1 });
        merge_metadata(
            &mut body,
            Some(&json!({ "extra": "x", "model": "override" })),
        );
        assert_eq!(body["extra"], json!("x"));
        assert_eq!(body["model"], json!("override"));
        assert_eq!(body["keep"], json!(1));
    }

    #[test]
    fn test_merge_metadata_ignores_none_and_non_objects() {
        let mut body = json!({ "a": 1 });
        merge_metadata(&mut body, None);
        merge_metadata(&mut body, Some(&json!("not-an-object")));
        assert_eq!(body, json!({ "a": 1 }));
    }
}
