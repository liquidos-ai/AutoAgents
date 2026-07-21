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
    /// Provider-specific request options passed through verbatim.
    pub metadata: Option<Value>,
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
