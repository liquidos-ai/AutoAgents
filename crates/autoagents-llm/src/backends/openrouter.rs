//! OpenRouter API client implementation for chat functionality.
//!
//! This module provides integration with OpenRouter's LLM models through their API.

use crate::builder::LLMBuilder;
use crate::http::ensure_success;
use crate::{
    LLMProvider,
    builder::LLMBackend,
    chat::{StructuredOutputFormat, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    image_generation::{
        GeneratedImage, ImageGenerationProvider, ImageGenerationRequest, ImageGenerationResponse,
        merge_metadata,
    },
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Deserialize;
use std::sync::Arc;

/// OpenRouter configuration for the generic provider
pub struct OpenRouterConfig;

impl OpenAIProviderConfig for OpenRouterConfig {
    const PROVIDER_NAME: &'static str = "OpenRouter";
    const DEFAULT_BASE_URL: &'static str = "https://openrouter.ai/api/v1/";
    const DEFAULT_MODEL: &'static str = "moonshotai/kimi-k2:free";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type OpenRouter = OpenAICompatibleProvider<OpenRouterConfig>;

impl OpenRouter {
    /// Creates a new OpenRouter client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tool_choice: Option<ToolChoice>,
        extra_body: Option<serde_json::Value>,
        _embedding_encoding_format: Option<String>,
        _embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
    ) -> Self {
        OpenAICompatibleProvider::<OpenRouterConfig>::new(
            api_key,
            base_url,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            top_p,
            top_k,
            tool_choice,
            reasoning_effort,
            None, // voice - not supported by OpenRouter
            extra_body,
            parallel_tool_calls,
            normalize_response,
            None, // embedding_encoding_format - not supported by OpenRouter
            None, // embedding_dimensions - not supported by OpenRouter
        )
    }
}

impl LLMProvider for OpenRouter {}

impl crate::HasConfig for OpenRouter {
    type Config = crate::NoConfig;
}

#[async_trait]
impl CompletionProvider for OpenRouter {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "OpenRouter completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenRouter {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for OpenRouter {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::missing_api_key(
                "Missing OpenRouter API key".to_string(),
            ));
        }

        let url = format!("{}/models", OpenRouterConfig::DEFAULT_BASE_URL);

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;
        let resp = ensure_success(resp, "OpenRouter").await?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::OpenRouter,
        };
        Ok(Box::new(result))
    }
}

/// Response body for an OpenRouter image generation request.
///
/// OpenRouter returns generated images through the chat completions endpoint:
/// each choice's message carries an `images` array whose entries hold a
/// base64 data URL under `image_url.url`.
#[derive(Deserialize, Debug)]
struct OpenRouterImageResponse {
    #[serde(default)]
    choices: Vec<OpenRouterImageChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterImageChoice {
    message: OpenRouterImageMessage,
}

#[derive(Deserialize, Debug)]
struct OpenRouterImageMessage {
    /// Generated images attached to the assistant message.
    #[serde(default)]
    images: Vec<OpenRouterImageData>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterImageData {
    image_url: OpenRouterImageUrl,
}

#[derive(Deserialize, Debug)]
struct OpenRouterImageUrl {
    /// Base64 data URL, e.g. `data:image/png;base64,<...>`.
    url: String,
}

/// Parses a `data:<mime>;base64,<data>` URL into its MIME type and raw bytes.
fn parse_image_data_url(url: &str) -> Result<(String, Vec<u8>), LLMError> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| LLMError::ResponseFormatError {
            message: "OpenRouter image URL is not a base64 data URL".to_string(),
            raw_response: url.to_string(),
        })?;
    let (meta, encoded) = rest
        .split_once(',')
        .ok_or_else(|| LLMError::ResponseFormatError {
            message: "OpenRouter image data URL is missing a comma separator".to_string(),
            raw_response: url.to_string(),
        })?;
    // meta looks like "image/png;base64" — the MIME type is the first segment.
    let mime_type = meta.split(';').next().unwrap_or("image/png").to_string();
    let data = BASE64
        .decode(encoded.as_bytes())
        .map_err(|e| LLMError::ResponseFormatError {
            message: format!("Failed to base64-decode OpenRouter image data: {e}"),
            raw_response: url.to_string(),
        })?;
    Ok((mime_type, data))
}

fn validate_openrouter_image_metadata(metadata: Option<&serde_json::Value>) -> Result<(), LLMError> {
    const RESERVED_KEYS: &[&str] = &["model", "messages", "modalities"];

    let Some(serde_json::Value::Object(map)) = metadata else {
        return Ok(());
    };

    if let Some(key) = RESERVED_KEYS.iter().find(|key| map.contains_key(**key)) {
        return Err(LLMError::invalid_request(format!(
            "OpenRouter image generation metadata cannot override reserved request field `{key}`"
        )));
    }

    Ok(())
}

#[async_trait]
impl ImageGenerationProvider for OpenRouter {
    /// Generates one or more images from a prompt via OpenRouter's chat
    /// completions endpoint with the `image` output modality.
    ///
    /// Requires an image-capable model (e.g. `google/gemini-2.5-flash-image`)
    /// via `request.model` or the provider's configured model; the text default
    /// model is rejected with [`LLMError::InvalidRequest`].
    async fn generate_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::missing_api_key(
                "Missing OpenRouter API key".to_string(),
            ));
        }

        if request.prompt.trim().is_empty() {
            return Err(LLMError::invalid_request(
                "Image generation prompt must not be empty".to_string(),
            ));
        }

        let model = request.model.as_deref().unwrap_or(&self.model);

        // The provider's default model is a text model that cannot produce
        // images. Reject it explicitly instead of sending an image request that
        // OpenRouter would reject with an opaque HTTP error.
        if model == OpenRouterConfig::DEFAULT_MODEL {
            return Err(LLMError::invalid_request(
                "OpenRouter image generation requires an image-capable model; set \
                 request.model (e.g. \"google/gemini-2.5-flash-image\") — the provider \
                 default is a text model that cannot generate images"
                    .to_string(),
            ));
        }

        // Prompt text first, followed by any input images as data-URL parts.
        let mut content_parts = vec![serde_json::json!({
            "type": "text",
            "text": request.prompt,
        })];
        if let Some(input_images) = &request.input_images {
            for image in input_images {
                let data_url = format!(
                    "data:{};base64,{}",
                    image.mime_type,
                    BASE64.encode(&image.data)
                );
                content_parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": data_url },
                }));
            }
        }

        let mut body = serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": content_parts }],
            "modalities": ["image", "text"],
        });

        // Merge only provider-specific options. Core request fields are reserved because
        // they were validated above and are required for image generation.
        validate_openrouter_image_metadata(request.metadata.as_ref())?;
        merge_metadata(&mut body, request.metadata.as_ref());

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        log::debug!("OpenRouter image HTTP status: {}", resp.status());

        let resp = ensure_success(resp, "OpenRouter").await?;
        let resp_text = resp.text().await?;

        let json_resp: OpenRouterImageResponse =
            serde_json::from_str(&resp_text).map_err(|e| LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenRouter image response: {e}"),
                raw_response: resp_text.clone(),
            })?;

        let mut images = Vec::new();
        for choice in &json_resp.choices {
            for image in &choice.message.images {
                let (mime_type, data) = parse_image_data_url(&image.image_url.url)?;
                images.push(GeneratedImage {
                    mime_type,
                    data,
                    metadata: serde_json::json!({ "model": model }),
                });
            }
        }

        if images.is_empty() {
            return Err(LLMError::ProviderError(
                "No image returned by OpenRouter".to_string(),
            ));
        }

        Ok(ImageGenerationResponse {
            images,
            backend: LLMBackend::OpenRouter,
        })
    }
}

impl LLMBuilder<OpenRouter> {
    pub fn build(self) -> Result<Arc<OpenRouter>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::invalid_request("No API key provided for OpenRouter".to_string())
        })?;

        let openrouter = OpenRouter::with_config(
            api_key,
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.tool_choice,
            self.extra_body,
            None, // embedding_encoding_format
            None, // embedding_dimensions
            self.reasoning_effort,
            self.enable_parallel_tool_use,
            self.normalize_response,
        );

        Ok(Arc::new(openrouter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::image_generation::{ImageGenerationRequest, ImageInput};
    use httpmock::{Method::POST, MockServer};
    use serde_json::json;

    #[test]
    fn test_openrouter_builder_requires_api_key() {
        let result = LLMBuilder::<OpenRouter>::new().build();
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("No API key provided"));
        }
    }

    #[tokio::test]
    async fn test_openrouter_list_models_missing_key() {
        let provider = OpenRouter::with_config(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        );
        let err = provider.list_models(None).await.unwrap_err();
        assert!(err.to_string().contains("Missing OpenRouter API key"));
    }

    fn test_openrouter_provider(server: &MockServer) -> OpenRouter {
        OpenRouter::with_config(
            "secret-key",
            Some(format!("{}/api/v1", server.base_url())),
            Some("google/gemini-2.5-flash-image".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn image_request(prompt: &str) -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: prompt.to_string(),
            model: None,
            input_images: None,
            metadata: None,
        }
    }

    #[test]
    fn test_parse_image_data_url_decodes_mime_and_bytes() {
        let encoded = BASE64.encode([0x89, 0x50, 0x4e, 0x47]);
        let url = format!("data:image/png;base64,{encoded}");
        let (mime, bytes) = parse_image_data_url(&url).expect("data url should parse");
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, vec![0x89, 0x50, 0x4e, 0x47]);
    }

    #[test]
    fn test_parse_image_data_url_rejects_non_data_url() {
        let err = parse_image_data_url("https://example.com/x.png").unwrap_err();
        assert!(matches!(err, LLMError::ResponseFormatError { .. }));
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_sends_bearer_and_returns_image() {
        let server = MockServer::start();
        let expected_bytes = vec![0x89, 0x50, 0x4e, 0x47];
        let data_url = format!("data:image/png;base64,{}", BASE64.encode(&expected_bytes));
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/chat/completions")
                .header("authorization", "Bearer secret-key");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "images": [{
                            "type": "image_url",
                            "image_url": { "url": data_url }
                        }]
                    }
                }]
            }));
        });
        let provider = test_openrouter_provider(&server);

        let response = provider
            .generate_image(&image_request("a blue teapot"))
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        assert_eq!(response.backend, LLMBackend::OpenRouter);
        assert_eq!(response.images[0].mime_type, "image/png");
        assert_eq!(response.images[0].data, expected_bytes);
        mock.assert();
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_uses_request_model_override() {
        let server = MockServer::start();
        let data_url = format!("data:image/png;base64,{}", BASE64.encode([1u8, 2, 3]));
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/chat/completions")
                .header("authorization", "Bearer secret-key")
                // The overriding model must be sent in the request body.
                .body_includes("openrouter/image-model");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "images": [{ "image_url": { "url": data_url } }]
                    }
                }]
            }));
        });
        let provider = test_openrouter_provider(&server);

        let mut request = image_request("override the model");
        request.model = Some("openrouter/image-model".to_string());

        let response = provider
            .generate_image(&request)
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        mock.assert();
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_supports_input_images() {
        let server = MockServer::start();
        let data_url = format!("data:image/png;base64,{}", BASE64.encode([9u8, 9, 9]));
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/chat/completions")
                .header("authorization", "Bearer secret-key")
                // Input image should be forwarded as an image_url data-URL part.
                .body_includes("image_url")
                .body_includes("data:image/png;base64");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "images": [{ "image_url": { "url": data_url } }]
                    }
                }]
            }));
        });
        let provider = test_openrouter_provider(&server);

        let mut request = image_request("edit this image");
        request.input_images = Some(vec![ImageInput {
            mime_type: "image/png".to_string(),
            data: vec![0x01, 0x02, 0x03],
        }]);

        let response = provider
            .generate_image(&request)
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        mock.assert();
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_merges_metadata_into_body() {
        let server = MockServer::start();
        let data_url = format!("data:image/png;base64,{}", BASE64.encode([1u8, 2, 3]));
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/chat/completions")
                // Caller-provided metadata key must appear in the request body.
                .body_includes("marker-42");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": { "images": [{ "image_url": { "url": data_url } }] }
                }]
            }));
        });
        let provider = test_openrouter_provider(&server);

        let mut request = image_request("with metadata");
        request.metadata = Some(json!({ "provider": { "sort": "marker-42" } }));

        let response = provider
            .generate_image(&request)
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        mock.assert();
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_missing_api_key() {
        let provider = OpenRouter::with_config(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        );

        let err = provider
            .generate_image(&image_request("no key"))
            .await
            .expect_err("missing api key should error");

        assert!(matches!(err, LLMError::AuthError { .. }));
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_returns_error_when_no_image() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/chat/completions")
                .header("authorization", "Bearer secret-key");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "here is a description instead of an image"
                    }
                }]
            }));
        });
        let provider = test_openrouter_provider(&server);

        let err = provider
            .generate_image(&image_request("text only"))
            .await
            .expect_err("response without image should error");

        assert!(matches!(
            err,
            LLMError::ProviderError(message) if message == "No image returned by OpenRouter"
        ));
        mock.assert();
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_rejects_default_text_model() {
        // Provider built with the default (text) model and no request override.
        let provider = OpenRouter::with_config(
            "secret-key",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let err = provider
            .generate_image(&image_request("a cat"))
            .await
            .expect_err("default text model should be rejected");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message.contains("image-capable model")
        ));
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_rejects_empty_prompt() {
        let server = MockServer::start();
        let provider = test_openrouter_provider(&server);

        let err = provider
            .generate_image(&image_request("   "))
            .await
            .expect_err("empty prompt should error");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message == "Image generation prompt must not be empty"
        ));
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_rejects_metadata_model_override() {
        let server = MockServer::start();
        let provider = test_openrouter_provider(&server);

        let mut request = image_request("blocked model override");
        request.metadata = Some(json!({
            "model": "moonshotai/kimi-k2:free"
        }));

        let err = provider
            .generate_image(&request)
            .await
            .expect_err("metadata model override should be rejected");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message.contains("reserved request field `model`")
        ));
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_rejects_metadata_modalities_override() {
        let server = MockServer::start();
        let provider = test_openrouter_provider(&server);

        let mut request = image_request("blocked modalities override");
        request.metadata = Some(json!({
            "modalities": ["text"]
        }));

        let err = provider
            .generate_image(&request)
            .await
            .expect_err("metadata modalities override should be rejected");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message.contains("reserved request field `modalities`")
        ));
    }

    #[tokio::test]
    async fn test_openrouter_generate_image_rejects_metadata_messages_override() {
        let server = MockServer::start();
        let provider = test_openrouter_provider(&server);

        let mut request = image_request("blocked messages override");
        request.metadata = Some(json!({
            "messages": []
        }));

        let err = provider
            .generate_image(&request)
            .await
            .expect_err("metadata messages override should be rejected");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message.contains("reserved request field `messages`")
        ));
    }
}
