//! LiteLLM proxy client implementation.
//!
//! [LiteLLM](https://docs.litellm.ai/docs/simple_proxy) is an OpenAI-compatible
//! AI gateway in front of 100+ providers (OpenAI, Anthropic, Bedrock, Vertex AI,
//! Azure, Ollama, ...). Model ids are the `model_name` aliases configured on the
//! proxy, so any model the proxy routes to works with `.model(...)`.
//!
//! Beyond chat and streaming (shared with the other OpenAI-compatible
//! backends) this backend supports:
//! - running without an API key, for proxies started without a master key,
//! - embeddings through the proxy's `/embeddings` endpoint,
//! - image generation through `/images/generations`,
//! - model discovery through `/models`.

use crate::builder::LLMBuilder;
use crate::http::ensure_success;
use crate::image_generation::{
    GeneratedImage, ImageGenerationProvider, ImageGenerationRequest, ImageGenerationResponse,
    merge_metadata,
};
use crate::{
    LLMProvider,
    builder::LLMBackend,
    chat::{StructuredOutputFormat, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{
        ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse,
        StandardModelListResponseInner,
    },
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// LiteLLM proxy configuration for the generic provider
pub struct LiteLLMConfig;

impl OpenAIProviderConfig for LiteLLMConfig {
    const PROVIDER_NAME: &'static str = "LiteLLM";
    const DEFAULT_BASE_URL: &'static str = "http://localhost:4000/v1/";
    const DEFAULT_MODEL: &'static str = "gpt-4o-mini";
    // LiteLLM translates these OpenAI parameters for every upstream provider.
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
    // A proxy started without `master_key` accepts unauthenticated requests.
    const REQUIRES_API_KEY: bool = false;
}

pub type LiteLLM = OpenAICompatibleProvider<LiteLLMConfig>;

impl LiteLLM {
    /// Creates a new LiteLLM proxy client with the specified configuration.
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
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
    ) -> Self {
        OpenAICompatibleProvider::<LiteLLMConfig>::new(
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
            None, // voice - not used by chat requests
            extra_body,
            parallel_tool_calls,
            normalize_response,
            embedding_encoding_format,
            embedding_dimensions,
        )
    }

    fn endpoint(&self, path: &str) -> Result<url::Url, LLMError> {
        self.base_url
            .join(path)
            .map_err(|e| LLMError::HttpError(e.to_string()))
    }

    fn post(&self, path: &str) -> Result<reqwest::RequestBuilder, LLMError> {
        let mut request = self.client.post(self.endpoint(path)?);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }
        Ok(request)
    }
}

impl LLMProvider for LiteLLM {}

impl crate::HasConfig for LiteLLM {
    type Config = crate::NoConfig;
}

#[async_trait]
impl CompletionProvider for LiteLLM {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Err(LLMError::ProviderError(
            "LiteLLM completion not implemented, use chat instead".into(),
        ))
    }
}

#[derive(Serialize)]
struct LiteLLMEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// `encoding_format: "float"` returns numbers; `"base64"` returns packed little-endian f32s.
#[derive(Deserialize)]
#[serde(untagged)]
enum LiteLLMEmbedding {
    Float(Vec<f32>),
    Base64(String),
}

impl LiteLLMEmbedding {
    fn into_vec(self) -> Result<Vec<f32>, LLMError> {
        match self {
            Self::Float(values) => Ok(values),
            Self::Base64(encoded) => {
                let bytes = BASE64.decode(encoded.as_bytes()).map_err(|e| {
                    LLMError::ResponseFormatError {
                        message: format!("Invalid base64 embedding from LiteLLM: {e}"),
                        raw_response: String::default(),
                    }
                })?;
                if bytes.len() % 4 != 0 {
                    return Err(LLMError::ResponseFormatError {
                        message: "Base64 embedding length is not a multiple of 4 bytes".to_string(),
                        raw_response: String::default(),
                    });
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
        }
    }
}

#[derive(Deserialize)]
struct LiteLLMEmbeddingData {
    embedding: LiteLLMEmbedding,
}

#[derive(Deserialize)]
struct LiteLLMEmbeddingResponse {
    data: Vec<LiteLLMEmbeddingData>,
}

#[async_trait]
impl EmbeddingProvider for LiteLLM {
    /// Embeds `input` with the configured model through the proxy's
    /// `/embeddings` endpoint. The model must be an embedding model on the
    /// proxy (for example a `text-embedding-3-small` alias).
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        let body = LiteLLMEmbeddingRequest {
            model: &self.model,
            input,
            encoding_format: Some(self.embedding_encoding_format.as_deref().unwrap_or("float")),
            dimensions: self.embedding_dimensions,
        };

        let resp = self.post("embeddings")?.json(&body).send().await?;
        let resp = ensure_success(resp, "LiteLLM").await?;
        let resp_text = resp.text().await?;
        let json_resp: LiteLLMEmbeddingResponse =
            serde_json::from_str(&resp_text).map_err(|e| LLMError::ResponseFormatError {
                message: format!("Failed to decode LiteLLM embedding response: {e}"),
                raw_response: resp_text.clone(),
            })?;

        json_resp
            .data
            .into_iter()
            .map(|d| d.embedding.into_vec())
            .collect()
    }
}

#[async_trait]
impl ModelsProvider for LiteLLM {
    /// Lists the model aliases the proxy serves (`/models`). With a virtual
    /// key, the proxy returns only the models that key may call.
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let mut http_request = self.client.get(self.endpoint("models")?);
        if !self.api_key.is_empty() {
            http_request = http_request.bearer_auth(&self.api_key);
        }
        let resp = http_request.send().await?;
        let resp = ensure_success(resp, "LiteLLM").await?;

        let mut inner: StandardModelListResponseInner = resp.json().await?;
        if let Some(filter) = request.and_then(|r| r.filter.as_ref()) {
            inner.data.retain(|entry| entry.id.contains(filter));
        }

        let result = StandardModelListResponse {
            inner,
            backend: LLMBackend::LiteLLM,
        };
        Ok(Box::new(result))
    }
}

#[derive(Deserialize)]
struct LiteLLMImageData {
    b64_json: Option<String>,
    revised_prompt: Option<String>,
}

#[derive(Deserialize)]
struct LiteLLMImageResponse {
    data: Vec<LiteLLMImageData>,
    #[serde(default)]
    output_format: Option<String>,
}

#[async_trait]
impl ImageGenerationProvider for LiteLLM {
    /// Generates images through the proxy's `/images/generations` endpoint,
    /// which LiteLLM routes to OpenAI, Azure, Vertex AI Imagen, Bedrock, and
    /// other image providers. Images are requested as base64.
    ///
    /// Uses `request.model`, falling back to the configured model. Image edits
    /// (`input_images`) are rejected with [`LLMError::InvalidRequest`]; use a
    /// chat request with image inputs for that.
    async fn generate_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LLMError> {
        if request.prompt.trim().is_empty() {
            return Err(LLMError::invalid_request(
                "Image generation prompt must not be empty".to_string(),
            ));
        }
        if request
            .input_images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
        {
            return Err(LLMError::invalid_request(
                "LiteLLM image generation does not support input images".to_string(),
            ));
        }

        let model = request.model.as_deref().unwrap_or(&self.model);
        let mut body = serde_json::json!({
            "model": model,
            "prompt": request.prompt,
            "response_format": "b64_json",
            // Image models that only return base64 (Vertex AI, Gemini) reject
            // `response_format`; let the proxy drop what a provider can't take.
            "drop_params": true,
        });
        merge_metadata(&mut body, request.metadata.as_ref());

        let resp = self.post("images/generations")?.json(&body).send().await?;
        let resp = ensure_success(resp, "LiteLLM").await?;
        let resp_text = resp.text().await?;
        let json_resp: LiteLLMImageResponse =
            serde_json::from_str(&resp_text).map_err(|e| LLMError::ResponseFormatError {
                message: format!("Failed to decode LiteLLM image response: {e}"),
                raw_response: resp_text.clone(),
            })?;

        let mime_type = match json_resp.output_format.as_deref() {
            Some("jpeg") | Some("jpg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => "image/png",
        };

        let mut images = Vec::new();
        for image in json_resp.data {
            let Some(b64) = image.b64_json else { continue };
            let data =
                BASE64
                    .decode(b64.as_bytes())
                    .map_err(|e| LLMError::ResponseFormatError {
                        message: format!("Invalid base64 image from LiteLLM: {e}"),
                        raw_response: String::default(),
                    })?;
            images.push(GeneratedImage {
                mime_type: mime_type.to_string(),
                data,
                metadata: serde_json::json!({
                    "model": model,
                    "revised_prompt": image.revised_prompt,
                }),
            });
        }

        if images.is_empty() {
            return Err(LLMError::ProviderError(
                "No image returned by LiteLLM".to_string(),
            ));
        }

        Ok(ImageGenerationResponse {
            images,
            backend: LLMBackend::LiteLLM,
        })
    }
}

impl LLMBuilder<LiteLLM> {
    /// Builds a LiteLLM proxy client. The API key (a proxy master or virtual
    /// key) is optional; the base URL defaults to `http://localhost:4000/v1/`.
    pub fn build(self) -> Result<Arc<LiteLLM>, LLMError> {
        let litellm = LiteLLM::with_config(
            self.api_key.unwrap_or_default(),
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.tool_choice,
            self.extra_body,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            self.reasoning_effort,
            self.enable_parallel_tool_use,
            self.normalize_response,
        );

        Ok(Arc::new(litellm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::chat::{ChatMessage, ChatProvider};
    use httpmock::prelude::*;

    fn test_provider(server: &MockServer, api_key: &str) -> LiteLLM {
        LiteLLM::with_config(
            api_key,
            Some(server.url("/v1")),
            Some("claude-sonnet".to_string()),
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

    fn chat_completion_body(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "claude-sonnet",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11 }
        })
    }

    #[test]
    fn test_litellm_builder_defaults_without_api_key() {
        let provider = LLMBuilder::<LiteLLM>::new().build().unwrap();
        assert!(provider.api_key.is_empty());
        assert_eq!(provider.model, LiteLLMConfig::DEFAULT_MODEL);
        assert_eq!(provider.base_url.as_str(), LiteLLMConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_litellm_builder_custom_values() {
        let provider = LLMBuilder::<LiteLLM>::new()
            .api_key("sk-virtual-key")
            .base_url("https://litellm.internal.example.com/v1")
            .model("bedrock-claude")
            .build()
            .unwrap();
        assert_eq!(provider.api_key, "sk-virtual-key");
        assert_eq!(
            provider.base_url.as_str(),
            "https://litellm.internal.example.com/v1/"
        );
        assert_eq!(provider.model, "bedrock-claude");
    }

    #[test]
    fn test_litellm_backend_parses_and_displays() {
        assert_eq!(
            "litellm".parse::<LLMBackend>().unwrap(),
            LLMBackend::LiteLLM
        );
        assert_eq!(LLMBackend::LiteLLM.to_string(), "litellm");
    }

    #[tokio::test]
    async fn test_litellm_chat_without_api_key_sends_no_auth_header() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .header_missing("authorization")
                .body_includes("\"model\":\"claude-sonnet\"");
            then.status(200).json_body(chat_completion_body("pong"));
        });

        let provider = test_provider(&server, "");
        let response = provider
            .chat(&[ChatMessage::user().content("ping").build()], None)
            .await
            .unwrap();
        mock.assert();
        assert_eq!(response.text().as_deref(), Some("pong"));
    }

    #[tokio::test]
    async fn test_litellm_chat_with_virtual_key_sends_bearer() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .header("authorization", "Bearer sk-virtual-key");
            then.status(200).json_body(chat_completion_body("ok"));
        });

        let provider = test_provider(&server, "sk-virtual-key");
        provider
            .chat(&[ChatMessage::user().content("hi").build()], None)
            .await
            .unwrap();
        mock.assert();
    }

    #[tokio::test]
    async fn test_litellm_embed_uses_embeddings_endpoint() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .json_body(serde_json::json!({
                    "model": "claude-sonnet",
                    "input": ["a", "b"],
                    "encoding_format": "float"
                }));
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [
                    { "object": "embedding", "index": 0, "embedding": [0.1, 0.2] },
                    { "object": "embedding", "index": 1, "embedding": [0.3, 0.4] }
                ],
                "model": "text-embedding-3-small"
            }));
        });

        let provider = test_provider(&server, "");
        let vectors = provider
            .embed(vec!["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        mock.assert();
        assert_eq!(vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[tokio::test]
    async fn test_litellm_embed_decodes_base64_vectors() {
        let server = MockServer::start();
        let packed: Vec<u8> = [0.5f32, -1.25f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .body_includes("\"encoding_format\":\"base64\"");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{ "object": "embedding", "index": 0, "embedding": BASE64.encode(&packed) }]
            }));
        });

        let provider = LiteLLM::with_config(
            "",
            Some(server.url("/v1")),
            Some("emb-small".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("base64".to_string()),
            None,
            None,
            None,
            None,
        );
        let vectors = provider.embed(vec!["a".to_string()]).await.unwrap();
        mock.assert();
        assert_eq!(vectors, vec![vec![0.5, -1.25]]);
    }

    #[tokio::test]
    async fn test_litellm_list_models_applies_filter() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/v1/models");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [
                    { "id": "claude-sonnet", "object": "model", "created": 0, "owned_by": "openai" },
                    { "id": "bedrock-claude", "object": "model", "created": 0, "owned_by": "openai" },
                    { "id": "gpt-4o-mini", "object": "model", "created": 0, "owned_by": "openai" }
                ]
            }));
        });

        let provider = test_provider(&server, "");
        let models = provider
            .list_models(Some(&ModelListRequest {
                filter: Some("claude".to_string()),
            }))
            .await
            .unwrap();
        assert_eq!(models.get_models(), vec!["claude-sonnet", "bedrock-claude"]);
    }

    #[tokio::test]
    async fn test_litellm_list_models() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models")
                .header("authorization", "Bearer sk-virtual-key");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [
                    { "id": "claude-sonnet", "object": "model", "created": 0, "owned_by": "openai" },
                    { "id": "bedrock-llama", "object": "model", "created": 0, "owned_by": "openai" }
                ]
            }));
        });

        let provider = test_provider(&server, "sk-virtual-key");
        let models = provider.list_models(None).await.unwrap();
        mock.assert();
        assert_eq!(models.get_models(), vec!["claude-sonnet", "bedrock-llama"]);
        assert_eq!(models.get_backend(), "litellm");
    }

    #[tokio::test]
    async fn test_litellm_generate_image_decodes_b64_and_merges_metadata() {
        let server = MockServer::start();
        let png = vec![0x89, b'P', b'N', b'G'];
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/images/generations")
                .json_body(serde_json::json!({
                    "model": "imagen",
                    "prompt": "a lighthouse",
                    "response_format": "b64_json",
                    "drop_params": true,
                    "size": "1024x1024"
                }));
            then.status(200).json_body(serde_json::json!({
                "created": 0,
                "data": [{ "b64_json": BASE64.encode(&png), "revised_prompt": "a red lighthouse" }]
            }));
        });

        let provider = test_provider(&server, "");
        let response = provider
            .generate_image(&ImageGenerationRequest {
                prompt: "a lighthouse".to_string(),
                model: Some("imagen".to_string()),
                input_images: None,
                metadata: Some(serde_json::json!({ "size": "1024x1024" })),
            })
            .await
            .unwrap();
        mock.assert();
        assert_eq!(response.backend, LLMBackend::LiteLLM);
        assert_eq!(response.images.len(), 1);
        assert_eq!(response.images[0].mime_type, "image/png");
        assert_eq!(response.images[0].data, png);
        assert_eq!(
            response.images[0].metadata["revised_prompt"],
            "a red lighthouse"
        );
    }

    #[tokio::test]
    async fn test_litellm_generate_image_rejects_input_images_and_empty_prompt() {
        let server = MockServer::start();
        let provider = test_provider(&server, "");

        let edit = provider
            .generate_image(&ImageGenerationRequest {
                prompt: "make it blue".to_string(),
                model: None,
                input_images: Some(vec![crate::image_generation::ImageInput {
                    mime_type: "image/png".to_string(),
                    data: vec![1, 2, 3],
                }]),
                metadata: None,
            })
            .await;
        assert!(matches!(edit, Err(LLMError::InvalidRequest { .. })));

        let empty = provider
            .generate_image(&ImageGenerationRequest {
                prompt: "  ".to_string(),
                model: None,
                input_images: None,
                metadata: None,
            })
            .await;
        assert!(matches!(empty, Err(LLMError::InvalidRequest { .. })));
    }

    #[tokio::test]
    async fn test_litellm_surfaces_proxy_auth_errors() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/v1/chat/completions");
            then.status(401).json_body(serde_json::json!({
                "error": {
                    "message": "Authentication Error, Invalid proxy server token passed.",
                    "type": "auth_error",
                    "code": "401"
                }
            }));
        });

        let provider = test_provider(&server, "sk-wrong");
        let result = provider
            .chat(&[ChatMessage::user().content("hi").build()], None)
            .await;
        assert!(result.is_err());
    }
}
