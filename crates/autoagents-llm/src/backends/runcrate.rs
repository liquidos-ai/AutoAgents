//! Runcrate API client implementation for chat and embeddings.
//!
//! This module provides integration with Runcrate inference for open-weight
//! models through its OpenAI-compatible API. Model ids use the
//! `organization/model` form, for example `deepseek-ai/DeepSeek-V4.1-Flash`
//! or `Qwen/Qwen3-Embedding-8B`.

use crate::builder::LLMBuilder;
use crate::http::ensure_success;
use crate::{
    LLMProvider,
    builder::LLMBackend,
    chat::{StructuredOutputFormat, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Runcrate configuration for the generic provider
pub struct RuncrateConfig;

impl OpenAIProviderConfig for RuncrateConfig {
    const PROVIDER_NAME: &'static str = "Runcrate";
    const DEFAULT_BASE_URL: &'static str = "https://api.runcrate.ai/v1/";
    const DEFAULT_MODEL: &'static str = "deepseek-ai/DeepSeek-V4.1-Flash";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type Runcrate = OpenAICompatibleProvider<RuncrateConfig>;

#[derive(Serialize)]
struct RuncrateEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct RuncrateEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct RuncrateEmbeddingResponse {
    data: Vec<RuncrateEmbeddingData>,
}

impl Runcrate {
    /// Creates a new Runcrate client with the specified configuration.
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
        OpenAICompatibleProvider::<RuncrateConfig>::new(
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
            None, // voice - not supported by Runcrate
            extra_body,
            parallel_tool_calls,
            normalize_response,
            embedding_encoding_format,
            embedding_dimensions,
        )
    }
}

impl LLMProvider for Runcrate {}

impl crate::HasConfig for Runcrate {
    type Config = crate::NoConfig;
}

#[async_trait]
impl CompletionProvider for Runcrate {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Err(LLMError::ProviderError(
            "Runcrate completion not implemented, use chat instead".into(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for Runcrate {
    /// Embeds the input with the configured model, for example
    /// `Qwen/Qwen3-Embedding-8B` or `BAAI/bge-m3`.
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::missing_api_key(
                "Missing Runcrate API key".to_string(),
            ));
        }

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let body = RuncrateEmbeddingRequest {
            model: &self.model,
            input,
            encoding_format: self.embedding_encoding_format.as_deref(),
            dimensions: self.embedding_dimensions,
        };

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let resp = ensure_success(resp, "Runcrate").await?;

        let json_resp: RuncrateEmbeddingResponse = resp.json().await?;
        Ok(json_resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl ModelsProvider for Runcrate {
    /// Lists the models available on Runcrate (`/models`).
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.get(url);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }

        let resp = request.send().await?;
        let resp = ensure_success(resp, "Runcrate").await?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::Runcrate,
        };
        Ok(Box::new(result))
    }
}

impl LLMBuilder<Runcrate> {
    pub fn build(self) -> Result<Arc<Runcrate>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::invalid_request("No API key provided for Runcrate".to_string())
        })?;

        let runcrate = Runcrate::with_config(
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
            self.embedding_encoding_format,
            self.embedding_dimensions,
            self.reasoning_effort,
            self.enable_parallel_tool_use,
            self.normalize_response,
        );

        Ok(Arc::new(runcrate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use httpmock::prelude::*;

    fn test_runcrate_provider(server: &MockServer, api_key: &str, model: Option<&str>) -> Runcrate {
        Runcrate::with_config(
            api_key,
            Some(server.url("/v1")),
            model.map(str::to_string),
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

    #[test]
    fn test_runcrate_builder_requires_api_key() {
        let result = LLMBuilder::<Runcrate>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided for Runcrate"));
    }

    #[test]
    fn test_runcrate_builder_defaults() {
        let provider = LLMBuilder::<Runcrate>::new()
            .api_key("test-key")
            .build()
            .unwrap();
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, RuncrateConfig::DEFAULT_MODEL);
        assert_eq!(provider.base_url.as_str(), RuncrateConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_runcrate_builder_custom_base_url_and_model() {
        let provider = LLMBuilder::<Runcrate>::new()
            .api_key("test-key")
            .base_url("https://example.com/v1")
            .model("zai-org/GLM-5.3")
            .build()
            .unwrap();
        assert_eq!(provider.base_url.as_str(), "https://example.com/v1/");
        assert_eq!(provider.model, "zai-org/GLM-5.3");
    }

    #[tokio::test]
    async fn test_runcrate_list_models() {
        let server = MockServer::start();
        let models = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models")
                .header("authorization", "Bearer test-key");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash", "created": 1700000000}]
            }));
        });

        let provider = test_runcrate_provider(&server, "test-key", None);
        let response = provider.list_models(None).await.unwrap();

        models.assert();
        assert_eq!(
            response.get_models(),
            vec!["deepseek-ai/DeepSeek-V4.1-Flash".to_string()]
        );
        assert_eq!(response.get_backend(), "runcrate");
    }

    #[tokio::test]
    async fn test_runcrate_list_models_returns_auth_error() {
        let server = MockServer::start();
        let models = server.mock(|when, then| {
            when.method(GET).path("/v1/models");
            then.status(401).json_body(serde_json::json!({
                "error": {"message": "Invalid API key"}
            }));
        });

        let provider = test_runcrate_provider(&server, "bad-key", None);
        let err = provider.list_models(None).await.err().unwrap();

        models.assert();
        assert!(matches!(
            err,
            LLMError::AuthError {
                status_code: Some(401),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_runcrate_embed_sends_model_and_input() {
        let server = MockServer::start();
        let embeddings = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .header("authorization", "Bearer test-key")
                .json_body(serde_json::json!({
                    "model": "Qwen/Qwen3-Embedding-8B",
                    "input": ["hello"]
                }));
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5]}]
            }));
        });

        let provider = test_runcrate_provider(&server, "test-key", Some("Qwen/Qwen3-Embedding-8B"));
        let vectors = provider.embed(vec!["hello".to_string()]).await.unwrap();

        embeddings.assert();
        assert_eq!(vectors, vec![vec![0.25, -0.5]]);
    }

    #[tokio::test]
    async fn test_runcrate_embed_requires_api_key() {
        let server = MockServer::start();
        let provider = test_runcrate_provider(&server, "", None);
        let err = provider
            .embed(vec!["hello".to_string()])
            .await
            .err()
            .unwrap();
        assert!(matches!(
            err,
            LLMError::AuthError {
                status_code: None,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_runcrate_complete_returns_explicit_error() {
        let server = MockServer::start();
        let provider = test_runcrate_provider(&server, "test-key", None);
        let req = CompletionRequest {
            prompt: "hello".to_string(),
            max_tokens: None,
            temperature: None,
        };
        let err = provider.complete(&req, None).await.err().unwrap();
        assert!(matches!(err, LLMError::ProviderError(_)));
    }
}
