//! Requesty API client implementation for chat functionality.
//!
//! This module provides integration with the Requesty LLM router through its
//! OpenAI-compatible API. Model ids use the `provider/model` form, for example
//! `openai/gpt-4o-mini` or `anthropic/claude-sonnet-4-5`.

use crate::builder::LLMBuilder;
use crate::http::ensure_success;
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
use std::sync::Arc;

/// Requesty configuration for the generic provider
pub struct RequestyConfig;

impl OpenAIProviderConfig for RequestyConfig {
    const PROVIDER_NAME: &'static str = "Requesty";
    const DEFAULT_BASE_URL: &'static str = "https://router.requesty.ai/v1/";
    const DEFAULT_MODEL: &'static str = "openai/gpt-4o-mini";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type Requesty = OpenAICompatibleProvider<RequestyConfig>;

impl Requesty {
    /// Creates a new Requesty client with the specified configuration.
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
        OpenAICompatibleProvider::<RequestyConfig>::new(
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
            None, // voice - not supported by Requesty
            extra_body,
            parallel_tool_calls,
            normalize_response,
            None, // embedding_encoding_format - not supported by Requesty
            None, // embedding_dimensions - not supported by Requesty
        )
    }

    /// Fetches a model list from the given path relative to the base URL.
    async fn fetch_models(&self, path: &str) -> Result<StandardModelListResponseInner, LLMError> {
        let url = self
            .base_url
            .join(path)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.get(url);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }

        let resp = request.send().await?;
        let resp = ensure_success(resp, "Requesty").await?;
        Ok(resp.json().await?)
    }
}

impl LLMProvider for Requesty {}

impl crate::HasConfig for Requesty {
    type Config = crate::NoConfig;
}

#[async_trait]
impl CompletionProvider for Requesty {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Requesty completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for Requesty {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Requesty {
    /// Lists Requesty managed models (`/models/managed`), falling back to the
    /// full catalog (`/models`) if the managed list is unavailable.
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let inner = match self.fetch_models("models/managed").await {
            Ok(inner) => inner,
            Err(err) => {
                log::debug!("Requesty managed models unavailable, using full catalog: {err}");
                self.fetch_models("models").await?
            }
        };

        let result = StandardModelListResponse {
            inner,
            backend: LLMBackend::Requesty,
        };
        Ok(Box::new(result))
    }
}

impl LLMBuilder<Requesty> {
    pub fn build(self) -> Result<Arc<Requesty>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::invalid_request("No API key provided for Requesty".to_string())
        })?;

        let requesty = Requesty::with_config(
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

        Ok(Arc::new(requesty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use httpmock::prelude::*;

    fn test_requesty_provider(server: &MockServer, api_key: &str) -> Requesty {
        Requesty::with_config(
            api_key,
            Some(server.url("/v1")),
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
        )
    }

    #[test]
    fn test_requesty_builder_requires_api_key() {
        let result = LLMBuilder::<Requesty>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided for Requesty"));
    }

    #[test]
    fn test_requesty_builder_defaults() {
        let provider = LLMBuilder::<Requesty>::new()
            .api_key("test-key")
            .build()
            .unwrap();
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, RequestyConfig::DEFAULT_MODEL);
        assert_eq!(provider.base_url.as_str(), RequestyConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_requesty_builder_custom_base_url_and_model() {
        let provider = LLMBuilder::<Requesty>::new()
            .api_key("test-key")
            .base_url("https://router.eu.requesty.ai/v1")
            .model("anthropic/claude-sonnet-4-5")
            .build()
            .unwrap();
        assert_eq!(
            provider.base_url.as_str(),
            "https://router.eu.requesty.ai/v1/"
        );
        assert_eq!(provider.model, "anthropic/claude-sonnet-4-5");
    }

    #[tokio::test]
    async fn test_requesty_list_models_uses_managed_endpoint() {
        let server = MockServer::start();
        let managed = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/managed")
                .header("authorization", "Bearer test-key");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"id": "claude-sonnet-4-5", "created": 1700000000}]
            }));
        });

        let provider = test_requesty_provider(&server, "test-key");
        let response = provider.list_models(None).await.unwrap();

        managed.assert();
        assert_eq!(response.get_models(), vec!["claude-sonnet-4-5".to_string()]);
        assert_eq!(response.get_backend(), "requesty");
    }

    #[tokio::test]
    async fn test_requesty_list_models_falls_back_to_full_catalog() {
        let server = MockServer::start();
        let managed = server.mock(|when, then| {
            when.method(GET).path("/v1/models/managed");
            then.status(500);
        });
        let catalog = server.mock(|when, then| {
            when.method(GET).path("/v1/models");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"id": "openai/gpt-4o-mini", "created": 1700000000}]
            }));
        });

        let provider = test_requesty_provider(&server, "test-key");
        let response = provider.list_models(None).await.unwrap();

        managed.assert();
        catalog.assert();
        assert_eq!(
            response.get_models(),
            vec!["openai/gpt-4o-mini".to_string()]
        );
    }

    #[tokio::test]
    async fn test_requesty_list_models_without_key() {
        let server = MockServer::start();
        let managed = server.mock(|when, then| {
            when.method(GET)
                .path("/v1/models/managed")
                .header_missing("authorization");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"id": "gpt-5.4-mini", "created": null}]
            }));
        });

        let provider = test_requesty_provider(&server, "");
        let response = provider.list_models(None).await.unwrap();

        managed.assert();
        assert_eq!(response.get_models(), vec!["gpt-5.4-mini".to_string()]);
    }
}
