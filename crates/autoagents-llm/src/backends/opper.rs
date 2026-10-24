//! Opper API client implementation for chat functionality.
//!
//! This module provides integration with the Opper LLM gateway through its
//! OpenAI-compatible API. Model ids are bare model names, for example
//! `claude-sonnet-4-6` or `gpt-5.4-mini`, and a `provider/model` id such as
//! `anthropic/claude-sonnet-4-6` pins one route.

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
use std::sync::Arc;

/// Opper configuration for the generic provider
pub struct OpperConfig;

impl OpenAIProviderConfig for OpperConfig {
    const PROVIDER_NAME: &'static str = "Opper";
    const DEFAULT_BASE_URL: &'static str = "https://api.opper.ai/v3/compat/";
    const DEFAULT_MODEL: &'static str = "claude-sonnet-4-6";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type Opper = OpenAICompatibleProvider<OpperConfig>;

impl Opper {
    /// Creates a new Opper client with the specified configuration.
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
        OpenAICompatibleProvider::<OpperConfig>::new(
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
            None, // voice - not supported by Opper
            extra_body,
            parallel_tool_calls,
            normalize_response,
            None, // embedding_encoding_format - not supported by Opper
            None, // embedding_dimensions - not supported by Opper
        )
    }
}

impl LLMProvider for Opper {}

impl crate::HasConfig for Opper {
    type Config = crate::NoConfig;
}

#[async_trait]
impl CompletionProvider for Opper {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Err(LLMError::ProviderError(
            "Opper completion not implemented, use chat instead".into(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for Opper {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Opper {
    /// Lists the Opper model catalog (`/models`) relative to the base URL.
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
        let resp = ensure_success(resp, "Opper").await?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::Opper,
        };
        Ok(Box::new(result))
    }
}

impl LLMBuilder<Opper> {
    pub fn build(self) -> Result<Arc<Opper>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::invalid_request("No API key provided for Opper".to_string())
        })?;

        let opper = Opper::with_config(
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

        Ok(Arc::new(opper))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::chat::{ChatMessage, ChatProvider};
    use httpmock::prelude::*;

    fn test_opper_provider(server: &MockServer, api_key: &str) -> Opper {
        Opper::with_config(
            api_key,
            Some(server.url("/v3/compat")),
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
    fn test_opper_builder_requires_api_key() {
        let result = LLMBuilder::<Opper>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided for Opper"));
    }

    #[test]
    fn test_opper_builder_defaults() {
        let provider = LLMBuilder::<Opper>::new()
            .api_key("test-key")
            .build()
            .unwrap();
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, OpperConfig::DEFAULT_MODEL);
        assert_eq!(provider.base_url.as_str(), OpperConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_opper_builder_custom_base_url_and_model() {
        let provider = LLMBuilder::<Opper>::new()
            .api_key("test-key")
            .base_url("https://gateway.example.com/v3/compat")
            .model("anthropic/claude-sonnet-4-6")
            .build()
            .unwrap();
        assert_eq!(
            provider.base_url.as_str(),
            "https://gateway.example.com/v3/compat/"
        );
        assert_eq!(provider.model, "anthropic/claude-sonnet-4-6");
    }

    #[tokio::test]
    async fn test_opper_list_models_uses_compat_endpoint() {
        let server = MockServer::start();
        let catalog = server.mock(|when, then| {
            when.method(GET)
                .path("/v3/compat/models")
                .header("authorization", "Bearer test-key");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"id": "claude-sonnet-4-6", "created": 1700000000}]
            }));
        });

        let provider = test_opper_provider(&server, "test-key");
        let response = provider.list_models(None).await.unwrap();

        catalog.assert();
        assert_eq!(response.get_models(), vec!["claude-sonnet-4-6".to_string()]);
        assert_eq!(response.get_backend(), "opper");
    }

    #[tokio::test]
    async fn test_opper_list_models_returns_auth_error() {
        let server = MockServer::start();
        let catalog = server.mock(|when, then| {
            when.method(GET).path("/v3/compat/models");
            then.status(401).json_body(serde_json::json!({
                "error": {"message": "Invalid API key"}
            }));
        });

        let provider = test_opper_provider(&server, "bad-key");
        let err = provider.list_models(None).await.err().unwrap();

        catalog.assert();
        assert!(matches!(
            err,
            LLMError::AuthError {
                status_code: Some(401),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_opper_list_models_returns_rate_limit() {
        let server = MockServer::start();
        let catalog = server.mock(|when, then| {
            when.method(GET).path("/v3/compat/models");
            then.status(429);
        });

        let provider = test_opper_provider(&server, "test-key");
        let err = provider.list_models(None).await.err().unwrap();

        catalog.assert();
        assert!(matches!(err, LLMError::RateLimitError { .. }));
    }

    #[tokio::test]
    async fn test_opper_chat_uses_compat_endpoint() {
        let server = MockServer::start();
        let chat = server.mock(|when, then| {
            when.method(POST)
                .path("/v3/compat/chat/completions")
                .header("authorization", "Bearer test-key")
                .body_includes("\"model\":\"claude-sonnet-4-6\"")
                .body_includes("\"response_format\"");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "{\"answer\":\"ok\"}"}
                }]
            }));
        });

        let provider = test_opper_provider(&server, "test-key");
        let messages = [ChatMessage::user().content("hello").build()];
        let schema = StructuredOutputFormat {
            name: "Answer".to_string(),
            description: None,
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            })),
            strict: Some(true),
        };
        let response = provider.chat(&messages, Some(schema)).await.unwrap();

        chat.assert();
        assert_eq!(response.text().as_deref(), Some("{\"answer\":\"ok\"}"));
    }

    #[tokio::test]
    async fn test_opper_complete_returns_explicit_error() {
        let server = MockServer::start();
        let provider = test_opper_provider(&server, "test-key");
        let req = CompletionRequest {
            prompt: "hello".to_string(),
            max_tokens: None,
            temperature: None,
        };
        let err = provider.complete(&req, None).await.err().unwrap();
        assert!(matches!(err, LLMError::ProviderError(_)));
    }

    #[tokio::test]
    async fn test_opper_list_models_without_key() {
        let server = MockServer::start();
        let catalog = server.mock(|when, then| {
            when.method(GET)
                .path("/v3/compat/models")
                .header_missing("authorization");
            then.status(200).json_body(serde_json::json!({
                "object": "list",
                "data": [{"id": "gpt-5.4-mini", "created": null}]
            }));
        });

        let provider = test_opper_provider(&server, "");
        let response = provider.list_models(None).await.unwrap();

        catalog.assert();
        assert_eq!(response.get_models(), vec!["gpt-5.4-mini".to_string()]);
    }
}
