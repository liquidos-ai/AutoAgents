//! OpenRouter API client implementation for chat functionality.
//!
//! This module provides integration with OpenRouter's LLM models through their API.

use crate::builder::LLMBuilder;
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

/// OpenRouter configuration for the generic provider
pub struct OpenRouterConfig;

impl OpenAIProviderConfig for OpenRouterConfig {
    const PROVIDER_NAME: &'static str = "OpenRouter";
    const DEFAULT_BASE_URL: &'static str = "https://openrouter.ai/api/v1/";
    const DEFAULT_MODEL: &'static str = "moonshotai/kimi-k2:free";
    const SUPPORTS_REASONING_EFFORT: bool = false;
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
            return Err(LLMError::AuthError(
                "Missing OpenRouter API key".to_string(),
            ));
        }

        let url = format!("{}/models", OpenRouterConfig::DEFAULT_BASE_URL);

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::OpenRouter,
        };
        Ok(Box::new(result))
    }
}

impl LLMBuilder<OpenRouter> {
    pub fn build(self) -> Result<Arc<OpenRouter>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for OpenRouter".to_string())
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
