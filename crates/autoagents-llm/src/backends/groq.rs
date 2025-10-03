//! Groq API client implementation for chat functionality.
//!
//! This module provides integration with Groq's LLM models through their API.

use crate::builder::LLMBuilder;
use crate::{
    builder::LLMBackend,
    chat::{StructuredOutputFormat, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
    LLMProvider,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Groq configuration for the generic provider
pub struct GroqConfig;

impl OpenAIProviderConfig for GroqConfig {
    const PROVIDER_NAME: &'static str = "Groq";
    const DEFAULT_BASE_URL: &'static str = "https://api.groq.com/openai/v1/";
    const DEFAULT_MODEL: &'static str = "llama3-8b-8192";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type Groq = OpenAICompatibleProvider<GroqConfig>;

// TODO: for groq usage goes in .x_groq.usage...
// /// Streaming response structures
// #[derive(Deserialize, Debug)]
// pub struct GroqChatStreamChunk {
//     pub choices: Vec<ChatStreamChoice>,
//     pub x_groq: Option<GroqMetadata>,
// }

// /// Streaming response structures
// #[derive(Deserialize, Debug)]
// pub struct GroqMetadata {
//     pub usage: Option<Usage>,
// }

impl Groq {
    /// Creates a new Groq client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tool_choice: Option<ToolChoice>,
        _embedding_encoding_format: Option<String>,
        _embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
    ) -> Self {
        OpenAICompatibleProvider::<GroqConfig>::new(
            api_key,
            base_url,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            top_p,
            top_k,
            tool_choice,
            reasoning_effort,
            None, // voice - not supported by Groq
            parallel_tool_calls,
            normalize_response,
            None, // embedding_encoding_format - not supported by Groq
            None, // embedding_dimensions - not supported by Groq
        )
    }
}

impl LLMProvider for Groq {}

#[async_trait]
impl CompletionProvider for Groq {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Groq completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for Groq {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Groq {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Groq API key".to_string()));
        }

        let url = format!("{}/models", GroqConfig::DEFAULT_BASE_URL);

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::Groq,
        };
        Ok(Box::new(result))
    }
}

impl LLMBuilder<Groq> {
    pub fn build(self) -> Result<Arc<Groq>, LLMError> {
        let api_key = self
            .api_key
            .ok_or_else(|| LLMError::InvalidRequest("No API key provided for Groq".to_string()))?;

        let groq = Groq::with_config(
            api_key,
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.top_p,
            self.top_k,
            self.tool_choice,
            None, // embedding_encoding_format
            None, // embedding_dimensions
            self.reasoning_effort,
            self.enable_parallel_tool_use,
            self.normalize_response,
        );

        Ok(Arc::new(groq))
    }
}
