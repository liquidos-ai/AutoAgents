//! DeepSeek API client implementation for chat and completion functionality.
//!
//! This module provides integration with DeepSeek's models through their API.
//! DeepSeek uses an OpenAI-compatible API, so we leverage the OpenAICompatibleProvider.

use crate::chat::{
    ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse, StructuredOutputFormat,
    Tool,
};
use crate::providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig};
use crate::{
    LLMProvider,
    builder::LLMBuilder,
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;

/// DeepSeek configuration for the OpenAI-compatible provider
struct DeepSeekConfig;

impl OpenAIProviderConfig for DeepSeekConfig {
    const PROVIDER_NAME: &'static str = "DeepSeek";
    const DEFAULT_BASE_URL: &'static str = "https://api.deepseek.com/v1/";
    const DEFAULT_MODEL: &'static str = "deepseek-chat";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

/// Client for DeepSeek API
pub struct DeepSeek {
    provider: OpenAICompatibleProvider<DeepSeekConfig>,
}

impl DeepSeek {
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
    ) -> Self {
        Self {
            provider: OpenAICompatibleProvider::new(
                api_key,
                None, // base_url - use default
                model,
                max_tokens,
                temperature,
                timeout_seconds,
                None, // top_p
                None, // top_k
                None, // tool_choice
                None, // reasoning_effort
                None, // voice
                None, // extra_body
                None, // parallel_tool_calls
                None, // normalize_response
                None, // embedding_encoding_format
                None, // embedding_dimensions
            ),
        }
    }

    /// Creates a new DeepSeek client with extended configuration options.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        tool_choice: Option<crate::chat::ToolChoice>,
    ) -> Self {
        Self {
            provider: OpenAICompatibleProvider::new(
                api_key,
                base_url,
                model,
                max_tokens,
                temperature,
                timeout_seconds,
                top_p,
                None, // top_k
                tool_choice,
                None, // reasoning_effort
                None, // voice
                None, // extra_body
                None, // parallel_tool_calls
                None, // normalize_response
                None, // embedding_encoding_format
                None, // embedding_dimensions
            ),
        }
    }

    /// Returns the API key
    pub fn api_key(&self) -> &str {
        &self.provider.api_key
    }

    /// Returns the model name
    pub fn model(&self) -> &str {
        &self.provider.model
    }
}

#[async_trait]
impl ChatProvider for DeepSeek {
    /// Sends a chat request to DeepSeek's API.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.provider.chat(messages, json_schema).await
    }

    /// Sends a chat request to DeepSeek's API with tools.
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.provider
            .chat_with_tools(messages, tools, json_schema)
            .await
    }

    /// Stream chat responses as a stream of strings
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        self.provider.chat_stream(messages, json_schema).await
    }

    /// Stream chat responses as structured objects
    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        self.provider
            .chat_stream_struct(messages, tools, json_schema)
            .await
    }

    /// Stream chat responses with tool support
    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        self.provider
            .chat_stream_with_tools(messages, tools, json_schema)
            .await
    }
}

#[async_trait]
impl CompletionProvider for DeepSeek {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        if self.api_key().is_empty() {
            return Err(LLMError::AuthError("Missing DeepSeek API key".into()));
        }
        Err(LLMError::ProviderError(
            "DeepSeek completion not implemented yet".into(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for DeepSeek {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for DeepSeek {}

impl LLMProvider for DeepSeek {}

impl LLMBuilder<DeepSeek> {
    pub fn build(self) -> Result<Arc<DeepSeek>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for DeepSeek".to_string())
        })?;

        let deepseek = DeepSeek::new(
            api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
        );

        Ok(Arc::new(deepseek))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::completion::CompletionRequest;

    #[test]
    fn test_new_defaults() {
        let client = DeepSeek::new("key", None, None, None, None);
        assert_eq!(client.api_key(), "key");
        assert_eq!(client.model(), "deepseek-chat");
    }

    #[test]
    fn test_new_with_options_overrides() {
        let client = DeepSeek::new_with_options(
            "key",
            Some("https://example.com/v9/".to_string()),
            Some("custom".to_string()),
            Some(111),
            Some(0.3),
            Some(9),
            Some(0.8),
            None,
        );
        assert_eq!(client.provider.model, "custom");
        assert_eq!(client.provider.base_url.as_str(), "https://example.com/v9/");
        assert_eq!(client.provider.max_tokens, Some(111));
        assert_eq!(client.provider.temperature, Some(0.3));
        assert_eq!(client.provider.top_p, Some(0.8));
    }

    #[tokio::test]
    async fn test_complete_missing_key() {
        let client = DeepSeek::new("", None, None, None, None);
        let err = client
            .complete(
                &CompletionRequest {
                    prompt: "hi".to_string(),
                    max_tokens: None,
                    temperature: None,
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Missing DeepSeek API key"));
    }

    #[tokio::test]
    async fn test_embed_not_supported() {
        let client = DeepSeek::new("key", None, None, None, None);
        let err = client.embed(vec!["hello".to_string()]).await.unwrap_err();
        assert!(err.to_string().contains("Embedding not supported"));
    }

    #[test]
    fn test_builder_requires_api_key() {
        let result = LLMBuilder::<DeepSeek>::new().build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No API key provided"));
    }
}
