//! DeepSeek API client implementation for chat and completion functionality.
//!
//! This module provides integration with DeepSeek's models through their API.

use crate::chat::StructuredOutputFormat;
use crate::ToolCall;
use crate::{
    builder::LLMBuilder,
    chat::{ChatResponse, Tool},
};
use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct DeepSeek {
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    client: Client,
}

#[derive(Serialize)]
struct DeepSeekChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct DeepSeekChatRequest<'a> {
    model: &'a str,
    messages: Vec<DeepSeekChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct DeepSeekChatResponse {
    choices: Vec<DeepSeekChatChoice>,
}

impl std::fmt::Display for DeepSeekChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Deserialize, Debug)]
struct DeepSeekChatChoice {
    message: DeepSeekChatMsg,
}

#[derive(Deserialize, Debug)]
struct DeepSeekChatMsg {
    content: String,
}
impl ChatResponse for DeepSeekChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| {
            if c.message.content.is_empty() {
                None
            } else {
                Some(c.message.content.clone())
            }
        })
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }
}

impl DeepSeek {
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or("deepseek-chat".to_string()),
            max_tokens,
            temperature,
            timeout_seconds,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for DeepSeek {
    /// Sends a chat request to DeepSeek's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing DeepSeek API key".to_string()));
        }

        let deepseek_msgs: Vec<DeepSeekChatMessage> = messages
            .iter()
            .map(|m| DeepSeekChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::System => "system",
                    ChatRole::Tool => "tool",
                },
                content: &m.content,
            })
            .collect();

        let body = DeepSeekChatRequest {
            model: &self.model,
            messages: deepseek_msgs,
            temperature: self.temperature,
            stream: false,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("DeepSeek request payload: {json}");
            }
        }

        let mut request = self
            .client
            .post("https://api.deepseek.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("DeepSeek HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        let json_resp: DeepSeekChatResponse = resp.json().await?;

        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to DeepSeek's API with tools.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `tools` - Optional slice of tools to use in the chat
    /// * `json_schema` - Optional response json format
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        unimplemented!("TODO: Yet to be implented")
    }
}

#[async_trait]
impl CompletionProvider for DeepSeek {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        if self.api_key.is_empty() {
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
