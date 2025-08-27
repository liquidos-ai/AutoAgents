//! Groq API client implementation for chat functionality.
//!
//! This module provides integration with Groq's LLM models through their API.

use crate::{
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, StructuredOutputFormat, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider, ToolCall,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Supported models for Groq API.
#[derive(Debug, Clone, Default)]
pub enum GroqModel {
    #[default]
    Llama33_70B,
    KimiK2,
}

impl From<GroqModel> for String {
    fn from(model: GroqModel) -> Self {
        match model {
            GroqModel::Llama33_70B => "llama-3.3-70b-versatile".to_string(),
            GroqModel::KimiK2 => "moonshotai/kimi-k2-instruct".to_string(),
        }
    }
}

impl From<String> for GroqModel {
    fn from(model: String) -> Self {
        match model.as_str() {
            "moonshotai/kimi-k2-instruct" => GroqModel::KimiK2,
            "llama-3.3-70b-versatile" => GroqModel::Llama33_70B,
            _ => GroqModel::Llama33_70B, // Default to Llama if unknown
        }
    }
}

/// Client for interacting with Groq's API.
pub struct Groq {
    pub api_key: String,
    pub model: GroqModel,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    client: Client,
}

#[derive(Serialize)]
struct GroqChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct GroqChatRequest<'a> {
    model: &'a str,
    messages: Vec<GroqChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct GroqChatResponse {
    choices: Vec<GroqChatChoice>,
}

#[derive(Deserialize, Debug)]
struct GroqChatChoice {
    message: GroqChatMsg,
}

#[derive(Deserialize, Debug)]
struct GroqChatMsg {
    content: String,
}

impl std::fmt::Display for GroqChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

impl ChatResponse for GroqChatResponse {
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
        todo!()
    }
}

#[allow(clippy::too_many_arguments)]
impl Groq {
    /// Creates a new Groq client with the specified configuration.
    pub fn new(
        api_key: impl Into<String>,
        model: Option<GroqModel>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_default(),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            top_p,
            top_k,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Groq {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Groq API key".to_string()));
        }

        if tools.is_some() {
            return Err(LLMError::NoToolSupport(
                "Doesn't support tools yet.".to_string(),
            ));
        }

        let mut groq_msgs: Vec<GroqChatMessage> = messages
            .iter()
            .map(|m| GroqChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                    ChatRole::System => "system",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            groq_msgs.insert(
                0,
                GroqChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let model_str = String::from(self.model.clone());
        let body = GroqChatRequest {
            model: &model_str,
            messages: groq_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: false,
            top_p: self.top_p,
            top_k: self.top_k,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Groq request payload: {json}");
            }
        }

        let mut request = self
            .client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key)
            .json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Groq HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp: GroqChatResponse = resp.json().await?;

        Ok(Box::new(json_resp))
    }
}

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
impl ModelsProvider for Groq {}

impl LLMProvider for Groq {}

impl LLMBuilder<Groq> {
    pub fn build(self) -> Result<Arc<Groq>, LLMError> {
        let api_key = self
            .api_key
            .ok_or_else(|| LLMError::InvalidRequest("No API key provided for Groq".to_string()))?;

        let model = self.model.map(GroqModel::from);

        let groq = Groq::new(
            api_key,
            model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.top_p,
            self.top_k,
        );

        Ok(Arc::new(groq))
    }
}
