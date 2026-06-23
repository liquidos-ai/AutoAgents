use crate::{
    LLMProvider,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    config::resolve_request_timeout,
    embedding::EmbeddingProvider,
    error::LLMError,
    http::ensure_success,
    models::ModelsProvider,
};
/// Implementation of the Phind LLM provider.
/// This module provides integration with Phind's language model API.
use crate::{
    ToolCall,
    builder::LLMBuilder,
    chat::{ChatResponse, StructuredOutputFormat, Tool},
};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, Response};
use serde_json::{Value, json};
use std::sync::Arc;

/// Represents a Phind LLM client with configuration options.
pub struct Phind {
    /// The model identifier to use (e.g. "Phind-70B")
    pub model: String,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for controlling randomness (0.0-1.0)
    pub temperature: Option<f32>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Base URL for the Phind API
    pub api_base_url: String,
    /// HTTP client for making requests
    client: Client,
}

#[derive(Debug)]
pub struct PhindResponse {
    content: String,
}

impl std::fmt::Display for PhindResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

impl ChatResponse for PhindResponse {
    fn text(&self) -> Option<String> {
        Some(self.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }
}

impl Phind {
    /// Creates a new Phind client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        api_base_url: Option<String>,
    ) -> Self {
        let timeout_seconds = resolve_request_timeout(timeout_seconds);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .expect("Failed to build reqwest Client");
        Self {
            model: model.unwrap_or_else(|| "Phind-70B".to_string()),
            max_tokens,
            temperature,
            timeout_seconds,
            top_p,
            top_k,
            api_base_url: api_base_url
                .unwrap_or_else(|| "https://extension.phind.com/agent/".to_string()),
            client,
        }
    }

    /// Creates the required headers for API requests.
    fn create_headers() -> Result<HeaderMap, LLMError> {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        headers.insert("User-Agent", HeaderValue::from_static(""));
        headers.insert("Accept", HeaderValue::from_static("*/*"));
        headers.insert("Accept-Encoding", HeaderValue::from_static("Identity"));
        Ok(headers)
    }

    /// Parses a single line from the streaming response.
    fn parse_line(line: &str) -> Option<String> {
        let data = line.strip_prefix("data: ")?;
        let json_value: Value = serde_json::from_str(data).ok()?;

        json_value
            .get("choices")?
            .as_array()?
            .first()?
            .get("delta")?
            .get("content")?
            .as_str()
            .map(String::from)
    }

    /// Parses the complete streaming response into a single string.
    fn parse_stream_response(response_text: &str) -> String {
        response_text
            .split('\n')
            .filter_map(Self::parse_line)
            .collect()
    }

    /// Interprets the API response and handles any errors.
    async fn interpret_response(
        &self,
        response: Response,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let response = ensure_success(response, "Phind").await?;
        let response_text = response.text().await?;
        let full_text = Self::parse_stream_response(&response_text);
        if full_text.is_empty() {
            Err(LLMError::ProviderError(
                "No completion choice returned.".to_string(),
            ))
        } else {
            Ok(Box::new(PhindResponse { content: full_text }))
        }
    }
}

/// Implementation of chat functionality for Phind.
#[async_trait]
impl ChatProvider for Phind {
    /// Sends a chat request to Phind's API.
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
        validate_text_only_messages(messages)?;

        let mut message_history = vec![];
        for m in messages {
            let role_str = match m.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::System => "system",
                ChatRole::Tool => "user",
            };
            message_history.push(json!({
                "content": m.content,
                "role": role_str
            }));
        }

        let payload = json!({
            "additional_extension_context": "",
            "allow_magic_buttons": true,
            "is_vscode_extension": true,
            "message_history": message_history,
            "requested_model": self.model,
            "user_input": messages
                .iter()
                .rev()
                .find(|m| m.role == ChatRole::User)
                .map(|m| m.content.clone())
                .unwrap_or_default(),
        });

        if log::log_enabled!(log::Level::Trace) {
            log::trace!("Phind request payload: {payload}");
        }

        let headers = Self::create_headers()?;
        let response = self
            .client
            .post(&self.api_base_url)
            .headers(headers)
            .json(&payload)
            .send()
            .await?;

        log::debug!("Phind HTTP status: {}", response.status());

        self.interpret_response(response).await
    }

    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::NoToolSupport(
            "Phind does not support tool calling".to_string(),
        ))
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Implementation of completion functionality for Phind.
#[async_trait]
impl CompletionProvider for Phind {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let chat_resp = self
            .chat(
                &[crate::chat::ChatMessage::user()
                    .content(_req.prompt.clone())
                    .build()],
                json_schema,
            )
            .await?;
        if let Some(text) = chat_resp.text() {
            Ok(CompletionResponse { text })
        } else {
            Err(LLMError::ProviderError(
                "No completion text returned by Phind".to_string(),
            ))
        }
    }
}

/// Implementation of embedding functionality for Phind.
#[cfg(feature = "phind")]
#[async_trait]
impl EmbeddingProvider for Phind {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Phind does not implement embeddings endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Phind {}

/// Implementation of the LLMProvider trait for Phind.
impl LLMProvider for Phind {}

impl crate::HasConfig for Phind {
    type Config = crate::NoConfig;
}

fn validate_text_only_messages(messages: &[ChatMessage]) -> Result<(), LLMError> {
    for msg in messages {
        match &msg.message_type {
            MessageType::Text => {}
            MessageType::ToolUse(_) | MessageType::ToolResult(_) => {
                return Err(LLMError::NoToolSupport(
                    "Phind does not support tool calling".to_string(),
                ));
            }
            _ => {
                return Err(LLMError::invalid_request(
                    "Multimodal input is not supported by the Phind backend".to_string(),
                ));
            }
        }
    }
    Ok(())
}

impl LLMBuilder<Phind> {
    pub fn build(self) -> Result<Arc<Phind>, LLMError> {
        let phind = Phind::new(
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.base_url,
        );

        Ok(Arc::new(phind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionCall;

    #[test]
    fn test_create_headers() {
        let headers = Phind::create_headers().unwrap();
        assert_eq!(
            headers.get("Content-Type").unwrap(),
            &HeaderValue::from_static("application/json")
        );
        assert_eq!(
            headers.get("Accept").unwrap(),
            &HeaderValue::from_static("*/*")
        );
    }

    #[test]
    fn test_parse_line_and_stream_response() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(Phind::parse_line(line), Some("Hello".to_string()));

        let response_text = [
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"data: {"choices":[{"delta":{"content":" "}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"World"}}]}"#,
        ]
        .join("\n");
        assert_eq!(Phind::parse_stream_response(&response_text), "Hello World");
    }

    #[test]
    fn test_validate_text_only_messages_rejects_multimodal() {
        let messages = [ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::ImageURL("https://example.com/image.png".to_string()),
            content: "describe".to_string(),
        }];

        let err = validate_text_only_messages(&messages).expect_err("image URL should be rejected");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message == "Multimodal input is not supported by the Phind backend"
        ));
    }

    #[test]
    fn test_validate_text_only_messages_rejects_tool_messages() {
        let messages = [ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::ToolUse(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "lookup".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            content: "tool".to_string(),
        }];

        let err = validate_text_only_messages(&messages).expect_err("tool use should be rejected");

        assert!(matches!(
            err,
            LLMError::NoToolSupport(message)
                if message == "Phind does not support tool calling"
        ));
    }

    #[tokio::test]
    async fn test_chat_with_tools_returns_no_tool_support() {
        let provider = Phind::new(None, None, None, None, None, None, None);
        let messages = [ChatMessage::user().content("hello").build()];

        let err = provider
            .chat_with_tools(&messages, None, None)
            .await
            .expect_err("Phind should report unsupported tool calling");

        assert!(matches!(
            err,
            LLMError::NoToolSupport(message)
                if message == "Phind does not support tool calling"
        ));
    }
}
