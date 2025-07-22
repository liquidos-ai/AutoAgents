/// Implementation of the Phind LLM provider.
/// This module provides integration with Phind's language model API.
use crate::{
    builder::LLMBuilder,
    chat::{ChatResponse, StructuredOutputFormat, Tool},
    ToolCall,
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
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::StatusCode;
use reqwest::{Client, Response};
use serde_json::{json, Value};
use std::sync::Arc;

/// Represents a Phind LLM client with configuration options.
pub struct Phind {
    /// The model identifier to use (e.g. "Phind-70B")
    pub model: String,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for controlling randomness (0.0-1.0)
    pub temperature: Option<f32>,
    /// System prompt to prepend to conversations
    pub system: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Whether to stream responses
    pub stream: Option<bool>,
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
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            model: model.unwrap_or_else(|| "Phind-70B".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            api_base_url: "https://https.extension.phind.com/agent/".to_string(),
            client: builder.build().expect("Failed to build reqwest Client"),
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
        let status = response.status();
        match status {
            StatusCode::OK => {
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
            _ => {
                let error_text = response.text().await?;
                let error_json: Value = serde_json::from_str(&error_text)
                    .unwrap_or_else(|_| json!({"error": {"message": "Unknown error"}}));

                let error_message = error_json
                    .get("error")
                    .and_then(|err| err.get("message"))
                    .and_then(|msg| msg.as_str())
                    .unwrap_or("Unexpected error from Phind")
                    .to_string();

                Err(LLMError::ProviderError(format!(
                    "APIError {status}: {error_message}"
                )))
            }
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
        let mut message_history = vec![];
        for m in messages {
            let role_str = match m.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
                ChatRole::System => "system",
            };
            message_history.push(json!({
                "content": m.content,
                "role": role_str
            }));
        }

        if let Some(system_prompt) = &self.system {
            message_history.insert(
                0,
                json!({
                    "content": system_prompt,
                    "role": "system"
                }),
            );
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
        let mut request = self
            .client
            .post(&self.api_base_url)
            .headers(headers)
            .json(&payload);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        log::debug!("Phind HTTP status: {}", response.status());

        self.interpret_response(response).await
    }

    /// Sends a chat request to Phind's API with tools.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `tools` - Optional slice of tools to use in the chat
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
        todo!()
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

impl LLMBuilder<Phind> {
    pub fn build(self) -> Result<Arc<Phind>, LLMError> {
        let phind = crate::backends::phind::Phind::new(
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.stream,
            self.top_p,
            self.top_k,
        );

        Ok(Arc::new(phind))
    }
}
