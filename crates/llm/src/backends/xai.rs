//! X.AI API client implementation for chat and completion functionality.
//!
//! This module provides integration with X.AI's models through their API.
//! It implements chat and completion capabilities using the X.AI API endpoints.

use crate::{
    builder::LLMBuilder,
    chat::{ChatResponse, Tool},
    ToolCall,
};
use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider,
};
use async_trait::async_trait;
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Client for interacting with X.AI's API.
///
/// This struct provides methods for making chat and completion requests to X.AI's language models.
/// It handles authentication, request configuration, and response parsing.
pub struct XAI {
    /// API key for authentication with X.AI services
    pub api_key: String,
    /// Model identifier to use for requests (e.g. "grok-2-latest")
    pub model: String,
    /// Maximum number of tokens to generate in responses
    pub max_tokens: Option<u32>,
    /// Temperature parameter for controlling response randomness (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Optional system prompt to provide context
    pub system: Option<String>,
    /// Request timeout duration in seconds
    pub timeout_seconds: Option<u64>,
    /// Top-p sampling parameter for controlling response diversity
    pub top_p: Option<f32>,
    /// Top-k sampling parameter for controlling response diversity
    pub top_k: Option<u32>,
    /// Embedding encoding format
    pub embedding_encoding_format: Option<String>,
    /// Embedding dimensions
    pub embedding_dimensions: Option<u32>,
    /// XAI search parameters
    pub xai_search_mode: Option<String>,
    /// XAI search sources
    pub xai_search_source_type: Option<String>,
    /// XAI search excluded websites
    pub xai_search_excluded_websites: Option<Vec<String>>,
    /// XAI search max results
    pub xai_search_max_results: Option<u32>,
    /// XAI search from date
    pub xai_search_from_date: Option<String>,
    /// XAI search to date
    pub xai_search_to_date: Option<String>,
    /// HTTP client for making API requests
    client: Client,
}

/// Search source configuration for search parameters
#[derive(Debug, Clone, serde::Serialize)]
pub struct XaiSearchSource {
    /// Type of source: "web" or "news"
    #[serde(rename = "type")]
    pub source_type: String,
    /// List of websites to exclude from this source
    pub excluded_websites: Option<Vec<String>>,
}

/// Search parameters for LLM providers that support search functionality
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct XaiSearchParameters {
    /// Search mode (e.g., "auto")
    pub mode: Option<String>,
    /// List of search sources with exclusions
    pub sources: Option<Vec<XaiSearchSource>>,
    /// Maximum number of search results to return
    pub max_search_results: Option<u32>,
    /// Start date for search results (format: "YYYY-MM-DD")
    pub from_date: Option<String>,
    /// End date for search results (format: "YYYY-MM-DD")
    pub to_date: Option<String>,
}

/// Individual message in an X.AI chat conversation.
#[derive(Serialize)]
struct XAIChatMessage<'a> {
    /// Role of the message sender (user, assistant, or system)
    role: &'a str,
    /// Content of the message
    content: &'a str,
}

/// Request payload for X.AI's chat API endpoint.
#[derive(Serialize)]
struct XAIChatRequest<'a> {
    /// Model identifier to use
    model: &'a str,
    /// Array of conversation messages
    messages: Vec<XAIChatMessage<'a>>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Temperature parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Whether to stream the response
    stream: bool,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<XAIResponseFormat>,
    /// Search parameters for search functionality
    #[serde(skip_serializing_if = "Option::is_none")]
    search_parameters: Option<&'a XaiSearchParameters>,
}

/// Response from X.AI's chat API endpoint.
#[derive(Deserialize, Debug)]
struct XAIChatResponse {
    /// Array of generated responses
    choices: Vec<XAIChatChoice>,
}

impl std::fmt::Display for XAIChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

impl ChatResponse for XAIChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().map(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }
}

/// Individual response choice from the chat API.
#[derive(Deserialize, Debug)]
struct XAIChatChoice {
    /// Message content and metadata
    message: XAIChatMsg,
}

/// Message content from a chat response.
#[derive(Deserialize, Debug)]
struct XAIChatMsg {
    /// Generated text content
    content: String,
}

#[derive(Debug, Serialize)]
struct XAIEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct XAIEmbeddingData {
    embedding: Vec<f32>,
}

/// Response from X.AI's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct XAIStreamResponse {
    /// Array of generated responses
    choices: Vec<XAIStreamChoice>,
}

/// Individual response choice from the streaming chat API.
#[derive(Deserialize, Debug)]
struct XAIStreamChoice {
    /// Delta content
    delta: XAIStreamDelta,
}

/// Delta content from a streaming chat response.
#[derive(Deserialize, Debug)]
struct XAIStreamDelta {
    /// Generated text content
    content: Option<String>,
}

#[derive(Deserialize)]
struct XAIEmbeddingResponse {
    data: Vec<XAIEmbeddingData>,
}

#[derive(Deserialize, Debug, Serialize)]
enum XAIResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

/// An object specifying the format that the model must output.
/// Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
/// Setting to `{ "type": "json_object" }` enables the older JSON mode, which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for models that support it.
/// The structured outputs feature is only supported for the `grok-2-latest` model.
#[derive(Deserialize, Debug, Serialize)]
struct XAIResponseFormat {
    #[serde(rename = "type")]
    response_type: XAIResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

impl XAI {
    /// Creates a new X.AI client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Authentication key for X.AI API access
    /// * `model` - Model identifier (defaults to "grok-2-latest" if None)
    /// * `max_tokens` - Maximum number of tokens to generate in responses
    /// * `temperature` - Sampling temperature for controlling randomness
    /// * `timeout_seconds` - Request timeout duration in seconds
    /// * `system` - System prompt for providing context
    /// * `stream` - Whether to enable streaming responses
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    /// * `json_schema` - JSON schema for structured output
    /// * `search_parameters` - Search parameters for search functionality
    ///
    /// # Returns
    ///
    /// A configured X.AI client instance ready to make API requests.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        xai_search_mode: Option<String>,
        xai_search_source_type: Option<String>,
        xai_search_excluded_websites: Option<Vec<String>>,
        xai_search_max_results: Option<u32>,
        xai_search_from_date: Option<String>,
        xai_search_to_date: Option<String>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or("grok-2-latest".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            top_p,
            top_k,
            embedding_encoding_format,
            embedding_dimensions,
            xai_search_mode,
            xai_search_source_type,
            xai_search_excluded_websites,
            xai_search_max_results,
            xai_search_from_date,
            xai_search_to_date,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }

    /// Sets the search mode for search-enabled providers.
    pub fn set_search_mode(mut self, mode: impl Into<String>) -> Self {
        self.xai_search_mode = Some(mode.into());
        self
    }

    /// Adds a search source with optional excluded websites.
    pub fn set_search_source(
        mut self,
        source_type: impl Into<String>,
        excluded_websites: Option<Vec<String>>,
    ) -> Self {
        self.xai_search_source_type = Some(source_type.into());
        self.xai_search_excluded_websites = excluded_websites;
        self
    }

    /// Sets the maximum number of search results.
    pub fn set_max_search_results(mut self, max: u32) -> Self {
        self.xai_search_max_results = Some(max);
        self
    }

    /// Sets the date range for search results.
    pub fn set_search_date_range(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.xai_search_from_date = Some(from.into());
        self.xai_search_to_date = Some(to.into());
        self
    }

    /// Sets the start date for search results (format: "YYYY-MM-DD").
    pub fn set_search_from_date(mut self, date: impl Into<String>) -> Self {
        self.xai_search_from_date = Some(date.into());
        self
    }

    /// Sets the end date for search results (format: "YYYY-MM-DD").
    pub fn set_search_to_date(mut self, date: impl Into<String>) -> Self {
        self.xai_search_to_date = Some(date.into());
        self
    }
}

#[async_trait]
impl ChatProvider for XAI {
    /// Sends a chat request to the X.AI API and returns the response.
    ///
    /// # Arguments
    ///
    /// * `messages` - Array of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// The generated response text, or an error if the request fails.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>, // XAI Does not support tools yet
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".to_string()));
        }

        let mut xai_msgs: Vec<XAIChatMessage> = messages
            .iter()
            .map(|m| XAIChatMessage {
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
            xai_msgs.insert(
                0,
                XAIChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        // OpenAI's structured output has some [odd requirements](https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat&lang=curl#supported-schemas).
        // There's currently no check for these, so we'll leave it up to the user to provide a valid schema.
        // Unknown if XAI requires these too, but since it copies everything else from OpenAI, it's likely.
        let response_format: Option<XAIResponseFormat> =
            json_schema.as_ref().map(|s| XAIResponseFormat {
                response_type: XAIResponseType::JsonSchema,
                json_schema: Some(s.clone()),
            });

        let search_parameters = XaiSearchParameters {
            mode: self.xai_search_mode.clone(),
            sources: Some(vec![XaiSearchSource {
                source_type: self
                    .xai_search_source_type
                    .clone()
                    .unwrap_or("web".to_string()),
                excluded_websites: self.xai_search_excluded_websites.clone(),
            }]),
            max_search_results: self.xai_search_max_results,
            from_date: self.xai_search_from_date.clone(),
            to_date: self.xai_search_to_date.clone(),
        };

        let body = XAIChatRequest {
            model: &self.model,
            messages: xai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: false,
            top_p: self.top_p,
            top_k: self.top_k,
            response_format,
            search_parameters: Some(&search_parameters),
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("XAI request payload: {json}");
            }
        }

        let mut request = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("XAI HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        let json_resp: XAIChatResponse = resp.json().await?;
        Ok(Box::new(json_resp))
    }

    /// Sends a streaming chat request to X.AI's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".to_string()));
        }

        let mut xai_msgs: Vec<XAIChatMessage> = messages
            .iter()
            .map(|m| XAIChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::System => "system",
                    ChatRole::Tool => "tool",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            xai_msgs.insert(
                0,
                XAIChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = XAIChatRequest {
            model: &self.model,
            messages: xai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            response_format: None,
            search_parameters: None,
        };

        let mut request = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("X.AI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(
            response,
            parse_xai_sse_chunk,
        ))
    }
}

#[async_trait]
impl CompletionProvider for XAI {
    /// Sends a completion request to X.AI's API.
    ///
    /// This functionality is currently not implemented.
    ///
    /// # Arguments
    ///
    /// * `_req` - The completion request parameters
    ///
    /// # Returns
    ///
    /// A placeholder response indicating the functionality is not implemented.
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "X.AI completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for XAI {
    async fn embed(&self, text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".into()));
        }

        let emb_format = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = XAIEmbeddingRequest {
            model: &self.model,
            input: text,
            encoding_format: Some(&emb_format),
            dimensions: self.embedding_dimensions,
        };

        let resp = self
            .client
            .post("https://api.x.ai/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: XAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[async_trait]
impl ModelsProvider for XAI {}

impl LLMProvider for XAI {}

/// Parses a Server-Sent Events (SSE) chunk from X.AI's streaming API.
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
///
/// # Returns
///
/// * `Ok(Some(String))` - Content token if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
fn parse_xai_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();

        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                return Ok(None);
            }

            match serde_json::from_str::<XAIStreamResponse>(data) {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            return Ok(Some(content.clone()));
                        }
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }

    Ok(None)
}

impl LLMBuilder<XAI> {
    pub fn build(self) -> Result<Arc<XAI>, LLMError> {
        let api_key = self
            .api_key
            .ok_or_else(|| LLMError::InvalidRequest("No API key provided for XAI".to_string()))?;

        let xai = crate::backends::xai::XAI::new(
            api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.top_p,
            self.top_k,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        Ok(Arc::new(xai))
    }
}
