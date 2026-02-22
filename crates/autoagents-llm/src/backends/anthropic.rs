//! Anthropic API client implementation for chat and completion functionality.
//!
//! This module provides integration with Anthropic's Claude models through their API.

use std::{collections::HashMap, sync::Arc};

use crate::{
    FunctionCall, ToolCall,
    builder::{LLMBackend, LLMBuilder},
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChunk,
        StructuredOutputFormat, Tool, ToolChoice, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for interacting with Anthropic's API.
///
/// Provides methods for chat and completion requests using Anthropic's models.
#[derive(Debug)]
pub struct Anthropic {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: bool,
    pub thinking_budget_tokens: Option<u32>,
    client: Client,
}

/// Anthropic-specific tool format that matches their API structure
#[derive(Serialize, Debug)]
struct AnthropicTool<'a> {
    name: &'a str,
    description: &'a str,
    #[serde(rename = "input_schema")]
    schema: &'a serde_json::Value,
}

/// Configuration for the thinking feature
#[derive(Serialize, Debug)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

/// Request payload for Anthropic's messages API endpoint.
#[derive(Serialize, Debug)]
struct AnthropicCompleteRequest<'a> {
    messages: Vec<AnthropicMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

/// Individual message in an Anthropic chat conversation.
#[derive(Serialize, Debug)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: Vec<MessageContent<'a>>,
}

#[derive(Serialize, Debug)]
struct MessageContent<'a> {
    #[serde(rename = "type")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ImageSource<'a>>,
    // tool use
    #[serde(skip_serializing_if = "Option::is_none", rename = "id")]
    tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "name")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "input")]
    tool_input: Option<Value>,
    // tool result
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_use_id")]
    tool_result_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<String>,
}

#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

#[derive(Serialize, Debug)]
struct ImageSource<'a> {
    #[serde(rename = "type")]
    source_type: &'a str,
    media_type: &'a str,
    data: String,
}

/// Response from Anthropic's messages API endpoint.
#[derive(Deserialize, Debug)]
struct AnthropicCompleteResponse {
    content: Vec<AnthropicContent>,
    usage: Option<AnthropicUsage>,
}

/// Usage information from Anthropic API response.
#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<u32>,
}

/// Content block within an Anthropic API response.
#[derive(Serialize, Deserialize, Debug)]
struct AnthropicContent {
    text: Option<String>,
    #[serde(rename = "type")]
    content_type: Option<String>,
    thinking: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
    id: Option<String>,
}

/// Response from Anthropic's streaming messages API endpoint.
/// https://docs.anthropic.com/en/api/messages-streaming
/// https://docs.anthropic.com/claude/reference/messages-streaming
#[derive(Deserialize, Debug)]
struct AnthropicStreamResponse {
    #[serde(rename = "type")]
    response_type: String,
    /// Index of the content block (for content_block_start, content_block_delta, content_block_stop)
    index: Option<usize>,
    /// Content block for content_block_start events
    content_block: Option<AnthropicStreamContentBlock>,
    /// Delta for content_block_delta and message_delta events
    delta: Option<AnthropicDelta>,
}

/// Content block within an Anthropic streaming content_block_start event.
#[derive(Deserialize, Debug)]
struct AnthropicStreamContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    /// Tool use ID (for tool_use blocks)
    id: Option<String>,
    /// Tool name (for tool_use blocks)
    name: Option<String>,
    /// Initial text (for text blocks, usually empty)
    #[allow(dead_code)]
    text: Option<String>,
}

/// Delta content within an Anthropic streaming response.
#[derive(Deserialize, Debug)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    /// Text content (for text_delta)
    text: Option<String>,
    /// Partial JSON string (for input_json_delta)
    partial_json: Option<String>,
    /// Stop reason (for message_delta)
    stop_reason: Option<String>,
}

impl std::fmt::Display for AnthropicCompleteResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for content in self.content.iter() {
            match content.content_type {
                Some(ref t) if t == "tool_use" => write!(
                    f,
                    "{{\n \"name\": {}, \"input\": {}\n}}",
                    content.name.clone().unwrap_or_default(),
                    content.input.clone().unwrap_or(serde_json::Value::Null)
                )?,
                Some(ref t) if t == "thinking" => {
                    write!(f, "{}", content.thinking.clone().unwrap_or_default())?
                }
                _ => write!(
                    f,
                    "{}",
                    self.content
                        .iter()
                        .map(|c| c.text.clone().unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join("\n")
                )?,
            }
        }
        Ok(())
    }
}

impl ChatResponse for AnthropicCompleteResponse {
    fn text(&self) -> Option<String> {
        Some(
            self.content
                .iter()
                .filter_map(|c| {
                    if c.content_type == Some("text".to_string()) || c.content_type.is_none() {
                        c.text.clone()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    fn thinking(&self) -> Option<String> {
        self.content
            .iter()
            .find(|c| c.content_type == Some("thinking".to_string()))
            .and_then(|c| c.thinking.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        match self
            .content
            .iter()
            .filter_map(|c| {
                if c.content_type == Some("tool_use".to_string()) {
                    Some(ToolCall {
                        id: c.id.clone().unwrap_or_default(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: c.name.clone().unwrap_or_default(),
                            arguments: serde_json::to_string(
                                &c.input.clone().unwrap_or(serde_json::Value::Null),
                            )
                            .unwrap_or_default(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<ToolCall>>()
        {
            v if v.is_empty() => None,
            v => Some(v),
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|anthropic_usage| {
            let cached_tokens = anthropic_usage.cache_creation_input_tokens.unwrap_or(0)
                + anthropic_usage.cache_read_input_tokens.unwrap_or(0);
            Usage {
                prompt_tokens: anthropic_usage.input_tokens,
                completion_tokens: anthropic_usage.output_tokens,
                total_tokens: anthropic_usage.input_tokens + anthropic_usage.output_tokens,
                completion_tokens_details: None,
                prompt_tokens_details: if cached_tokens > 0 {
                    Some(crate::chat::PromptTokensDetails {
                        cached_tokens: Some(cached_tokens),
                        audio_tokens: None,
                    })
                } else {
                    None
                },
            }
        })
    }
}

impl Anthropic {
    /// Converts a slice of ChatMessage into Anthropic's message format.
    ///
    /// This helper method handles all message types including text, images, PDFs,
    /// tool use, and tool results.
    fn convert_messages_to_anthropic<'a>(messages: &'a [ChatMessage]) -> Vec<AnthropicMessage<'a>> {
        messages
            .iter()
            .filter(|m| m.role != ChatRole::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::System => unreachable!("system messages are filtered before mapping"),
                    ChatRole::Tool => "user",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(raw_bytes) => {
                        vec![MessageContent {
                            message_type: Some("document"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: "application/pdf",
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    MessageType::ImageURL(url) => vec![MessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url }),
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|c| MessageContent {
                            message_type: Some("tool_use"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: Some(c.id.clone()),
                            tool_input: Some(
                                serde_json::from_str(&c.function.arguments)
                                    .unwrap_or_else(|_| c.function.arguments.clone().into()),
                            ),
                            tool_name: Some(c.function.name.clone()),
                            tool_result_id: None,
                            tool_output: None,
                        })
                        .collect(),
                    MessageType::ToolResult(responses) => responses
                        .iter()
                        .map(|r| MessageContent {
                            message_type: Some("tool_result"),
                            text: None,
                            image_url: None,
                            source: None,
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: Some(r.id.clone()),
                            tool_output: Some(r.function.arguments.clone()),
                        })
                        .collect(),
                },
            })
            .collect()
    }

    /// Prepares Anthropic tools and tool_choice from the provided tools and instance configuration.
    ///
    /// Returns a tuple of (anthropic_tools, final_tool_choice) ready for the API request.
    fn prepare_tools_and_choice<'a>(
        tools: Option<&'a [Tool]>,
        tool_choice: &Option<ToolChoice>,
    ) -> (
        Option<Vec<AnthropicTool<'a>>>,
        Option<HashMap<String, String>>,
    ) {
        let anthropic_tools = tools.map(|slice| {
            slice
                .iter()
                .map(|tool| AnthropicTool {
                    name: &tool.function.name,
                    description: &tool.function.description,
                    schema: &tool.function.parameters,
                })
                .collect::<Vec<_>>()
        });

        let tool_choice_map = match tool_choice {
            Some(ToolChoice::Auto) => {
                Some(HashMap::from([("type".to_string(), "auto".to_string())]))
            }
            Some(ToolChoice::Any) => Some(HashMap::from([("type".to_string(), "any".to_string())])),
            Some(ToolChoice::Tool(tool_name)) => Some(HashMap::from([
                ("type".to_string(), "tool".to_string()),
                ("name".to_string(), tool_name.clone()),
            ])),
            Some(ToolChoice::None) => {
                Some(HashMap::from([("type".to_string(), "none".to_string())]))
            }
            None => None,
        };

        let final_tool_choice = if anthropic_tools.is_some() {
            tool_choice_map
        } else {
            None
        };

        (anthropic_tools, final_tool_choice)
    }

    /// Creates a new Anthropic client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key for authentication
    /// * `model` - Model identifier (defaults to "claude-3-sonnet-20240229")
    /// * `max_tokens` - Maximum tokens in response (defaults to 300)
    /// * `temperature` - Sampling temperature (defaults to 0.7)
    /// * `timeout_seconds` - Request timeout in seconds (defaults to 30)
    /// *
    /// * `thinking_budget_tokens` - Budget tokens for thinking (optional)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tool_choice: Option<ToolChoice>,
        reasoning: Option<bool>,
        thinking_budget_tokens: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| "claude-3-sonnet-20240229".to_string()),
            max_tokens: max_tokens.unwrap_or(300),
            temperature: temperature.unwrap_or(0.7),
            timeout_seconds: timeout_seconds.unwrap_or(30),
            top_p,
            top_k,
            tool_choice,
            reasoning: reasoning.unwrap_or(false),
            thinking_budget_tokens,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Anthropic {
    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    /// * `json_schema` - Optional json_schema for the response format
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages = Self::convert_messages_to_anthropic(messages);
        if anthropic_messages.is_empty() {
            return Err(LLMError::InvalidRequest(
                "At least one non-system message is required".to_string(),
            ));
        }
        let (anthropic_tools, final_tool_choice) =
            Self::prepare_tools_and_choice(tools, &self.tool_choice);

        let thinking = if self.reasoning {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.thinking_budget_tokens.unwrap_or(16000),
            })
        } else {
            None
        };

        let system_message = messages
            .iter()
            .find(|msg| msg.role == ChatRole::System)
            .map(|msg| msg.content.as_str());

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: system_message,
            stream: Some(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: anthropic_tools,
            tool_choice: final_tool_choice,
            thinking,
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(&req_body)
        {
            log::trace!("Anthropic request payload: {}", json);
        }

        log::debug!("Anthropic request: POST /v1/messages");
        let resp = request.send().await?;
        log::debug!("Anthropic HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        let body = resp.text().await?;
        let json_resp: AnthropicCompleteResponse = serde_json::from_str(&body)
            .map_err(|e| LLMError::HttpError(format!("Failed to parse JSON: {e}")))?;
        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `json_schema` - Optional json_schema for the response format
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None, json_schema).await
    }

    /// Sends a streaming chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `json_schema` - Optional json_schema for the response format
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| m.role != ChatRole::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::System => unreachable!("system messages are filtered before mapping"),
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "user",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Pdf(_) => unimplemented!(),
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: image_mime.mime_type(),
                                data: BASE64.encode(raw_bytes),
                            }),
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: None,
                            tool_output: None,
                        }]
                    }
                    _ => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                },
            })
            .collect();

        if anthropic_messages.is_empty() {
            return Err(LLMError::InvalidRequest(
                "At least one non-system message is required".to_string(),
            ));
        }

        let system_message = messages
            .iter()
            .find(|msg| msg.role == ChatRole::System)
            .map(|msg| msg.content.as_str());

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: system_message,
            stream: Some(true),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: None,
            tool_choice: None,
            thinking: None,
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Anthropic API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(crate::chat::create_sse_stream(
            response,
            parse_anthropic_sse_chunk,
        ))
    }

    /// Sends a streaming chat request with tool support.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools available for the model to use
    /// * `json_schema` - Optional json_schema for the response format
    ///
    ///
    /// # Returns
    ///
    /// A stream of `StreamChunk` items or an error
    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages = Self::convert_messages_to_anthropic(messages);
        if anthropic_messages.is_empty() {
            return Err(LLMError::InvalidRequest(
                "At least one non-system message is required".to_string(),
            ));
        }
        let (anthropic_tools, final_tool_choice) =
            Self::prepare_tools_and_choice(tools, &self.tool_choice);

        let system_message = messages
            .iter()
            .find(|msg| msg.role == ChatRole::System)
            .map(|msg| msg.content.as_str());

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: system_message,
            stream: Some(true),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: anthropic_tools,
            tool_choice: final_tool_choice,
            thinking: None, // Thinking not supported with streaming tools
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(&req_body)
        {
            log::trace!("Anthropic streaming request payload: {}", json);
        }

        log::debug!("Anthropic request: POST /v1/messages (streaming with tools)");
        let response = request.send().await?;
        log::debug!("Anthropic HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Anthropic API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(create_anthropic_tool_stream(response))
    }
}

/// Creates an SSE stream that parses Anthropic tool use events into StreamChunk.
fn create_anthropic_tool_stream(
    response: reqwest::Response,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>> {
    use futures::stream::StreamExt;

    let stream = response
        .bytes_stream()
        .scan(
            (String::default(), Vec::default(), HashMap::default()),
            move |(buffer, utf8_buffer, tool_states), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        utf8_buffer.extend_from_slice(&bytes);

                        match String::from_utf8(utf8_buffer.clone()) {
                            Ok(text) => {
                                buffer.push_str(&text);
                                utf8_buffer.clear();
                            }
                            Err(e) => {
                                let valid_up_to = e.utf8_error().valid_up_to();
                                if valid_up_to > 0 {
                                    let valid =
                                        String::from_utf8_lossy(&utf8_buffer[..valid_up_to]);
                                    buffer.push_str(&valid);
                                    utf8_buffer.drain(..valid_up_to);
                                }
                            }
                        }

                        let mut results = Vec::new();

                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos + 2].to_string();
                            buffer.drain(..pos + 2);

                            match parse_anthropic_sse_chunk_with_tools(&event, tool_states) {
                                Ok(Some(chunk)) => results.push(Ok(chunk)),
                                Ok(None) => {}
                                Err(e) => results.push(Err(e)),
                            }
                        }

                        Some(results)
                    }
                    Err(e) => Some(vec![Err(LLMError::HttpError(e.to_string()))]),
                };

                async move { result }
            },
        )
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}

#[async_trait]
impl CompletionProvider for Anthropic {
    /// Sends a completion request to Anthropic's API.
    ///
    /// Converts the completion request into a chat message format.
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        unimplemented!()
    }
}

#[async_trait]
impl EmbeddingProvider for Anthropic {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicModelListResponse {
    data: Vec<AnthropicModelEntry>,
}

impl ModelListResponse for AnthropicModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|m| m.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::Anthropic
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicModelEntry {
    created_at: DateTime<Utc>,
    id: String,
    #[serde(flatten)]
    extra: Value,
}

impl ModelListRawEntry for AnthropicModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[async_trait]
impl ModelsProvider for Anthropic {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let resp = self
            .client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;

        let result: AnthropicModelListResponse = resp.json().await?;

        Ok(Box::new(result))
    }
}

impl crate::LLMProvider for Anthropic {}

/// Parses a Server-Sent Events (SSE) chunk from Anthropic's streaming API.
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
fn parse_anthropic_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<AnthropicStreamResponse>(data) {
                Ok(response) => {
                    if response.response_type == "content_block_delta"
                        && let Some(delta) = response.delta
                        && let Some(text) = delta.text
                    {
                        return Ok(Some(text));
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }
    Ok(None)
}

/// State for tracking tool use blocks during streaming
#[derive(Debug, Default)]
struct ToolUseState {
    /// Tool ID
    id: String,
    /// Tool name
    name: String,
    /// Accumulated JSON input
    json_buffer: String,
}

/// Parses Anthropic SSE chunks with tool use support.
///
/// This parser handles all Anthropic streaming event types including:
/// - `content_block_start` with `type: "text"` or `type: "tool_use"`
/// - `content_block_delta` with `type: "text_delta"` or `type: "input_json_delta"`
/// - `content_block_stop`
/// - `message_delta` with `stop_reason`
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
/// * `tool_states` - Mutable reference to a map tracking tool use state by index
///
/// # Returns
///
/// * `Ok(Some(StreamChunk))` - A stream chunk if one was parsed
/// * `Ok(None)` - If chunk should be skipped
/// * `Err(LLMError)` - If parsing fails
fn parse_anthropic_sse_chunk_with_tools(
    chunk: &str,
    tool_states: &mut HashMap<usize, ToolUseState>,
) -> Result<Option<StreamChunk>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<AnthropicStreamResponse>(data) {
                Ok(response) => {
                    match response.response_type.as_str() {
                        "content_block_start" => {
                            if let (Some(index), Some(content_block)) =
                                (response.index, response.content_block)
                                && content_block.block_type == "tool_use"
                            {
                                let id = content_block.id.unwrap_or_default();
                                let name = content_block.name.unwrap_or_default();

                                // Store state for this tool use block
                                tool_states.insert(
                                    index,
                                    ToolUseState {
                                        id: id.clone(),
                                        name: name.clone(),
                                        json_buffer: String::default(),
                                    },
                                );

                                return Ok(Some(StreamChunk::ToolUseStart { index, id, name }));
                            }
                            // For text blocks, we just wait for content_block_delta
                        }
                        "content_block_delta" => {
                            if let (Some(index), Some(delta)) = (response.index, response.delta) {
                                // Check delta type
                                match delta.delta_type.as_deref() {
                                    Some("text_delta") => {
                                        if let Some(text) = delta.text {
                                            return Ok(Some(StreamChunk::Text(text)));
                                        }
                                    }
                                    Some("input_json_delta") => {
                                        if let Some(partial_json) = delta.partial_json {
                                            // Accumulate JSON in state
                                            if let Some(state) = tool_states.get_mut(&index) {
                                                state.json_buffer.push_str(&partial_json);
                                            }
                                            return Ok(Some(StreamChunk::ToolUseInputDelta {
                                                index,
                                                partial_json,
                                            }));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        "content_block_stop" => {
                            if let Some(index) = response.index {
                                // If we have tool state for this index, emit ToolUseComplete
                                if let Some(state) = tool_states.remove(&index) {
                                    // Anthropic API requires tool_use.input to be a JSON object.
                                    // If no input deltas were received (tool has no parameters),
                                    // default to empty object "{}" instead of empty string "".
                                    let arguments = if state.json_buffer.is_empty() {
                                        "{}".to_string()
                                    } else {
                                        state.json_buffer
                                    };
                                    let tool_call = ToolCall {
                                        id: state.id,
                                        call_type: "function".to_string(),
                                        function: FunctionCall {
                                            name: state.name,
                                            arguments,
                                        },
                                    };
                                    return Ok(Some(StreamChunk::ToolUseComplete {
                                        index,
                                        tool_call,
                                    }));
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(delta) = response.delta
                                && let Some(stop_reason) = delta.stop_reason
                            {
                                return Ok(Some(StreamChunk::Done { stop_reason }));
                            }
                        }
                        _ => {}
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }
    Ok(None)
}

impl LLMBuilder<Anthropic> {
    pub fn build(self) -> Result<Arc<Anthropic>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for Anthropic".to_string())
        })?;

        let anthro = Anthropic::new(
            api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.tool_choice,
            self.reasoning,
            self.reasoning_budget_tokens,
        );

        Ok(Arc::new(anthro))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{FunctionTool, ImageMime};

    #[test]
    fn test_parse_stream_text_delta() {
        let chunk = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::Text(text)) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text chunk, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_stream_tool_use_start() {
        let chunk = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01ABC", "name": "get_weather", "input": {}}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseStart { index, id, name }) => {
                assert_eq!(index, 1);
                assert_eq!(id, "toolu_01ABC");
                assert_eq!(name, "get_weather");
            }
            _ => panic!("Expected ToolUseStart chunk, got {:?}", result),
        }

        // Verify state was stored
        assert!(tool_states.contains_key(&1));
        assert_eq!(tool_states[&1].id, "toolu_01ABC");
        assert_eq!(tool_states[&1].name, "get_weather");
    }

    #[test]
    fn test_parse_stream_tool_use_input_delta() {
        let chunk = r#"event: content_block_delta
            data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"location\":"}}"#;
        let mut tool_states = HashMap::default();
        // Pre-populate state as if tool_use_start was already processed
        tool_states.insert(
            1,
            ToolUseState {
                id: "toolu_01ABC".to_string(),
                name: "get_weather".to_string(),
                json_buffer: String::default(),
            },
        );

        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            }) => {
                assert_eq!(index, 1);
                assert_eq!(partial_json, "{\"location\":");
            }
            _ => panic!("Expected ToolUseInputDelta chunk, got {:?}", result),
        }

        // Verify JSON was accumulated
        assert_eq!(tool_states[&1].json_buffer, "{\"location\":");
    }

    #[test]
    fn test_parse_stream_tool_use_complete() {
        let chunk = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let mut tool_states = HashMap::new();
        // Pre-populate state with accumulated JSON
        tool_states.insert(
            1,
            ToolUseState {
                id: "toolu_01ABC".to_string(),
                name: "get_weather".to_string(),
                json_buffer: r#"{"location": "Paris"}"#.to_string(),
            },
        );

        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseComplete { index, tool_call }) => {
                assert_eq!(index, 1);
                assert_eq!(tool_call.id, "toolu_01ABC");
                assert_eq!(tool_call.function.name, "get_weather");
                assert_eq!(tool_call.function.arguments, r#"{"location": "Paris"}"#);
            }
            _ => panic!("Expected ToolUseComplete chunk, got {:?}", result),
        }

        // Verify state was removed
        assert!(!tool_states.contains_key(&1));
    }

    #[test]
    fn test_parse_stream_tool_use_complete_empty_arguments() {
        // Regression test: tools with no parameters should return "{}" not ""
        // Empty string arguments cause Anthropic API to reject with:
        // "tool_use.input: Input should be a valid dictionary"
        let chunk = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let mut tool_states = HashMap::default();
        // Pre-populate state with EMPTY json_buffer (no input_json_delta events received)
        tool_states.insert(
            1,
            ToolUseState {
                id: "toolu_01XYZ".to_string(),
                name: "get_current_time".to_string(),
                json_buffer: String::default(), // Empty - tool has no parameters
            },
        );

        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseComplete { index, tool_call }) => {
                assert_eq!(index, 1);
                assert_eq!(tool_call.id, "toolu_01XYZ");
                assert_eq!(tool_call.function.name, "get_current_time");
                // CRITICAL: arguments must be "{}" not "" for Anthropic API compatibility
                assert_eq!(
                    tool_call.function.arguments, "{}",
                    "Empty arguments should default to '{{}}' not empty string"
                );
            }
            _ => panic!("Expected ToolUseComplete chunk, got {:?}", result),
        }

        // Verify state was removed
        assert!(!tool_states.contains_key(&1));
    }

    #[test]
    fn test_parse_stream_done_tool_use() {
        let chunk = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::Done { stop_reason }) => {
                assert_eq!(stop_reason, "tool_use");
            }
            _ => panic!("Expected Done chunk, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_stream_done_end_turn() {
        let chunk = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::Done { stop_reason }) => {
                assert_eq!(stop_reason, "end_turn");
            }
            _ => panic!("Expected Done chunk, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_stream_full_tool_use_sequence() {
        let mut tool_states = HashMap::new();

        // 1. Tool use start
        let start_chunk = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01ABC", "name": "get_weather", "input": {}}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(start_chunk, &mut tool_states).unwrap();
        assert!(matches!(result, Some(StreamChunk::ToolUseStart { .. })));

        // 2. Input JSON deltas
        let delta1 = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"loc"}}

"#;
        let _ = parse_anthropic_sse_chunk_with_tools(delta1, &mut tool_states).unwrap();

        let delta2 = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "ation\": \"Paris\"}"}}

"#;
        let _ = parse_anthropic_sse_chunk_with_tools(delta2, &mut tool_states).unwrap();

        // Verify accumulated JSON
        assert_eq!(tool_states[&1].json_buffer, "{\"location\": \"Paris\"}");

        // 3. Content block stop
        let stop_chunk = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(stop_chunk, &mut tool_states).unwrap();

        match result {
            Some(StreamChunk::ToolUseComplete { tool_call, .. }) => {
                assert_eq!(tool_call.function.arguments, "{\"location\": \"Paris\"}");
            }
            _ => panic!("Expected ToolUseComplete"),
        }

        // 4. Message delta with stop reason
        let done_chunk = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(done_chunk, &mut tool_states).unwrap();
        assert!(matches!(
            result,
            Some(StreamChunk::Done {
                stop_reason
            }) if stop_reason == "tool_use"
        ));
    }

    #[test]
    fn test_parse_stream_mixed_text_and_tool() {
        let mut tool_states = HashMap::new();

        // Text delta first
        let text_chunk = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I'll check the weather"}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(text_chunk, &mut tool_states).unwrap();
        assert!(matches!(result, Some(StreamChunk::Text(t)) if t == "I'll check the weather"));

        // Then tool use
        let tool_start = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01XYZ", "name": "weather", "input": {}}}

"#;
        let result = parse_anthropic_sse_chunk_with_tools(tool_start, &mut tool_states).unwrap();
        assert!(
            matches!(result, Some(StreamChunk::ToolUseStart { name, .. }) if name == "weather")
        );
    }

    #[test]
    fn test_parse_stream_ignores_message_start() {
        let chunk = r#"event: message_start
data: {"type": "message_start", "message": {"id": "msg_123", "type": "message", "role": "assistant"}}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_ignores_ping() {
        let chunk = r#"event: ping
data: {"type": "ping"}

"#;
        let mut tool_states = HashMap::new();
        let result = parse_anthropic_sse_chunk_with_tools(chunk, &mut tool_states).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_convert_messages_to_anthropic_tool_result() {
        let msg = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::ToolResult(vec![ToolCall {
                id: "tool_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "lookup".to_string(),
                    arguments: "{\"q\":\"value\"}".to_string(),
                },
            }]),
            content: "result".to_string(),
        };

        let messages = vec![msg];
        let converted = Anthropic::convert_messages_to_anthropic(&messages);
        assert_eq!(converted.len(), 1);
        let content = &converted[0].content[0];
        assert_eq!(content.message_type, Some("tool_result"));
        assert_eq!(content.tool_result_id.as_deref(), Some("tool_1"));
        assert_eq!(content.tool_output.as_deref(), Some("{\"q\":\"value\"}"));
    }

    #[test]
    fn test_convert_messages_to_anthropic_tool_use_parses_input() {
        let msg = ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::ToolUse(vec![ToolCall {
                id: "tool_call".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "lookup".to_string(),
                    arguments: "{\"q\":\"value\"}".to_string(),
                },
            }]),
            content: "call".to_string(),
        };

        let messages = [msg];
        let converted = Anthropic::convert_messages_to_anthropic(&messages);
        let content = &converted[0].content[0];
        assert_eq!(content.message_type, Some("tool_use"));
        assert_eq!(content.tool_use_id.as_deref(), Some("tool_call"));
        assert_eq!(content.tool_name.as_deref(), Some("lookup"));
        assert_eq!(content.tool_input, Some(serde_json::json!({"q": "value"})));
    }

    #[test]
    fn test_prepare_tools_and_choice_auto() {
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "desc".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        };
        let tools_list = [tool];
        let (tools, choice) =
            Anthropic::prepare_tools_and_choice(Some(&tools_list), &Some(ToolChoice::Auto));
        assert!(tools.is_some());
        assert_eq!(choice.unwrap().get("type"), Some(&"auto".to_string()));
    }

    #[test]
    fn test_prepare_tools_and_choice_specific_tool() {
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "desc".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "q": { "type": "string" } }
                }),
            },
        };
        let tools_list = [tool];
        let (tools, choice) = Anthropic::prepare_tools_and_choice(
            Some(&tools_list),
            &Some(ToolChoice::Tool("lookup".to_string())),
        );
        assert!(tools.is_some());
        let choice = choice.unwrap();
        assert_eq!(choice.get("type"), Some(&"tool".to_string()));
        assert_eq!(choice.get("name"), Some(&"lookup".to_string()));
    }

    #[test]
    fn test_prepare_tools_and_choice_ignored_without_tools() {
        let (tools, choice) = Anthropic::prepare_tools_and_choice(None, &Some(ToolChoice::Auto));
        assert!(tools.is_none());
        assert!(choice.is_none());
    }

    #[test]
    fn test_convert_messages_to_anthropic_image_and_pdf() {
        let image = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Image((ImageMime::PNG, vec![1, 2, 3])),
            content: "img".to_string(),
        };
        let pdf = ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Pdf(vec![9, 8, 7]),
            content: "doc".to_string(),
        };

        let messages = [image, pdf];
        let converted = Anthropic::convert_messages_to_anthropic(&messages);
        assert_eq!(converted.len(), 2);

        let image_content = &converted[0].content[0];
        assert_eq!(image_content.message_type, Some("image"));
        assert_eq!(
            image_content.source.as_ref().expect("source").media_type,
            "image/png"
        );

        let pdf_content = &converted[1].content[0];
        assert_eq!(pdf_content.message_type, Some("document"));
        assert_eq!(
            pdf_content.source.as_ref().expect("source").media_type,
            "application/pdf"
        );
    }

    #[test]
    fn test_anthropic_builder_requires_api_key() {
        let err = LLMBuilder::<Anthropic>::new().build().unwrap_err();
        assert!(err.to_string().contains("No API key provided"));
    }
}
