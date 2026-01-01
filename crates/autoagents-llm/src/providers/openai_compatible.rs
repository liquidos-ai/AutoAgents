//! OpenAI-compatible API client base implementation
//!
//! This module provides a generic base for OpenAI-compatible APIs that can be reused
//! across multiple providers like OpenAI, Mistral, XAI, Groq, DeepSeek, etc.

use crate::chat::{StreamChoice, StreamChunk as ChatStreamChunk, StreamDelta};
use crate::error::LLMError;
use crate::FunctionCall;
use crate::{
    chat::ChatResponse,
    chat::{
        ChatMessage, ChatProvider, ChatRole, MessageType, StreamResponse, StructuredOutputFormat,
        Tool, ToolChoice, Usage,
    },
    default_call_type, ToolCall,
};
use async_trait::async_trait;
use either::*;
use futures::{stream::Stream, StreamExt};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;

/// Generic OpenAI-compatible provider
///
/// This struct provides a base implementation for any OpenAI-compatible API.
/// Different providers can customize behavior by implementing the `OpenAICompatibleConfig` trait.
pub struct OpenAICompatibleProvider<T: OpenAIProviderConfig> {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<String>,
    pub voice: Option<String>,
    pub extra_body: serde_json::Map<String, serde_json::Value>,
    pub parallel_tool_calls: bool,
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub normalize_response: bool,
    pub client: Client,
    _phantom: PhantomData<T>,
}

/// Configuration trait for OpenAI-compatible providers
///
/// This trait allows different providers to customize behavior while reusing
/// the common OpenAI-compatible implementation.
pub trait OpenAIProviderConfig: Send + Sync {
    /// The name of the provider (e.g., "OpenAI", "Mistral", "XAI")
    const PROVIDER_NAME: &'static str;
    /// Default base URL for the provider
    const DEFAULT_BASE_URL: &'static str;
    /// Default model for the provider
    const DEFAULT_MODEL: &'static str;
    /// Chat completions endpoint path (usually "chat/completions")
    const CHAT_ENDPOINT: &'static str = "chat/completions";
    /// Whether this provider supports reasoning effort
    const SUPPORTS_REASONING_EFFORT: bool = false;
    /// Whether this provider supports structured output
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;
    /// Whether this provider supports parallel tool calls
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    /// Whether this provider supports stream options (like include_usage)
    const SUPPORTS_STREAM_OPTIONS: bool = false;
    /// Custom headers to add to requests
    fn custom_headers() -> Option<Vec<(String, String)>> {
        None
    }
}

/// Generic OpenAI-compatible chat message
#[derive(Serialize, Debug)]
pub struct OpenAIChatMessage<'a> {
    pub role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    pub content: Option<Either<Vec<OpenAIMessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct OpenAIMessageContent<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrlContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    pub tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    pub tool_output: Option<&'a str>,
}

#[derive(Serialize, Debug)]
pub struct ImageUrlContent {
    pub url: String,
}

/// Generic OpenAI-compatible chat request
#[derive(Serialize, Debug)]
pub struct OpenAIChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<OpenAIChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenAIStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(flatten)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

/// Generic OpenAI-compatible chat response
#[derive(Deserialize, Debug)]
pub struct OpenAIChatResponse {
    pub choices: Vec<OpenAIChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIChatChoice {
    pub message: OpenAIChatMsg,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIChatMsg {
    #[allow(dead_code)]
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug, Serialize)]
pub enum OpenAIResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    pub response_type: OpenAIResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<StructuredOutputFormat>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct OpenAIStreamOptions {
    pub include_usage: bool,
}

/// Streaming response structures
#[derive(Deserialize, Debug)]
pub struct StreamChunk {
    pub choices: Vec<OpenAIStreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIStreamChoice {
    pub delta: OpenAIStreamDelta,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIStreamDelta {
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<StreamToolCall>>,
}

/// Tool call represents a function call that an LLM wants to make.
/// This is a standardized structure used across all providers.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct StreamToolCall {
    /// The ID of the tool call.
    pub id: Option<String>,
    /// The type of the tool call (defaults to "function" if not provided).
    #[serde(rename = "type", default = "default_call_type")]
    pub call_type: String,
    /// The function to call.
    pub function: StreamFunctionCall,
}

/// FunctionCall contains details about which function to call and with what arguments.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct StreamFunctionCall {
    /// The name of the function to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    pub arguments: String,
}

impl From<StructuredOutputFormat> for OpenAIResponseFormat {
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        match structured_response_format.schema {
            None => OpenAIResponseFormat {
                response_type: OpenAIResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                schema = if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                    schema
                } else {
                    schema
                };
                OpenAIResponseFormat {
                    response_type: OpenAIResponseType::JsonSchema,
                    json_schema: Some(StructuredOutputFormat {
                        name: structured_response_format.name,
                        description: structured_response_format.description,
                        schema: Some(schema),
                        strict: structured_response_format.strict,
                    }),
                }
            }
        }
    }
}

impl ChatResponse for OpenAIChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

impl std::fmt::Display for OpenAIChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (
            &self.choices.first().unwrap().message.content,
            &self.choices.first().unwrap().message.tool_calls,
        ) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl<T: OpenAIProviderConfig> OpenAICompatibleProvider<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        voice: Option<String>,
        extra_body: Option<serde_json::Value>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        let extra_body = match extra_body {
            Some(serde_json::Value::Object(map)) => map,
            _ => serde_json::Map::new(), // Should we panic here?
        };
        Self {
            api_key: api_key.into(),
            base_url: Url::parse(&format!(
                "{}/",
                base_url
                    .unwrap_or_else(|| T::DEFAULT_BASE_URL.to_owned())
                    .trim_end_matches("/")
            ))
            .expect("Failed to parse base URL"),
            model: model.unwrap_or_else(|| T::DEFAULT_MODEL.to_string()),
            max_tokens,
            temperature,
            timeout_seconds,
            top_p,
            top_k,
            tool_choice,
            reasoning_effort,
            voice,
            extra_body,
            parallel_tool_calls: parallel_tool_calls.unwrap_or(false),
            normalize_response: normalize_response.unwrap_or(true),
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            _phantom: PhantomData,
        }
    }

    pub fn prepare_messages(&self, messages: &[ChatMessage]) -> Vec<OpenAIChatMessage<'_>> {
        let openai_msgs: Vec<OpenAIChatMessage> = messages
            .iter()
            .flat_map(|msg| {
                if let MessageType::ToolResult(ref results) = msg.message_type {
                    // Expand ToolResult into multiple messages
                    results
                        .iter()
                        .map(|result| OpenAIChatMessage {
                            role: "tool",
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        })
                        .collect::<Vec<_>>()
                } else {
                    // Convert single message
                    vec![chat_message_to_openai_message(msg.clone())]
                }
            })
            .collect();
        openai_msgs
    }
}

#[async_trait]
impl<T: OpenAIProviderConfig> ChatProvider for OpenAICompatibleProvider<T> {
    /// Perform a chat request with tool calls
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!(
                "Missing {} API key",
                T::PROVIDER_NAME
            )));
        }
        let openai_msgs = self.prepare_messages(messages);
        let response_format: Option<OpenAIResponseFormat> = if T::SUPPORTS_STRUCTURED_OUTPUT {
            json_schema.clone().map(|s| s.into())
        } else {
            None
        };
        let request_tools = tools.map(|t| t.to_vec());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };
        let reasoning_effort = if T::SUPPORTS_REASONING_EFFORT {
            self.reasoning_effort.clone()
        } else {
            None
        };
        let parallel_tool_calls = if T::SUPPORTS_PARALLEL_TOOL_CALLS {
            Some(self.parallel_tool_calls)
        } else {
            None
        };
        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: false,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort,
            response_format,
            stream_options: None,
            parallel_tool_calls,
            extra_body: self.extra_body.clone(),
        };
        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);
        // Add custom headers if provider specifies them
        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("{} request payload: {}", T::PROVIDER_NAME, json);
            }
        }
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        log::debug!("{} HTTP status: {}", T::PROVIDER_NAME, response.status());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode {} API response: {e}", T::PROVIDER_NAME),
                raw_response: resp_text,
            }),
        }
    }

    /// Perform a chat request without tool calls
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None, json_schema).await
    }

    /// Stream chat responses as a stream of strings
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let struct_stream = self.chat_stream_struct(messages, None, json_schema).await?;
        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(stream_response) => {
                    if let Some(choice) = stream_response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            if !content.is_empty() {
                                return Some(Ok(content.clone()));
                            }
                        }
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::pin(content_stream))
    }

    /// Stream chat responses as `ChatMessage` structured objects, including usage information
    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!(
                "Missing {} API key",
                T::PROVIDER_NAME
            )));
        }
        let openai_msgs = self.prepare_messages(messages);
        let request_tools = tools.map(|t| t.to_vec());

        let response_format: Option<OpenAIResponseFormat> = if T::SUPPORTS_STRUCTURED_OUTPUT {
            json_schema.clone().map(|s| s.into())
        } else {
            None
        };

        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: if T::SUPPORTS_REASONING_EFFORT {
                self.reasoning_effort.clone()
            } else {
                None
            },
            response_format,
            stream_options: if T::SUPPORTS_STREAM_OPTIONS {
                Some(OpenAIStreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            parallel_tool_calls: if T::SUPPORTS_PARALLEL_TOOL_CALLS {
                Some(self.parallel_tool_calls)
            } else {
                None
            },
            extra_body: self.extra_body.clone(),
        };
        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);
        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("{} request payload: {}", T::PROVIDER_NAME, json);
            }
        }
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        log::debug!("{} HTTP status: {}", T::PROVIDER_NAME, response.status());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }
        Ok(create_sse_stream(response, self.normalize_response))
    }

    /// Sends a streaming chat request with tool support.
    ///
    /// Returns a stream of `StreamChunk` which can be text deltas or tool call events.
    /// This method provides a unified interface for streaming with tools across
    /// OpenAI-compatible providers.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools available for the model to use
    ///
    /// # Returns
    ///
    /// A stream of `StreamChunk` items or an error
    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!(
                "Missing {} API key",
                T::PROVIDER_NAME
            )));
        }

        let openai_msgs = self.prepare_messages(messages);

        let requested_tools = tools.map(|t| t.to_vec());

        let response_format: Option<OpenAIResponseFormat> = if T::SUPPORTS_STRUCTURED_OUTPUT {
            json_schema.clone().map(|s| s.into())
        } else {
            None
        };

        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: requested_tools,
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: if T::SUPPORTS_REASONING_EFFORT {
                self.reasoning_effort.clone()
            } else {
                None
            },
            response_format,
            stream_options: if T::SUPPORTS_STREAM_OPTIONS {
                Some(OpenAIStreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            parallel_tool_calls: if T::SUPPORTS_PARALLEL_TOOL_CALLS {
                Some(self.parallel_tool_calls)
            } else {
                None
            },
            extra_body: self.extra_body.clone(),
        };

        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!(
                    "{} streaming with tools request: {}",
                    T::PROVIDER_NAME,
                    json
                );
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        log::debug!(
            "{} request: POST {} (streaming with tools)",
            T::PROVIDER_NAME,
            T::CHAT_ENDPOINT
        );
        let response = request.send().await?;
        log::debug!("{} HTTP status: {}", T::PROVIDER_NAME, response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }

        Ok(create_openai_tool_stream(response))
    }
}

/// State for tracking tool use blocks during OpenAI-compatible streaming
#[derive(Debug, Default)]
struct OpenAIToolUseState {
    /// Tool ID
    id: String,
    /// Tool name
    name: String,
    /// Accumulated JSON arguments
    arguments_buffer: String,
    /// Whether we've emitted the start event
    started: bool,
}

/// Creates an SSE stream that parses OpenAI-compatible tool use events into ChatStreamChunk.
fn create_openai_tool_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, LLMError>> + Send>> {
    let stream = response
        .bytes_stream()
        .scan(
            (String::new(), HashMap::<usize, OpenAIToolUseState>::new()),
            move |(buffer, tool_states), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&text);

                        let mut results = Vec::new();

                        // Process complete SSE events (separated by double newlines)
                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos].to_string();
                            buffer.drain(..pos + 2);

                            // Also handle \r\n\r\n
                            let event = event.trim();
                            if event.is_empty() {
                                continue;
                            }

                            match parse_openai_sse_chunk_with_tools(event, tool_states) {
                                Ok(chunks) => results.extend(chunks.into_iter().map(Ok)),
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

/// Parses OpenAI-compatible SSE chunks with tool use support.
///
/// OpenAI streams tool calls as deltas with:
/// - `tool_calls[].index` - identifies which tool call
/// - `tool_calls[].id` - tool call ID (first chunk only)
/// - `tool_calls[].function.name` - function name (first chunk only)
/// - `tool_calls[].function.arguments` - partial JSON arguments (streamed)
/// - `finish_reason: "tool_calls"` - signals completion
fn parse_openai_sse_chunk_with_tools(
    event: &str,
    tool_states: &mut HashMap<usize, OpenAIToolUseState>,
) -> Result<Vec<ChatStreamChunk>, LLMError> {
    let mut results = Vec::new();

    for line in event.lines() {
        let line = line.trim();

        // Accept both "data: " and "data:" prefixes (tolerant parsing).
        let data_opt = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:").map(|d| d.trim_start()));

        if let Some(data) = data_opt {
            let data_trimmed = data.trim();

            // End-of-stream sentinel.
            if data_trimmed == "[DONE]" {
                // Emit any remaining tool completions
                for (index, state) in tool_states.drain() {
                    if state.started {
                        results.push(ChatStreamChunk::ToolUseComplete {
                            index,
                            tool_call: ToolCall {
                                id: state.id,
                                call_type: "function".to_string(),
                                function: FunctionCall {
                                    name: state.name,
                                    arguments: state.arguments_buffer,
                                },
                            },
                        });
                    }
                }

                // No usage available here (usage is often sent in a prior event).
                results.push(ChatStreamChunk::Done {
                    stop_reason: "end_turn".to_string(),
                });
                return Ok(results);
            }

            // Try to parse JSON chunk; return parse error so caller can turn into an Err item.
            let chunk: OpenAIToolStreamChunk = serde_json::from_str(data_trimmed)
                .map_err(|e| LLMError::JsonError(e.to_string()))?;

            // Capture usage if present (we may emit it either immediately or after completions).
            let mut usage_opt = chunk.usage.clone();

            for choice in &chunk.choices {
                // Handle text content
                if let Some(content) = &choice.delta.content {
                    if !content.is_empty() {
                        results.push(ChatStreamChunk::Text(content.clone()));
                    }
                }

                // Handle tool calls (per-index)
                if let Some(tool_calls) = &choice.delta.tool_calls {
                    for tc in tool_calls {
                        let index = tc.index.unwrap_or(0);
                        let state = tool_states.entry(index).or_default();

                        // First chunk can contain id and name
                        if let Some(id) = &tc.id {
                            state.id = id.clone();
                        }
                        if let Some(name) = &tc.function.name {
                            state.name = name.clone();

                            // Emit ToolUseStart if not already started
                            if !state.started {
                                state.started = true;
                                results.push(ChatStreamChunk::ToolUseStart {
                                    index,
                                    id: state.id.clone(),
                                    name: state.name.clone(),
                                });
                            }
                        }

                        // Accumulate arguments and emit the delta event so consumers can stream partial JSON.
                        if !tc.function.arguments.is_empty() {
                            state.arguments_buffer.push_str(&tc.function.arguments);
                            results.push(ChatStreamChunk::ToolUseInputDelta {
                                index,
                                partial_json: tc.function.arguments.clone(),
                            });
                        }
                    }
                }

                // Handle finish_reason (map to Done stop_reason and flush completions)
                if let Some(finish_reason) = &choice.finish_reason {
                    // Emit tool completions before Done
                    for (index, state) in tool_states.drain() {
                        if state.started {
                            results.push(ChatStreamChunk::ToolUseComplete {
                                index,
                                tool_call: ToolCall {
                                    id: state.id,
                                    call_type: "function".to_string(),
                                    function: FunctionCall {
                                        name: state.name,
                                        arguments: state.arguments_buffer,
                                    },
                                },
                            });
                        }
                    }

                    // If usage was present in this chunk, emit it now (after completions, before Done).
                    if let Some(u) = usage_opt.take() {
                        results.push(ChatStreamChunk::Usage(u));
                    }

                    let stop_reason = match finish_reason.as_str() {
                        "tool_calls" => "tool_use",
                        "stop" => "end_turn",
                        other => other,
                    };
                    results.push(ChatStreamChunk::Done {
                        stop_reason: stop_reason.to_string(),
                    });
                }
            }

            // If no finish_reason was found but usage was present in this chunk,
            // emit the usage now (consume it so it is not emitted again).
            if let Some(u) = usage_opt.take() {
                results.push(ChatStreamChunk::Usage(u));
            }
        }
    }

    Ok(results)
}

/// OpenAI streaming chunk structure for tool parsing
#[derive(Debug, Deserialize)]
struct OpenAIToolStreamChunk {
    choices: Vec<OpenAIToolStreamChoice>,
    /// Optional usage metadata (often appears in its own SSE event).
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolStreamChoice {
    delta: OpenAIToolStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: OpenAIToolStreamFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolStreamFunction {
    name: Option<String>,
    #[serde(default)]
    arguments: String,
}

/// Create OpenAICompatibleChatMessage` that doesn't borrow from any temporary variables
pub fn chat_message_to_openai_message(chat_msg: ChatMessage) -> OpenAIChatMessage<'static> {
    OpenAIChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::System => "system",
            ChatRole::Tool => "user",
        },
        tool_call_id: None,
        content: match &chat_msg.message_type {
            MessageType::Text => Some(Right(chat_msg.content.clone())),
            MessageType::Image(_) => unreachable!(),
            MessageType::Pdf(_) => unimplemented!(),
            MessageType::ImageURL(url) => Some(Left(vec![OpenAIMessageContent {
                message_type: Some("image_url"),
                text: None,
                image_url: Some(ImageUrlContent { url: url.clone() }),
                tool_output: None,
                tool_call_id: None,
            }])),
            MessageType::ToolUse(_) => None,
            MessageType::ToolResult(_) => None,
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                let owned_calls: Vec<ToolCall> = calls
                    .iter()
                    .map(|c| ToolCall {
                        id: c.id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: c.function.name.clone(),
                            arguments: c.function.arguments.clone(),
                        },
                    })
                    .collect();
                Some(owned_calls)
            }
            _ => None,
        },
    }
}

/// Creates a structured SSE stream that returns `StreamResponse` objects
///
/// Buffer required to accumulate JSON payload lines that are split across multiple SSE chunks
pub fn create_sse_stream(
    response: reqwest::Response,
    normalize_response: bool,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>> {
    struct SSEStreamParser {
        event_buffer: String,
        tool_buffer: ToolCall,
        usage: Option<Usage>,
        results: Vec<Result<StreamResponse, LLMError>>,
        normalize_response: bool,
    }
    impl SSEStreamParser {
        fn new(normalize_response: bool) -> Self {
            Self {
                event_buffer: String::new(),
                usage: None,
                results: Vec::new(),
                tool_buffer: ToolCall {
                    id: String::new(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: String::new(),
                        arguments: String::new(),
                    },
                },
                normalize_response,
            }
        }

        /// Push the current `tool_buffer` as a `StreamResponse` and reset it
        fn push_tool_call(&mut self) {
            if self.normalize_response && !self.tool_buffer.function.name.is_empty() {
                self.results.push(Ok(StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: None,
                            tool_calls: Some(vec![self.tool_buffer.clone()]),
                        },
                    }],
                    usage: None,
                }));
            }
            self.tool_buffer = ToolCall {
                id: String::new(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: String::new(),
                    arguments: String::new(),
                },
            };
        }

        /// Parse the accumulated event_buffer as one SSE event
        fn parse_event(&mut self) {
            let mut data_payload = String::new();
            for line in self.event_buffer.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        self.push_tool_call();
                        if let Some(usage) = self.usage.clone() {
                            self.results.push(Ok(StreamResponse {
                                choices: vec![StreamChoice {
                                    delta: StreamDelta {
                                        content: None,
                                        tool_calls: None,
                                    },
                                }],
                                usage: Some(usage),
                            }));
                        }
                        return;
                    }
                    data_payload.push_str(data);
                } else {
                    data_payload.push_str(line);
                }
            }
            if data_payload.is_empty() {
                return;
            }
            if let Ok(response) = serde_json::from_str::<StreamChunk>(&data_payload) {
                if let Some(resp_usage) = response.usage.clone() {
                    self.usage = Some(resp_usage);
                }
                for choice in &response.choices {
                    let content = choice.delta.content.clone();
                    // Map StreamToolCall (some fields are optional) to ToolCall
                    let tool_calls: Option<Vec<ToolCall>> =
                        choice.delta.tool_calls.clone().map(|calls| {
                            calls
                                .into_iter()
                                .map(|c| ToolCall {
                                    id: c.id.unwrap_or_default(),
                                    call_type: c.call_type,
                                    function: FunctionCall {
                                        name: c.function.name.unwrap_or_default(),
                                        arguments: c.function.arguments,
                                    },
                                })
                                .collect::<Vec<ToolCall>>()
                        });
                    if content.is_some() || tool_calls.is_some() {
                        if self.normalize_response && tool_calls.is_some() {
                            // If normalize_response is enabled, accumulate tool call outputs
                            if let Some(calls) = &tool_calls {
                                for call in calls {
                                    // println!("Accumulating tool call: {:?}", call);
                                    if !call.function.name.is_empty() {
                                        self.push_tool_call();
                                        self.tool_buffer.function.name = call.function.name.clone();
                                    }
                                    if !call.function.arguments.is_empty() {
                                        self.tool_buffer
                                            .function
                                            .arguments
                                            .push_str(&call.function.arguments);
                                    }
                                    if !call.id.is_empty() {
                                        self.tool_buffer.id = call.id.clone();
                                    }
                                    if !call.call_type.is_empty() {
                                        self.tool_buffer.call_type = call.call_type.clone();
                                    }
                                }
                            }
                        } else {
                            self.push_tool_call();
                            self.results.push(Ok(StreamResponse {
                                choices: vec![StreamChoice {
                                    delta: StreamDelta {
                                        content,
                                        tool_calls,
                                    },
                                }],
                                usage: None,
                            }));
                        }
                    }
                }
            }
        }
    }

    let bytes_stream = response.bytes_stream();
    let stream = bytes_stream
        .scan(SSEStreamParser::new(normalize_response), |parser, chunk| {
            let results = match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        let line = line.trim_end();
                        if line.is_empty() {
                            // Blank line: end of event, parse accumulated event_buffer
                            parser.parse_event();
                            parser.event_buffer.clear();
                        } else {
                            parser.event_buffer.push_str(line);
                            parser.event_buffer.push('\n');
                        }
                    }
                    parser.results.drain(..).collect::<Vec<_>>()
                }
                Err(e) => vec![Err(LLMError::HttpError(e.to_string()))],
            };
            futures::future::ready(Some(results))
        })
        .flat_map(futures::stream::iter);
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_stream_text_delta() {
        let event = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let mut tool_states = HashMap::new();
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 1);
        match &results[0] {
            ChatStreamChunk::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text chunk, got {:?}", results[0]),
        }
    }

    #[test]
    fn test_parse_openai_stream_tool_call_start() {
        let event = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#;
        let mut tool_states = HashMap::new();
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 1);
        match &results[0] {
            ChatStreamChunk::ToolUseStart { index, id, name } => {
                assert_eq!(*index, 0);
                assert_eq!(id, "call_abc123");
                assert_eq!(name, "get_weather");
            }
            _ => panic!("Expected ToolUseStart chunk, got {:?}", results[0]),
        }

        // Verify state was stored
        assert!(tool_states.contains_key(&0));
        assert_eq!(tool_states[&0].id, "call_abc123");
        assert_eq!(tool_states[&0].name, "get_weather");
        assert!(tool_states[&0].started);
    }

    #[test]
    fn test_parse_openai_stream_tool_call_arguments_delta() {
        // First, set up tool state as if start was already processed
        let mut tool_states = HashMap::new();
        tool_states.insert(
            0,
            OpenAIToolUseState {
                id: "call_abc123".to_string(),
                name: "get_weather".to_string(),
                arguments_buffer: String::new(),
                started: true,
            },
        );

        let event = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]},"finish_reason":null}]}"#;
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 1);
        match &results[0] {
            ChatStreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            } => {
                assert_eq!(*index, 0);
                assert_eq!(partial_json, "{\"location\":");
            }
            _ => panic!("Expected ToolUseInputDelta chunk, got {:?}", results[0]),
        }

        // Verify arguments were accumulated
        assert_eq!(tool_states[&0].arguments_buffer, "{\"location\":");
    }

    #[test]
    fn test_parse_openai_stream_finish_reason_tool_calls() {
        let mut tool_states = HashMap::new();
        tool_states.insert(
            0,
            OpenAIToolUseState {
                id: "call_abc123".to_string(),
                name: "get_weather".to_string(),
                arguments_buffer: r#"{"location": "Paris"}"#.to_string(),
                started: true,
            },
        );

        let event = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        // Should have ToolUseComplete and Done
        assert_eq!(results.len(), 2);

        match &results[0] {
            ChatStreamChunk::ToolUseComplete { index, tool_call } => {
                assert_eq!(*index, 0);
                assert_eq!(tool_call.id, "call_abc123");
                assert_eq!(tool_call.function.name, "get_weather");
                assert_eq!(tool_call.function.arguments, r#"{"location": "Paris"}"#);
            }
            _ => panic!("Expected ToolUseComplete chunk, got {:?}", results[0]),
        }

        match &results[1] {
            ChatStreamChunk::Done { stop_reason } => {
                assert_eq!(stop_reason, "tool_use");
            }
            _ => panic!("Expected Done chunk, got {:?}", results[1]),
        }

        // Verify state was cleared
        assert!(tool_states.is_empty());
    }

    #[test]
    fn test_parse_openai_stream_finish_reason_stop() {
        let event = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let mut tool_states = HashMap::new();
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 1);
        match &results[0] {
            ChatStreamChunk::Done { stop_reason } => {
                assert_eq!(stop_reason, "end_turn");
            }
            _ => panic!("Expected Done chunk, got {:?}", results[0]),
        }
    }

    #[test]
    fn test_parse_openai_stream_done_marker() {
        let event = "data: [DONE]";
        let mut tool_states = HashMap::new();
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 1);
        match &results[0] {
            ChatStreamChunk::Done { stop_reason } => {
                assert_eq!(stop_reason, "end_turn");
            }
            _ => panic!("Expected Done chunk, got {:?}", results[0]),
        }
    }

    #[test]
    fn test_parse_openai_stream_done_marker_with_pending_tool() {
        let mut tool_states = HashMap::new();
        tool_states.insert(
            0,
            OpenAIToolUseState {
                id: "call_xyz".to_string(),
                name: "some_function".to_string(),
                arguments_buffer: "{}".to_string(),
                started: true,
            },
        );

        let event = "data: [DONE]";
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        // Should emit ToolUseComplete before Done
        assert_eq!(results.len(), 2);
        assert!(matches!(
            &results[0],
            ChatStreamChunk::ToolUseComplete { .. }
        ));
        assert!(matches!(&results[1], ChatStreamChunk::Done { .. }));
    }

    #[test]
    fn test_parse_openai_stream_full_tool_sequence() {
        let mut tool_states = HashMap::new();

        // 1. Tool call start with name
        let start_event = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#;
        let results = parse_openai_sse_chunk_with_tools(start_event, &mut tool_states).unwrap();
        assert!(
            matches!(&results[0], ChatStreamChunk::ToolUseStart { name, .. } if name == "get_weather")
        );

        // 2. Arguments delta 1
        let delta1 = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}"#;
        let _ = parse_openai_sse_chunk_with_tools(delta1, &mut tool_states).unwrap();

        // 3. Arguments delta 2
        let delta2 = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"Tokyo\"}"}}]},"finish_reason":null}]}"#;
        let _ = parse_openai_sse_chunk_with_tools(delta2, &mut tool_states).unwrap();

        // Verify accumulated arguments
        assert_eq!(tool_states[&0].arguments_buffer, "{\"location\":\"Tokyo\"}");

        // 4. Finish reason
        let finish_event = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let results = parse_openai_sse_chunk_with_tools(finish_event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 2);
        match &results[0] {
            ChatStreamChunk::ToolUseComplete { tool_call, .. } => {
                assert_eq!(tool_call.function.arguments, "{\"location\":\"Tokyo\"}");
            }
            _ => panic!("Expected ToolUseComplete"),
        }
        assert!(matches!(
            &results[1],
            ChatStreamChunk::Done { stop_reason } if stop_reason == "tool_use"
        ));
    }

    #[test]
    fn test_parse_openai_stream_parallel_tool_calls() {
        let mut tool_states = HashMap::new();

        // Two tool calls in one chunk
        let event = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}},{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}]},"finish_reason":null}]}"#;
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert_eq!(results.len(), 2);
        assert!(
            matches!(&results[0], ChatStreamChunk::ToolUseStart { index: 0, name, .. } if name == "get_weather")
        );
        assert!(
            matches!(&results[1], ChatStreamChunk::ToolUseStart { index: 1, name, .. } if name == "get_time")
        );

        // Verify both states exist
        assert!(tool_states.contains_key(&0));
        assert!(tool_states.contains_key(&1));
    }

    #[test]
    fn test_parse_openai_stream_ignores_empty_content() {
        let event = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#;
        let mut tool_states = HashMap::new();
        let results = parse_openai_sse_chunk_with_tools(event, &mut tool_states).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_vllm_stream_tool_calls() {
        // vLLM includes extra fields like reasoning_content, type, token_ids
        let mut tool_states = HashMap::new();

        // First chunk from vLLM - just role, empty content
        let first_chunk = r#"data: {"id":"chatcmpl-be8d6d925ff14741","object":"chat.completion.chunk","created":1765374283,"model":"Qwen/Qwen2.5-Coder-7B-Instruct-AWQ","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":null},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null}"#;
        let results = parse_openai_sse_chunk_with_tools(first_chunk, &mut tool_states).unwrap();
        assert!(results.is_empty(), "First chunk should produce no results");

        // Second chunk - tool call start with id, type, index, function.name and function.arguments
        let tool_start = r#"data: {"id":"chatcmpl-be8d6d925ff14741","object":"chat.completion.chunk","created":1765374283,"model":"Qwen/Qwen2.5-Coder-7B-Instruct-AWQ","choices":[{"index":0,"delta":{"reasoning_content":null,"tool_calls":[{"id":"chatcmpl-tool-a331788bab1045a8","type":"function","index":0,"function":{"name":"db_list_databases","arguments":"{\"catalog\":"}}]},"logprobs":null,"finish_reason":null,"token_ids":null}]}"#;
        let results = parse_openai_sse_chunk_with_tools(tool_start, &mut tool_states).unwrap();

        // Should have ToolUseStart and ToolUseInputDelta
        assert!(
            !results.is_empty(),
            "Expected at least 1 result, got {:?}",
            results
        );
        assert!(
            matches!(&results[0], ChatStreamChunk::ToolUseStart { name, .. } if name == "db_list_databases"),
            "Expected ToolUseStart, got {:?}",
            results[0]
        );

        // Arguments delta
        let args_delta = r#"data: {"id":"chatcmpl-be8d6d925ff14741","object":"chat.completion.chunk","created":1765374283,"model":"Qwen/Qwen2.5-Coder-7B-Instruct-AWQ","choices":[{"index":0,"delta":{"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":"\"default\"}"}}]},"logprobs":null,"finish_reason":null,"token_ids":null}]}"#;
        let results = parse_openai_sse_chunk_with_tools(args_delta, &mut tool_states).unwrap();
        assert!(
            matches!(&results[0], ChatStreamChunk::ToolUseInputDelta { partial_json, .. } if partial_json == "\"default\"}"),
            "Expected ToolUseInputDelta, got {:?}",
            results
        );

        // Finish with stop reason
        let finish = r#"data: {"id":"chatcmpl-be8d6d925ff14741","object":"chat.completion.chunk","created":1765374283,"model":"Qwen/Qwen2.5-Coder-7B-Instruct-AWQ","choices":[{"index":0,"delta":{"reasoning_content":null,"tool_calls":[{"index":0,"function":{"arguments":""}}]},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}]}"#;
        let results = parse_openai_sse_chunk_with_tools(finish, &mut tool_states).unwrap();

        // Should have ToolUseComplete and Done
        assert!(
            results.len() >= 2,
            "Expected ToolUseComplete and Done, got {:?}",
            results
        );
        assert!(
            matches!(&results[0], ChatStreamChunk::ToolUseComplete { tool_call, .. } if tool_call.function.name == "db_list_databases"),
            "Expected ToolUseComplete, got {:?}",
            results[0]
        );
    }
}
