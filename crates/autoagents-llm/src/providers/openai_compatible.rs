//! OpenAI-compatible API client base implementation
//!
//! This module provides a generic base for OpenAI-compatible APIs that can be reused
//! across multiple providers like OpenAI, Mistral, XAI, Groq, DeepSeek, etc.

use crate::chat::{StreamChoice, StreamDelta, StreamToolCallDelta, StreamToolCallFunction};
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
use std::marker::PhantomData;

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
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<String>,
    #[allow(dead_code)]
    pub voice: Option<String>,
    pub parallel_tool_calls: bool,
    #[allow(dead_code)]
    pub embedding_encoding_format: Option<String>,
    #[allow(dead_code)]
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
        // Ensure description is present (required by some providers like Groq)
        let description = structured_response_format
            .description
            .or_else(|| Some("Generated schema".to_string()));

        match structured_response_format.schema {
            None => OpenAIResponseFormat {
                response_type: OpenAIResponseType::JsonSchema,
                json_schema: Some(StructuredOutputFormat {
                    name: structured_response_format.name,
                    description,
                    schema: structured_response_format.schema,
                    strict: structured_response_format.strict,
                }),
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
                        description,
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
                    write!(f, "{tool_call:?}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call:?}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl<T: OpenAIProviderConfig> OpenAICompatibleProvider<T> {
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn new(
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
        reasoning_effort: Option<String>,
        voice: Option<String>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            base_url: Url::parse(&base_url.unwrap_or_else(|| T::DEFAULT_BASE_URL.to_owned()))
                .expect("Failed to parse base URL"),
            model: model.unwrap_or_else(|| T::DEFAULT_MODEL.to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            top_p,
            top_k,
            tool_choice,
            reasoning_effort,
            voice,
            parallel_tool_calls: parallel_tool_calls.unwrap_or(false),
            normalize_response: normalize_response.unwrap_or(true),
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            _phantom: PhantomData,
        }
    }

    pub fn prepare_messages(&self, messages: &[ChatMessage]) -> Vec<OpenAIChatMessage<'_>> {
        let mut openai_msgs: Vec<OpenAIChatMessage> = messages
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
        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
                    role: "system",
                    content: Some(Left(vec![OpenAIMessageContent {
                        message_type: Some("text"),
                        text: Some(system.as_str()),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
        openai_msgs
    }
}

#[async_trait]
impl<T: OpenAIProviderConfig> ChatProvider for OpenAICompatibleProvider<T> {
    /// Perform a chat request
    async fn chat(
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
            json_schema
                .clone()
                .or_else(|| json_schema.clone())
                .map(|s| s.into())
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

    /// Stream chat responses as a stream of strings
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let struct_stream = self
            .chat_stream_struct(messages, tools, json_schema)
            .await?;
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
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };
        let response_format: Option<OpenAIResponseFormat> = if T::SUPPORTS_STRUCTURED_OUTPUT {
            json_schema
                .clone()
                .or_else(|| json_schema.clone())
                .map(|s| s.into())
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
            tool_choice: request_tool_choice,
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
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
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
}

/// Create OpenAICompatibleChatMessage` that doesn't borrow from any temporary variables
pub fn chat_message_to_openai_message(chat_msg: ChatMessage) -> OpenAIChatMessage<'static> {
    OpenAIChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::System => "system",
            ChatRole::Tool => "tool",
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
                // Convert ToolCall to StreamToolCallDelta for normalized output
                let tool_call_delta = StreamToolCallDelta {
                    index: 0,
                    function: Some(StreamToolCallFunction {
                        name: self.tool_buffer.function.name.clone(),
                        arguments: self.tool_buffer.function.arguments.clone(),
                    }),
                };
                self.results.push(Ok(StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: None,
                            tool_calls: Some(vec![tool_call_delta]),
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

                    if self.normalize_response {
                        // Map StreamToolCall (some fields are optional) to ToolCall for normalization
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
                        }
                    } else {
                        // When not normalizing, convert StreamToolCall to StreamToolCallDelta
                        let tool_call_deltas: Option<Vec<StreamToolCallDelta>> =
                            choice.delta.tool_calls.clone().map(|calls| {
                                calls
                                    .into_iter()
                                    .enumerate()
                                    .map(|(idx, c)| StreamToolCallDelta {
                                        index: idx,
                                        function: c.function.name.map(|name| {
                                            StreamToolCallFunction {
                                                name,
                                                arguments: c.function.arguments,
                                            }
                                        }),
                                    })
                                    .collect()
                            });

                        if content.is_some() || tool_call_deltas.is_some() {
                            self.push_tool_call();
                            self.results.push(Ok(StreamResponse {
                                choices: vec![StreamChoice {
                                    delta: StreamDelta {
                                        content,
                                        tool_calls: tool_call_deltas,
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
