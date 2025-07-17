//! OpenAI API client implementation for chat and completion functionality.
//!
//! This module provides integration with OpenAI's GPT models through their API.

use crate::{
    builder::LLMBackend,
    chat::Tool,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    LLMProvider,
};
use crate::{
    builder::LLMBuilder,
    chat::{ChatResponse, ChatStreamResponse, ToolChoice},
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use either::*;
use futures::stream::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Client for interacting with OpenAI's API.
///
/// Provides methods for chat and completion requests using OpenAI's models.
pub struct OpenAI {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    /// Embedding parameters
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub reasoning_effort: Option<String>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    pub voice: Option<String>,
    pub enable_web_search: Option<bool>,
    pub web_search_context_size: Option<String>,
    pub web_search_user_location_type: Option<String>,
    pub web_search_user_location_approximate_country: Option<String>,
    pub web_search_user_location_approximate_city: Option<String>,
    pub web_search_user_location_approximate_region: Option<String>,
    client: Client,
}

/// Individual message in an OpenAI chat conversation.
#[derive(Serialize, Debug)]
struct OpenAIChatMessage<'a> {
    #[allow(dead_code)]
    role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<MessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIFunctionCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
struct OpenAIFunctionPayload<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize, Debug)]
struct OpenAIFunctionCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    content_type: &'a str,
    function: OpenAIFunctionPayload<'a>,
}

#[derive(Serialize, Debug)]
struct MessageContent<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<&'a str>,
}

/// Individual image message in an OpenAI chat conversation.
#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// Request payload for OpenAI's chat API endpoint.
#[derive(Serialize, Debug)]
struct OpenAIChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAIChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_search_options: Option<OpenAIWebSearchOptions>,
}

impl std::fmt::Display for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  \"id\": \"{}\",\n  \"type\": \"{}\",\n  \"function\": {}\n}}",
            self.id, self.call_type, self.function
        )
    }
}

impl std::fmt::Display for FunctionCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  \"name\": \"{}\",\n  \"arguments\": {}\n}}",
            self.name, self.arguments
        )
    }
}

/// Response from OpenAI's chat API endpoint.
#[derive(Deserialize, Debug)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChatChoice>,
}

/// Individual choice within an OpenAI chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatChoice {
    message: OpenAIChatMsg,
}

/// Message content within an OpenAI chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatMsg {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}
#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

/// Response from OpenAI's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct OpenAIChatStreamResponse {
    choices: Vec<OpenAIChatStreamChoice>,
}

/// Individual choice within an OpenAI streaming chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatStreamChoice {
    delta: OpenAIChatStreamDelta,
}

/// Delta content within an OpenAI streaming chat API response.
/// Delta content within an OpenAI streaming chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<crate::chat::ToolCallChunk>>,
}

/// An object specifying the format that the model must output.
///Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
/// Setting to `{ "type": "json_object" }` enables the older JSON mode, which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for models that support it.
#[derive(Deserialize, Debug, Serialize)]
enum OpenAIResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    response_type: OpenAIResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

#[derive(Deserialize, Debug, Serialize)]
struct OpenAIWebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_location: Option<UserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_context_size: Option<String>,
}

#[derive(Deserialize, Debug, Serialize)]
struct UserLocation {
    #[serde(rename = "type")]
    location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    approximate: Option<ApproximateLocation>,
}

#[derive(Deserialize, Debug, Serialize)]
struct ApproximateLocation {
    country: String,
    city: String,
    region: String,
}

impl From<StructuredOutputFormat> for OpenAIResponseFormat {
    /// Modify the schema to ensure that it meets OpenAI's requirements.
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        // It's possible to pass a StructuredOutputJsonSchema without an actual schema.
        // In this case, just pass the StructuredOutputJsonSchema object without modifying it.
        match structured_response_format.schema {
            None => OpenAIResponseFormat {
                response_type: OpenAIResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                // Although [OpenAI's specifications](https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat#additionalproperties-false-must-always-be-set-in-objects) say that the "additionalProperties" field is required, my testing shows that it is not.
                // Just to be safe, add it to the schema if it is missing.
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
}

impl std::fmt::Display for OpenAIChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (
            &self.choices.first().unwrap().message.content,
            &self.choices.first().unwrap().message.tool_calls,
        ) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{}", tool_call)?;
                }
                write!(f, "{}", content)
            }
            (Some(content), None) => write!(f, "{}", content),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{}", tool_call)?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl OpenAI {
    /// Creates a new OpenAI client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `model` - Model to use (defaults to "gpt-3.5-turbo")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    /// * `embedding_encoding_format` - Format for embedding outputs
    /// * `embedding_dimensions` - Dimensions for embedding vectors
    /// * `tools` - Function tools that the model can use
    /// * `tool_choice` - Determines how the model uses tools
    /// * `reasoning_effort` - Reasoning effort level
    /// * `json_schema` - JSON schema for structured output
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        voice: Option<String>,
        enable_web_search: Option<bool>,
        web_search_context_size: Option<String>,
        web_search_user_location_type: Option<String>,
        web_search_user_location_approximate_country: Option<String>,
        web_search_user_location_approximate_city: Option<String>,
        web_search_user_location_approximate_region: Option<String>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            base_url: Url::parse(
                &base_url.unwrap_or_else(|| "https://api.openai.com/v1/".to_owned()),
            )
            .expect("Failed to prase base Url"),
            model: model.unwrap_or("gpt-3.5-turbo".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            reasoning_effort,
            json_schema,
            voice,
            enable_web_search,
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
        }
    }

    /// Enable web search
    pub fn set_enable_web_search(mut self, enable: bool) -> Self {
        self.enable_web_search = Some(enable);
        self
    }

    /// Set the web search context
    pub fn set_web_search_context_size(mut self, context_size: impl Into<String>) -> Self {
        self.web_search_context_size = Some(context_size.into());
        self
    }

    /// Set the web search user location type
    pub fn set_web_search_user_location_type(mut self, location_type: impl Into<String>) -> Self {
        self.web_search_user_location_type = Some(location_type.into());
        self
    }

    /// Set the web search user location approximate country
    pub fn set_web_search_user_location_approximate_country(
        mut self,
        country: impl Into<String>,
    ) -> Self {
        self.web_search_user_location_approximate_country = Some(country.into());
        self
    }

    /// Set the web search user location approximate city
    pub fn set_web_search_user_location_approximate_city(
        mut self,
        city: impl Into<String>,
    ) -> Self {
        self.web_search_user_location_approximate_city = Some(city.into());
        self
    }

    /// Set the web search user location approximate region
    pub fn set_web_search_user_location_approximate_region(
        mut self,
        region: impl Into<String>,
    ) -> Self {
        self.web_search_user_location_approximate_region = Some(region.into());
        self
    }
}

#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamResponse, LLMError>> + Send>>, LLMError> {
        let mut openai_messages = Vec::new();
        if let Some(system_prompt) = &self.system {
            openai_messages.push(OpenAIChatMessage {
                role: "system",
                content: Some(Either::Right(system_prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        for msg in messages {
            match &msg.message_type {
                MessageType::ToolResult(calls) => {
                    for call in calls {
                        openai_messages.push(OpenAIChatMessage {
                            role: "tool",
                            content: Some(Either::Right(call.function.arguments.clone())),
                            tool_calls: None,
                            tool_call_id: Some(call.id.clone()),
                        });
                    }
                }
                _ => {
                    let role = match msg.role {
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        ChatRole::System => "system",
                        ChatRole::Tool => "tool", // Should not happen due to the outer match
                    };

                    let (content, tool_calls) = match &msg.message_type {
                        MessageType::Text => (Some(Either::Right(msg.content.clone())), None),
                        MessageType::ToolUse(calls) => (
                            Some(Either::Right(msg.content.clone())),
                            Some(
                                calls
                                    .iter()
                                    .map(|c| OpenAIFunctionCall {
                                        id: &c.id,
                                        content_type: "function",
                                        function: OpenAIFunctionPayload {
                                            name: &c.function.name,
                                            arguments: &c.function.arguments,
                                        },
                                    })
                                    .collect(),
                            ),
                        ),
                        _ => (None, None), // ToolResult is handled above
                    };

                    openai_messages.push(OpenAIChatMessage {
                        role,
                        content,
                        tool_calls,
                        tool_call_id: None,
                    });
                }
            }
        }

        let request = OpenAIChatRequest {
            model: &self.model,
            messages: openai_messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true, // Explicitly set stream to true
            top_p: self.top_p,
            top_k: self.top_k,
            tools: tools.map(|t| t.to_vec()),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            response_format: json_schema.map(OpenAIResponseFormat::from),
            web_search_options: None, // Add web search options if needed
        };

        let mut url = self.base_url.clone();
        url.path_segments_mut()
            .unwrap()
            .push("chat")
            .push("completions");

        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(LLMError::ProviderError(error_body));
        }

        let stream = response.bytes_stream();

        let response_stream = async_stream::try_stream! {
            let mut buffer = Vec::new();
            for await chunk in stream {
                buffer.extend_from_slice(&chunk.map_err(|e| LLMError::HttpError(e.to_string()))?);

                while let Some(i) = buffer.iter().position(|&b| b == b'\n') {
                    let line = buffer.drain(..=i).collect::<Vec<u8>>();
                    let line_str = String::from_utf8_lossy(&line).trim().to_string();

                    if line_str.starts_with("data: ") {
                        let json_str = &line_str[6..];

                        if json_str == "[DONE]" {
                            break;
                        }

                        match serde_json::from_str::<OpenAIChatStreamResponse>(json_str) {
                            Ok(stream_response) => {
                                if let Some(choice) = stream_response.choices.into_iter().next() {
                                    let delta = choice.delta;
                                    let text = delta.content.unwrap_or_default();
                                    let tool_calls = delta.tool_calls.unwrap_or_default();

                                    yield ChatStreamResponse {
                                        text,
                                        tool_calls,
                                        finish_reason: None, // OpenAI doesn't seem to provide this per chunk
                                    };
                                }
                            }
                            Err(e) => {
                                yield Err(LLMError::ResponseFormatError{ message: e.to_string(), raw_response: json_str.to_string() })?;
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(response_stream))
    }

    /// Sends a chat request to OpenAI's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
        }

        // Clone the messages to have an owned mutable vector.
        let messages = messages.to_vec();

        let mut openai_msgs: Vec<OpenAIChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    openai_msgs.push(
                        // Clone strings to own them
                        OpenAIChatMessage {
                            role: "tool",
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        },
                    );
                }
            } else {
                openai_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
                    role: "system",
                    content: Some(Left(vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(system),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let response_format: Option<OpenAIResponseFormat> = json_schema.clone().map(|s| s.into());

        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());

        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let web_search_options = if self.enable_web_search.unwrap_or(false) {
            let loc_type_opt = self
                .web_search_user_location_type
                .as_ref()
                .filter(|t| matches!(t.as_str(), "exact" | "approximate"));

            let country = self.web_search_user_location_approximate_country.as_ref();
            let city = self.web_search_user_location_approximate_city.as_ref();
            let region = self.web_search_user_location_approximate_region.as_ref();

            let approximate = if [country, city, region].iter().any(|v| v.is_some()) {
                Some(ApproximateLocation {
                    country: country.cloned().unwrap_or_default(),
                    city: city.cloned().unwrap_or_default(),
                    region: region.cloned().unwrap_or_default(),
                })
            } else {
                None
            };

            let user_location = loc_type_opt.map(|loc_type| UserLocation {
                location_type: loc_type.clone(),
                approximate,
            });

            Some(OpenAIWebSearchOptions {
                search_context_size: self.web_search_context_size.clone(),
                user_location,
            })
        } else {
            None
        };

        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
            response_format,
            web_search_options,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("OpenAI request payload: {}", json);
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        log::debug!("OpenAI HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {}", status),
                raw_response: error_text,
            });
        }

        // Parse the successful response
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI API response: {}", e),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None, json_schema).await
    }

    /// Sends a streaming chat request to OpenAI's API.
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
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
        }

        let messages = messages.to_vec();
        let mut openai_msgs: Vec<OpenAIChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    openai_msgs.push(OpenAIChatMessage {
                        role: "tool",
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                openai_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
                    role: "system",
                    content: Some(Left(vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(system),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            response_format: None,
            web_search_options: None,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {}", status),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(response, parse_sse_chunk))
    }
}

// Create an owned OpenAIChatMessage that doesn't borrow from any temporary variables
fn chat_message_to_api_message(chat_msg: ChatMessage) -> OpenAIChatMessage<'static> {
    // For other message types, create an owned OpenAIChatMessage
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
            // Image case is handled separately above
            MessageType::Image(_) => unreachable!(),
            MessageType::Pdf(_) => unimplemented!(),
            MessageType::ImageURL(url) => {
                // Clone the URL to create an owned version
                let owned_url = url.clone();
                // Leak the string to get a 'static reference
                let url_str = Box::leak(owned_url.into_boxed_str());
                Some(Left(vec![MessageContent {
                    message_type: Some("image_url"),
                    text: None,
                    image_url: Some(ImageUrlContent { url: url_str }),
                    tool_output: None,
                    tool_call_id: None,
                }]))
            }
            MessageType::ToolUse(_) => None,
            MessageType::ToolResult(_) => None,
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                let owned_calls: Vec<OpenAIFunctionCall<'static>> = calls
                    .iter()
                    .map(|c| {
                        let owned_id = c.id.clone();
                        let owned_name = c.function.name.clone();
                        let owned_args = c.function.arguments.clone();

                        // Need to leak these strings to create 'static references
                        // This is a deliberate choice to solve the lifetime issue
                        // The small memory leak is acceptable in this context
                        let id_str = Box::leak(owned_id.into_boxed_str());
                        let name_str = Box::leak(owned_name.into_boxed_str());
                        let args_str = Box::leak(owned_args.into_boxed_str());

                        OpenAIFunctionCall {
                            id: id_str,
                            content_type: "function",
                            function: OpenAIFunctionPayload {
                                name: name_str,
                                arguments: args_str,
                            },
                        }
                    })
                    .collect();
                Some(owned_calls)
            }
            _ => None,
        },
    }
}

#[async_trait]
impl CompletionProvider for OpenAI {
    /// Sends a completion request to OpenAI's API.
    ///
    /// Currently not implemented.
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "OpenAI completion not implemented.".into(),
        })
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        let emb_format = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = OpenAIEmbeddingRequest {
            model: self.model.clone(),
            input,
            encoding_format: Some(emb_format),
            dimensions: self.embedding_dimensions,
        };

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OpenAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAIModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for OpenAIModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|t| chrono::DateTime::from_timestamp(t as i64, 0).unwrap_or_default())
            .unwrap_or_default()
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAIModelListResponse {
    pub data: Vec<OpenAIModelEntry>,
}

impl ModelListResponse for OpenAIModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|e| e.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::OpenAI
    }
}

#[async_trait]
impl ModelsProvider for OpenAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .get(url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = resp.json::<OpenAIModelListResponse>().await?;

        Ok(Box::new(result))
    }
}

impl LLMProvider for OpenAI {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

/// Parses a Server-Sent Events (SSE) chunk from OpenAI's streaming API.
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
fn parse_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    let mut collected_content = String::new();

    for line in chunk.lines() {
        let line = line.trim();

        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                if collected_content.is_empty() {
                    return Ok(None);
                } else {
                    return Ok(Some(collected_content));
                }
            }

            match serde_json::from_str::<OpenAIChatStreamResponse>(data) {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
    }

    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(collected_content))
    }
}

impl LLMBuilder<OpenAI> {
    /// Set the voice.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    pub fn build(self) -> Result<Arc<OpenAI>, LLMError> {
        let (tools, tool_choice) = self.validate_tool_config()?;
        let key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for OpenAI".to_string())
        })?;
        let openai = OpenAI::new(
            key,
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.stream,
            self.top_p,
            self.top_k,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            tools,
            tool_choice,
            self.reasoning_effort,
            self.json_schema,
            self.voice,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        Ok(Arc::new(openai))
    }
}
