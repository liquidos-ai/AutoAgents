// This file will be completely replaced to fix the syntax errors

//! Google Gemini API client implementation for chat and completion functionality.
//!
//! This module provides integration with Google's Gemini models through their API.
//! It implements chat, completion and embedding capabilities via the Gemini API.
//!
//! # Features
//! - Chat conversations with system prompts and message history
//! - Text completion requests
//! - Configuration options for temperature, tokens, top_p, top_k etc.
//! - Streaming support
//!
//! ```

use crate::{
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
        Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    FunctionCall, LLMProvider, ToolCall,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Client for interacting with Google's Gemini API.
///
/// This struct holds the configuration and state needed to make requests to the Gemini API.
/// It implements the [`ChatProvider`], [`CompletionProvider`], and [`EmbeddingProvider`] traits.
pub struct Google {
    /// API key for authentication with Google's API
    pub api_key: String,
    /// Model identifier (e.g. "gemini-1.5-flash")
    pub model: String,
    /// Maximum number of tokens to generate in responses
    pub max_tokens: Option<u32>,
    /// Sampling temperature between 0.0 and 1.0
    pub temperature: Option<f32>,
    /// Optional system prompt to set context
    pub system: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Whether to stream responses
    pub stream: Option<bool>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    /// Available tools for function calling
    pub tools: Option<Vec<Tool>>,
    /// HTTP client for making API requests
    client: Client,
}

/// Request body for chat completions
#[derive(Serialize)]
struct GoogleChatRequest<'a> {
    /// List of conversation messages
    contents: Vec<GoogleChatContent<'a>>,
    /// Optional generation parameters
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    generation_config: Option<GoogleGenerationConfig>,
    /// Tools that the model can use
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GoogleTool>>,
}

/// Individual message in a chat conversation
#[derive(Serialize)]
struct GoogleChatContent<'a> {
    /// Role of the message sender ("user", "model", or "system")
    role: &'a str,
    /// Content parts of the message
    parts: Vec<GoogleContentPart<'a>>,
}

/// Text content within a chat message
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
enum GoogleContentPart<'a> {
    /// The actual text content
    #[serde(rename = "text")]
    Text(&'a str),
    InlineData(GoogleInlineData),
    FunctionCall(GoogleFunctionCall),
    #[serde(rename = "functionResponse")]
    FunctionResponse(GoogleFunctionResponse),
}

#[derive(Serialize)]
struct GoogleInlineData {
    mime_type: String,
    data: String,
}

/// Configuration parameters for text generation
#[derive(Serialize)]
struct GoogleGenerationConfig {
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    max_output_tokens: Option<u32>,
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f32>,
    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    top_k: Option<u32>,
    /// The MIME type of the response
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<GoogleResponseMimeType>,
    /// A schema for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
}

/// Response from the chat completion API
#[derive(Deserialize, Debug)]
struct GoogleChatResponse {
    /// Generated completion candidates
    candidates: Vec<GoogleCandidate>,
}

/// Response from the streaming chat completion API
#[derive(Deserialize, Debug)]
struct GoogleStreamResponse {
    /// Generated completion candidates
    candidates: Option<Vec<GoogleCandidate>>,
}

impl std::fmt::Display for GoogleChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.text(), self.tool_calls()) {
            (Some(text), Some(tool_calls)) => {
                for call in tool_calls {
                    write!(f, "{call}")?;
                }
                write!(f, "{text}")
            }
            (Some(text), None) => write!(f, "{text}"),
            (None, Some(tool_calls)) => {
                for call in tool_calls {
                    write!(f, "{call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

/// Individual completion candidate
#[derive(Deserialize, Debug)]
struct GoogleCandidate {
    /// Content of the candidate response
    content: GoogleResponseContent,
}

/// Content block within a response
#[derive(Deserialize, Debug)]
struct GoogleResponseContent {
    /// Parts making up the content (might be absent when only function calls are present)
    #[serde(default)]
    parts: Vec<GoogleResponsePart>,
    /// Function calls if any are used - can be a single object or array
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GoogleFunctionCall>,
    /// Function calls as array (newer format in some responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    function_calls: Option<Vec<GoogleFunctionCall>>,
}

impl ChatResponse for GoogleChatResponse {
    fn text(&self) -> Option<String> {
        self.candidates
            .first()
            .map(|c| c.content.parts.iter().map(|p| p.text.clone()).collect())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.candidates.first().and_then(|c| {
            // First check for function calls at the part level (new API format)
            let part_function_calls: Vec<ToolCall> = c
                .content
                .parts
                .iter()
                .filter_map(|part| {
                    part.function_call.as_ref().map(|f| ToolCall {
                        id: format!("call_{}", f.name),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: f.name.clone(),
                            arguments: serde_json::to_string(&f.args).unwrap_or_default(),
                        },
                    })
                })
                .collect();

            if !part_function_calls.is_empty() {
                return Some(part_function_calls);
            }

            // Otherwise check for function_calls/function_call at the content level (older format)
            if let Some(fc) = &c.content.function_calls {
                // Process array of function calls
                Some(
                    fc.iter()
                        .map(|f| ToolCall {
                            id: format!("call_{}", f.name),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: f.name.clone(),
                                arguments: serde_json::to_string(&f.args).unwrap_or_default(),
                            },
                        })
                        .collect(),
                )
            } else {
                c.content.function_call.as_ref().map(|f| {
                    vec![ToolCall {
                        id: format!("call_{}", f.name),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: f.name.clone(),
                            arguments: serde_json::to_string(&f.args).unwrap_or_default(),
                        },
                    }]
                })
            }
        })
    }
}

/// Individual part of response content
#[derive(Deserialize, Debug)]
struct GoogleResponsePart {
    /// Text content of this part (may be absent if functionCall is present)
    #[serde(default)]
    text: String,
    /// Function call contained in this part
    #[serde(rename = "functionCall")]
    function_call: Option<GoogleFunctionCall>,
}

/// MIME type of the response
#[derive(Deserialize, Debug, Serialize)]
enum GoogleResponseMimeType {
    /// Plain text response
    #[serde(rename = "text/plain")]
    PlainText,
    /// JSON response
    #[serde(rename = "application/json")]
    Json,
    /// ENUM as a string response in the response candidates.
    #[serde(rename = "text/x.enum")]
    Enum,
}

/// Google's function calling tool definition
#[derive(Serialize, Debug)]
struct GoogleTool {
    /// The function declarations array
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GoogleFunctionDeclaration>,
}

/// Google function declaration, similar to OpenAI's function definition
#[derive(Serialize, Debug)]
struct GoogleFunctionDeclaration {
    /// Name of the function
    name: String,
    /// Description of what the function does
    description: String,
    /// Parameters for the function
    parameters: GoogleFunctionParameters,
}

impl From<&crate::chat::Tool> for GoogleFunctionDeclaration {
    fn from(tool: &crate::chat::Tool) -> Self {
        let properties_value = tool
            .function
            .parameters
            .get("properties")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        GoogleFunctionDeclaration {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: GoogleFunctionParameters {
                schema_type: "object".to_string(),
                properties: properties_value,
                required: tool
                    .function
                    .parameters
                    .get("required")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect::<Vec<String>>()
                    })
                    .unwrap_or_default(),
            },
        }
    }
}

/// Google function parameters schema
#[derive(Serialize, Debug)]
struct GoogleFunctionParameters {
    /// The type of parameters object (usually "object")
    #[serde(rename = "type")]
    schema_type: String,
    /// Map of parameter names to their properties
    properties: Value,
    /// List of required parameter names
    required: Vec<String>,
}

/// Google function call object in response
#[derive(Deserialize, Debug, Serialize)]
struct GoogleFunctionCall {
    /// Name of the function to call
    name: String,
    /// Arguments for the function call as structured JSON
    #[serde(default)]
    args: Value,
}

/// Google function response wrapper for function results
///
/// Format follows Google's Gemini API specification for function calling results:
/// https://ai.google.dev/docs/function_calling
///
/// The expected format is:
/// {
///   "role": "function",
///   "parts": [{
///     "functionResponse": {
///       "name": "function_name",
///       "response": {
///         "name": "function_name",
///         "content": { ... } // JSON content returned by the function
///       }
///     }
///   }]
/// }
#[derive(Deserialize, Debug, Serialize)]
struct GoogleFunctionResponse {
    /// Name of the function that was called
    name: String,
    /// Response from the function as structured JSON
    response: GoogleFunctionResponseContent,
}

#[derive(Deserialize, Debug, Serialize)]
struct GoogleFunctionResponseContent {
    /// Name of the function that was called
    name: String,
    /// Content of the function response
    content: Value,
}

/// Request body for embedding content
#[derive(Serialize)]
struct GoogleEmbeddingRequest<'a> {
    model: &'a str,
    content: GoogleEmbeddingContent<'a>,
}

#[derive(Serialize)]
struct GoogleEmbeddingContent<'a> {
    parts: Vec<GoogleContentPart<'a>>,
}

/// Response from the embedding API
#[derive(Deserialize)]
struct GoogleEmbeddingResponse {
    embedding: GoogleEmbedding,
}

#[derive(Deserialize)]
struct GoogleEmbedding {
    values: Vec<f32>,
}

impl Google {
    /// Creates a new Google Gemini client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Google API key for authentication
    /// * `model` - Model identifier (defaults to "gemini-1.5-flash")
    /// * `max_tokens` - Maximum tokens in response
    /// * `temperature` - Sampling temperature between 0.0 and 1.0
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt to set context
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    /// * `json_schema` - JSON schema for structured output
    /// * `tools` - Function tools that the model can use
    ///
    /// # Returns
    ///
    /// A new `Google` client instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| "gemini-1.5-flash".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            json_schema,
            tools,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Google {
    /// Sends a chat request to Google's Gemini API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Google API key".to_string()));
        }

        let mut chat_contents = Vec::with_capacity(messages.len());

        // Add system message if present
        if let Some(system) = &self.system {
            chat_contents.push(GoogleChatContent {
                role: "user",
                parts: vec![GoogleContentPart::Text(system)],
            });
        }

        // Add conversation messages in pairs to maintain context
        for msg in messages {
            // For tool results, we need to use "function" role
            let role = match &msg.message_type {
                MessageType::ToolResult(_) => "function",
                _ => match msg.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "model",
                    ChatRole::Tool => "tool",
                    ChatRole::System => "system",
                },
            };

            chat_contents.push(GoogleChatContent {
                role,
                parts: match &msg.message_type {
                    MessageType::Text => vec![GoogleContentPart::Text(&msg.content)],
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: image_mime.mime_type().to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    MessageType::ImageURL(_) => unimplemented!(),
                    MessageType::Pdf(raw_bytes) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: "application/pdf".to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|call| {
                            GoogleContentPart::FunctionCall(GoogleFunctionCall {
                                name: call.function.name.clone(),
                                args: serde_json::from_str(&call.function.arguments)
                                    .unwrap_or(serde_json::Value::Null),
                            })
                        })
                        .collect(),
                    MessageType::ToolResult(result) => result
                        .iter()
                        .map(|result| {
                            let parsed_args =
                                serde_json::from_str::<Value>(&result.function.arguments)
                                    .unwrap_or(serde_json::Value::Null);

                            GoogleContentPart::FunctionResponse(GoogleFunctionResponse {
                                name: result.function.name.clone(),
                                response: GoogleFunctionResponseContent {
                                    name: result.function.name.clone(),
                                    content: parsed_args,
                                },
                            })
                        })
                        .collect(),
                },
            });
        }

        // Remove generation_config if empty to avoid validation errors
        let generation_config = if self.max_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && json_schema.is_none()
        {
            None
        } else {
            // If json_schema and json_schema.schema are not None, use json_schema.schema as the response schema and set response_mime_type to JSON
            // Google's API doesn't need the schema to have a "name" field, so we can just use the schema directly.
            let (response_mime_type, response_schema) = if let Some(json_schema) = &json_schema {
                if let Some(schema) = &json_schema.schema {
                    // If the schema has an "additionalProperties" field (as required by OpenAI), remove it as Google's API doesn't support it
                    let mut schema = schema.clone();

                    if let Some(obj) = schema.as_object_mut() {
                        obj.remove("additionalProperties");
                    }

                    (Some(GoogleResponseMimeType::Json), Some(schema))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            Some(GoogleGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                response_mime_type,
                response_schema,
            })
        };

        let req_body = GoogleChatRequest {
            contents: chat_contents,
            generation_config,
            tools: None,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Google Gemini request payload: {json}");
            }
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            model = self.model,
            key = self.api_key
        );

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Google Gemini HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        // Get the raw response text for debugging
        let resp_text = resp.text().await?;

        // Try to parse the response
        let json_resp: Result<GoogleChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => {
                // Return a more descriptive error with the raw response
                Err(LLMError::ResponseFormatError {
                    message: format!("Failed to decode Google API response: {e}"),
                    raw_response: resp_text,
                })
            }
        }
    }

    /// Sends a chat request to Google's Gemini API with tools.
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
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Google API key".to_string()));
        }

        let mut chat_contents = Vec::with_capacity(messages.len());

        // Add system message if present
        if let Some(system) = &self.system {
            chat_contents.push(GoogleChatContent {
                role: "user",
                parts: vec![GoogleContentPart::Text(system)],
            });
        }

        // Add conversation messages in pairs to maintain context
        for msg in messages {
            // For tool results, we need to use "function" role
            let role = match &msg.message_type {
                MessageType::ToolResult(_) => "function",
                _ => match msg.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "model",
                    ChatRole::Tool => "tool",
                    ChatRole::System => "system",
                },
            };

            chat_contents.push(GoogleChatContent {
                role,
                parts: match &msg.message_type {
                    MessageType::Text => vec![GoogleContentPart::Text(&msg.content)],
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: image_mime.mime_type().to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    MessageType::ImageURL(_) => unimplemented!(),
                    MessageType::Pdf(raw_bytes) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: "application/pdf".to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|call| {
                            GoogleContentPart::FunctionCall(GoogleFunctionCall {
                                name: call.function.name.clone(),
                                args: serde_json::from_str(&call.function.arguments)
                                    .unwrap_or(serde_json::Value::Null),
                            })
                        })
                        .collect(),
                    MessageType::ToolResult(result) => result
                        .iter()
                        .map(|result| {
                            let parsed_args =
                                serde_json::from_str::<Value>(&result.function.arguments)
                                    .unwrap_or(serde_json::Value::Null);

                            GoogleContentPart::FunctionResponse(GoogleFunctionResponse {
                                name: result.function.name.clone(),
                                response: GoogleFunctionResponseContent {
                                    name: result.function.name.clone(),
                                    content: parsed_args,
                                },
                            })
                        })
                        .collect(),
                },
            });
        }

        // Convert tools to Google's format if provided
        let google_tools = tools.map(|t| {
            vec![GoogleTool {
                function_declarations: t.iter().map(GoogleFunctionDeclaration::from).collect(),
            }]
        });

        // Build generation config
        let generation_config = {
            // If json_schema and json_schema.schema are not None, use json_schema.schema as the response schema and set response_mime_type to JSON
            // Google's API doesn't need the schema to have a "name" field, so we can just use the schema directly.
            let (response_mime_type, response_schema) = if let Some(json_schema) = &json_schema {
                if let Some(schema) = &json_schema.schema {
                    // If the schema has an "additionalProperties" field (as required by OpenAI), remove it as Google's API doesn't support it
                    let mut schema = schema.clone();

                    if let Some(obj) = schema.as_object_mut() {
                        obj.remove("additionalProperties");
                    }

                    (Some(GoogleResponseMimeType::Json), Some(schema))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            Some(GoogleGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                response_mime_type,
                response_schema,
            })
        };

        let req_body = GoogleChatRequest {
            contents: chat_contents,
            generation_config,
            tools: google_tools,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Google Gemini request payload (tool): {json}");
            }
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            model = self.model,
            key = self.api_key

        );

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Google Gemini HTTP status (tool): {}", resp.status());

        let resp = resp.error_for_status()?;

        // Get the raw response text for debugging
        let resp_text = resp.text().await?;

        // Try to parse the response
        let json_resp: Result<GoogleChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => {
                // Return a more descriptive error with the raw response
                Err(LLMError::ResponseFormatError {
                    message: format!("Failed to decode Google API response: {e}"),
                    raw_response: resp_text,
                })
            }
        }
    }

    /// Sends a streaming chat request to Google's Gemini API.
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
            return Err(LLMError::AuthError("Missing Google API key".to_string()));
        }

        let mut chat_contents = Vec::with_capacity(messages.len());

        if let Some(system) = &self.system {
            chat_contents.push(GoogleChatContent {
                role: "user",
                parts: vec![GoogleContentPart::Text(system)],
            });
        }

        for msg in messages {
            let role = match msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
                ChatRole::Tool => "tool",
                ChatRole::System => "system",
            };

            chat_contents.push(GoogleChatContent {
                role,
                parts: match &msg.message_type {
                    MessageType::Text => vec![GoogleContentPart::Text(&msg.content)],
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: image_mime.mime_type().to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    MessageType::Pdf(raw_bytes) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: "application/pdf".to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    _ => vec![GoogleContentPart::Text(&msg.content)],
                },
            });
        }

        let generation_config = if self.max_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
        {
            None
        } else {
            Some(GoogleGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                response_mime_type: None,
                response_schema: None,
            })
        };

        let req_body = GoogleChatRequest {
            contents: chat_contents,
            generation_config,
            tools: None,
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={key}",
            model = self.model,
            key = self.api_key
        );

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Google API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(
            response,
            parse_google_sse_chunk,
        ))
    }
}

#[async_trait]
impl CompletionProvider for Google {
    /// Performs a completion request using the chat endpoint.
    ///
    /// # Arguments
    ///
    /// * `req` - Completion request parameters
    ///
    /// # Returns
    ///
    /// The completion response or an error
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let chat_message = ChatMessage::user().content(req.prompt.clone()).build();
        if let Some(text) = self.chat(&[chat_message], json_schema).await?.text() {
            Ok(CompletionResponse { text })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by Google".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for Google {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Google API key".to_string()));
        }

        let mut embeddings = Vec::new();

        // Process each text separately as Gemini API accepts one text at a time
        for text in texts {
            let req_body = GoogleEmbeddingRequest {
                model: "models/text-embedding-004",
                content: GoogleEmbeddingContent {
                    parts: vec![GoogleContentPart::Text(&text)],
                },
            };

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={}",
                self.api_key
            );

            let resp = self
                .client
                .post(&url)
                .json(&req_body)
                .send()
                .await?
                .error_for_status()?;

            let embedding_resp: GoogleEmbeddingResponse = resp.json().await?;
            embeddings.push(embedding_resp.embedding.values);
        }

        Ok(embeddings)
    }
}

impl LLMProvider for Google {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

/// Parses a Server-Sent Events (SSE) chunk from Google's streaming API.
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
fn parse_google_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();

        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<GoogleStreamResponse>(data) {
                Ok(response) => {
                    if let Some(candidates) = response.candidates {
                        if let Some(candidate) = candidates.first() {
                            if let Some(part) = candidate.content.parts.first() {
                                if !part.text.is_empty() {
                                    return Ok(Some(part.text.clone()));
                                }
                            }
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

#[async_trait]
impl ModelsProvider for Google {}

impl LLMBuilder<Google> {
    pub fn build(self) -> Result<Arc<Google>, LLMError> {
        let (tools, _) = self.validate_tool_config()?;
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for Google".to_string())
        })?;

        let google = crate::backends::google::Google::new(
            api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.stream,
            self.top_p,
            self.top_k,
            self.json_schema,
            tools,
        );

        Ok(Arc::new(google))
    }
}
