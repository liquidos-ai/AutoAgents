//! Ollama API client implementation for chat and completion functionality.
//!
//! This module provides integration with Ollama's local LLM server through its API.

use crate::{
    FunctionCall, ToolCall,
    builder::LLMBuilder,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
        Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::{EmbeddingBuilder, EmbeddingProvider},
    error::LLMError,
    models::ModelsProvider,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Provider-specific configuration for the Ollama backend.
#[derive(Debug, Default, Clone)]
pub struct OllamaConfig {
    pub keep_alive: Option<String>,
    pub system: Option<String>,
    pub think: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<i64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub num_ctx: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub min_p: Option<f32>,
}

/// Client for interacting with Ollama's API.
///
/// Provides methods for chat and completion requests using Ollama's models.
pub struct Ollama {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub keep_alive: Option<String>,
    pub system: Option<String>,
    pub think: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<i64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub num_ctx: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub min_p: Option<f32>,
    client: Client,
}

/// Request payload for Ollama's chat API endpoint.
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: String,
    messages: Vec<OllamaChatMessage<'a>>,
    stream: bool,
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<OllamaResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    keep_alive: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
}

/// Individual message in an Ollama chat conversation.
#[derive(Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCallRequest>>,
}

/// Response from Ollama's API endpoints.
#[derive(Deserialize, Debug)]
struct OllamaResponse {
    content: Option<String>,
    response: Option<String>,
    message: Option<OllamaChatResponseMessage>,
}

impl std::fmt::Display for OllamaResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let empty = String::default();
        let text = self
            .content
            .as_ref()
            .or(self.response.as_ref())
            .or(self.message.as_ref().map(|m| &m.content))
            .unwrap_or(&empty);

        // Write tool calls if present
        if let Some(message) = &self.message
            && let Some(tool_calls) = &message.tool_calls
        {
            for tc in tool_calls {
                writeln!(
                    f,
                    "{{\"name\": \"{}\", \"arguments\": {}}}",
                    tc.function.name,
                    serde_json::to_string_pretty(&tc.function.arguments).unwrap_or_default()
                )?;
            }
        }

        write!(f, "{text}")
    }
}

impl ChatResponse for OllamaResponse {
    fn text(&self) -> Option<String> {
        self.content
            .as_ref()
            .or(self.response.as_ref())
            .or(self.message.as_ref().map(|m| &m.content))
            .map(|s| s.to_string())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.message.as_ref().and_then(|msg| {
            msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .enumerate()
                    .map(|(idx, tc)| ToolCall {
                        id: format!("call_{}_{}", tc.function.name, idx),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_default(),
                        },
                    })
                    .collect()
            })
        })
    }
}

/// Message content within an Ollama chat API response.
#[derive(Deserialize, Debug)]
struct OllamaChatResponseMessage {
    content: String,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Request payload for Ollama's generate API endpoint.
#[derive(Serialize)]
struct OllamaGenerateRequest<'a> {
    model: String,
    prompt: &'a str,
    raw: bool,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
enum OllamaResponseType {
    Json,
    StructuredOutput(Value),
}

impl Serialize for OllamaResponseType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            OllamaResponseType::Json => serializer.serialize_str("json"),
            OllamaResponseType::StructuredOutput(schema) => schema.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for OllamaResponseType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(text) if text == "json" => Ok(OllamaResponseType::Json),
            other => Ok(OllamaResponseType::StructuredOutput(other)),
        }
    }
}

#[derive(Deserialize, Debug, Serialize)]
struct OllamaResponseFormat(OllamaResponseType);

/// Ollama's tool format
#[derive(Serialize, Debug)]
struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,

    pub function: OllamaFunctionTool,
}

#[derive(Serialize, Debug)]
struct OllamaFunctionTool {
    /// Name of the tool
    name: String,
    /// Description of what the tool does
    description: String,
    /// Parameters for the tool
    parameters: OllamaParameters,
}

impl From<&crate::chat::Tool> for OllamaTool {
    fn from(tool: &crate::chat::Tool) -> Self {
        let properties_value = tool
            .function
            .parameters
            .get("properties")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        let required_fields = tool
            .function
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        OllamaTool {
            tool_type: "function".to_owned(),
            function: OllamaFunctionTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: OllamaParameters {
                    schema_type: "object".to_string(),
                    properties: properties_value,
                    required: required_fields,
                },
            },
        }
    }
}

/// Ollama's parameters schema
#[derive(Serialize, Debug)]
struct OllamaParameters {
    /// The type of parameters object (usually "object")
    #[serde(rename = "type")]
    schema_type: String,
    /// Map of parameter names to their properties
    properties: Value,
    /// List of required parameter names
    required: Vec<String>,
}

/// Ollama's tool call response
#[derive(Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Serialize, Debug)]
struct OllamaToolCallRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: String,
    function: OllamaFunctionCallRequest,
}

#[derive(Serialize, Debug)]
struct OllamaFunctionCallRequest {
    name: String,
    arguments: Value,
}

#[derive(Deserialize, Debug)]
struct OllamaFunctionCall {
    /// Name of the tool that was called
    name: String,
    /// Arguments provided to the tool
    arguments: Value,
}

fn tool_args_to_value(args: &str) -> Value {
    serde_json::from_str(args).unwrap_or_else(|_| Value::String(args.to_string()))
}

fn chat_message_to_ollama_message<'a>(msg: &'a ChatMessage) -> OllamaChatMessage<'a> {
    OllamaChatMessage {
        role: match msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::System => "system",
            ChatRole::Tool => "tool",
        },
        content: match msg.message_type {
            MessageType::Text => &msg.content,
            MessageType::ToolUse(_) => "",
            MessageType::ToolResult(_) => &msg.content,
            _ => &msg.content,
        },
        tool_calls: match &msg.message_type {
            MessageType::ToolUse(calls) => Some(
                calls
                    .iter()
                    .map(|call| OllamaToolCallRequest {
                        id: Some(call.id.clone()),
                        call_type: "function".to_string(),
                        function: OllamaFunctionCallRequest {
                            name: call.function.name.clone(),
                            arguments: tool_args_to_value(&call.function.arguments),
                        },
                    })
                    .collect(),
            ),
            _ => None,
        },
    }
}

impl Ollama {
    /// Creates a new Ollama client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the Ollama server
    /// * `api_key` - Optional API key for authentication
    /// * `model` - Model name to use (defaults to "llama3.1")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `json_schema` - JSON schema for structured output
    /// * `tools` - Function tools that the model can use
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        keep_alive: Option<String>,
        system: Option<String>,
        think: Option<bool>,
        stop: Option<Vec<String>>,
        seed: Option<i64>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        num_ctx: Option<u32>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<i32>,
        min_p: Option<f32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            base_url: base_url.into(),
            api_key,
            model: model.unwrap_or_else(|| "llama3.1".to_string()),
            temperature,
            max_tokens,
            timeout_seconds,
            top_p,
            top_k,
            keep_alive,
            system,
            think,
            stop,
            seed,
            presence_penalty,
            frequency_penalty,
            num_ctx,
            repeat_penalty,
            repeat_last_n,
            min_p,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let chat_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .flat_map(|msg| {
                if let MessageType::ToolResult(ref results) = msg.message_type {
                    results
                        .iter()
                        .map(|result| OllamaChatMessage {
                            role: "tool",
                            content: &result.function.arguments,
                            tool_calls: None,
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![chat_message_to_ollama_message(msg)]
                }
            })
            .collect();

        // Convert tools to Ollama format if provided
        let ollama_tools = tools.map(|t| t.iter().map(OllamaTool::from).collect());

        // Ollama doesn't require the "name" field in the schema, so we just use the schema itself
        let format = if let Some(schema) = &json_schema {
            schema.schema.as_ref().map(|schema| {
                OllamaResponseFormat(OllamaResponseType::StructuredOutput(schema.clone()))
            })
        } else {
            None
        };

        let keep_alive: String = self.keep_alive.clone().unwrap_or_else(|| "0".into());

        let req_body = OllamaChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            stream: false,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
                top_p: self.top_p,
                top_k: self.top_k,
                num_ctx: self.num_ctx,
                seed: self.seed,
                stop: self.stop.clone(),
                repeat_penalty: self.repeat_penalty,
                repeat_last_n: self.repeat_last_n,
                presence_penalty: self.presence_penalty,
                frequency_penalty: self.frequency_penalty,
                min_p: self.min_p,
            }),
            keep_alive,
            format,
            tools: ollama_tools,
            system: self.system.clone(),
            think: self.think,
        };

        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(&req_body)
        {
            log::trace!("Ollama request payload (tools): {json}");
        }

        let url = format!("{}/api/chat", self.base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Ollama HTTP status (tools): {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp = resp.json::<OllamaResponse>().await?;

        Ok(Box::new(json_resp))
    }
}

#[async_trait]
impl ChatProvider for Ollama {
    /// Sends a chat request to Ollama's API.
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
        self.chat_with_tools(messages, None, json_schema).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, tools, json_schema).await
    }
}

#[async_trait]
impl CompletionProvider for Ollama {
    /// Sends a completion request to Ollama's API.
    ///
    /// # Arguments
    ///
    /// * `req` - The completion request containing the prompt
    ///
    /// # Returns
    ///
    /// The completion response containing the generated text or an error
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/api/generate", self.base_url);

        let req_body = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: &req.prompt,
            raw: true,
            stream: false,
            system: self.system.clone(),
            think: self.think,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
                top_p: self.top_p,
                top_k: self.top_k,
                num_ctx: self.num_ctx,
                seed: self.seed,
                stop: self.stop.clone(),
                repeat_penalty: self.repeat_penalty,
                repeat_last_n: self.repeat_last_n,
                presence_penalty: self.presence_penalty,
                frequency_penalty: self.frequency_penalty,
                min_p: self.min_p,
            }),
        };

        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?
            .error_for_status()?;
        let json_resp: OllamaResponse = resp.json().await?;

        if let Some(answer) = json_resp.response.or(json_resp.content) {
            Ok(CompletionResponse { text: answer })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by Ollama".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for Ollama {
    async fn embed(&self, text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/api/embed", self.base_url);

        let body = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: text,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OllamaEmbeddingResponse = resp.json().await?;
        Ok(json_resp.embeddings)
    }
}

#[async_trait]
impl ModelsProvider for Ollama {}

impl crate::LLMProvider for Ollama {}

impl crate::HasConfig for Ollama {
    type Config = OllamaConfig;
}

impl LLMBuilder<Ollama> {
    pub fn keep_alive(mut self, v: impl Into<String>) -> Self {
        self.config.keep_alive = Some(v.into());
        self
    }

    /// Sets the system message override.
    pub fn system(mut self, v: impl Into<String>) -> Self {
        self.config.system = Some(v.into());
        self
    }

    /// Enables or disables thinking/reasoning mode.
    pub fn think(mut self, v: bool) -> Self {
        self.config.think = Some(v);
        self
    }

    /// Sets stop sequences.
    pub fn stop(mut self, v: Vec<String>) -> Self {
        self.config.stop = Some(v);
        self
    }

    /// Sets a fixed seed for reproducible output.
    pub fn seed(mut self, v: i64) -> Self {
        self.config.seed = Some(v);
        self
    }

    /// Sets presence penalty.
    pub fn presence_penalty(mut self, v: f32) -> Self {
        self.config.presence_penalty = Some(v);
        self
    }

    /// Sets frequency penalty.
    pub fn frequency_penalty(mut self, v: f32) -> Self {
        self.config.frequency_penalty = Some(v);
        self
    }

    /// Sets the context window size.
    pub fn num_ctx(mut self, n: u32) -> Self {
        self.config.num_ctx = Some(n);
        self
    }

    /// Sets the repetition penalty.
    pub fn repeat_penalty(mut self, v: f32) -> Self {
        self.config.repeat_penalty = Some(v);
        self
    }

    /// Sets the look-back window for the repetition penalty.
    pub fn repeat_last_n(mut self, n: i32) -> Self {
        self.config.repeat_last_n = Some(n);
        self
    }

    /// Sets the min probability threshold (alternative to top_p).
    pub fn min_p(mut self, v: f32) -> Self {
        self.config.min_p = Some(v);
        self
    }

    pub fn build(self) -> Result<Arc<Ollama>, LLMError> {
        let url = self
            .base_url
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let ollama = Ollama::new(
            url,
            self.api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.config.keep_alive,
            self.config.system,
            self.config.think,
            self.config.stop,
            self.config.seed,
            self.config.presence_penalty,
            self.config.frequency_penalty,
            self.config.num_ctx,
            self.config.repeat_penalty,
            self.config.repeat_last_n,
            self.config.min_p,
        );

        Ok(Arc::new(ollama))
    }
}

impl EmbeddingBuilder<Ollama> {
    /// Build an Ollama embedding provider.
    pub fn build(self) -> Result<Arc<Ollama>, LLMError> {
        let model = self.model.ok_or_else(|| {
            LLMError::InvalidRequest("No model provided for Ollama embeddings".to_string())
        })?;

        let provider = Ollama::new(
            self.base_url
                .unwrap_or_else(|| "http://localhost:11434".to_string()),
            self.api_key,
            Some(model),
            None,
            None,
            self.timeout_seconds,
            None,
            None,
            None, // keep_alive
            None, // system
            None, // think
            None, // stop
            None, // seed
            None, // presence_penalty
            None, // frequency_penalty
            None, // num_ctx
            None, // repeat_penalty
            None, // repeat_last_n
            None, // min_p
        );

        Ok(Arc::new(provider))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{FunctionTool, Tool};
    use httpmock::{Method::POST, MockServer};
    use serde_json::json;

    #[test]
    fn test_ollama_response_text_priority() {
        let response = OllamaResponse {
            content: Some("content".to_string()),
            response: Some("response".to_string()),
            message: Some(OllamaChatResponseMessage {
                content: "message".to_string(),
                tool_calls: None,
            }),
        };
        assert_eq!(response.text(), Some("content".to_string()));

        let response = OllamaResponse {
            content: None,
            response: Some("response".to_string()),
            message: Some(OllamaChatResponseMessage {
                content: "message".to_string(),
                tool_calls: None,
            }),
        };
        assert_eq!(response.text(), Some("response".to_string()));

        let response = OllamaResponse {
            content: None,
            response: None,
            message: Some(OllamaChatResponseMessage {
                content: "message".to_string(),
                tool_calls: None,
            }),
        };
        assert_eq!(response.text(), Some("message".to_string()));
    }

    #[test]
    fn test_ollama_tool_calls_conversion() {
        let response = OllamaResponse {
            content: None,
            response: None,
            message: Some(OllamaChatResponseMessage {
                content: "tool".to_string(),
                tool_calls: Some(vec![OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: "lookup".to_string(),
                        arguments: json!({"q":"value"}),
                    },
                }]),
            }),
        };
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert!(calls[0].function.arguments.contains("\"q\""));
    }

    #[test]
    fn test_ollama_display_includes_tool_calls() {
        let response = OllamaResponse {
            content: Some("hello".to_string()),
            response: None,
            message: Some(OllamaChatResponseMessage {
                content: "ignored".to_string(),
                tool_calls: Some(vec![OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: "lookup".to_string(),
                        arguments: json!({"q":"value"}),
                    },
                }]),
            }),
        };
        let output = format!("{response}");
        assert!(output.contains("lookup"));
        assert!(output.contains("hello"));
    }

    #[test]
    fn test_ollama_tool_from_tool() {
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
        let ollama_tool = OllamaTool::from(&tool);
        assert_eq!(ollama_tool.tool_type, "function");
        assert_eq!(ollama_tool.function.name, "lookup");
    }

    #[test]
    fn test_ollama_response_format_serialization() {
        let format = OllamaResponseFormat(OllamaResponseType::Json);
        let serialized = serde_json::to_value(&format).unwrap();
        assert_eq!(serialized, json!("json"));
    }

    #[tokio::test]
    async fn test_ollama_chat_complete_and_embed_use_mock_server() {
        let server = MockServer::start();
        let provider = Ollama::new(
            server.base_url(),
            None,
            Some("llama3.2".to_string()),
            Some(128),
            Some(0.3),
            Some(5),
            Some(0.9),
            Some(20),
            Some("10m".to_string()),
            Some("system prompt".to_string()),
            Some(true),
            Some(vec!["STOP".to_string()]),
            Some(7),
            Some(0.1),
            Some(0.2),
            Some(2048),
            Some(1.1),
            Some(32),
            Some(0.05),
        );

        let chat_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/chat")
                .body_includes("\"keep_alive\":\"10m\"")
                .body_includes("\"system\":\"system prompt\"")
                .body_includes("\"think\":true")
                .body_includes("\"tools\"")
                .body_includes("\"format\"");
            then.status(200).json_body(json!({
                "message": {
                    "content": "ollama reply",
                    "tool_calls": [{
                        "function": {
                            "name": "lookup",
                            "arguments": { "q": "value" }
                        }
                    }]
                }
            }));
        });

        let messages = vec![ChatMessage::user().content("hello").build()];
        let response = provider
            .chat_with_tools(
                &messages,
                Some(&[Tool {
                    tool_type: "function".to_string(),
                    function: FunctionTool {
                        name: "lookup".to_string(),
                        description: "desc".to_string(),
                        parameters: json!({
                            "type": "object",
                            "properties": {
                                "q": { "type": "string" }
                            }
                        }),
                    },
                }]),
                Some(StructuredOutputFormat {
                    name: "Answer".to_string(),
                    description: None,
                    schema: Some(json!({
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        }
                    })),
                    strict: Some(true),
                }),
            )
            .await
            .expect("chat should succeed");
        assert_eq!(response.text().as_deref(), Some("ollama reply"));
        assert_eq!(
            response.tool_calls().expect("tool calls should exist")[0]
                .function
                .name,
            "lookup"
        );
        chat_mock.assert();

        let complete_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/generate")
                .body_includes("\"prompt\":\"finish this\"");
            then.status(200).json_body(json!({
                "response": "generated answer"
            }));
        });

        let completion = provider
            .complete(
                &CompletionRequest {
                    prompt: "finish this".to_string(),
                    max_tokens: None,
                    temperature: None,
                },
                None,
            )
            .await
            .expect("completion should succeed");
        assert_eq!(completion.text, "generated answer");
        complete_mock.assert();

        let embed_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/embed")
                .body_includes("\"model\":\"llama3.2\"");
            then.status(200).json_body(json!({
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]
                ]
            }));
        });

        let embeddings = provider
            .embed(vec!["a".to_string(), "b".to_string()])
            .await
            .expect("embedding should succeed");
        assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(embeddings[1], vec![0.4, 0.5, 0.6]);
        embed_mock.assert();
    }

    #[tokio::test]
    async fn test_ollama_missing_base_url_and_empty_completion_response_error() {
        let provider = Ollama::new(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
        );
        let messages = vec![ChatMessage::user().content("hello").build()];
        assert!(matches!(
            provider.chat_with_tools(&messages, None, None).await,
            Err(LLMError::InvalidRequest(message)) if message == "Missing base_url"
        ));
        assert!(matches!(
            provider
                .complete(
                    &CompletionRequest {
                        prompt: "prompt".to_string(),
                        max_tokens: None,
                        temperature: None,
                    },
                    None
                )
                .await,
            Err(LLMError::InvalidRequest(message)) if message == "Missing base_url"
        ));
        assert!(matches!(
            provider.embed(vec!["hello".to_string()]).await,
            Err(LLMError::InvalidRequest(message)) if message == "Missing base_url"
        ));

        let server = MockServer::start();
        let provider = Ollama::new(
            server.base_url(),
            None,
            Some("llama3.2".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mock = server.mock(|when, then| {
            when.method(POST).path("/api/generate");
            then.status(200).json_body(json!({
                "message": {
                    "content": "",
                    "tool_calls": null
                }
            }));
        });

        let err = provider
            .complete(
                &CompletionRequest {
                    prompt: "prompt".to_string(),
                    max_tokens: None,
                    temperature: None,
                },
                None,
            )
            .await
            .expect_err("missing response text should fail");
        assert!(
            matches!(err, LLMError::ProviderError(message) if message == "No answer returned by Ollama")
        );
        mock.assert();
    }
}
