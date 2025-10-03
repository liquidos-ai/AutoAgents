//! Azure OpenAI API client implementation for chat and completion functionality.
//!
//! This module provides integration with Azure OpenAI's GPT models through their API.

use crate::{
    builder::LLMBuilder,
    chat::{ChatResponse, ToolChoice},
    FunctionCall, ToolCall,
};
use crate::{
    chat::Tool,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use either::*;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Client for interacting with Azure OpenAI's API.
///
/// Provides methods for chat and completion requests using Azure OpenAI's models.
pub struct AzureOpenAI {
    pub api_key: String,
    pub api_version: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tool_choice: Option<ToolChoice>,
    /// Embedding parameters
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub reasoning_effort: Option<String>,
    client: Client,
}

/// Individual message in an OpenAI chat conversation.
#[derive(Serialize, Debug)]
struct AzureOpenAIChatMessage<'a> {
    #[allow(dead_code)]
    role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<AzureMessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<AzureOpenAIToolCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl<'a> From<&'a ChatMessage> for AzureOpenAIChatMessage<'a> {
    fn from(chat_msg: &'a ChatMessage) -> Self {
        Self {
            role: match chat_msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::System => "system",
                ChatRole::Tool => "tool",
            },
            tool_call_id: None,
            content: match &chat_msg.message_type {
                MessageType::Text => Some(Right(chat_msg.content.clone())),
                MessageType::Image((image_mime, raw_bytes)) => {
                    // Convert raw bytes to base64 data URL
                    let base64_data = BASE64_STANDARD.encode(raw_bytes);
                    let data_url =
                        format!("data:{};base64,{}", image_mime.mime_type(), base64_data);
                    let data_url_str = Box::leak(data_url.into_boxed_str());
                    Some(Left(vec![AzureMessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url: data_url_str }),
                        tool_output: None,
                        tool_call_id: None,
                    }]))
                }
                MessageType::Pdf(_) => unimplemented!(),
                MessageType::ImageURL(url) => {
                    // Clone the URL to create an owned version

                    Some(Left(vec![AzureMessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url }),
                        tool_output: None,
                        tool_call_id: None,
                    }]))
                }
                MessageType::ToolUse(_) => None,
                MessageType::ToolResult(_) => None,
            },
            tool_calls: match &chat_msg.message_type {
                MessageType::ToolUse(calls) => {
                    let owned_calls: Vec<AzureOpenAIToolCall> =
                        calls.iter().map(|c| c.into()).collect();
                    Some(owned_calls)
                }
                _ => None,
            },
        }
    }
}

#[derive(Serialize, Debug)]
struct AzureOpenAIFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

impl<'a> From<&'a FunctionCall> for AzureOpenAIFunctionCall<'a> {
    fn from(value: &'a FunctionCall) -> Self {
        Self {
            name: &value.name,
            arguments: &value.arguments,
        }
    }
}

#[derive(Serialize, Debug)]
struct AzureOpenAIToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    content_type: &'a str,
    function: AzureOpenAIFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for AzureOpenAIToolCall<'a> {
    fn from(value: &'a ToolCall) -> Self {
        Self {
            id: &value.id,
            content_type: "function",
            function: AzureOpenAIFunctionCall::from(&value.function),
        }
    }
}

#[derive(Serialize, Debug)]
struct AzureMessageContent<'a> {
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

/// Request payload for Azure OpenAI's chat API endpoint.
#[derive(Serialize, Debug)]
struct AzureOpenAIChatRequest<'a> {
    model: &'a str,
    messages: Vec<AzureOpenAIChatMessage<'a>>,
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
}

/// Response from OpenAI's chat API endpoint.
#[derive(Deserialize, Debug)]
struct AzureOpenAIChatResponse {
    choices: Vec<AzureOpenAIChatChoice>,
}

/// Individual choice within an OpenAI chat API response.
#[derive(Deserialize, Debug)]
struct AzureOpenAIChatChoice {
    message: AzureOpenAIChatMsg,
}

/// Message content within an OpenAI chat API response.
#[derive(Deserialize, Debug)]
struct AzureOpenAIChatMsg {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
struct AzureOpenAIEmbeddingData {
    embedding: Vec<f32>,
}
#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingResponse {
    data: Vec<AzureOpenAIEmbeddingData>,
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

impl ChatResponse for AzureOpenAIChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }
}

impl std::fmt::Display for AzureOpenAIChatResponse {
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

impl AzureOpenAI {
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
        api_version: impl Into<String>,
        deployment_id: impl Into<String>,
        endpoint: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }

        let endpoint = endpoint.into();
        let deployment_id = deployment_id.into();

        Self {
            api_key: api_key.into(),
            api_version: api_version.into(),
            base_url: Url::parse(&format!("{endpoint}/openai/deployments/{deployment_id}/"))
                .expect("Failed to parse base Url"),
            model: model.unwrap_or("gpt-3.5-turbo".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            top_p,
            top_k,
            tool_choice,
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            reasoning_effort,
        }
    }

    fn build_chat_completion_request<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        tools: Option<&'a [Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        _stream: bool,
        _stream_options: Option<AzureOpenAIChatRequest>,
    ) -> Result<AzureOpenAIChatRequest<'a>, LLMError> {
        let mut openai_msgs: Vec<AzureOpenAIChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    openai_msgs.push(
                        // Clone strings to own them
                        AzureOpenAIChatMessage {
                            role: "tool",
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        },
                    );
                }
            } else {
                openai_msgs.push(msg.into())
            }
        }

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                AzureOpenAIChatMessage {
                    role: "system",
                    content: Some(Left(vec![AzureMessageContent {
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

        // Build the response format object
        let response_format: Option<OpenAIResponseFormat> = json_schema.clone().map(|s| s.into());

        let request_tools = tools.map(|t| t.to_vec());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        Ok(AzureOpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: false,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
            response_format,
        })
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
            return Err(LLMError::AuthError(
                "Missing Azure OpenAI API key".to_string(),
            ));
        }

        let body = self.build_chat_completion_request(messages, tools, json_schema, false, None)?;

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Azure OpenAI request payload: {json}");
            }
        }

        let mut url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        url.query_pairs_mut()
            .append_pair("api-version", &self.api_version);

        let mut request = self
            .client
            .post(url)
            .header("api-key", &self.api_key)
            .json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        // Send the request
        let response = request.send().await?;

        log::debug!("Azure OpenAI HTTP status: {}", response.status());

        // If we got a non-200 response, let's get the error details
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        // Parse the successful response
        let resp_text = response.text().await?;
        let json_resp: Result<AzureOpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode Azure OpenAI API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }
}

#[async_trait]
impl ChatProvider for AzureOpenAI {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, tools, json_schema).await
    }
}

#[async_trait]
impl CompletionProvider for AzureOpenAI {
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

#[cfg(feature = "azure_openai")]
#[async_trait]
impl EmbeddingProvider for AzureOpenAI {
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

        let mut url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        url.query_pairs_mut()
            .append_pair("api-version", &self.api_version);

        let resp = self
            .client
            .post(url)
            .header("api-key", &self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OpenAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

impl LLMProvider for AzureOpenAI {}

#[async_trait]
impl ModelsProvider for AzureOpenAI {}

impl LLMBuilder<AzureOpenAI> {
    pub fn build(self) -> Result<Arc<AzureOpenAI>, LLMError> {
        let endpoint = self.base_url.ok_or_else(|| {
            LLMError::InvalidRequest("No API endpoint provided for Azure OpenAI".into())
        })?;

        let key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for Azure OpenAI".to_string())
        })?;

        let api_version = self.api_version.ok_or_else(|| {
            LLMError::InvalidRequest("No API version provided for Azure OpenAI".to_string())
        })?;

        let deployment = self.deployment_id.ok_or_else(|| {
            LLMError::InvalidRequest("No deployment ID provided for Azure OpenAI".into())
        })?;

        let provider = AzureOpenAI::new(
            key,
            api_version,
            deployment,
            endpoint,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.system,
            self.top_p,
            self.top_k,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            self.tool_choice,
            self.reasoning_effort,
        );

        Ok(Arc::new(provider))
    }
}
