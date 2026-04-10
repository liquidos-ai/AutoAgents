//! OpenAI API client implementation using the OpenAI-compatible base.
//!
//! This module provides integration with OpenAI's GPT models through their API.

use crate::builder::{LLMBackend, LLMBuilder};
use crate::chat::Usage;
use crate::embedding::EmbeddingBuilder;
use crate::providers::openai_compatible::{
    OpenAIChatMessage, OpenAIChatResponse, OpenAICompatibleProvider, OpenAIProviderConfig,
    OpenAIResponseFormat, OpenAIStreamOptions, create_sse_stream,
};
use crate::{
    FunctionCall, LLMProvider, ToolCall,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChoice, StreamChunk,
        StreamDelta, StreamResponse, StructuredOutputFormat, Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

/// OpenAI chat API mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAIApiMode {
    /// Use the Responses API.
    #[default]
    Responses,
    /// Use the legacy chat/completions endpoint.
    ChatCompletions,
}

/// Provider-specific configuration for the OpenAI builder.
#[derive(Debug, Default, Clone)]
pub struct OpenAIConfig {
    pub voice: Option<String>,
    pub api_mode: OpenAIApiMode,
}

/// Internal OpenAI provider config (for OpenAICompatibleProvider).
struct OpenAIInternalCfg;

impl OpenAIProviderConfig for OpenAIInternalCfg {
    const PROVIDER_NAME: &'static str = "OpenAI";
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1/";
    const DEFAULT_MODEL: &'static str = "gpt-4.1-nano";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

/// Client for OpenAI API.
pub struct OpenAI {
    provider: OpenAICompatibleProvider<OpenAIInternalCfg>,
    pub api_mode: OpenAIApiMode,
    pub enable_web_search: bool,
    pub web_search_context_size: Option<String>,
    pub web_search_user_location_type: Option<String>,
    pub web_search_user_location_approximate_country: Option<String>,
    pub web_search_user_location_approximate_city: Option<String>,
    pub web_search_user_location_approximate_region: Option<String>,
}

/// Legacy chat/completions tool representation.
#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum OpenAITool {
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: crate::chat::FunctionTool,
    },
    WebSearch {
        #[serde(rename = "type")]
        tool_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
}

/// Responses API tool representation.
#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
enum OpenAIResponsesTool {
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        name: String,
        description: String,
        parameters: Value,
        strict: bool,
    },
    WebSearch {
        #[serde(rename = "type")]
        tool_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
}

#[derive(Deserialize, Debug, Serialize, Clone)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approximate: Option<ApproximateLocation>,
}

#[derive(Deserialize, Debug, Serialize, Clone)]
pub struct ApproximateLocation {
    pub country: String,
    pub city: String,
    pub region: String,
}

/// Request payload for OpenAI's chat/completions endpoint.
#[derive(Serialize, Debug)]
pub struct OpenAIAPIChatRequest<'a> {
    pub model: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<OpenAIChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenAIStreamOptions>,
    #[serde(flatten)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize, Debug)]
struct OpenAIResponsesRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: Vec<OpenAIResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAIResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAIResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<OpenAIResponsesTextConfig>,
    store: bool,
    #[serde(flatten)]
    extra_body: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
enum OpenAIResponsesInputItem {
    Message(OpenAIResponsesMessageInput),
    FunctionCall {
        #[serde(rename = "type")]
        item_type: &'static str,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
    FunctionCallOutput {
        #[serde(rename = "type")]
        item_type: &'static str,
        call_id: String,
        output: String,
    },
}

#[derive(Serialize, Debug)]
struct OpenAIResponsesMessageInput {
    role: String,
    content: OpenAIResponsesMessageContent,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
enum OpenAIResponsesMessageContent {
    Text(String),
    Parts(Vec<OpenAIResponsesMessagePart>),
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum OpenAIResponsesMessagePart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
}

#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
enum OpenAIResponsesToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        tool_type: &'static str,
        name: String,
    },
}

#[derive(Serialize, Debug, Clone)]
struct OpenAIResponsesReasoning {
    effort: String,
}

#[derive(Serialize, Debug, Clone)]
struct OpenAIResponsesTextConfig {
    format: OpenAIResponsesTextFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAIResponsesTextFormat {
    #[serde(rename = "type")]
    format_type: OpenAIResponsesTextFormatType,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum OpenAIResponsesTextFormatType {
    #[serde(rename = "json_schema")]
    JsonSchema,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponsesResponse {
    #[serde(default)]
    output: Vec<OpenAIResponsesOutputItem>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    output_text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponsesOutputItem {
    #[serde(rename = "type")]
    item_type: String,
    #[serde(default)]
    content: Vec<OpenAIResponsesOutputContent>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    summary: Vec<OpenAIResponsesReasoningSummary>,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponsesOutputContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponsesReasoningSummary {
    #[serde(default)]
    text: Option<String>,
}

impl std::fmt::Display for OpenAIResponsesResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.text(), self.tool_calls()) {
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

impl ChatResponse for OpenAIResponsesResponse {
    fn text(&self) -> Option<String> {
        if let Some(text) = &self.output_text
            && !text.is_empty()
        {
            return Some(text.clone());
        }

        let text = self
            .output
            .iter()
            .filter(|item| item.item_type == "message")
            .flat_map(|item| item.content.iter())
            .filter(|part| {
                matches!(
                    part.content_type.as_str(),
                    "output_text" | "text" | "input_text"
                )
            })
            .filter_map(|part| part.text.clone())
            .collect::<Vec<_>>()
            .concat();

        if text.is_empty() { None } else { Some(text) }
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        let tool_calls = self
            .output
            .iter()
            .filter(|item| item.item_type == "function_call")
            .filter_map(|item| {
                Some(ToolCall {
                    id: item.call_id.clone()?,
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: item.name.clone()?,
                        arguments: item.arguments.clone().unwrap_or_default(),
                    },
                })
            })
            .collect::<Vec<_>>();

        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        }
    }

    fn thinking(&self) -> Option<String> {
        let thinking = self
            .output
            .iter()
            .filter(|item| item.item_type == "reasoning")
            .flat_map(|item| item.summary.iter())
            .filter_map(|summary| summary.text.clone())
            .collect::<Vec<_>>()
            .concat();

        if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

impl From<StructuredOutputFormat> for OpenAIResponsesTextFormat {
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        let schema = structured_response_format
            .schema
            .map(normalize_responses_json_schema);

        Self {
            format_type: OpenAIResponsesTextFormatType::JsonSchema,
            name: structured_response_format.name,
            description: structured_response_format.description,
            schema,
            strict: structured_response_format.strict,
        }
    }
}

fn normalize_responses_json_schema(mut schema: Value) -> Value {
    normalize_responses_json_schema_inner(&mut schema);
    schema
}

fn normalize_responses_json_schema_inner(schema: &mut Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };

    let existing_required = object
        .get("required")
        .and_then(Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<std::collections::HashSet<_>>()
        })
        .unwrap_or_default();

    if let Some(properties_value) = object.get_mut("properties")
        && let Some(properties) = properties_value.as_object_mut()
    {
        let property_names = properties.keys().cloned().collect::<Vec<_>>();
        for property_name in &property_names {
            if let Some(property_schema) = properties.get_mut(property_name) {
                normalize_responses_json_schema_inner(property_schema);
                if !existing_required.contains(property_name) {
                    make_schema_nullable(property_schema);
                }
            }
        }

        object.insert(
            "required".to_string(),
            Value::Array(property_names.into_iter().map(Value::String).collect()),
        );
        object
            .entry("additionalProperties".to_string())
            .or_insert(Value::Bool(false));
    }

    if let Some(items) = object.get_mut("items") {
        normalize_responses_json_schema_inner(items);
    }

    if let Some(any_of) = object.get_mut("anyOf").and_then(Value::as_array_mut) {
        for branch in any_of {
            normalize_responses_json_schema_inner(branch);
        }
    }

    if let Some(one_of) = object.get_mut("oneOf").and_then(Value::as_array_mut) {
        for branch in one_of {
            normalize_responses_json_schema_inner(branch);
        }
    }

    if let Some(all_of) = object.get_mut("allOf").and_then(Value::as_array_mut) {
        for branch in all_of {
            normalize_responses_json_schema_inner(branch);
        }
    }
}

fn make_schema_nullable(schema: &mut Value) {
    if schema_is_nullable(schema) {
        return;
    }

    let original = schema.take();
    *schema = serde_json::json!({
        "anyOf": [
            original,
            { "type": "null" }
        ]
    });
}

fn schema_is_nullable(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };

    if object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| ty == "null")
    {
        return true;
    }

    if object
        .get("type")
        .and_then(Value::as_array)
        .is_some_and(|types| types.iter().any(|ty| ty.as_str() == Some("null")))
    {
        return true;
    }

    object
        .get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|variants| variants.iter().any(schema_is_nullable))
}

#[derive(Debug, Default)]
struct ResponsesToolCallState {
    call_id: String,
    item_id: Option<String>,
    name: String,
    arguments_buffer: String,
    started: bool,
}

#[derive(Debug, Default)]
struct ResponsesStreamState {
    tool_states: HashMap<usize, ResponsesToolCallState>,
    saw_tool_completion: bool,
    finished: bool,
}

impl OpenAI {
    /// Creates a new OpenAI client with the specified configuration.
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
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tool_choice: Option<ToolChoice>,
        normalize_response: Option<bool>,
        reasoning_effort: Option<String>,
        voice: Option<String>,
        api_mode: OpenAIApiMode,
        extra_body: Option<serde_json::Value>,
        enable_web_search: Option<bool>,
        web_search_context_size: Option<String>,
        web_search_user_location_type: Option<String>,
        web_search_user_location_approximate_country: Option<String>,
        web_search_user_location_approximate_city: Option<String>,
        web_search_user_location_approximate_region: Option<String>,
    ) -> Result<Self, LLMError> {
        let api_key_str = api_key.into();
        if api_key_str.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
        }

        Ok(Self {
            provider: OpenAICompatibleProvider::<OpenAIInternalCfg>::new(
                api_key_str,
                base_url,
                model,
                max_tokens,
                temperature,
                timeout_seconds,
                top_p,
                top_k,
                tool_choice,
                reasoning_effort,
                voice,
                extra_body,
                None,
                normalize_response,
                embedding_encoding_format,
                embedding_dimensions,
            ),
            api_mode,
            enable_web_search: enable_web_search.unwrap_or(false),
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
        })
    }
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

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        match self.api_mode {
            OpenAIApiMode::Responses => {
                self.chat_with_tools_responses(messages, tools, json_schema)
                    .await
            }
            OpenAIApiMode::ChatCompletions => {
                self.chat_with_tools_legacy(messages, tools, json_schema)
                    .await
            }
        }
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        let web_search_tool = self.build_web_search_tool();
        self.chat_with_hosted_tools(input, vec![web_search_tool])
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        let struct_stream = self.chat_stream_struct(messages, None, json_schema).await?;
        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(stream_response) => {
                    if let Some(choice) = stream_response.choices.first()
                        && let Some(content) = &choice.delta.content
                        && !content.is_empty()
                    {
                        return Some(Ok(content.clone()));
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::pin(content_stream))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        match self.api_mode {
            OpenAIApiMode::Responses => {
                let chunk_stream = self
                    .chat_stream_with_tools_responses(messages, tools, json_schema)
                    .await?;
                let struct_stream = chunk_stream.filter_map(|result| async move {
                    match result {
                        Ok(StreamChunk::Text(content)) => Some(Ok(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: Some(content),
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                            }],
                            usage: None,
                        })),
                        Ok(StreamChunk::ReasoningContent(content)) => Some(Ok(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: None,
                                    reasoning_content: Some(content),
                                    tool_calls: None,
                                },
                            }],
                            usage: None,
                        })),
                        Ok(StreamChunk::ToolUseComplete { tool_call, .. }) => {
                            Some(Ok(StreamResponse {
                                choices: vec![StreamChoice {
                                    delta: StreamDelta {
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: Some(vec![tool_call]),
                                    },
                                }],
                                usage: None,
                            }))
                        }
                        Ok(StreamChunk::Usage(usage)) => Some(Ok(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                            }],
                            usage: Some(usage),
                        })),
                        Ok(StreamChunk::Done { .. })
                        | Ok(StreamChunk::ToolUseStart { .. })
                        | Ok(StreamChunk::ToolUseInputDelta { .. }) => None,
                        Err(err) => Some(Err(err)),
                    }
                });
                Ok(Box::pin(struct_stream))
            }
            OpenAIApiMode::ChatCompletions => {
                self.chat_stream_struct_legacy(messages, tools, json_schema)
                    .await
            }
        }
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        match self.api_mode {
            OpenAIApiMode::Responses => {
                self.chat_stream_with_tools_responses(messages, tools, json_schema)
                    .await
            }
            OpenAIApiMode::ChatCompletions => {
                self.provider
                    .chat_stream_with_tools(messages, tools, json_schema)
                    .await
            }
        }
    }
}

#[async_trait]
impl CompletionProvider for OpenAI {
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

#[async_trait]
impl ModelsProvider for OpenAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url()
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client()
            .get(url)
            .bearer_auth(self.api_key())
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::OpenAI,
        };
        Ok(Box::new(result))
    }
}

impl LLMProvider for OpenAI {}

impl crate::HasConfig for OpenAI {
    type Config = OpenAIConfig;
}

impl OpenAI {
    pub fn api_key(&self) -> &str {
        &self.provider.api_key
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }

    pub fn base_url(&self) -> &reqwest::Url {
        &self.provider.base_url
    }

    pub fn timeout_seconds(&self) -> Option<u64> {
        self.provider.timeout_seconds
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.provider.client
    }

    fn build_legacy_function_tools(&self, tools: Option<&[Tool]>) -> Option<Vec<OpenAITool>> {
        let mut openai_tools = Vec::new();
        if let Some(tools) = tools {
            for tool in tools {
                openai_tools.push(OpenAITool::Function {
                    tool_type: tool.tool_type.clone(),
                    function: tool.function.clone(),
                });
            }
        }

        if openai_tools.is_empty() {
            None
        } else {
            Some(openai_tools)
        }
    }

    fn build_responses_function_tools(
        &self,
        tools: Option<&[Tool]>,
    ) -> Option<Vec<OpenAIResponsesTool>> {
        let mut openai_tools = Vec::new();
        if let Some(tools) = tools {
            for tool in tools {
                openai_tools.push(OpenAIResponsesTool::Function {
                    tool_type: tool.tool_type.clone(),
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                    strict: false,
                });
            }
        }

        if openai_tools.is_empty() {
            None
        } else {
            Some(openai_tools)
        }
    }

    fn resolve_legacy_tool_choice_for_request(
        &self,
        tools: &Option<Vec<OpenAITool>>,
    ) -> Option<ToolChoice> {
        if tools.is_some() {
            self.provider.tool_choice.clone()
        } else {
            None
        }
    }

    fn resolve_responses_tool_choice_for_request(
        &self,
        tools: &Option<Vec<OpenAIResponsesTool>>,
    ) -> Option<OpenAIResponsesToolChoice> {
        if tools.is_none() {
            return None;
        }

        self.provider
            .tool_choice
            .as_ref()
            .map(OpenAIResponsesToolChoice::from)
    }

    fn build_web_search_tool(&self) -> OpenAIResponsesTool {
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
        OpenAIResponsesTool::WebSearch {
            tool_type: "web_search_preview".to_string(),
            user_location,
        }
    }

    fn build_responses_request(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        stream: bool,
    ) -> Result<OpenAIResponsesRequest, LLMError> {
        let (instructions, input) = self.build_responses_input_items(messages)?;
        let request_tools = self.build_responses_function_tools(tools);
        let tool_choice = self.resolve_responses_tool_choice_for_request(&request_tools);
        let text = json_schema.map(|schema| OpenAIResponsesTextConfig {
            format: schema.into(),
        });

        Ok(OpenAIResponsesRequest {
            model: self.provider.model.clone(),
            instructions,
            input,
            max_output_tokens: self.provider.max_tokens,
            temperature: self.provider.temperature,
            stream,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: request_tools,
            tool_choice,
            reasoning: self
                .provider
                .reasoning_effort
                .clone()
                .map(|effort| OpenAIResponsesReasoning { effort }),
            text,
            store: false,
            extra_body: self.provider.extra_body.clone(),
        })
    }

    fn build_responses_request_for_input(
        &self,
        input: Vec<OpenAIResponsesInputItem>,
        tools: Option<Vec<OpenAIResponsesTool>>,
        stream: bool,
    ) -> OpenAIResponsesRequest {
        OpenAIResponsesRequest {
            model: self.provider.model.clone(),
            instructions: None,
            input,
            max_output_tokens: self.provider.max_tokens,
            temperature: self.provider.temperature,
            stream,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools,
            tool_choice: None,
            reasoning: self
                .provider
                .reasoning_effort
                .clone()
                .map(|effort| OpenAIResponsesReasoning { effort }),
            text: None,
            store: false,
            extra_body: self.provider.extra_body.clone(),
        }
    }

    fn build_responses_input_items(
        &self,
        messages: &[ChatMessage],
    ) -> Result<(Option<String>, Vec<OpenAIResponsesInputItem>), LLMError> {
        let mut instructions_parts = Vec::new();
        let mut input_items = Vec::new();
        let mut index = 0usize;

        while let Some(message) = messages.get(index) {
            if message.role == ChatRole::System {
                instructions_parts.push(message.content.clone());
                index += 1;
            } else {
                break;
            }
        }

        for message in &messages[index..] {
            match &message.message_type {
                MessageType::Text => {
                    input_items.push(OpenAIResponsesInputItem::Message(
                        OpenAIResponsesMessageInput {
                            role: self.responses_role_for_message(message).to_string(),
                            content: OpenAIResponsesMessageContent::Text(message.content.clone()),
                        },
                    ));
                }
                MessageType::Image((mime, bytes)) => {
                    let mut parts = Vec::new();
                    if !message.content.is_empty() {
                        parts.push(OpenAIResponsesMessagePart::InputText {
                            text: message.content.clone(),
                        });
                    }
                    parts.push(OpenAIResponsesMessagePart::InputImage {
                        image_url: format!(
                            "data:{};base64,{}",
                            mime.mime_type(),
                            BASE64.encode(bytes)
                        ),
                    });
                    input_items.push(OpenAIResponsesInputItem::Message(
                        OpenAIResponsesMessageInput {
                            role: self.responses_role_for_message(message).to_string(),
                            content: OpenAIResponsesMessageContent::Parts(parts),
                        },
                    ));
                }
                MessageType::ImageURL(url) => {
                    let mut parts = Vec::new();
                    if !message.content.is_empty() {
                        parts.push(OpenAIResponsesMessagePart::InputText {
                            text: message.content.clone(),
                        });
                    }
                    parts.push(OpenAIResponsesMessagePart::InputImage {
                        image_url: url.clone(),
                    });
                    input_items.push(OpenAIResponsesInputItem::Message(
                        OpenAIResponsesMessageInput {
                            role: self.responses_role_for_message(message).to_string(),
                            content: OpenAIResponsesMessageContent::Parts(parts),
                        },
                    ));
                }
                MessageType::Pdf(_) => {
                    return Err(LLMError::InvalidRequest(
                        "PDF input is not supported by the OpenAI Responses backend".to_string(),
                    ));
                }
                MessageType::ToolUse(tool_calls) => {
                    if !message.content.is_empty() {
                        input_items.push(OpenAIResponsesInputItem::Message(
                            OpenAIResponsesMessageInput {
                                role: "assistant".to_string(),
                                content: OpenAIResponsesMessageContent::Text(
                                    message.content.clone(),
                                ),
                            },
                        ));
                    }
                    for tool_call in tool_calls {
                        input_items.push(OpenAIResponsesInputItem::FunctionCall {
                            item_type: "function_call",
                            call_id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            arguments: tool_call.function.arguments.clone(),
                            status: "completed",
                        });
                    }
                }
                MessageType::ToolResult(results) => {
                    for result in results {
                        input_items.push(OpenAIResponsesInputItem::FunctionCallOutput {
                            item_type: "function_call_output",
                            call_id: result.id.clone(),
                            output: result.function.arguments.clone(),
                        });
                    }
                }
            }
        }

        let instructions = if instructions_parts.is_empty() {
            None
        } else {
            Some(instructions_parts.join("\n\n"))
        };

        Ok((instructions, input_items))
    }

    fn responses_role_for_message(&self, message: &ChatMessage) -> &'static str {
        match message.role {
            ChatRole::System => "developer",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "assistant",
        }
    }

    async fn send_responses_request(
        &self,
        body: &OpenAIResponsesRequest,
    ) -> Result<reqwest::Response, LLMError> {
        let url = self
            .provider
            .base_url
            .join("responses")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(body);

        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(body)
        {
            log::trace!("OpenAI Responses payload: {}", json);
        }

        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;
        log::debug!("OpenAI Responses HTTP status: {}", response.status());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI Responses API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(response)
    }

    async fn chat_with_tools_responses(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let body = self.build_responses_request(messages, tools, json_schema, false)?;
        let response = self.send_responses_request(&body).await?;
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIResponsesResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI Responses API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat_with_tools_legacy(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let openai_msgs = self.provider.prepare_messages(messages);
        let response_format: Option<OpenAIResponseFormat> = json_schema.clone().map(|s| s.into());
        let final_tools = self.build_legacy_function_tools(tools);
        let request_tool_choice = self.resolve_legacy_tool_choice_for_request(&final_tools);
        let body = OpenAIAPIChatRequest {
            model: self.provider.model.as_str(),
            messages: openai_msgs,
            input: None,
            max_completion_tokens: self.provider.max_tokens,
            max_output_tokens: None,
            temperature: self.provider.temperature,
            stream: false,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: final_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.provider.reasoning_effort.clone(),
            response_format,
            stream_options: None,
            extra_body: self.provider.extra_body.clone(),
        };
        let url = self
            .provider
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body);
        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(&body)
        {
            log::trace!("OpenAI request payload: {}", json);
        }
        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        log::debug!("OpenAI HTTP status: {}", response.status());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat_stream_struct_legacy(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        let openai_msgs = self.provider.prepare_messages(messages);
        let openai_tools = self.build_legacy_function_tools(tools);
        let response_schema: Option<OpenAIResponseFormat> = json_schema.map(|schema| schema.into());
        let body = OpenAIAPIChatRequest {
            model: &self.provider.model,
            messages: openai_msgs,
            input: None,
            max_completion_tokens: self.provider.max_tokens,
            max_output_tokens: None,
            temperature: self.provider.temperature,
            stream: true,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: openai_tools,
            tool_choice: self.provider.tool_choice.clone(),
            reasoning_effort: self.provider.reasoning_effort.clone(),
            response_format: response_schema,
            stream_options: Some(OpenAIStreamOptions {
                include_usage: true,
            }),
            extra_body: self.provider.extra_body.clone(),
        };
        let url = self
            .provider
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body);
        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(create_sse_stream(
            response,
            self.provider.normalize_response,
        ))
    }

    async fn chat_stream_with_tools_responses(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        let body = self.build_responses_request(messages, tools, json_schema, true)?;
        let response = self.send_responses_request(&body).await?;
        Ok(create_responses_tool_stream(response))
    }

    async fn chat_with_hosted_tools(
        &self,
        input: String,
        hosted_tools: Vec<OpenAIResponsesTool>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let body = self.build_responses_request_for_input(
            vec![OpenAIResponsesInputItem::Message(
                OpenAIResponsesMessageInput {
                    role: "user".to_string(),
                    content: OpenAIResponsesMessageContent::Text(input),
                },
            )],
            Some(hosted_tools),
            false,
        );
        let response = self.send_responses_request(&body).await?;
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIResponsesResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI Responses API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }
}

fn find_sse_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|pos| (pos, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|pos| (pos, 4));

    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(boundary), None) | (None, Some(boundary)) => Some(boundary),
        (None, None) => None,
    }
}

fn parse_sse_data_payload(event: &str) -> Option<String> {
    let mut payload = String::default();
    for line in event.lines() {
        let line = line.trim();
        let data_opt = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:").map(|d| d.trim_start()));
        if let Some(data) = data_opt {
            payload.push_str(data);
        }
    }

    if payload.is_empty() {
        None
    } else {
        Some(payload)
    }
}

fn usage_from_value(value: &Value) -> Option<Usage> {
    if let Some(usage_value) = value.get("usage") {
        serde_json::from_value(usage_value.clone()).ok()
    } else {
        value
            .get("response")
            .and_then(|response| response.get("usage"))
            .and_then(|usage| serde_json::from_value(usage.clone()).ok())
    }
}

fn finalize_responses_stream(
    state: &mut ResponsesStreamState,
    include_done: bool,
) -> Vec<StreamChunk> {
    let mut results = Vec::new();
    for (index, tool_state) in state.tool_states.drain() {
        if !tool_state.name.is_empty() {
            state.saw_tool_completion = true;
            results.push(StreamChunk::ToolUseComplete {
                index,
                tool_call: ToolCall {
                    id: tool_state.call_id,
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: tool_state.name,
                        arguments: tool_state.arguments_buffer,
                    },
                },
            });
        }
    }

    if include_done {
        state.finished = true;
        results.push(StreamChunk::Done {
            stop_reason: if state.saw_tool_completion {
                "tool_use".to_string()
            } else {
                "end_turn".to_string()
            },
        });
    }

    results
}

fn parse_responses_sse_event(
    event: &str,
    state: &mut ResponsesStreamState,
) -> Result<Vec<StreamChunk>, LLMError> {
    let Some(data_payload) = parse_sse_data_payload(event) else {
        return Ok(Vec::new());
    };

    let data_trimmed = data_payload.trim();
    if data_trimmed == "[DONE]" {
        if state.finished {
            return Ok(Vec::new());
        }
        return Ok(finalize_responses_stream(state, true));
    }

    let value: Value = serde_json::from_str(data_trimmed)?;
    let event_type =
        value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| LLMError::ResponseFormatError {
                message: "Responses stream event missing type".to_string(),
                raw_response: data_trimmed.to_string(),
            })?;

    match event_type {
        "response.output_text.delta" => Ok(parse_responses_text_delta(&value, false)),
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            Ok(parse_responses_text_delta(&value, true))
        }
        "response.output_item.added" => Ok(handle_responses_output_item_added(&value, state)),
        "response.function_call_arguments.delta" => Ok(
            handle_responses_function_call_arguments_delta(&value, state),
        ),
        "response.function_call_arguments.done" | "response.output_item.done" => Ok(
            handle_responses_function_call_completion(event_type, &value, state),
        ),
        "response.completed" | "response.done" => Ok(handle_responses_done(&value, state)),
        "error" | "response.failed" => Err(LLMError::ProviderError(data_trimmed.to_string())),
        _ => Ok(Vec::new()),
    }
}

fn parse_responses_text_delta(value: &Value, reasoning: bool) -> Vec<StreamChunk> {
    let Some(delta) = value.get("delta").and_then(Value::as_str) else {
        return Vec::new();
    };
    if delta.is_empty() {
        return Vec::new();
    }

    vec![if reasoning {
        StreamChunk::ReasoningContent(delta.to_string())
    } else {
        StreamChunk::Text(delta.to_string())
    }]
}

fn responses_output_index(value: &Value) -> usize {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

fn handle_responses_output_item_added(
    value: &Value,
    state: &mut ResponsesStreamState,
) -> Vec<StreamChunk> {
    let output_index = responses_output_index(value);
    let item = value.get("item").cloned().unwrap_or(Value::Null);
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return Vec::new();
    }

    let tool_state = state.tool_states.entry(output_index).or_default();
    tool_state.call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    tool_state.item_id = item
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    tool_state.name = item
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    if tool_state.started || tool_state.name.is_empty() {
        return Vec::new();
    }

    tool_state.started = true;
    vec![StreamChunk::ToolUseStart {
        index: output_index,
        id: tool_state.call_id.clone(),
        name: tool_state.name.clone(),
    }]
}

fn handle_responses_function_call_arguments_delta(
    value: &Value,
    state: &mut ResponsesStreamState,
) -> Vec<StreamChunk> {
    let output_index = responses_output_index(value);
    let delta = value
        .get("delta")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let tool_state = state.tool_states.entry(output_index).or_default();
    if tool_state.item_id.is_none() {
        tool_state.item_id = value
            .get("item_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }

    if delta.is_empty() {
        return Vec::new();
    }

    tool_state.arguments_buffer.push_str(delta);
    vec![StreamChunk::ToolUseInputDelta {
        index: output_index,
        partial_json: delta.to_string(),
    }]
}

fn handle_responses_function_call_completion(
    event_type: &str,
    value: &Value,
    state: &mut ResponsesStreamState,
) -> Vec<StreamChunk> {
    let output_index = responses_output_index(value);
    let item = value.get("item").cloned().unwrap_or(Value::Null);
    let is_function_call = event_type == "response.function_call_arguments.done"
        || item.get("type").and_then(Value::as_str) == Some("function_call");
    if !is_function_call {
        return Vec::new();
    }

    let tool_state = state.tool_states.entry(output_index).or_default();
    if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
        tool_state.call_id = call_id.to_string();
    }
    if let Some(item_id) = item.get("id").and_then(Value::as_str) {
        tool_state.item_id = Some(item_id.to_string());
    }
    if let Some(name) = item.get("name").and_then(Value::as_str) {
        tool_state.name = name.to_string();
    }
    if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
        tool_state.arguments_buffer = arguments.to_string();
    }

    complete_responses_tool_call(output_index, state)
}

fn complete_responses_tool_call(
    output_index: usize,
    state: &mut ResponsesStreamState,
) -> Vec<StreamChunk> {
    let Some(tool_state) = state.tool_states.remove(&output_index) else {
        return Vec::new();
    };
    if tool_state.name.is_empty() {
        return Vec::new();
    }

    state.saw_tool_completion = true;
    vec![StreamChunk::ToolUseComplete {
        index: output_index,
        tool_call: ToolCall {
            id: tool_state.call_id,
            call_type: "function".to_string(),
            function: FunctionCall {
                name: tool_state.name,
                arguments: tool_state.arguments_buffer,
            },
        },
    }]
}

fn handle_responses_done(value: &Value, state: &mut ResponsesStreamState) -> Vec<StreamChunk> {
    let mut results = Vec::new();
    if let Some(usage) = usage_from_value(value) {
        results.push(StreamChunk::Usage(usage));
    }
    results.extend(finalize_responses_stream(state, true));
    results
}

fn create_responses_tool_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>> {
    let stream = response
        .bytes_stream()
        .scan(
            (Vec::<u8>::new(), ResponsesStreamState::default()),
            |(buffer, state), chunk| {
                let items = match chunk {
                    Ok(bytes) => {
                        let mut items = Vec::new();
                        buffer.extend_from_slice(&bytes);
                        while let Some((pos, delimiter_len)) = find_sse_event_boundary(buffer) {
                            let event_bytes = buffer[..pos].to_vec();
                            buffer.drain(..pos + delimiter_len);

                            let event = String::from_utf8_lossy(&event_bytes).into_owned();
                            match parse_responses_sse_event(event.trim(), state) {
                                Ok(results) => {
                                    items.extend(results.into_iter().map(Ok));
                                }
                                Err(err) => items.push(Err(err)),
                            }
                        }
                        items
                    }
                    Err(err) => vec![Err(LLMError::HttpError(err.to_string()))],
                };
                futures::future::ready(Some(items))
            },
        )
        .flat_map(futures::stream::iter);
    Box::pin(stream)
}

impl From<&ToolChoice> for OpenAIResponsesToolChoice {
    fn from(value: &ToolChoice) -> Self {
        match value {
            ToolChoice::Any => Self::Mode("required".to_string()),
            ToolChoice::Auto => Self::Mode("auto".to_string()),
            ToolChoice::None => Self::Mode("none".to_string()),
            ToolChoice::Tool(name) => Self::Function {
                tool_type: "function",
                name: name.clone(),
            },
        }
    }
}

impl LLMBuilder<OpenAI> {
    /// Set the voice.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.config.voice = Some(voice.into());
        self
    }

    /// Select which OpenAI chat API shape to use.
    pub fn api_mode(mut self, api_mode: OpenAIApiMode) -> Self {
        self.config.api_mode = api_mode;
        self
    }

    pub fn build(self) -> Result<Arc<OpenAI>, LLMError> {
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
            self.top_p,
            self.top_k,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            self.tool_choice,
            self.normalize_response,
            self.reasoning_effort,
            self.config.voice,
            self.config.api_mode,
            self.extra_body,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        Ok(Arc::new(openai))
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.provider.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        let emb_format = self
            .provider
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = OpenAIEmbeddingRequest {
            model: self.provider.model.to_string(),
            input,
            encoding_format: Some(emb_format),
            dimensions: self.provider.embedding_dimensions,
        };

        let url = self
            .provider
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OpenAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

impl EmbeddingBuilder<OpenAI> {
    /// Build an OpenAI embedding provider.
    pub fn build(self) -> Result<Arc<OpenAI>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for OpenAI".to_string())
        })?;

        let model = self
            .model
            .unwrap_or_else(|| "text-embedding-3-small".to_string());

        let provider = OpenAI::new(
            api_key,
            self.base_url,
            Some(model),
            None,
            None,
            self.timeout_seconds,
            None,
            None,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            None,
            None,
            None,
            None,
            OpenAIApiMode::Responses,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        Ok(Arc::new(provider))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::chat::{FunctionTool, ToolChoice};
    use either::Either::Right;
    use futures::StreamExt;
    use httpmock::{
        Method::{GET, POST},
        MockServer,
    };
    use serde_json::json;

    fn sample_function_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup data".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string", "description": "query" }
                    },
                    "required": ["q"]
                }),
            },
        }
    }

    fn sample_schema() -> StructuredOutputFormat {
        StructuredOutputFormat {
            name: "Answer".to_string(),
            description: Some("Structured answer".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            })),
            strict: Some(true),
        }
    }

    fn openai_provider(base_url: String, api_mode: OpenAIApiMode) -> OpenAI {
        OpenAI::new(
            "key",
            Some(base_url),
            Some("gpt-5".to_string()),
            Some(256),
            Some(0.2),
            Some(5),
            Some(0.8),
            Some(16),
            Some("float".to_string()),
            Some(3),
            Some(ToolChoice::Auto),
            Some(true),
            Some("medium".to_string()),
            None,
            api_mode,
            Some(json!({"seed": 7})),
            Some(true),
            Some("high".to_string()),
            Some("approximate".to_string()),
            Some("US".to_string()),
            Some("SF".to_string()),
            Some("CA".to_string()),
        )
        .expect("openai provider should build")
    }

    #[test]
    fn test_legacy_openai_tool_serialization() {
        let tool = OpenAITool::Function {
            tool_type: "function".to_string(),
            function: sample_function_tool().function,
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized.get("type"), Some(&json!("function")));
        assert!(serialized.get("function").is_some());
    }

    #[test]
    fn test_responses_function_tool_serialization() {
        let provider = OpenAI::new(
            "key",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(ToolChoice::Auto),
            None,
            None,
            None,
            OpenAIApiMode::Responses,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let tools = provider
            .build_responses_function_tools(Some(&[sample_function_tool()]))
            .unwrap();
        let serialized = serde_json::to_value(&tools[0]).unwrap();
        assert_eq!(serialized.get("type"), Some(&json!("function")));
        assert_eq!(serialized.get("name"), Some(&json!("lookup")));
        assert_eq!(serialized.get("strict"), Some(&json!(false)));
    }

    #[test]
    fn test_openai_default_api_mode_is_responses() {
        assert_eq!(OpenAIConfig::default().api_mode, OpenAIApiMode::Responses);
    }

    #[test]
    fn test_openai_new_requires_api_key() {
        let result = OpenAI::new(
            "",
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
            OpenAIApiMode::Responses,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(matches!(result, Err(LLMError::AuthError(_))));
    }

    #[test]
    fn test_openai_api_chat_request_serialization() {
        let msg = OpenAIChatMessage {
            role: "user",
            content: Some(Right("hello".to_string())),
            tool_calls: None,
            tool_call_id: None,
        };

        let request = OpenAIAPIChatRequest {
            model: "gpt-test",
            messages: vec![msg],
            input: None,
            max_completion_tokens: Some(10),
            max_output_tokens: None,
            temperature: Some(0.2),
            stream: false,
            top_p: Some(0.9),
            top_k: Some(40),
            tools: None,
            tool_choice: Some(ToolChoice::Auto),
            reasoning_effort: None,
            response_format: None,
            stream_options: None,
            extra_body: serde_json::Map::new(),
        };

        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized.get("model"), Some(&json!("gpt-test")));
        assert_eq!(
            serialized
                .get("messages")
                .and_then(|m| m.as_array())
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_responses_request_serialization_uses_instructions_and_store_false() {
        let provider = OpenAI::new(
            "key",
            None,
            Some("gpt-5".to_string()),
            Some(256),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(ToolChoice::Auto),
            Some(true),
            Some("medium".to_string()),
            None,
            OpenAIApiMode::Responses,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "hello".to_string(),
            },
        ];

        let request = provider
            .build_responses_request(&messages, Some(&[sample_function_tool()]), None, false)
            .unwrap();
        let serialized = serde_json::to_value(&request).unwrap();

        assert_eq!(
            serialized.get("instructions"),
            Some(&json!("You are helpful."))
        );
        assert_eq!(serialized.get("store"), Some(&json!(false)));
        assert_eq!(
            serialized
                .get("input")
                .and_then(Value::as_array)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            serialized
                .get("reasoning")
                .and_then(|value| value.get("effort")),
            Some(&json!("medium"))
        );
    }

    #[test]
    fn test_responses_request_serialization_uses_text_format() {
        let provider = OpenAI::new(
            "key",
            None,
            Some("gpt-5".to_string()),
            Some(256),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(ToolChoice::Auto),
            Some(true),
            None,
            None,
            OpenAIApiMode::Responses,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "Jane, 54 years old".to_string(),
        }];
        let schema = StructuredOutputFormat {
            name: "person".to_string(),
            description: Some("Person payload".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            })),
            strict: Some(true),
        };

        let request = provider
            .build_responses_request(&messages, None, Some(schema), false)
            .unwrap();
        let serialized = serde_json::to_value(&request).unwrap();

        assert_eq!(
            serialized
                .get("text")
                .and_then(|text| text.get("format"))
                .and_then(|format| format.get("type")),
            Some(&json!("json_schema"))
        );
        assert_eq!(
            serialized
                .get("text")
                .and_then(|text| text.get("format"))
                .and_then(|format| format.get("schema"))
                .and_then(|schema| schema.get("additionalProperties")),
            Some(&json!(false))
        );
    }

    #[test]
    fn test_responses_text_format_normalizes_optional_fields_for_openai() {
        let schema = StructuredOutputFormat {
            name: "MathAgentOutput".to_string(),
            description: Some("Math agent output".to_string()),
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "value": { "type": "integer" },
                    "explanation": { "type": "string" },
                    "generic": { "type": "string" }
                },
                "required": ["value", "explanation"]
            })),
            strict: Some(true),
        };

        let format: OpenAIResponsesTextFormat = schema.into();
        let schema = format.schema.expect("schema");

        let mut required = schema
            .get("required")
            .and_then(Value::as_array)
            .cloned()
            .expect("required");
        required.sort_by(|left, right| {
            left.as_str()
                .unwrap_or_default()
                .cmp(right.as_str().unwrap_or_default())
        });
        assert_eq!(
            required,
            vec![json!("explanation"), json!("generic"), json!("value")]
        );
        assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
        assert_eq!(
            schema
                .get("properties")
                .and_then(|properties| properties.get("generic"))
                .and_then(|generic| generic.get("anyOf"))
                .and_then(Value::as_array)
                .map(|variants| variants.len()),
            Some(2)
        );
        assert_eq!(
            schema
                .get("properties")
                .and_then(|properties| properties.get("generic"))
                .and_then(|generic| generic.get("anyOf"))
                .and_then(Value::as_array)
                .and_then(|variants| variants.get(1))
                .and_then(|variant| variant.get("type")),
            Some(&json!("null"))
        );
    }

    #[test]
    fn test_responses_input_translation_preserves_tool_history() {
        let provider = OpenAI::new(
            "key",
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
            OpenAIApiMode::Responses,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let messages = vec![
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "lookup".to_string(),
                        arguments: "{\"q\":\"value\"}".to_string(),
                    },
                }]),
                content: "Checking...".to_string(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "lookup".to_string(),
                        arguments: "{\"result\":42}".to_string(),
                    },
                }]),
                content: String::new(),
            },
        ];

        let (_, input) = provider.build_responses_input_items(&messages).unwrap();
        let serialized = serde_json::to_value(&input).unwrap();
        let input = serialized.as_array().unwrap();

        assert_eq!(input.len(), 3);
        assert_eq!(input[0].get("role"), Some(&json!("assistant")));
        assert_eq!(input[1].get("type"), Some(&json!("function_call")));
        assert_eq!(input[1].get("call_id"), Some(&json!("call_1")));
        assert_eq!(input[2].get("type"), Some(&json!("function_call_output")));
        assert_eq!(input[2].get("call_id"), Some(&json!("call_1")));
    }

    #[test]
    fn test_openai_builder_requires_api_key() {
        let result = LLMBuilder::<OpenAI>::new().build();
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("No API key provided"));
        }
    }

    #[test]
    fn test_builder_can_override_api_mode() {
        let llm = LLMBuilder::<OpenAI>::new()
            .api_key("key")
            .api_mode(OpenAIApiMode::ChatCompletions)
            .build()
            .unwrap();
        assert_eq!(llm.api_mode, OpenAIApiMode::ChatCompletions);
    }

    #[test]
    fn test_build_web_search_tool_with_location() {
        let provider = OpenAI::new(
            "key",
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
            OpenAIApiMode::Responses,
            None,
            Some(true),
            None,
            Some("approximate".to_string()),
            Some("US".to_string()),
            Some("SF".to_string()),
            Some("CA".to_string()),
        )
        .unwrap();

        let tool = provider.build_web_search_tool();
        match tool {
            OpenAIResponsesTool::WebSearch { user_location, .. } => {
                let loc = user_location.expect("location");
                assert_eq!(loc.location_type, "approximate");
                let approx = loc.approximate.expect("approx");
                assert_eq!(approx.country, "US");
                assert_eq!(approx.city, "SF");
                assert_eq!(approx.region, "CA");
            }
            _ => panic!("expected web search tool"),
        }
    }

    #[test]
    fn test_parse_responses_sse_text_delta() {
        let mut state = ResponsesStreamState::default();
        let event = r#"data: {"type":"response.output_text.delta","delta":"Hello"}"#;
        let results = parse_responses_sse_event(event, &mut state).unwrap();
        assert!(matches!(&results[0], StreamChunk::Text(text) if text == "Hello"));
    }

    #[test]
    fn test_parse_responses_sse_function_call_sequence() {
        let mut state = ResponsesStreamState::default();

        let start = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let start_results = parse_responses_sse_event(start, &mut state).unwrap();
        assert!(
            matches!(&start_results[0], StreamChunk::ToolUseStart { id, name, .. } if id == "call_1" && name == "lookup")
        );

        let delta = r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{\"q\":\"v"}"#;
        let delta_results = parse_responses_sse_event(delta, &mut state).unwrap();
        assert!(
            matches!(&delta_results[0], StreamChunk::ToolUseInputDelta { partial_json, .. } if partial_json == "{\"q\":\"v")
        );

        let done = r#"data: {"type":"response.function_call_arguments.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"value\"}"}}"#;
        let done_results = parse_responses_sse_event(done, &mut state).unwrap();
        match &done_results[0] {
            StreamChunk::ToolUseComplete { tool_call, .. } => {
                assert_eq!(tool_call.id, "call_1");
                assert_eq!(tool_call.function.name, "lookup");
                assert_eq!(tool_call.function.arguments, "{\"q\":\"value\"}");
            }
            other => panic!("Expected ToolUseComplete, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_responses_sse_done_with_usage() {
        let mut state = ResponsesStreamState::default();
        let event = r#"data: {"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#;
        let results = parse_responses_sse_event(event, &mut state).unwrap();

        assert_eq!(results.len(), 2);
        assert!(matches!(&results[0], StreamChunk::Usage(usage) if usage.total_tokens == 5));
        assert!(
            matches!(&results[1], StreamChunk::Done { stop_reason } if stop_reason == "end_turn")
        );
    }

    #[test]
    fn test_openai_responses_response_helpers() {
        let response = OpenAIResponsesResponse {
            output: vec![
                OpenAIResponsesOutputItem {
                    item_type: "reasoning".to_string(),
                    content: Vec::new(),
                    name: None,
                    arguments: None,
                    call_id: None,
                    summary: vec![OpenAIResponsesReasoningSummary {
                        text: Some("Thought".to_string()),
                    }],
                },
                OpenAIResponsesOutputItem {
                    item_type: "function_call".to_string(),
                    content: Vec::new(),
                    name: Some("lookup".to_string()),
                    arguments: Some("{\"q\":\"value\"}".to_string()),
                    call_id: Some("call_1".to_string()),
                    summary: Vec::new(),
                },
                OpenAIResponsesOutputItem {
                    item_type: "message".to_string(),
                    content: vec![OpenAIResponsesOutputContent {
                        content_type: "output_text".to_string(),
                        text: Some("Hello".to_string()),
                    }],
                    name: None,
                    arguments: None,
                    call_id: None,
                    summary: Vec::new(),
                },
            ],
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }),
            output_text: None,
        };

        assert_eq!(response.text().as_deref(), Some("Hello"));
        assert_eq!(response.thinking().as_deref(), Some("Thought"));
        let tool_calls = response.tool_calls().unwrap();
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "lookup");
    }

    #[test]
    fn test_openai_responses_response_ignores_function_call_without_call_id() {
        let response = OpenAIResponsesResponse {
            output: vec![OpenAIResponsesOutputItem {
                item_type: "function_call".to_string(),
                content: Vec::new(),
                name: Some("lookup".to_string()),
                arguments: Some("{\"q\":\"value\"}".to_string()),
                call_id: None,
                summary: Vec::new(),
            }],
            usage: None,
            output_text: None,
        };

        assert!(response.tool_calls().is_none());
    }

    #[test]
    fn test_build_responses_input_items_support_images_and_reject_pdf() {
        use crate::chat::ImageMime;

        let provider = openai_provider(
            "https://example.com/v1".to_string(),
            OpenAIApiMode::Responses,
        );
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "lookup".to_string(),
                arguments: "{\"q\":\"value\"}".to_string(),
            },
        };
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "Be precise".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Image((ImageMime::PNG, vec![1, 2, 3])),
                content: "caption".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::ImageURL("https://example.com/image.png".to_string()),
                content: "describe".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(vec![tool_call.clone()]),
                content: "calling tool".to_string(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(vec![tool_call]),
                content: String::new(),
            },
        ];

        let (instructions, input) = provider
            .build_responses_input_items(&messages)
            .expect("responses input should build");
        let serialized = serde_json::to_value(&input).expect("input should serialize");
        let input = serialized.as_array().expect("input should be array");

        assert_eq!(instructions.as_deref(), Some("Be precise"));
        assert_eq!(input.len(), 5);
        assert_eq!(input[0]["role"], json!("user"));
        assert_eq!(input[0]["content"][0]["text"], json!("caption"));
        assert!(
            input[0]["content"][1]["image_url"]
                .as_str()
                .expect("inline image should exist")
                .starts_with("data:image/png;base64,")
        );
        assert_eq!(input[1]["content"][0]["text"], json!("describe"));
        assert_eq!(
            input[1]["content"][1]["image_url"],
            json!("https://example.com/image.png")
        );
        assert_eq!(input[2]["content"], json!("calling tool"));
        assert_eq!(input[3]["type"], json!("function_call"));
        assert_eq!(input[4]["type"], json!("function_call_output"));

        let pdf_messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Pdf(vec![1, 2, 3]),
            content: "doc".to_string(),
        }];
        let err = provider
            .build_responses_input_items(&pdf_messages)
            .expect_err("pdf input should be rejected");
        assert!(
            err.to_string()
                .contains("PDF input is not supported by the OpenAI Responses backend")
        );
    }

    #[tokio::test]
    async fn test_responses_chat_stream_and_web_search_use_mock_server() {
        let server = MockServer::start();
        let base_url = format!("{}/v1", server.base_url());

        let chat_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/responses")
                .body_includes("\"stream\":false")
                .body_includes("\"tools\"")
                .body_includes("\"tool_choice\":\"auto\"")
                .body_includes("\"seed\":7");
            then.status(200).json_body(json!({
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{ "text": "plan" }]
                    },
                    {
                        "type": "function_call",
                        "name": "lookup",
                        "arguments": "{\"q\":\"value\"}",
                        "call_id": "call_1"
                    },
                    {
                        "type": "message",
                        "content": [{ "type": "output_text", "text": "hello responses" }]
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5
                }
            }));
        });

        let provider = openai_provider(base_url.clone(), OpenAIApiMode::Responses);
        let messages = vec![ChatMessage::user().content("hello").build()];
        let tools = vec![sample_function_tool()];
        let response = provider
            .chat_with_tools(&messages, Some(&tools), Some(sample_schema()))
            .await
            .expect("responses chat should succeed");
        assert_eq!(response.text().as_deref(), Some("hello responses"));
        assert_eq!(response.thinking().as_deref(), Some("plan"));
        assert_eq!(
            response.tool_calls().expect("tool calls should exist")[0]
                .function
                .name,
            "lookup"
        );
        chat_mock.assert();

        let stream_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/responses")
                .body_includes("\"stream\":true");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello \"}\n\n\
                     data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"think\"}\n\n\
                     data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
                     data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"value\\\"}\"}}\n\n\
                     data: {\"type\":\"response.done\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":3,\"total_tokens\":5}}}\n\n\
                     data: [DONE]\n\n",
                );
        });

        let mut stream = provider
            .chat_stream_struct(&messages, Some(&tools), Some(sample_schema()))
            .await
            .expect("responses stream should build");

        let first = stream
            .next()
            .await
            .expect("text delta should exist")
            .expect("text delta should be ok");
        assert_eq!(first.choices[0].delta.content.as_deref(), Some("hello "));

        let second = stream
            .next()
            .await
            .expect("reasoning delta should exist")
            .expect("reasoning delta should be ok");
        assert_eq!(
            second.choices[0].delta.reasoning_content.as_deref(),
            Some("think")
        );

        let third = stream
            .next()
            .await
            .expect("tool completion should exist")
            .expect("tool completion should be ok");
        assert_eq!(
            third.choices[0]
                .delta
                .tool_calls
                .as_ref()
                .expect("tool call")[0]
                .function
                .name,
            "lookup"
        );

        let fourth = stream
            .next()
            .await
            .expect("usage chunk should exist")
            .expect("usage chunk should be ok");
        assert_eq!(
            fourth.usage.as_ref().map(|usage| usage.total_tokens),
            Some(5)
        );
        assert!(stream.next().await.is_none());
        stream_mock.assert();

        let web_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/responses")
                .body_includes("\"web_search_preview\"")
                .body_includes("\"country\":\"US\"")
                .body_includes("\"city\":\"SF\"");
            then.status(200).json_body(json!({
                "output": [{
                    "type": "message",
                    "content": [{ "type": "output_text", "text": "web answer" }]
                }]
            }));
        });

        let response = provider
            .chat_with_web_search("current news".to_string())
            .await
            .expect("web search request should succeed");
        assert_eq!(response.text().as_deref(), Some("web answer"));
        web_mock.assert();
    }

    #[tokio::test]
    async fn test_responses_error_status_invalid_json_list_models_and_embed() {
        let server = MockServer::start();
        let base_url = format!("{}/v1", server.base_url());
        let provider = openai_provider(base_url.clone(), OpenAIApiMode::Responses);
        let messages = vec![ChatMessage::user().content("hello").build()];

        let error_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/responses")
                .body_includes("\"stream\":false");
            then.status(502).body("bad gateway");
        });
        let err = provider
            .chat_with_tools(&messages, None, None)
            .await
            .expect_err("error status should fail");
        match err {
            LLMError::ResponseFormatError {
                message,
                raw_response,
            } => {
                assert!(message.contains("returned error status"));
                assert_eq!(raw_response, "bad gateway");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        error_mock.assert();

        let server = MockServer::start();
        let base_url = format!("{}/v1", server.base_url());
        let provider = openai_provider(base_url.clone(), OpenAIApiMode::Responses);
        let invalid_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/responses")
                .body_includes("\"stream\":false");
            then.status(200).body("not-json");
        });
        let err = provider
            .chat_with_tools(&messages, None, None)
            .await
            .expect_err("invalid json should fail");
        match err {
            LLMError::ResponseFormatError {
                message,
                raw_response,
            } => {
                assert!(message.contains("Failed to decode"));
                assert_eq!(raw_response, "not-json");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        invalid_mock.assert();

        let models_mock = server.mock(|when, then| {
            when.method(GET).path("/v1/models");
            then.status(200).json_body(json!({
                "data": [
                    { "id": "gpt-4.1", "created": 1 },
                    { "id": "gpt-4o-mini", "created": 2 }
                ]
            }));
        });
        let models = provider
            .list_models(None)
            .await
            .expect("model list should succeed");
        assert_eq!(
            models.get_models(),
            vec!["gpt-4.1".to_string(), "gpt-4o-mini".to_string()]
        );
        models_mock.assert();

        let embed_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .body_includes("\"encoding_format\":\"float\"")
                .body_includes("\"dimensions\":3");
            then.status(200).json_body(json!({
                "data": [
                    { "embedding": [0.1, 0.2, 0.3] },
                    { "embedding": [0.4, 0.5, 0.6] }
                ]
            }));
        });
        let embeddings = provider
            .embed(vec!["alpha".to_string(), "beta".to_string()])
            .await
            .expect("embeddings should succeed");
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![0.1, 0.2, 0.3]);
        embed_mock.assert();
    }

    #[tokio::test]
    async fn test_legacy_chat_and_stream_use_mock_server() {
        let server = MockServer::start();
        let base_url = format!("{}/v1", server.base_url());
        let provider = openai_provider(base_url.clone(), OpenAIApiMode::ChatCompletions);
        let messages = vec![ChatMessage::user().content("hello").build()];
        let tools = vec![sample_function_tool()];

        let chat_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_includes("\"stream\":false")
                .body_includes("\"response_format\"");
            then.status(200).json_body(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "legacy reply",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"q\":\"value\"}"
                            }
                        }]
                    }
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3
                }
            }));
        });

        let response = provider
            .chat_with_tools(&messages, Some(&tools), Some(sample_schema()))
            .await
            .expect("legacy chat should succeed");
        assert_eq!(response.text().as_deref(), Some("legacy reply"));
        assert_eq!(
            response.tool_calls().expect("legacy tool calls")[0]
                .function
                .name,
            "lookup"
        );
        chat_mock.assert();

        let stream_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_includes("\"stream\":true");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"legacy stream\"}}]}\n\n\
                     data: [DONE]\n\n",
                );
        });

        let mut stream = provider
            .chat_stream_struct(&messages, Some(&tools), Some(sample_schema()))
            .await
            .expect("legacy stream should build");
        let first = stream
            .next()
            .await
            .expect("legacy stream chunk should exist")
            .expect("legacy stream chunk should be ok");
        assert_eq!(
            first.choices[0].delta.content.as_deref(),
            Some("legacy stream")
        );
        assert!(stream.next().await.is_none());
        stream_mock.assert();
    }
}
