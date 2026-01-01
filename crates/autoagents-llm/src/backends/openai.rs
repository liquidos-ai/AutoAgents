//! OpenAI API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with OpenAI's GPT models through their API.

use crate::builder::{LLMBackend, LLMBuilder};
use crate::chat::Usage;
use crate::embedding::EmbeddingBuilder;
use crate::providers::openai_compatible::{
    create_sse_stream, OpenAIChatMessage, OpenAIChatResponse, OpenAICompatibleProvider,
    OpenAIProviderConfig, OpenAIResponseFormat, OpenAIStreamOptions,
};
use crate::{
    chat::{
        ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse,
        StructuredOutputFormat, Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    LLMProvider, ToolCall,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// OpenAI configuration for the generic provider
struct OpenAIConfig;

impl OpenAIProviderConfig for OpenAIConfig {
    const PROVIDER_NAME: &'static str = "OpenAI";
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1/";
    const DEFAULT_MODEL: &'static str = "gpt-4.1-nano";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

// NOTE: OpenAI cannot directly use the OpenAICompatibleProvider type alias, as it needs specific fields

/// Client for OpenAI API
pub struct OpenAI {
    // Delegate to the generic provider for common functionality
    provider: OpenAICompatibleProvider<OpenAIConfig>,
    pub enable_web_search: bool,
    pub web_search_context_size: Option<String>,
    pub web_search_user_location_type: Option<String>,
    pub web_search_user_location_approximate_country: Option<String>,
    pub web_search_user_location_approximate_city: Option<String>,
    pub web_search_user_location_approximate_region: Option<String>,
}

/// OpenAI-specific tool that can be either a function tool or a web search tool
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

/// Response for chat with web search
#[derive(Deserialize, Debug)]
pub struct OpenAIWebSearchChatResponse {
    pub output: Vec<OpenAIWebSearchOutput>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIWebSearchOutput {
    pub content: Option<Vec<OpenAIWebSearchContent>>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIWebSearchContent {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub text: String,
}

impl std::fmt::Display for OpenAIWebSearchChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = self.text() {
            write!(f, "{text}")
        } else {
            write!(f, "No response content")
        }
    }
}

impl ChatResponse for OpenAIWebSearchChatResponse {
    fn text(&self) -> Option<String> {
        self.output
            .last()
            .and_then(|output| output.content.as_ref())
            .and_then(|content| content.last())
            .map(|content| content.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None // Web search responses don't contain tool calls
    }

    fn thinking(&self) -> Option<String> {
        None
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

#[derive(Deserialize, Debug, Serialize)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approximate: Option<ApproximateLocation>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct ApproximateLocation {
    pub country: String,
    pub city: String,
    pub region: String,
}

/// Request payload for OpenAI's chat API endpoint.
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
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tool_choice: Option<ToolChoice>,
        normalize_response: Option<bool>,
        reasoning_effort: Option<String>,
        voice: Option<String>,
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
        Ok(OpenAI {
            provider: <OpenAICompatibleProvider<OpenAIConfig>>::new(
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
                None, // parallel_tool_calls
                normalize_response,
                embedding_encoding_format,
                embedding_dimensions,
            ),
            enable_web_search: enable_web_search.unwrap_or(false),
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
        })
    }
}

// OpenAI-specific implementations that don't fit in the generic provider

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

// Delegate other provider traits to the internal provider
#[async_trait]
impl ChatProvider for OpenAI {
    /// Chat with tool calls enabled
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Use the common prepare_messages method from the OpenAI-compatible provider
        let openai_msgs = self.provider.prepare_messages(messages);
        let response_format: Option<OpenAIResponseFormat> = json_schema.clone().map(|s| s.into());
        // Convert regular tools to OpenAI format
        let tool_calls = tools.map(|t| t.to_vec());
        let mut openai_tools: Vec<OpenAITool> = Vec::new();
        // Add regular function tools
        if let Some(tools) = &tool_calls {
            for tool in tools {
                openai_tools.push(OpenAITool::Function {
                    tool_type: tool.tool_type.clone(),
                    function: tool.function.clone(),
                });
            }
        }
        let final_tools = if openai_tools.is_empty() {
            None
        } else {
            Some(openai_tools)
        };
        let request_tool_choice = if final_tools.is_some() {
            self.provider.tool_choice.clone()
        } else {
            None
        };
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
        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("OpenAI request payload: {}", json);
            }
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
        // Parse the successful response
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

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Build web search tool configuration
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
        let web_search_tool = OpenAITool::WebSearch {
            tool_type: "web_search_preview".to_string(),
            user_location,
        };
        self.chat_with_hosted_tools(input, vec![web_search_tool])
            .await
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
        let openai_msgs = self.provider.prepare_messages(messages);
        // Convert regular tools to OpenAI format for streaming
        let openai_tools: Option<Vec<OpenAITool>> = tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| OpenAITool::Function {
                    tool_type: tool.tool_type.clone(),
                    function: tool.function.clone(),
                })
                .collect()
        });
        let repsonse_schema: Option<OpenAIResponseFormat> = json_schema.map(|schema| schema.into());
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
            response_format: repsonse_schema,
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

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
    {
        // Delegate to the inner OpenAICompatibleProvider which has the full implementation
        self.provider
            .chat_stream_with_tools(messages, tools, json_schema)
            .await
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

// Helper methods to access provider fields
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

    /// Chat with OpenAI-hosted tools using the `/responses` endpoint
    ///
    /// # Arguments
    ///
    /// * `input` - The input message
    /// * `hosted_tools` - List of OpenAI hosted tools to use
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    pub async fn chat_with_hosted_tools(
        &self,
        input: String,
        hosted_tools: Vec<OpenAITool>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let body = OpenAIAPIChatRequest {
            model: self.provider.model.as_str(),
            messages: Vec::new(), // Empty for hosted tools
            input: Some(input),
            max_completion_tokens: None,
            max_output_tokens: self.provider.max_tokens,
            temperature: self.provider.temperature,
            stream: false,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: Some(hosted_tools),
            tool_choice: self.provider.tool_choice.clone(),
            reasoning_effort: self.provider.reasoning_effort.clone(),
            response_format: None, // Hosted tools don't use structured output
            stream_options: None,
            extra_body: self.provider.extra_body.clone(),
        };

        let url = self
            .provider
            .base_url
            .join("responses") // Use responses endpoint for hosted tools
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("OpenAI hosted tools request payload: {}", json);
            }
        }

        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;
        log::debug!("OpenAI hosted tools HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI hosted tools API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIWebSearchChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI hosted tools API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }
}

impl LLMBuilder<OpenAI> {
    /// Set the voice.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
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
            self.voice,
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
