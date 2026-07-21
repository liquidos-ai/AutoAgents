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
    FunctionCall, LLMProvider, ToolCall,
    builder::{LLMBackend, LLMBuilder},
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StructuredOutputFormat,
        Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    config::resolve_request_timeout,
    embedding::{EmbeddingBuilder, EmbeddingProvider},
    error::LLMError,
    http::ensure_success,
    image_generation::{
        GeneratedImage, ImageGenerationProvider, ImageGenerationRequest, ImageGenerationResponse,
    },
    models::ModelsProvider,
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures::stream::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

const DEFAULT_GOOGLE_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const GOOGLE_API_KEY_HEADER: &str = "x-goog-api-key";

/// Client for interacting with Google's Gemini API.
///
/// This struct holds the configuration and state needed to make requests to the Gemini API.
/// It implements the [`ChatProvider`], [`CompletionProvider`], and [`EmbeddingProvider`] traits.
#[derive(Debug)]
pub struct Google {
    /// API key for authentication with Google's API
    pub api_key: String,
    /// Model identifier (e.g. "gemini-1.5-flash")
    pub model: String,
    /// Maximum number of tokens to generate in responses
    pub max_tokens: Option<u32>,
    /// Sampling temperature between 0.0 and 1.0
    pub temperature: Option<f32>,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Base URL for the Gemini API.
    api_base_url: String,
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
    #[serde(rename = "mimeType")]
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
    /// Token usage metadata per Gemini API spec (always present on
    /// non-streaming responses; present on the terminal streaming chunk).
    #[serde(rename = "usageMetadata", default)]
    usage_metadata: Option<GoogleUsageMetadata>,
}

/// Response from the streaming chat completion API.
///
/// Note: Gemini surfaces `usageMetadata` on the terminal streaming chunk too,
/// but the current SSE path (`parse_google_sse_chunk`) only extracts text.
/// Streaming usage tracking would require accumulating the terminal chunk —
/// out of scope for the initial usage() patch; tracked separately.
#[derive(Deserialize, Debug)]
struct GoogleStreamResponse {
    /// Generated completion candidates
    candidates: Option<Vec<GoogleCandidate>>,
}

/// Token accounting fields from the Gemini API response.
/// See <https://ai.google.dev/api/generate-content#UsageMetadata>.
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleUsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
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

    fn usage(&self) -> Option<crate::chat::Usage> {
        self.usage_metadata.as_ref().map(|u| crate::chat::Usage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            // Prefer Gemini's reported total (covers thinking tokens, etc).
            // Fall back to sum if the API ever omits it.
            total_tokens: if u.total_token_count > 0 {
                u.total_token_count
            } else {
                u.prompt_token_count
                    .saturating_add(u.candidates_token_count)
            },
            completion_tokens_details: None,
            prompt_tokens_details: None,
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
        top_p: Option<f32>,
        top_k: Option<u32>,
    ) -> Self {
        let timeout_seconds = resolve_request_timeout(timeout_seconds);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .expect("Failed to build reqwest Client");
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| "gemini-1.5-flash".to_string()),
            max_tokens,
            temperature,
            timeout_seconds,
            top_p,
            top_k,
            api_base_url: DEFAULT_GOOGLE_API_BASE_URL.to_string(),
            client,
        }
    }

    fn model_endpoint_url(&self, model: &str, method: &str) -> Result<Url, LLMError> {
        let raw_url = format!(
            "{}/v1beta/models/{model}:{method}",
            self.api_base_url.trim_end_matches('/')
        );
        Url::parse(&raw_url).map_err(|err| LLMError::HttpError(err.to_string()))
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
            return Err(LLMError::missing_api_key(
                "Missing Google API key".to_string(),
            ));
        }

        let chat_contents = build_google_chat_contents(messages)?;
        let google_tools = build_google_tools(tools);
        let generation_config = build_generation_config_with_schema(
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            json_schema.as_ref(),
        );

        let req_body = GoogleChatRequest {
            contents: chat_contents,
            generation_config,
            tools: google_tools,
        };

        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "{}",
                crate::request_diagnostics::summarize_json_request(
                    "Google Gemini",
                    "chat request",
                    &req_body
                )
            );
        }

        let url = self.model_endpoint_url(&self.model, "generateContent")?;

        let resp = self
            .client
            .post(url)
            .header(GOOGLE_API_KEY_HEADER, &self.api_key)
            .json(&req_body)
            .send()
            .await?;

        log::debug!("Google Gemini HTTP status (tool): {}", resp.status());

        let resp = ensure_success(resp, "Google").await?;

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
}

#[async_trait]
impl ChatProvider for Google {
    /// Sends a chat request to Google's Gemini API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `json_schema` - Optional Response json schema
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

    /// Sends a streaming chat request to Google's Gemini API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `json_schema` - Optional Response json schema
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
            return Err(LLMError::missing_api_key(
                "Missing Google API key".to_string(),
            ));
        }

        let chat_contents = build_google_stream_contents(messages)?;
        let generation_config = build_generation_config_for_stream(
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
        );

        let req_body = GoogleChatRequest {
            contents: chat_contents,
            generation_config,
            tools: None,
        };

        let mut url = self.model_endpoint_url(&self.model, "streamGenerateContent")?;
        url.query_pairs_mut().append_pair("alt", "sse");

        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "{}",
                crate::request_diagnostics::summarize_json_request(
                    "Google Gemini",
                    "stream request",
                    &req_body
                )
            );
        }

        let response = self
            .client
            .post(url)
            .header(GOOGLE_API_KEY_HEADER, &self.api_key)
            .json(&req_body)
            .send()
            .await?;

        let response = ensure_success(response, "Google").await?;

        Ok(crate::chat::create_sse_stream(
            response,
            parse_google_sse_chunk,
        ))
    }

    fn model(&self) -> &str {
        &self.model
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
            return Err(LLMError::missing_api_key(
                "Missing Google API key".to_string(),
            ));
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

            let url = self.model_endpoint_url("text-embedding-004", "embedContent")?;

            let resp = self
                .client
                .post(url)
                .header(GOOGLE_API_KEY_HEADER, &self.api_key)
                .json(&req_body)
                .send()
                .await?;
            let resp = ensure_success(resp, "Google").await?;

            let embedding_resp: GoogleEmbeddingResponse = resp.json().await?;
            embeddings.push(embedding_resp.embedding.values);
        }

        Ok(embeddings)
    }
}

/// Request body for Gemini image generation via the `generateContent` endpoint.
#[derive(Serialize)]
struct GoogleImageRequest<'a> {
    /// Prompt text and any input image parts.
    contents: Vec<GoogleChatContent<'a>>,
    /// Generation config requesting image output.
    #[serde(rename = "generationConfig")]
    generation_config: GoogleImageGenerationConfig,
}

/// Generation config that asks Gemini to return image (and text) modalities.
#[derive(Serialize)]
struct GoogleImageGenerationConfig {
    /// Requested response modalities, e.g. `["TEXT", "IMAGE"]`.
    #[serde(rename = "responseModalities")]
    response_modalities: Vec<String>,
}

/// Response body for a Gemini image generation request.
#[derive(Deserialize, Debug)]
struct GoogleImageResponse {
    /// Generated candidates; each may carry inline image data.
    #[serde(default)]
    candidates: Vec<GoogleImageCandidate>,
}

/// Individual image generation candidate.
#[derive(Deserialize, Debug)]
struct GoogleImageCandidate {
    /// Content block containing the response parts.
    content: GoogleImageContent,
}

/// Content block within an image generation candidate.
#[derive(Deserialize, Debug)]
struct GoogleImageContent {
    /// Parts making up the content (text and/or inline image data).
    #[serde(default)]
    parts: Vec<GoogleImagePart>,
}

/// A single part of an image generation response.
#[derive(Deserialize, Debug)]
struct GoogleImagePart {
    /// Inline image data, when this part carries a generated image.
    /// Gemini returns `inlineData`; the `inline_data` alias covers snake_case payloads.
    #[serde(default, rename = "inlineData", alias = "inline_data")]
    inline_data: Option<GoogleInlineImageData>,
}

/// Base64-encoded inline image data returned by Gemini.
#[derive(Deserialize, Debug)]
struct GoogleInlineImageData {
    /// MIME type of the image (e.g. `image/png`).
    #[serde(default, rename = "mimeType", alias = "mime_type")]
    mime_type: Option<String>,
    /// Base64-encoded image bytes.
    data: String,
}

#[async_trait]
impl ImageGenerationProvider for Google {
    /// Generates one or more images from a prompt using Gemini's
    /// `generateContent` endpoint with image response modality.
    ///
    /// # Arguments
    ///
    /// * `request` - Prompt, optional model override and optional input images
    ///
    /// # Returns
    ///
    /// The generated images (with preserved MIME types) or an error
    async fn generate_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::missing_api_key(
                "Missing Google API key".to_string(),
            ));
        }

        if request.prompt.trim().is_empty() {
            return Err(LLMError::invalid_request(
                "Image generation prompt must not be empty".to_string(),
            ));
        }

        let model = request.model.as_deref().unwrap_or(&self.model);

        // Prompt text first, followed by any input images as inline data parts.
        let mut parts: Vec<GoogleContentPart<'_>> = vec![GoogleContentPart::Text(&request.prompt)];
        if let Some(input_images) = &request.input_images {
            for image in input_images {
                parts.push(GoogleContentPart::InlineData(GoogleInlineData {
                    mime_type: image.mime_type.clone(),
                    data: BASE64.encode(&image.data),
                }));
            }
        }

        let req_body = GoogleImageRequest {
            contents: vec![GoogleChatContent {
                role: "user",
                parts,
            }],
            generation_config: GoogleImageGenerationConfig {
                response_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
            },
        };

        let url = self.model_endpoint_url(model, "generateContent")?;

        let resp = self
            .client
            .post(url)
            .header(GOOGLE_API_KEY_HEADER, &self.api_key)
            .json(&req_body)
            .send()
            .await?;

        log::debug!("Google Gemini image HTTP status: {}", resp.status());

        let resp = ensure_success(resp, "Google").await?;
        let resp_text = resp.text().await?;

        let json_resp: GoogleImageResponse =
            serde_json::from_str(&resp_text).map_err(|e| LLMError::ResponseFormatError {
                message: format!("Failed to decode Google image response: {e}"),
                raw_response: resp_text.clone(),
            })?;

        let mut images = Vec::new();
        for candidate in &json_resp.candidates {
            for part in &candidate.content.parts {
                if let Some(inline) = &part.inline_data {
                    let data = BASE64.decode(inline.data.as_bytes()).map_err(|e| {
                        LLMError::ResponseFormatError {
                            message: format!("Failed to base64-decode Google image data: {e}"),
                            raw_response: resp_text.clone(),
                        }
                    })?;
                    let mime_type = inline
                        .mime_type
                        .clone()
                        .unwrap_or_else(|| "image/png".to_string());
                    images.push(GeneratedImage {
                        mime_type,
                        data,
                        metadata: serde_json::json!({ "model": model }),
                    });
                }
            }
        }

        if images.is_empty() {
            return Err(LLMError::ProviderError(
                "No image returned by Google".to_string(),
            ));
        }

        Ok(ImageGenerationResponse {
            images,
            backend: LLMBackend::Google,
        })
    }
}

impl LLMProvider for Google {}

impl crate::HasConfig for Google {
    type Config = crate::NoConfig;
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
                    if let Some(candidates) = response.candidates
                        && let Some(candidate) = candidates.first()
                        && let Some(part) = candidate.content.parts.first()
                        && !part.text.is_empty()
                    {
                        return Ok(Some(part.text.clone()));
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }

    Ok(None)
}

fn build_google_chat_contents(
    messages: &[ChatMessage],
) -> Result<Vec<GoogleChatContent<'_>>, LLMError> {
    let mut chat_contents = Vec::with_capacity(messages.len());

    for msg in messages {
        let role = match &msg.message_type {
            MessageType::ToolResult(_) => "function",
            _ => match msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "model",
                ChatRole::System => "user",
                ChatRole::Tool => "user",
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
                MessageType::ImageURL(_) => {
                    return Err(LLMError::invalid_request(
                        "Image URL input is not supported by the Google Gemini backend".to_string(),
                    ));
                }
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
                        let parsed_args = serde_json::from_str::<Value>(&result.function.arguments)
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

    Ok(chat_contents)
}

fn build_google_stream_contents(
    messages: &[ChatMessage],
) -> Result<Vec<GoogleChatContent<'_>>, LLMError> {
    build_google_chat_contents(messages)
}

fn build_google_tools(tools: Option<&[Tool]>) -> Option<Vec<GoogleTool>> {
    tools.map(|t| {
        vec![GoogleTool {
            function_declarations: t.iter().map(GoogleFunctionDeclaration::from).collect(),
        }]
    })
}

fn build_generation_config_with_schema(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    json_schema: Option<&StructuredOutputFormat>,
) -> Option<GoogleGenerationConfig> {
    let (response_mime_type, response_schema) = if let Some(json_schema) = json_schema {
        if let Some(schema) = &json_schema.schema {
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
        max_output_tokens: max_tokens,
        temperature,
        top_p,
        top_k,
        response_mime_type,
        response_schema,
    })
}

fn build_generation_config_for_stream(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
) -> Option<GoogleGenerationConfig> {
    if max_tokens.is_none() && temperature.is_none() && top_p.is_none() && top_k.is_none() {
        None
    } else {
        Some(GoogleGenerationConfig {
            max_output_tokens: max_tokens,
            temperature,
            top_p,
            top_k,
            response_mime_type: None,
            response_schema: None,
        })
    }
}

#[async_trait]
impl ModelsProvider for Google {}

impl LLMBuilder<Google> {
    pub fn build(self) -> Result<Arc<Google>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::invalid_request("No API key provided for Google".to_string())
        })?;

        let google = Google::new(
            api_key,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
        );

        Ok(Arc::new(google))
    }
}

impl EmbeddingBuilder<Google> {
    /// Build a Google embedding provider.
    pub fn build(self) -> Result<Arc<Google>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::invalid_request("No API key provided for Google".to_string())
        })?;

        let provider = Google::new(
            api_key,
            self.model,
            None,
            None,
            self.timeout_seconds,
            None,
            None,
        );

        Ok(Arc::new(provider))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{FunctionTool, Tool};
    use futures::StreamExt;
    use httpmock::{Method::POST, MockServer};
    use serde_json::json;

    fn test_google_provider(server: &MockServer) -> Google {
        let mut provider = Google::new(
            "secret-key",
            Some("gemini-test".to_string()),
            None,
            None,
            None,
            None,
            None,
        );
        provider.api_base_url = server.base_url();
        provider
    }

    #[test]
    fn test_google_function_declaration_from_tool() {
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "desc".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string", "description": "query" }
                    },
                    "required": ["q"]
                }),
            },
        };

        let decl = GoogleFunctionDeclaration::from(&tool);
        assert_eq!(decl.name, "lookup");
        assert_eq!(decl.description, "desc");
        assert_eq!(decl.parameters.required, vec!["q".to_string()]);
    }

    #[test]
    fn test_google_chat_response_text_and_tool_calls() {
        let response = GoogleChatResponse {
            candidates: vec![GoogleCandidate {
                content: GoogleResponseContent {
                    parts: vec![GoogleResponsePart {
                        text: "hi".to_string(),
                        function_call: None,
                    }],
                    function_call: Some(GoogleFunctionCall {
                        name: "lookup".to_string(),
                        args: json!({"q":"value"}),
                    }),
                    function_calls: None,
                },
            }],
            usage_metadata: None,
        };

        assert_eq!(response.text(), Some("hi".to_string()));
        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert!(format!("{response}").contains("hi"));
    }

    #[test]
    fn test_google_chat_response_tool_calls_array() {
        let response = GoogleChatResponse {
            candidates: vec![GoogleCandidate {
                content: GoogleResponseContent {
                    parts: vec![GoogleResponsePart {
                        text: "".to_string(),
                        function_call: None,
                    }],
                    function_call: None,
                    function_calls: Some(vec![GoogleFunctionCall {
                        name: "sum".to_string(),
                        args: json!({"a": 1, "b": 2}),
                    }]),
                },
            }],
            usage_metadata: None,
        };

        let calls = response.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "sum");
        assert!(calls[0].function.arguments.contains("\"a\""));
    }

    #[test]
    fn test_parse_google_sse_chunk_extracts_text() {
        let chunk =
            r#"data: {"candidates":[{"content":{"parts":[{"text":"token"}]}}]}"#.to_string();

        let parsed = parse_google_sse_chunk(&chunk).unwrap();
        assert_eq!(parsed, Some("token".to_string()));
    }

    #[test]
    fn test_parse_google_sse_chunk_ignores_invalid_json() {
        let chunk = "data: {not-json}\n\n";
        let parsed = parse_google_sse_chunk(chunk).unwrap();
        assert!(parsed.is_none());
    }

    #[test]
    fn test_parse_google_sse_chunk_returns_none_for_empty_candidates() {
        let chunk = r#"data: {"candidates":[]}"#;
        let parsed = parse_google_sse_chunk(chunk).unwrap();
        assert!(parsed.is_none());
    }

    #[test]
    fn test_build_google_chat_contents_tool_use_and_result() {
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
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(vec![tool_call.clone()]),
                content: "call".to_string(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(vec![tool_call]),
                content: "result".to_string(),
            },
        ];

        let contents = build_google_chat_contents(&messages).expect("tool messages should convert");
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role, "model");
        match &contents[0].parts[0] {
            GoogleContentPart::FunctionCall(call) => {
                assert_eq!(call.name, "lookup");
                assert_eq!(call.args, json!({"q": "value"}));
            }
            _ => panic!("unexpected part"),
        }

        assert_eq!(contents[1].role, "function");
        match &contents[1].parts[0] {
            GoogleContentPart::FunctionResponse(resp) => {
                assert_eq!(resp.name, "lookup");
                assert_eq!(resp.response.content, json!({"q": "value"}));
            }
            _ => panic!("unexpected part"),
        }
    }

    #[test]
    fn test_build_google_stream_contents_tool_use_and_result() {
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
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(vec![tool_call.clone()]),
                content: "call".to_string(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(vec![tool_call]),
                content: "result".to_string(),
            },
        ];

        let contents =
            build_google_stream_contents(&messages).expect("tool messages should convert");
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role, "model");
        match &contents[0].parts[0] {
            GoogleContentPart::FunctionCall(call) => {
                assert_eq!(call.name, "lookup");
                assert_eq!(call.args, json!({"q": "value"}));
            }
            _ => panic!("unexpected part"),
        }

        assert_eq!(contents[1].role, "function");
        match &contents[1].parts[0] {
            GoogleContentPart::FunctionResponse(resp) => {
                assert_eq!(resp.name, "lookup");
                assert_eq!(resp.response.content, json!({"q": "value"}));
            }
            _ => panic!("unexpected part"),
        }
    }

    #[test]
    fn test_build_google_chat_contents_rejects_image_url() {
        let messages = [ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::ImageURL("https://example.com/image.png".to_string()),
            content: "describe".to_string(),
        }];

        let err = match build_google_chat_contents(&messages) {
            Ok(_) => panic!("Image URL should be rejected"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message == "Image URL input is not supported by the Google Gemini backend"
        ));
    }

    #[test]
    fn test_build_google_stream_contents_rejects_image_url() {
        let messages = [ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::ImageURL("https://example.com/image.png".to_string()),
            content: "describe".to_string(),
        }];

        let err = match build_google_stream_contents(&messages) {
            Ok(_) => panic!("Image URL should be rejected"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message == "Image URL input is not supported by the Google Gemini backend"
        ));
    }

    #[test]
    fn test_build_generation_config_strips_additional_properties() {
        let schema = StructuredOutputFormat {
            name: "Test".to_string(),
            description: None,
            schema: Some(json!({
                "type": "object",
                "additionalProperties": true,
                "properties": { "foo": { "type": "string" } }
            })),
            strict: Some(true),
        };

        let config =
            build_generation_config_with_schema(Some(1), Some(0.2), None, None, Some(&schema))
                .expect("config");
        assert!(matches!(
            config.response_mime_type,
            Some(GoogleResponseMimeType::Json)
        ));
        let schema = config.response_schema.expect("schema");
        assert!(schema.get("additionalProperties").is_none());
    }

    #[test]
    fn test_build_generation_config_for_stream_none() {
        let config = build_generation_config_for_stream(None, None, None, None);
        assert!(config.is_none());
    }

    #[test]
    fn test_build_google_tools_wraps_tools() {
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "desc".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "q": { "type": "string" } }
                }),
            },
        };
        let tools = build_google_tools(Some(&[tool])).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function_declarations.len(), 1);
    }

    #[test]
    fn test_google_usage_returns_none_when_metadata_absent() {
        let response = GoogleChatResponse {
            candidates: vec![],
            usage_metadata: None,
        };
        assert!(response.usage().is_none());
    }

    #[test]
    fn test_google_usage_maps_metadata_to_usage() {
        let response = GoogleChatResponse {
            candidates: vec![],
            usage_metadata: Some(GoogleUsageMetadata {
                prompt_token_count: 12,
                candidates_token_count: 34,
                total_token_count: 46,
            }),
        };
        let usage = response.usage().expect("usage populated");
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 34);
        assert_eq!(usage.total_tokens, 46);
    }

    #[test]
    fn test_google_usage_falls_back_to_sum_when_total_zero() {
        // Defensive — if Gemini ever returns the components without total.
        let response = GoogleChatResponse {
            candidates: vec![],
            usage_metadata: Some(GoogleUsageMetadata {
                prompt_token_count: 5,
                candidates_token_count: 7,
                total_token_count: 0,
            }),
        };
        let usage = response.usage().expect("usage populated");
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn test_google_chat_response_deserializes_usage_metadata_from_api_payload() {
        // Sample non-streaming generateContent response per Gemini API docs.
        // See https://ai.google.dev/api/generate-content#UsageMetadata
        let payload = json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": "Hello!" }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 22,
                "totalTokenCount": 33
            }
        });
        let response: GoogleChatResponse =
            serde_json::from_value(payload).expect("realistic Gemini payload must deserialize");
        let usage = response
            .usage()
            .expect("usage_metadata must populate Usage");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 22);
        assert_eq!(usage.total_tokens, 33);
    }

    #[test]
    fn test_google_chat_response_deserializes_without_usage_metadata() {
        // Legacy / cached responses may omit usageMetadata; must not fail.
        let payload = json!({
            "candidates": [{
                "content": { "parts": [{ "text": "ok" }] }
            }]
        });
        let response: GoogleChatResponse = serde_json::from_value(payload)
            .expect("response without usageMetadata must still deserialize");
        assert!(response.usage().is_none());
    }

    #[test]
    fn test_google_model_endpoint_url_does_not_include_api_key() {
        let server = MockServer::start();
        let provider = test_google_provider(&server);

        let url = provider
            .model_endpoint_url("gemini-test", "generateContent")
            .expect("url should build");

        assert_eq!(
            url.as_str(),
            format!(
                "{}/v1beta/models/gemini-test:generateContent",
                server.base_url()
            )
        );
        assert!(url.query().is_none());
        assert!(!url.as_str().contains("secret-key"));
        assert!(!url.as_str().contains("key="));
    }

    #[tokio::test]
    async fn test_google_chat_sends_api_key_header_not_query_param() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/gemini-test:generateContent")
                .header(GOOGLE_API_KEY_HEADER, "secret-key");
            then.status(200).json_body(json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "ok" }]
                    }
                }]
            }));
        });
        let provider = test_google_provider(&server);
        let messages = [ChatMessage::user().content("hello").build()];

        let response = provider
            .chat(&messages, None)
            .await
            .expect("chat should succeed");

        assert_eq!(response.text().as_deref(), Some("ok"));
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_stream_keeps_alt_query_and_sends_api_key_header() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/gemini-test:streamGenerateContent")
                .query_param("alt", "sse")
                .header(GOOGLE_API_KEY_HEADER, "secret-key");
            then.status(200)
                .body("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n");
        });
        let provider = test_google_provider(&server);
        let messages = [ChatMessage::user().content("hello").build()];

        let mut stream = provider
            .chat_stream(&messages, None)
            .await
            .expect("stream should start");

        let first = stream
            .next()
            .await
            .expect("stream should emit")
            .expect("stream item should decode");
        assert_eq!(first, "ok");
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_embeddings_send_api_key_header_not_query_param() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/text-embedding-004:embedContent")
                .header(GOOGLE_API_KEY_HEADER, "secret-key");
            then.status(200).json_body(json!({
                "embedding": {
                    "values": [0.1, 0.2]
                }
            }));
        });
        let provider = test_google_provider(&server);

        let embeddings = provider
            .embed(vec!["hello".to_string()])
            .await
            .expect("embedding should succeed");

        assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_generate_image_sends_api_key_header() {
        use crate::image_generation::ImageGenerationRequest;

        let server = MockServer::start();
        let expected_bytes = vec![0x89, 0x50, 0x4e, 0x47];
        let encoded = BASE64.encode(&expected_bytes);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/gemini-test:generateContent")
                .header(GOOGLE_API_KEY_HEADER, "secret-key");
            then.status(200).json_body(json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": encoded
                            }
                        }]
                    }
                }]
            }));
        });
        let provider = test_google_provider(&server);

        let request = ImageGenerationRequest {
            prompt: "a red apple".to_string(),
            model: None,
            input_images: None,
            metadata: None,
        };

        let response = provider
            .generate_image(&request)
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        assert_eq!(response.backend, LLMBackend::Google);
        assert_eq!(response.images[0].mime_type, "image/png");
        assert_eq!(response.images[0].data, expected_bytes);
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_generate_image_uses_request_model_override() {
        use crate::image_generation::ImageGenerationRequest;

        let server = MockServer::start();
        let encoded = BASE64.encode([1u8, 2, 3]);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/gemini-image-test:generateContent")
                .header(GOOGLE_API_KEY_HEADER, "secret-key");
            then.status(200).json_body(json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": { "mimeType": "image/png", "data": encoded }
                        }]
                    }
                }]
            }));
        });
        // Provider default model is gemini-test; request overrides it.
        let provider = test_google_provider(&server);

        let request = ImageGenerationRequest {
            prompt: "an override".to_string(),
            model: Some("gemini-image-test".to_string()),
            input_images: None,
            metadata: None,
        };

        let response = provider
            .generate_image(&request)
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_generate_image_supports_input_images() {
        use crate::image_generation::{ImageGenerationRequest, ImageInput};

        let server = MockServer::start();
        let encoded = BASE64.encode([9u8, 9, 9]);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/gemini-test:generateContent")
                .header(GOOGLE_API_KEY_HEADER, "secret-key")
                // The request body should carry the input image as an inlineData part.
                .body_includes("inlineData")
                .body_includes("mimeType");
            then.status(200).json_body(json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": { "mimeType": "image/png", "data": encoded }
                        }]
                    }
                }]
            }));
        });
        let provider = test_google_provider(&server);

        let request = ImageGenerationRequest {
            prompt: "edit this".to_string(),
            model: None,
            input_images: Some(vec![ImageInput {
                mime_type: "image/png".to_string(),
                data: vec![0x01, 0x02, 0x03],
            }]),
            metadata: None,
        };

        let response = provider
            .generate_image(&request)
            .await
            .expect("image generation should succeed");

        assert_eq!(response.images.len(), 1);
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_generate_image_missing_api_key() {
        use crate::image_generation::ImageGenerationRequest;

        let mut provider = Google::new(
            "secret-key",
            Some("gemini-test".to_string()),
            None,
            None,
            None,
            None,
            None,
        );
        provider.api_key = String::new();

        let request = ImageGenerationRequest {
            prompt: "no key".to_string(),
            model: None,
            input_images: None,
            metadata: None,
        };

        let err = provider
            .generate_image(&request)
            .await
            .expect_err("missing api key should error");

        assert!(matches!(err, LLMError::AuthError { .. }));
    }

    #[tokio::test]
    async fn test_google_generate_image_returns_error_when_no_image() {
        use crate::image_generation::ImageGenerationRequest;

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1beta/models/gemini-test:generateContent")
                .header(GOOGLE_API_KEY_HEADER, "secret-key");
            then.status(200).json_body(json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "here is your image" }]
                    }
                }]
            }));
        });
        let provider = test_google_provider(&server);

        let request = ImageGenerationRequest {
            prompt: "text only".to_string(),
            model: None,
            input_images: None,
            metadata: None,
        };

        let err = provider
            .generate_image(&request)
            .await
            .expect_err("response without image should error");

        assert!(matches!(
            err,
            LLMError::ProviderError(message) if message == "No image returned by Google"
        ));
        mock.assert();
    }

    #[tokio::test]
    async fn test_google_generate_image_rejects_empty_prompt() {
        use crate::image_generation::ImageGenerationRequest;

        let server = MockServer::start();
        let provider = test_google_provider(&server);

        let request = ImageGenerationRequest {
            prompt: "   ".to_string(),
            model: None,
            input_images: None,
            metadata: None,
        };

        let err = provider
            .generate_image(&request)
            .await
            .expect_err("empty prompt should error");

        assert!(matches!(
            err,
            LLMError::InvalidRequest { message, .. }
                if message == "Image generation prompt must not be empty"
        ));
    }
}
