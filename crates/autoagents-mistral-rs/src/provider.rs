//! MistralRsProvider implementation with LLMProvider traits

use crate::{
    config::{MistralRsConfig, MistralRsConfigBuilder},
    conversion::{
        convert_messages, convert_tool_calls, convert_vision_messages, MistralRsResponse,
    },
    error::convert_anyhow_error,
    models::{ModelSource, ModelType},
};
use autoagents_llm::{
    async_trait,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, StreamChoice, StreamDelta, StreamResponse,
        StructuredOutputFormat, Tool, Usage as ChatUsage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    FunctionCall, LLMProvider, ToolCall,
};
use futures::{stream::Stream, StreamExt};
use mistralrs::{
    CalledFunction, ChatCompletionChunkResponse, ChatCompletionResponse, ChunkChoice, Constraint,
    Function, GgufModelBuilder, IsqType, PagedAttentionMetaBuilder, RequestBuilder, Response,
    TextMessageRole, TextModelBuilder, ToolCallResponse, ToolCallType,
    ToolChoice as MistralToolChoice, ToolType, Usage as MistralUsage, VisionModelBuilder,
};
use std::sync::Arc;
use tokio::sync::mpsc;

/// MistralRs provider for local LLM inference
pub struct MistralRsProvider {
    model: Arc<mistralrs::Model>,
    config: MistralRsConfig,
}

impl MistralRsProvider {
    /// Create provider from HuggingFace model repository
    pub async fn from_hf(repo_id: impl Into<String>) -> Result<Self, LLMError> {
        let config = MistralRsConfigBuilder::new()
            .model_source(ModelSource::HuggingFace {
                repo_id: repo_id.into(),
                revision: None,
                model_type: crate::models::ModelType::Auto,
            })
            .build();

        Self::from_config(config).await
    }

    /// Create provider from GGUF files
    pub async fn from_gguf(
        model_dir: impl Into<String>,
        files: Vec<String>,
    ) -> Result<Self, LLMError> {
        let config = MistralRsConfigBuilder::new()
            .model_source(ModelSource::Gguf {
                model_dir: model_dir.into(),
                files,
                tokenizer: None,
                chat_template: None,
            })
            .build();

        Self::from_config(config).await
    }

    /// Create provider from configuration
    pub async fn from_config(config: MistralRsConfig) -> Result<Self, LLMError> {
        let model = Self::build_model(&config).await?;

        Ok(Self {
            model: Arc::new(model),
            config,
        })
    }

    /// Get a builder for advanced configuration
    pub fn builder() -> MistralRsProviderBuilder {
        MistralRsProviderBuilder::new()
    }

    /// Internal method to build the mistralrs model
    async fn build_model(config: &MistralRsConfig) -> Result<mistralrs::Model, LLMError> {
        match &config.model_source {
            ModelSource::HuggingFace { repo_id, .. } => {
                let detected_type = config.model_source.detect_model_type();
                match detected_type {
                    crate::models::ModelType::Vision => {
                        Self::build_vision_model(repo_id, config).await
                    }
                    _ => Self::build_hf_model(repo_id, config).await,
                }
            }
            ModelSource::Gguf {
                model_dir,
                files,
                tokenizer,
                chat_template,
            } => Self::build_gguf_model(model_dir, files, tokenizer, chat_template, config).await,
        }
    }

    /// Build HuggingFace text model
    async fn build_hf_model(
        repo_id: &str,
        config: &MistralRsConfig,
    ) -> Result<mistralrs::Model, LLMError> {
        let mut builder = TextModelBuilder::new(repo_id.to_string());

        // Apply ISQ if specified
        if let Some(isq) = config.isq_type {
            builder = builder.with_isq(isq);
        }

        // Apply paged attention if enabled
        if config.paged_attention {
            builder = builder
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                .map_err(convert_anyhow_error)?;
        }

        // Apply logging if enabled
        if config.logging {
            builder = builder.with_logging();
        }

        // Build the model
        builder.build().await.map_err(convert_anyhow_error)
    }

    /// Build vision model
    async fn build_vision_model(
        repo_id: &str,
        config: &MistralRsConfig,
    ) -> Result<mistralrs::Model, LLMError> {
        let mut builder = VisionModelBuilder::new(repo_id.to_string());

        // Apply ISQ if specified
        if let Some(isq) = config.isq_type {
            builder = builder.with_isq(isq);
        }

        // Apply logging if enabled
        if config.logging {
            builder = builder.with_logging();
        }

        // Build the model
        builder.build().await.map_err(convert_anyhow_error)
    }

    /// Build GGUF model
    async fn build_gguf_model(
        model_dir: &str,
        files: &[String],
        tokenizer: &Option<String>,
        chat_template: &Option<String>,
        config: &MistralRsConfig,
    ) -> Result<mistralrs::Model, LLMError> {
        let mut builder = GgufModelBuilder::new(model_dir, files.to_vec());

        // Apply tokenizer if specified
        if let Some(tok) = tokenizer {
            builder = builder.with_tok_model_id(tok);
        }

        // Apply chat template if specified
        if let Some(template) = chat_template {
            builder = builder.with_chat_template(template);
        }

        // Apply paged attention if enabled
        if config.paged_attention {
            builder = builder
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                .map_err(convert_anyhow_error)?;
        }

        // Apply logging if enabled
        if config.logging {
            builder = builder.with_logging();
        }

        // Build the model
        builder.build().await.map_err(convert_anyhow_error)
    }

    /// Get reference to the configuration
    pub fn config(&self) -> &MistralRsConfig {
        &self.config
    }

    fn prepare_messages_with_system(&self, messages: &[ChatMessage]) -> Vec<ChatMessage> {
        let mut all_messages = Vec::new();

        if let Some(system_prompt) = &self.config.system_prompt {
            let has_system_message = messages
                .iter()
                .any(|msg| msg.role == autoagents_llm::chat::ChatRole::System);

            if !has_system_message {
                all_messages.push(ChatMessage {
                    role: autoagents_llm::chat::ChatRole::System,
                    message_type: autoagents_llm::chat::MessageType::Text,
                    content: system_prompt.clone(),
                });
            }
        }

        all_messages.extend_from_slice(messages);
        all_messages
    }

    fn build_stream_request(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<RequestBuilder, LLMError> {
        let all_messages = self.prepare_messages_with_system(messages);
        let is_vision_model = self.config.model_source.detect_model_type() == ModelType::Vision;
        let has_images = all_messages.iter().any(|msg| {
            matches!(
                msg.message_type,
                autoagents_llm::chat::MessageType::Image(_)
                    | autoagents_llm::chat::MessageType::ImageURL(_)
            )
        });

        let request_builder = if tools.is_some() || json_schema.is_some() {
            build_request_builder(&all_messages, tools, json_schema)?
        } else if is_vision_model || has_images {
            let vision_messages = convert_vision_messages(&all_messages, &self.model)
                .map_err(convert_anyhow_error)?;
            RequestBuilder::from(vision_messages)
        } else {
            let text_messages = convert_messages(&all_messages);
            RequestBuilder::from(text_messages)
        };

        Ok(self.apply_sampling_params(request_builder))
    }

    fn apply_sampling_params(&self, request_builder: RequestBuilder) -> RequestBuilder {
        let mut builder = request_builder;

        if let Some(max_tokens) = self.config.max_tokens {
            builder = builder.set_sampler_max_len(max_tokens as usize);
        }
        if let Some(temperature) = self.config.temperature {
            builder = builder.set_sampler_temperature(temperature as f64);
        }
        if let Some(top_p) = self.config.top_p {
            builder = builder.set_sampler_topp(f64::from(top_p));
        }
        if let Some(top_k) = self.config.top_k {
            builder = builder.set_sampler_topk(top_k as usize);
        }

        builder
    }

    fn convert_usage(usage: MistralUsage) -> ChatUsage {
        ChatUsage {
            prompt_tokens: usage.prompt_tokens as u32,
            completion_tokens: usage.completion_tokens as u32,
            total_tokens: usage.total_tokens as u32,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        }
    }

    fn chunk_to_stream_response(chunk: ChatCompletionChunkResponse) -> Option<StreamResponse> {
        let choices: Vec<StreamChoice> = chunk
            .choices
            .into_iter()
            .filter_map(|choice: ChunkChoice| {
                let tool_calls = choice.delta.tool_calls.map(|calls| {
                    calls
                        .into_iter()
                        .map(|call| ToolCall {
                            id: call.id,
                            call_type: call.tp.to_string(),
                            function: FunctionCall {
                                name: call.function.name,
                                arguments: call.function.arguments,
                            },
                        })
                        .collect::<Vec<_>>()
                });

                let content = choice.delta.content;
                let has_tool_calls = tool_calls
                    .as_ref()
                    .map(|calls| !calls.is_empty())
                    .unwrap_or(false);
                let has_content = content.as_ref().map(|c| !c.is_empty()).unwrap_or(false);

                if !has_content && !has_tool_calls {
                    return None;
                }

                Some(StreamChoice {
                    delta: StreamDelta {
                        content,
                        tool_calls,
                    },
                })
            })
            .collect();

        if choices.is_empty() && chunk.usage.is_none() {
            return None;
        }

        Some(StreamResponse {
            choices,
            usage: chunk.usage.map(Self::convert_usage),
        })
    }

    fn done_to_stream_response(done: ChatCompletionResponse) -> StreamResponse {
        StreamResponse {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: None,
                    tool_calls: None,
                },
            }],
            usage: Some(Self::convert_usage(done.usage)),
        }
    }

    fn spawn_response_stream(
        model: Arc<mistralrs::Model>,
        request_builder: RequestBuilder,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<MistralStreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<MistralStreamEvent, LLMError>>();

        tokio::spawn(async move {
            match model.stream_chat_request(request_builder).await {
                Ok(mut response_stream) => {
                    while let Some(response) = response_stream.next().await {
                        match response {
                            Response::Chunk(chunk) => {
                                if tx.send(Ok(MistralStreamEvent::Chunk(chunk))).is_err() {
                                    break;
                                }
                            }
                            Response::Done(done) => {
                                let _ = tx.send(Ok(MistralStreamEvent::Done(done)));
                                break;
                            }
                            Response::InternalError(err) => {
                                let _ = tx.send(Err(LLMError::ProviderError(format!(
                                    "Mistral.rs internal error: {}",
                                    err
                                ))));
                                break;
                            }
                            Response::ValidationError(err) => {
                                let _ = tx.send(Err(LLMError::InvalidRequest(err.to_string())));
                                break;
                            }
                            Response::ModelError(msg, _) => {
                                let _ = tx.send(Err(LLMError::ProviderError(msg)));
                                break;
                            }
                            Response::CompletionModelError(msg, _) => {
                                let _ = tx.send(Err(LLMError::ProviderError(format!(
                                    "Mistral.rs completion error: {}",
                                    msg
                                ))));
                                break;
                            }
                            Response::CompletionDone(_)
                            | Response::CompletionChunk(_)
                            | Response::Speech { .. }
                            | Response::ImageGeneration(_)
                            | Response::Raw { .. } => {
                                let _ = tx.send(Err(LLMError::Generic(
                                    "Unexpected streaming response variant for chat request"
                                        .to_string(),
                                )));
                                break;
                            }
                            Response::Embeddings { .. } => {
                                let _ = tx.send(Err(LLMError::Generic(
                                    "Embedding streaming response variant for chat request"
                                        .to_string(),
                                )));
                                break;
                            }
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(convert_anyhow_error(err)));
                }
            }
        });

        let output_stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(output_stream)
    }
}

enum MistralStreamEvent {
    Chunk(ChatCompletionChunkResponse),
    Done(ChatCompletionResponse),
}

/// Builder for MistralRsProvider
pub struct MistralRsProviderBuilder {
    config_builder: MistralRsConfigBuilder,
}

impl MistralRsProviderBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config_builder: MistralRsConfigBuilder::new(),
        }
    }

    /// Set the model source
    pub fn model_source(mut self, source: ModelSource) -> Self {
        self.config_builder = self.config_builder.model_source(source);
        self
    }

    /// Set ISQ type
    pub fn with_isq(mut self, isq: IsqType) -> Self {
        self.config_builder = self.config_builder.with_isq(isq);
        self
    }

    /// Enable paged attention
    pub fn with_paged_attention(mut self) -> Self {
        self.config_builder = self.config_builder.with_paged_attention();
        self
    }

    /// Enable logging
    pub fn with_logging(mut self) -> Self {
        self.config_builder = self.config_builder.with_logging();
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config_builder = self.config_builder.max_tokens(tokens);
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config_builder = self.config_builder.temperature(temp);
        self
    }

    /// Set top-p
    pub fn top_p(mut self, p: f32) -> Self {
        self.config_builder = self.config_builder.top_p(p);
        self
    }

    /// Set top-k
    pub fn top_k(mut self, k: u32) -> Self {
        self.config_builder = self.config_builder.top_k(k);
        self
    }

    /// Set system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.system_prompt(prompt);
        self
    }

    /// Build the provider
    pub async fn build(self) -> Result<MistralRsProvider, LLMError> {
        let config = self.config_builder.build();
        MistralRsProvider::from_config(config).await
    }
}

impl Default for MistralRsProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert AutoAgents ChatRole to mistral.rs TextMessageRole for RequestBuilder
fn convert_role_for_request(role: &autoagents_llm::chat::ChatRole) -> TextMessageRole {
    match role {
        autoagents_llm::chat::ChatRole::System => TextMessageRole::System,
        autoagents_llm::chat::ChatRole::User => TextMessageRole::User,
        autoagents_llm::chat::ChatRole::Assistant => TextMessageRole::Assistant,
        autoagents_llm::chat::ChatRole::Tool => TextMessageRole::User,
    }
}

/// Convert AutoAgents Tool to mistral.rs Tool
fn convert_tools(tools: &[Tool]) -> Vec<mistralrs::Tool> {
    tools
        .iter()
        .map(|tool| mistralrs::Tool {
            tp: ToolType::Function,
            function: Function {
                name: tool.function.name.clone(),
                description: Some(tool.function.description.clone()),
                parameters: Some(
                    serde_json::from_value(tool.function.parameters.clone()).unwrap_or_default(),
                ),
            },
        })
        .collect()
}

/// Build a RequestBuilder with tools and/or structured output constraint
fn build_request_builder(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    json_schema: Option<StructuredOutputFormat>,
) -> Result<RequestBuilder, LLMError> {
    let mut request = RequestBuilder::new();

    // Add all messages
    for msg in messages {
        let role = convert_role_for_request(&msg.role);

        // Handle different message types
        let content = match &msg.message_type {
            autoagents_llm::chat::MessageType::Text => msg.content.clone(),
            autoagents_llm::chat::MessageType::Image(_) => {
                format!("[Image: {}]", msg.content)
            }
            autoagents_llm::chat::MessageType::ImageURL(url) => {
                format!("[Image URL: {}] {}", url, msg.content)
            }
            autoagents_llm::chat::MessageType::Pdf(_) => {
                format!("[PDF Document] {}", msg.content)
            }
            autoagents_llm::chat::MessageType::ToolUse(tool_calls) => {
                // For tool use messages, add them with tool calls
                if !tool_calls.is_empty() {
                    // Convert to mistral.rs ToolCallResponse format
                    let mistral_tool_calls: Vec<ToolCallResponse> = tool_calls
                        .iter()
                        .enumerate()
                        .map(|(index, tc)| ToolCallResponse {
                            index,
                            id: tc.id.clone(),
                            tp: ToolCallType::Function,
                            function: CalledFunction {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        })
                        .collect();

                    request = request.add_message_with_tool_call(
                        role,
                        msg.content.clone(),
                        mistral_tool_calls,
                    );
                    continue;
                }
                msg.content.clone()
            }
            autoagents_llm::chat::MessageType::ToolResult(tool_results) => {
                // For tool results, add them as tool messages
                for tc in tool_results {
                    request =
                        request.add_tool_message(tc.function.arguments.clone(), tc.id.clone());
                }
                continue;
            }
        };

        request = request.add_message(role, content);
    }

    // Set tools if provided
    if let Some(tools) = tools {
        let mistral_tools = convert_tools(tools);
        request = request
            .set_tools(mistral_tools)
            .set_tool_choice(MistralToolChoice::Auto);
    }

    if tools.is_none() {
        if let Some(schema) = json_schema {
            if let Some(json_schema) = schema.schema {
                request = request.set_constraint(Constraint::JsonSchema(json_schema));
            }
        }
    }

    Ok(request)
}

#[async_trait]
impl ChatProvider for MistralRsProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Prepare messages with system prompt if needed
        let mut all_messages = Vec::new();

        // If a system_prompt is configured and there's no system message in messages,
        // prepend it as the first system message
        if let Some(system_prompt) = &self.config.system_prompt {
            let has_system_message = messages
                .iter()
                .any(|msg| msg.role == autoagents_llm::chat::ChatRole::System);

            if !has_system_message {
                all_messages.push(ChatMessage {
                    role: autoagents_llm::chat::ChatRole::System,
                    message_type: autoagents_llm::chat::MessageType::Text,
                    content: system_prompt.clone(),
                });
            }
        }
        all_messages.extend_from_slice(messages);

        // Detect if this is a vision model by checking model type
        let is_vision_model = self.config.model_source.detect_model_type() == ModelType::Vision;

        // Check if messages contain images
        let has_images = all_messages.iter().any(|msg| {
            matches!(
                msg.message_type,
                autoagents_llm::chat::MessageType::Image(_)
            )
        });

        // Send chat request based on model type and message content
        // Use RequestBuilder if tools or structured output is needed
        let response = if tools.is_some() || json_schema.is_some() {
            // Use RequestBuilder for tools or structured output
            let request = build_request_builder(&all_messages, tools, json_schema)?;
            self.model
                .send_chat_request(request)
                .await
                .map_err(convert_anyhow_error)?
        } else if is_vision_model || has_images {
            // Use vision messages for vision models or when images are present
            let vision_messages = convert_vision_messages(&all_messages, &self.model)
                .map_err(convert_anyhow_error)?;
            self.model
                .send_chat_request(vision_messages)
                .await
                .map_err(convert_anyhow_error)?
        } else {
            // Use text messages for text-only models
            let text_messages = convert_messages(&all_messages);
            self.model
                .send_chat_request(text_messages)
                .await
                .map_err(convert_anyhow_error)?
        };

        // Extract text and tool calls from response
        let first_choice = response.choices.first();
        let text = first_choice
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();

        let tool_calls = first_choice
            .and_then(|choice| choice.message.tool_calls.as_ref())
            .map(|tcs| convert_tool_calls(tcs));

        Ok(Box::new(MistralRsResponse { text, tool_calls }))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let request_builder = self.build_stream_request(messages, None, json_schema)?;
        let response_stream = Self::spawn_response_stream(self.model.clone(), request_builder);

        let content_stream = response_stream.filter_map(|event| async move {
            match event {
                Ok(MistralStreamEvent::Chunk(chunk)) => chunk
                    .choices
                    .into_iter()
                    .find_map(|choice| choice.delta.content)
                    .filter(|content| !content.is_empty())
                    .map(Ok),
                Ok(MistralStreamEvent::Done(_)) => None,
                Err(err) => Some(Err(err)),
            }
        });

        Ok(Box::pin(content_stream))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        let request_builder = self.build_stream_request(messages, tools, json_schema)?;
        let response_stream = Self::spawn_response_stream(self.model.clone(), request_builder);

        let struct_stream = response_stream.filter_map(|event| async move {
            match event {
                Ok(MistralStreamEvent::Chunk(chunk)) => {
                    Self::chunk_to_stream_response(chunk).map(Ok)
                }
                Ok(MistralStreamEvent::Done(done)) => Some(Ok(Self::done_to_stream_response(done))),
                Err(err) => Some(Err(err)),
            }
        });

        Ok(Box::pin(struct_stream))
    }
}

#[async_trait]
impl CompletionProvider for MistralRsProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        // For completion, we convert it to a single user message
        let messages = vec![ChatMessage::user().content(&req.prompt).build()];

        let text_messages = convert_messages(&messages);

        // Send chat request
        let response = self
            .model
            .send_chat_request(text_messages)
            .await
            .map_err(convert_anyhow_error)?;

        // Extract text from response
        let text = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();

        Ok(CompletionResponse { text })
    }
}

#[async_trait]
impl EmbeddingProvider for MistralRsProvider {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        // mistral.rs TextModel doesn't support embeddings
        // This would require a separate EmbeddingModel
        Err(LLMError::NoToolSupport(
            "Embedding not supported for TextModel. Use a dedicated embedding model.".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for MistralRsProvider {}

impl LLMProvider for MistralRsProvider {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = MistralRsProvider::builder();
        drop(builder);
    }

    #[test]
    fn test_builder_configuration() {
        let builder = MistralRsProvider::builder()
            .with_logging()
            .with_paged_attention()
            .max_tokens(1024)
            .temperature(0.8);

        drop(builder);
    }

    #[test]
    fn test_provider_builder_default() {
        let builder1 = MistralRsProviderBuilder::default();
        let builder2 = MistralRsProviderBuilder::new();

        drop(builder1);
        drop(builder2);
    }
}
