//! LlamaCppProvider implementation with LLMProvider traits.

use crate::{
    builder::LlamaCppProviderBuilder,
    config::{LlamaCppConfig, LlamaCppConfigBuilder},
    conversion::{LlamaCppResponse, PromptData, build_fallback_prompt, build_openai_messages_json},
    error::LlamaCppProviderError,
    models::ModelSource,
};
use autoagents_llm::{
    FunctionCall, LLMProvider, ToolCall, async_trait,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, MessageType, StreamChoice, StreamChunk,
        StreamDelta, StreamResponse, StructuredOutputFormat, Tool, Usage as ChatUsage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
};
use futures::{StreamExt, stream::Stream};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::params::LlamaModelParams,
    model::{AddBos, GrammarTriggerType, LlamaChatTemplate, LlamaModel},
    openai::OpenAIChatTemplateParams,
    sampling::LlamaSampler,
    token::LlamaToken,
};
#[cfg(feature = "mtmd")]
use llama_cpp_2::{
    model::LlamaChatMessage,
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker},
};
use serde::Deserialize;
use serde_json::Value;
#[cfg(feature = "mtmd")]
use std::ffi::CString;
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
    path::Path,
    pin::Pin,
    sync::{Arc, OnceLock},
};
use tokio::sync::mpsc;

const JSON_GRAMMAR: &str = include_str!("grammars/json.gbnf");
const DEFAULT_N_BATCH: u32 = 64;

/// Llama.cpp provider for local LLM inference.
pub struct LlamaCppProvider {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    config: LlamaCppConfig,
}

struct GenerationResult {
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: String,
}

enum StreamEvent {
    Token(String),
    Delta(String),
    Usage(ChatUsage),
    Done { stop_reason: String },
}

type TokenCallback = Box<dyn FnMut(&str) -> Result<(), LlamaCppProviderError> + Send>;
type DeltaCallback = Box<dyn FnMut(&str) -> Result<(), LlamaCppProviderError> + Send>;

struct GenerationParams<'a> {
    prompt: &'a PromptData,
    use_json_grammar: bool,
    max_tokens: u32,
    temperature: Option<f32>,
    on_token: Option<TokenCallback>,
}

#[cfg(feature = "mtmd")]
struct MtmdGenerationParams<'a> {
    prompt: &'a str,
    marker: &'a str,
    images: &'a [Vec<u8>],
    max_tokens: u32,
    temperature: Option<f32>,
    on_token: Option<TokenCallback>,
}

struct ChatGenerationParams<'a> {
    template_result: &'a llama_cpp_2::model::ChatTemplateResult,
    max_tokens: u32,
    temperature: Option<f32>,
    on_delta: Option<DeltaCallback>,
}

enum ChatPrompt {
    OpenAI(llama_cpp_2::model::ChatTemplateResult),
    Fallback {
        prompt: PromptData,
        use_json_grammar: bool,
    },
}

#[derive(Debug, Deserialize)]
struct OpenAICompatMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAICompatDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    #[serde(rename = "type")]
    call_type: Option<String>,
    function: Option<OpenAIFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionDelta {
    name: Option<String>,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
    started: bool,
}

impl LlamaCppProvider {
    /// Create provider from GGUF model path.
    pub async fn from_gguf(model_path: impl Into<String>) -> Result<Self, LLMError> {
        let config = LlamaCppConfigBuilder::new().model_path(model_path).build();
        Self::from_config(config).await
    }

    /// Create provider from configuration.
    pub async fn from_config(mut config: LlamaCppConfig) -> Result<Self, LLMError> {
        if config.mmproj_path.is_none()
            && let ModelSource::HuggingFace {
                repo_id,
                mmproj_filename: Some(mmproj_filename),
                ..
            } = &config.model_source
        {
            let mmproj_path =
                crate::huggingface::resolve_hf_file(repo_id, mmproj_filename, &config)
                    .map_err(LLMError::from)?;
            config.mmproj_path = Some(mmproj_path);
        }

        let backend = initialize_backend()?;
        let model = load_model(backend.clone(), &config).await?;
        Ok(Self {
            backend,
            model,
            config,
        })
    }

    /// Get a builder for advanced configuration.
    pub fn builder() -> LlamaCppProviderBuilder {
        LlamaCppProviderBuilder::new()
    }

    /// Get reference to the configuration.
    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    fn prepare_messages(&self, messages: &[ChatMessage]) -> Vec<ChatMessage> {
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

    fn prepare_fallback_messages(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<&StructuredOutputFormat>,
    ) -> Vec<ChatMessage> {
        let mut all_messages = self.prepare_messages(messages);

        if let Some(schema) = json_schema {
            let mut schema_hint =
                format!("Return a valid JSON response for schema '{}'.", schema.name);
            if let Some(description) = &schema.description {
                schema_hint.push_str(&format!(" {description}"));
            }
            if let Some(json_schema) = &schema.schema {
                schema_hint.push_str(&format!(" Schema: {json_schema}"));
            }
            all_messages.push(ChatMessage {
                role: autoagents_llm::chat::ChatRole::System,
                message_type: autoagents_llm::chat::MessageType::Text,
                content: schema_hint,
            });
        }

        all_messages
    }

    fn ensure_supported_messages(&self, messages: &[ChatMessage]) -> Result<(), LLMError> {
        for message in messages {
            match &message.message_type {
                MessageType::Text | MessageType::ToolUse(_) | MessageType::ToolResult(_) => {}
                MessageType::Image(_) => {
                    #[cfg(feature = "mtmd")]
                    {
                        if self.config.mmproj_path.is_some() {
                            continue;
                        }
                    }
                    return Err(LLMError::InvalidRequest(
                        "llama.cpp backend does not support image inputs without MTMD and mmproj configured"
                            .to_string(),
                    ));
                }
                MessageType::ImageURL(_) | MessageType::Pdf(_) => {
                    return Err(LLMError::InvalidRequest(
                        "llama.cpp backend does not support image URL or PDF inputs".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn build_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<&StructuredOutputFormat>,
    ) -> Result<ChatPrompt, LLMError> {
        self.ensure_supported_messages(messages)?;
        let template = match self.resolve_chat_template() {
            Ok(template) => Some(template),
            Err(err) => {
                if tools.is_some() || json_schema.is_some() || self.config.force_json_grammar {
                    return Err(err);
                }
                None
            }
        };

        if let Some(template) = template {
            let messages = self.prepare_messages(messages);
            let messages_json = build_openai_messages_json(&messages).map_err(LLMError::from)?;
            let tools_json = match tools {
                Some(tools) if !tools.is_empty() => {
                    Some(serde_json::to_string(tools).map_err(|err| {
                        LLMError::ProviderError(format!("Failed to serialize tools: {err}"))
                    })?)
                }
                _ => None,
            };

            let json_schema_value = json_schema
                .and_then(|schema| schema.schema.as_ref())
                .map(Value::to_string);
            let grammar_value = if json_schema_value.is_none() && self.config.force_json_grammar {
                Some(JSON_GRAMMAR.to_string())
            } else {
                None
            };

            let parse_tool_calls =
                tools_json.is_some() && json_schema_value.is_none() && grammar_value.is_none();
            let params = OpenAIChatTemplateParams {
                messages_json: messages_json.as_str(),
                tools_json: tools_json.as_deref(),
                tool_choice: None,
                json_schema: json_schema_value.as_deref(),
                grammar: grammar_value.as_deref(),
                reasoning_format: None,
                chat_template_kwargs: None,
                add_generation_prompt: true,
                use_jinja: true,
                parallel_tool_calls: false,
                enable_thinking: true,
                add_bos: false,
                add_eos: false,
                parse_tool_calls,
            };

            let result = self
                .model
                .apply_chat_template_oaicompat(&template, &params)
                .map_err(|err| {
                    LLMError::ProviderError(format!(
                        "Failed to apply OpenAI-compatible chat template: {err}"
                    ))
                })?;

            Ok(ChatPrompt::OpenAI(result))
        } else {
            let fallback_messages = self.prepare_fallback_messages(messages, json_schema);
            let prompt = PromptData {
                prompt: build_fallback_prompt(&fallback_messages),
                add_bos: AddBos::Always,
            };
            let use_json_grammar = json_schema.is_some() || self.config.force_json_grammar;
            Ok(ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            })
        }
    }

    fn has_mtmd_media(messages: &[ChatMessage]) -> bool {
        messages
            .iter()
            .any(|message| matches!(message.message_type, MessageType::Image(_)))
    }

    #[cfg(feature = "mtmd")]
    fn build_mtmd_prompt(
        &self,
        messages: &[ChatMessage],
    ) -> Result<(String, Vec<Vec<u8>>, String), LLMError> {
        let template = self.resolve_chat_template()?;
        let mut chat = Vec::new();
        let mut images = Vec::new();
        let default_marker = mtmd_default_marker().to_string();
        let marker = self
            .config
            .media_marker
            .as_deref()
            .unwrap_or(&default_marker)
            .to_string();

        for message in self.prepare_messages(messages) {
            let mut content = message.content.clone();
            match message.message_type {
                MessageType::Text => {}
                MessageType::Image((_, bytes)) => {
                    images.push(bytes);
                    if !content.contains(&marker) {
                        content.push_str(&marker);
                    }
                }
                //TODO: Get a FIX
                MessageType::ToolUse(_) | MessageType::ToolResult(_) => {
                    return Err(LLMError::InvalidRequest(
                        "MTMD path does not support tool calls".to_string(),
                    ));
                }
                MessageType::ImageURL(_) | MessageType::Pdf(_) => {
                    return Err(LLMError::InvalidRequest(
                        "MTMD path only supports raw image inputs".to_string(),
                    ));
                }
            }

            let role = match message.role {
                autoagents_llm::chat::ChatRole::System => "system",
                autoagents_llm::chat::ChatRole::User => "user",
                autoagents_llm::chat::ChatRole::Assistant => "assistant",
                autoagents_llm::chat::ChatRole::Tool => "tool",
            };

            let chat_msg = LlamaChatMessage::new(role.to_string(), content)
                .map_err(|err| LLMError::ProviderError(format!("Invalid chat message: {err}")))?;
            chat.push(chat_msg);
        }

        let prompt = self
            .model
            .apply_chat_template(&template, &chat, true)
            .map_err(|err| {
                LLMError::ProviderError(format!("Failed to apply chat template: {err}"))
            })?;

        Ok((prompt, images, marker))
    }

    fn resolve_chat_template(&self) -> Result<LlamaChatTemplate, LLMError> {
        if let Some(template) = self.config.chat_template.as_deref() {
            return LlamaChatTemplate::new(template)
                .map_err(|err| LLMError::ProviderError(format!("Invalid chat template: {err}")));
        }

        self.model.chat_template(None).map_err(|err| {
            LLMError::ProviderError(format!("Model does not provide a chat template: {err}"))
        })
    }

    fn build_usage(prompt_tokens: u32, completion_tokens: u32) -> ChatUsage {
        ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        }
    }

    fn resolve_max_tokens(&self, max_tokens_override: Option<u32>) -> u32 {
        max_tokens_override
            .or(self.config.max_tokens)
            .unwrap_or(512)
    }

    fn resolve_temperature(&self, temperature_override: Option<f32>) -> Option<f32> {
        temperature_override.or(self.config.temperature)
    }

    async fn generate_completion_response(
        &self,
        prompt: PromptData,
        use_json_grammar: bool,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Result<GenerationResult, LLMError> {
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);

        let mut result = tokio::task::spawn_blocking(
            move || -> Result<GenerationResult, LlamaCppProviderError> {
                generate_text(
                    &model,
                    &backend,
                    &config,
                    GenerationParams {
                        prompt: &prompt,
                        use_json_grammar,
                        max_tokens,
                        temperature,
                        on_token: None,
                    },
                )
            },
        )
        .await
        .map_err(|err| LLMError::ProviderError(format!("Generation task failed: {err}")))?
        .map_err(LLMError::from)?;

        if use_json_grammar && let Some(extracted) = extract_json_payload(&result.text) {
            result.text = extracted;
        }

        Ok(result)
    }

    async fn generate_chat_completion(
        &self,
        template_result: llama_cpp_2::model::ChatTemplateResult,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Result<GenerationResult, LLMError> {
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);

        tokio::task::spawn_blocking(move || -> Result<GenerationResult, LlamaCppProviderError> {
            generate_chat_text(
                &model,
                &backend,
                &config,
                ChatGenerationParams {
                    template_result: &template_result,
                    max_tokens,
                    temperature,
                    on_delta: None,
                },
            )
        })
        .await
        .map_err(|err| LLMError::ProviderError(format!("Generation task failed: {err}")))?
        .map_err(LLMError::from)
    }

    fn spawn_fallback_stream(
        &self,
        prompt: PromptData,
        use_json_grammar: bool,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let emitted_any = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tx_tokens = tx.clone();

        tokio::spawn(async move {
            let emitted_any = Arc::clone(&emitted_any);
            let emitted_any_for_blocking = Arc::clone(&emitted_any);
            let result = tokio::task::spawn_blocking(
                move || -> Result<GenerationResult, LlamaCppProviderError> {
                    let mut json_started = false;
                    let emitted_any = emitted_any_for_blocking;
                    let on_token: Option<TokenCallback> = Some(Box::new(move |token: &str| {
                        if use_json_grammar && !json_started {
                            if let Some(start) = token.find('{').or_else(|| token.find('[')) {
                                json_started = true;
                                let suffix = &token[start..];
                                if !suffix.is_empty() {
                                    tx_tokens
                                        .send(Ok(StreamEvent::Token(suffix.to_string())))
                                        .map_err(|_| {
                                            LlamaCppProviderError::Inference(
                                                "Stream receiver dropped".to_string(),
                                            )
                                        })?;
                                    emitted_any.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            return Ok(());
                        }

                        tx_tokens
                            .send(Ok(StreamEvent::Token(token.to_string())))
                            .map_err(|_| {
                                LlamaCppProviderError::Inference(
                                    "Stream receiver dropped".to_string(),
                                )
                            })?;
                        emitted_any.store(true, std::sync::atomic::Ordering::Relaxed);
                        Ok(())
                    })
                        as TokenCallback);
                    generate_text(
                        &model,
                        &backend,
                        &config,
                        GenerationParams {
                            prompt: &prompt,
                            use_json_grammar,
                            max_tokens,
                            temperature,
                            on_token,
                        },
                    )
                },
            )
            .await;

            match result {
                Ok(Ok(generation)) => {
                    if use_json_grammar && !emitted_any.load(std::sync::atomic::Ordering::Relaxed) {
                        let mut text = generation.text;
                        if let Some(extracted) = extract_json_payload(&text) {
                            text = extracted;
                        }
                        let _ = tx.send(Ok(StreamEvent::Token(text)));
                    }
                    let usage = LlamaCppProvider::build_usage(
                        generation.prompt_tokens,
                        generation.completion_tokens,
                    );
                    let _ = tx.send(Ok(StreamEvent::Usage(usage)));
                    let stop_reason = if generation.finish_reason == "length" {
                        "length".to_string()
                    } else {
                        "end_turn".to_string()
                    };
                    let _ = tx.send(Ok(StreamEvent::Done { stop_reason }));
                }
                Ok(Err(err)) => {
                    let _ = tx.send(Err(LLMError::from(err)));
                }
                Err(err) => {
                    let _ = tx.send(Err(LLMError::ProviderError(format!(
                        "Generation task failed: {err}"
                    ))));
                }
            }
        });

        let output_stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(output_stream)
    }

    #[cfg(feature = "mtmd")]
    fn spawn_mtmd_stream(
        &self,
        prompt: String,
        images: Vec<Vec<u8>>,
        marker: String,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let tx_tokens = tx.clone();

        tokio::spawn(async move {
            let result = tokio::task::spawn_blocking(
                move || -> Result<GenerationResult, LlamaCppProviderError> {
                    let on_token: Option<TokenCallback> = Some(Box::new(move |token: &str| {
                        tx_tokens
                            .send(Ok(StreamEvent::Token(token.to_string())))
                            .map_err(|_| {
                                LlamaCppProviderError::Inference(
                                    "Stream receiver dropped".to_string(),
                                )
                            })?;
                        Ok(())
                    })
                        as TokenCallback);

                    generate_mtmd_text(
                        &model,
                        &backend,
                        &config,
                        MtmdGenerationParams {
                            prompt: &prompt,
                            marker: &marker,
                            images: &images,
                            max_tokens,
                            temperature,
                            on_token,
                        },
                    )
                },
            )
            .await;

            match result {
                Ok(Ok(generation)) => {
                    let usage = LlamaCppProvider::build_usage(
                        generation.prompt_tokens,
                        generation.completion_tokens,
                    );
                    let _ = tx.send(Ok(StreamEvent::Usage(usage)));
                    let _ = tx.send(Ok(StreamEvent::Done {
                        stop_reason: generation.finish_reason,
                    }));
                }
                Ok(Err(err)) => {
                    let _ = tx.send(Err(LLMError::from(err)));
                }
                Err(err) => {
                    let _ = tx.send(Err(LLMError::ProviderError(format!(
                        "Generation task failed: {err}"
                    ))));
                }
            }
        });

        let output_stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(output_stream)
    }

    fn spawn_chat_stream(
        &self,
        template_result: llama_cpp_2::model::ChatTemplateResult,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let tx_deltas = tx.clone();

        tokio::spawn(async move {
            let result = tokio::task::spawn_blocking(
                move || -> Result<(GenerationResult, String), LlamaCppProviderError> {
                    let on_delta: Option<DeltaCallback> = Some(Box::new(move |delta: &str| {
                        tx_deltas
                            .send(Ok(StreamEvent::Delta(delta.to_string())))
                            .map_err(|_| {
                                LlamaCppProviderError::Inference(
                                    "Stream receiver dropped".to_string(),
                                )
                            })?;
                        Ok(())
                    }));

                    let generation = generate_chat_text(
                        &model,
                        &backend,
                        &config,
                        ChatGenerationParams {
                            template_result: &template_result,
                            max_tokens,
                            temperature,
                            on_delta,
                        },
                    )?;

                    let message_json = template_result
                        .parse_response_oaicompat(&generation.text, false)
                        .map_err(|err| {
                            LlamaCppProviderError::Template(format!(
                                "Failed to parse response: {err}"
                            ))
                        })?;
                    let message: OpenAICompatMessage = serde_json::from_str(&message_json)
                        .map_err(|err| {
                            LlamaCppProviderError::Template(format!(
                                "Failed to decode parsed message: {err}"
                            ))
                        })?;

                    let stop_reason = if generation.finish_reason == "length" {
                        "length".to_string()
                    } else if message
                        .tool_calls
                        .as_ref()
                        .map(|calls| !calls.is_empty())
                        .unwrap_or(false)
                    {
                        "tool_use".to_string()
                    } else {
                        "end_turn".to_string()
                    };

                    Ok((generation, stop_reason))
                },
            )
            .await;

            match result {
                Ok(Ok((generation, stop_reason))) => {
                    let usage = LlamaCppProvider::build_usage(
                        generation.prompt_tokens,
                        generation.completion_tokens,
                    );
                    let _ = tx.send(Ok(StreamEvent::Usage(usage)));
                    let _ = tx.send(Ok(StreamEvent::Done { stop_reason }));
                }
                Ok(Err(err)) => {
                    let _ = tx.send(Err(LLMError::from(err)));
                }
                Err(err) => {
                    let _ = tx.send(Err(LLMError::ProviderError(format!(
                        "Generation task failed: {err}"
                    ))));
                }
            }
        });

        let output_stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(output_stream)
    }
}

#[async_trait]
impl ChatProvider for LlamaCppProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if Self::has_mtmd_media(messages) {
            if tools.is_some() || json_schema.is_some() {
                return Err(LLMError::InvalidRequest(
                    "MTMD path does not support tools or structured outputs".to_string(),
                ));
            }
            #[cfg(feature = "mtmd")]
            {
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let config = self.config.clone();
                let model = self.model.clone();
                let backend = self.backend.clone();
                let max_tokens = self.resolve_max_tokens(None);
                let temperature = self.resolve_temperature(None);
                let result = tokio::task::spawn_blocking(move || {
                    generate_mtmd_text(
                        &model,
                        &backend,
                        &config,
                        MtmdGenerationParams {
                            prompt: &prompt,
                            marker: &marker,
                            images: &images,
                            max_tokens,
                            temperature,
                            on_token: None,
                        },
                    )
                })
                .await
                .map_err(|err| LLMError::ProviderError(format!("Generation task failed: {err}")))?
                .map_err(LLMError::from)?;

                let usage = Some(Self::build_usage(
                    result.prompt_tokens,
                    result.completion_tokens,
                ));

                return Ok(Box::new(LlamaCppResponse {
                    content: Some(result.text),
                    tool_calls: None,
                    usage,
                }));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::InvalidRequest(
                    "MTMD feature is not enabled for llama.cpp backend".to_string(),
                ));
            }
        }

        let prompt = self.build_chat_prompt(messages, tools, json_schema.as_ref())?;
        match prompt {
            ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            } => {
                if tools.is_some() {
                    return Err(LLMError::NoToolSupport(
                        "Tool calls require a chat template".to_string(),
                    ));
                }
                let result = self
                    .generate_completion_response(prompt, use_json_grammar, None, None)
                    .await?;
                let usage = Some(Self::build_usage(
                    result.prompt_tokens,
                    result.completion_tokens,
                ));

                Ok(Box::new(LlamaCppResponse {
                    content: Some(result.text),
                    tool_calls: None,
                    usage,
                }))
            }
            ChatPrompt::OpenAI(template_result) => {
                let result = self
                    .generate_chat_completion(template_result.clone(), None, None)
                    .await?;
                let message_json = template_result
                    .parse_response_oaicompat(&result.text, false)
                    .map_err(|err| {
                        LLMError::ProviderError(format!("Failed to parse response: {err}"))
                    })?;
                let message: OpenAICompatMessage =
                    serde_json::from_str(&message_json).map_err(|err| {
                        LLMError::ProviderError(format!("Failed to decode response: {err}"))
                    })?;

                let usage = Some(Self::build_usage(
                    result.prompt_tokens,
                    result.completion_tokens,
                ));

                Ok(Box::new(LlamaCppResponse {
                    content: message.content,
                    tool_calls: message.tool_calls,
                    usage,
                }))
            }
        }
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if Self::has_mtmd_media(messages) {
            #[cfg(feature = "mtmd")]
            {
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let response_stream = self.spawn_mtmd_stream(prompt, images, marker, None, None);
                let content_stream = response_stream.filter_map(|event| async move {
                    match event {
                        Ok(StreamEvent::Token(token)) => Some(Ok(token)),
                        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                            Ok(parsed) => {
                                parsed.content.filter(|content| !content.is_empty()).map(Ok)
                            }
                            Err(err) => Some(Err(err)),
                        },
                        Ok(StreamEvent::Usage(_)) | Ok(StreamEvent::Done { .. }) => None,
                        Err(err) => Some(Err(err)),
                    }
                });
                return Ok(Box::pin(content_stream));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::InvalidRequest(
                    "MTMD feature is not enabled for llama.cpp backend".to_string(),
                ));
            }
        }
        let prompt = self.build_chat_prompt(messages, None, json_schema.as_ref())?;
        let response_stream = match prompt {
            ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            } => self.spawn_fallback_stream(prompt, use_json_grammar, None, None),
            ChatPrompt::OpenAI(template_result) => {
                self.spawn_chat_stream(template_result, None, None)
            }
        };

        let content_stream = response_stream.filter_map(|event| async move {
            match event {
                Ok(StreamEvent::Token(token)) => Some(Ok(token)),
                Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                    Ok(parsed) => parsed.content.filter(|content| !content.is_empty()).map(Ok),
                    Err(err) => Some(Err(err)),
                },
                Ok(StreamEvent::Usage(_)) | Ok(StreamEvent::Done { .. }) => None,
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
        if Self::has_mtmd_media(messages) {
            #[cfg(feature = "mtmd")]
            {
                if tools.is_some() || json_schema.is_some() {
                    return Err(LLMError::InvalidRequest(
                        "MTMD path does not support tools or structured outputs".to_string(),
                    ));
                }
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let response_stream = self.spawn_mtmd_stream(prompt, images, marker, None, None);
                let struct_stream = response_stream.filter_map(|event| async move {
                    match event {
                        Ok(StreamEvent::Token(token)) => Some(Ok(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta {
                                    content: Some(token),
                                    tool_calls: None,
                                },
                            }],
                            usage: None,
                        })),
                        Ok(StreamEvent::Usage(_)) | Ok(StreamEvent::Done { .. }) => None,
                        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                            Ok(parsed) => parsed.content.filter(|content| !content.is_empty()).map(
                                |content| {
                                    Ok(StreamResponse {
                                        choices: vec![StreamChoice {
                                            delta: StreamDelta {
                                                content: Some(content),
                                                tool_calls: None,
                                            },
                                        }],
                                        usage: None,
                                    })
                                },
                            ),
                            Err(err) => Some(Err(err)),
                        },
                        Err(err) => Some(Err(err)),
                    }
                });
                return Ok(Box::pin(struct_stream));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::InvalidRequest(
                    "MTMD feature is not enabled for llama.cpp backend".to_string(),
                ));
            }
        }
        let prompt = self.build_chat_prompt(messages, tools, json_schema.as_ref())?;
        let response_stream = match prompt {
            ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            } => self.spawn_fallback_stream(prompt, use_json_grammar, None, None),
            ChatPrompt::OpenAI(template_result) => {
                self.spawn_chat_stream(template_result, None, None)
            }
        };

        let struct_stream = response_stream
            .scan(
                HashMap::<usize, ToolCallState>::new(),
                |tool_states, event| {
                    let mut outputs = Vec::new();
                    match event {
                        Ok(StreamEvent::Token(token)) => {
                            outputs.push(Ok(StreamResponse {
                                choices: vec![StreamChoice {
                                    delta: StreamDelta {
                                        content: Some(token),
                                        tool_calls: None,
                                    },
                                }],
                                usage: None,
                            }));
                        }
                        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                            Ok(parsed) => {
                                if let Some(content) = parsed.content
                                    && !content.is_empty()
                                {
                                    outputs.push(Ok(StreamResponse {
                                        choices: vec![StreamChoice {
                                            delta: StreamDelta {
                                                content: Some(content),
                                                tool_calls: None,
                                            },
                                        }],
                                        usage: None,
                                    }));
                                }

                                if let Some(tool_calls) = parsed.tool_calls {
                                    let mut updated_calls = Vec::new();
                                    for call in tool_calls {
                                        let index = call.index.unwrap_or(0);
                                        let call_type = call
                                            .call_type
                                            .unwrap_or_else(|| "function".to_string());
                                        let state = tool_states.entry(index).or_default();
                                        if let Some(id) = call.id {
                                            state.id = id;
                                        }
                                        if let Some(function) = call.function {
                                            if let Some(name) = function.name {
                                                state.name = name;
                                            }
                                            if !function.arguments.is_empty() {
                                                state.arguments.push_str(&function.arguments);
                                            }
                                        }
                                        if !state.id.is_empty()
                                            || !state.name.is_empty()
                                            || !state.arguments.is_empty()
                                        {
                                            updated_calls.push(ToolCall {
                                                id: state.id.clone(),
                                                call_type,
                                                function: FunctionCall {
                                                    name: state.name.clone(),
                                                    arguments: state.arguments.clone(),
                                                },
                                            });
                                        }
                                    }

                                    if !updated_calls.is_empty() {
                                        outputs.push(Ok(StreamResponse {
                                            choices: vec![StreamChoice {
                                                delta: StreamDelta {
                                                    content: None,
                                                    tool_calls: Some(updated_calls),
                                                },
                                            }],
                                            usage: None,
                                        }));
                                    }
                                }
                            }
                            Err(err) => {
                                outputs.push(Err(err));
                            }
                        },
                        Ok(StreamEvent::Usage(usage)) => {
                            outputs.push(Ok(StreamResponse {
                                choices: vec![StreamChoice {
                                    delta: StreamDelta {
                                        content: None,
                                        tool_calls: None,
                                    },
                                }],
                                usage: Some(usage),
                            }));
                        }
                        Ok(StreamEvent::Done { .. }) => {}
                        Err(err) => outputs.push(Err(err)),
                    }
                    futures::future::ready(Some(outputs))
                },
            )
            .flat_map(futures::stream::iter);

        Ok(Box::pin(struct_stream))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        if Self::has_mtmd_media(messages) {
            if tools.is_some() || json_schema.is_some() {
                return Err(LLMError::InvalidRequest(
                    "MTMD path does not support tools or structured outputs".to_string(),
                ));
            }
            #[cfg(feature = "mtmd")]
            {
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let response_stream = self.spawn_mtmd_stream(prompt, images, marker, None, None);
                let stream = response_stream.filter_map(|event| async move {
                    match event {
                        Ok(StreamEvent::Token(token)) => Some(Ok(StreamChunk::Text(token))),
                        Ok(StreamEvent::Usage(usage)) => Some(Ok(StreamChunk::Usage(usage))),
                        Ok(StreamEvent::Done { stop_reason }) => {
                            Some(Ok(StreamChunk::Done { stop_reason }))
                        }
                        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                            Ok(parsed) => parsed
                                .content
                                .filter(|content| !content.is_empty())
                                .map(|content| Ok(StreamChunk::Text(content))),
                            Err(err) => Some(Err(err)),
                        },
                        Err(err) => Some(Err(err)),
                    }
                });
                return Ok(Box::pin(stream));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::InvalidRequest(
                    "MTMD feature is not enabled for llama.cpp backend".to_string(),
                ));
            }
        }

        let prompt = self.build_chat_prompt(messages, tools, json_schema.as_ref())?;
        let response_stream = match prompt {
            ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            } => {
                if tools.is_some() {
                    return Err(LLMError::NoToolSupport(
                        "Tool calls require a chat template".to_string(),
                    ));
                }
                self.spawn_fallback_stream(prompt, use_json_grammar, None, None)
            }
            ChatPrompt::OpenAI(template_result) => {
                self.spawn_chat_stream(template_result, None, None)
            }
        };

        let stream = response_stream
            .scan(
                HashMap::<usize, ToolCallState>::new(),
                |tool_states, event| {
                    let mut outputs = Vec::new();
                    match event {
                        Ok(StreamEvent::Token(token)) => {
                            outputs.push(Ok(StreamChunk::Text(token)));
                        }
                        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                            Ok(parsed) => {
                                if let Some(content) = parsed.content
                                    && !content.is_empty()
                                {
                                    outputs.push(Ok(StreamChunk::Text(content)));
                                }

                                if let Some(tool_calls) = parsed.tool_calls {
                                    for call in tool_calls {
                                        let index = call.index.unwrap_or(0);
                                        let state = tool_states.entry(index).or_default();
                                        if let Some(id) = call.id {
                                            state.id = id;
                                        }
                                        if let Some(function) = call.function {
                                            if let Some(name) = function.name {
                                                state.name = name;
                                                if !state.started {
                                                    state.started = true;
                                                    outputs.push(Ok(StreamChunk::ToolUseStart {
                                                        index,
                                                        id: state.id.clone(),
                                                        name: state.name.clone(),
                                                    }));
                                                }
                                            }
                                            if !function.arguments.is_empty() {
                                                state.arguments.push_str(&function.arguments);
                                                outputs.push(Ok(StreamChunk::ToolUseInputDelta {
                                                    index,
                                                    partial_json: function.arguments,
                                                }));
                                            }
                                        }
                                    }
                                }
                            }
                            Err(err) => outputs.push(Err(err)),
                        },
                        Ok(StreamEvent::Usage(usage)) => {
                            outputs.push(Ok(StreamChunk::Usage(usage)));
                        }
                        Ok(StreamEvent::Done { stop_reason }) => {
                            for (index, state) in tool_states.drain() {
                                if state.started {
                                    outputs.push(Ok(StreamChunk::ToolUseComplete {
                                        index,
                                        tool_call: ToolCall {
                                            id: state.id,
                                            call_type: "function".to_string(),
                                            function: FunctionCall {
                                                name: state.name,
                                                arguments: state.arguments,
                                            },
                                        },
                                    }));
                                }
                            }
                            outputs.push(Ok(StreamChunk::Done { stop_reason }));
                        }
                        Err(err) => outputs.push(Err(err)),
                    }
                    futures::future::ready(Some(outputs))
                },
            )
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

fn parse_openai_delta(delta: &str) -> Result<OpenAICompatDelta, LLMError> {
    serde_json::from_str(delta).map_err(|err| LLMError::JsonError(err.to_string()))
}

#[async_trait]
impl CompletionProvider for LlamaCppProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let prompt = PromptData {
            prompt: req.prompt.clone(),
            add_bos: AddBos::Always,
        };
        let use_json_grammar = json_schema.is_some() || self.config.force_json_grammar;
        let result = self
            .generate_completion_response(prompt, use_json_grammar, req.max_tokens, req.temperature)
            .await?;

        Ok(CompletionResponse { text: result.text })
    }
}

#[async_trait]
impl EmbeddingProvider for LlamaCppProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, LlamaCppProviderError> {
            let mut embeddings = Vec::with_capacity(input.len());
            for text in input {
                let embedding = generate_embedding(&model, &backend, &config, &text)?;
                embeddings.push(embedding);
            }
            Ok(embeddings)
        })
        .await
        .map_err(|err| LLMError::ProviderError(format!("Embedding task failed: {err}")))?
        .map_err(LLMError::from)
    }
}

#[async_trait]
impl ModelsProvider for LlamaCppProvider {}

impl LLMProvider for LlamaCppProvider {}

fn initialize_backend() -> Result<Arc<LlamaBackend>, LlamaCppProviderError> {
    static BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();
    if let Some(backend) = BACKEND.get() {
        return Ok(backend.clone());
    }

    let mut backend = LlamaBackend::init().map_err(|err| {
        LlamaCppProviderError::Other(format!("Failed to initialize llama backend: {err}"))
    })?;
    if !llama_logs_enabled() {
        backend.void_logs();
    }
    let backend = Arc::new(backend);
    let _ = BACKEND.set(backend.clone());
    Ok(backend)
}

fn llama_logs_enabled() -> bool {
    log::log_enabled!(log::Level::Info)
}

async fn load_model(
    backend: Arc<LlamaBackend>,
    config: &LlamaCppConfig,
) -> Result<Arc<LlamaModel>, LLMError> {
    let model_source = config.model_source.clone();
    let config = config.clone();
    tokio::task::spawn_blocking(move || -> Result<LlamaModel, LlamaCppProviderError> {
        let params = build_model_params(&config)?;
        let model_path = resolve_model_path(&model_source, &config)?;
        let path = Path::new(&model_path);
        LlamaModel::load_from_file(&backend, path, &params)
            .map_err(|err| LlamaCppProviderError::ModelLoad(err.to_string()))
    })
    .await
    .map_err(|err| LLMError::ProviderError(format!("Model load task failed: {err}")))?
    .map(Arc::new)
    .map_err(LLMError::from)
}

fn build_model_params(config: &LlamaCppConfig) -> Result<LlamaModelParams, LlamaCppProviderError> {
    let mut params = LlamaModelParams::default();

    if let Some(layers) = config.n_gpu_layers {
        params = params.with_n_gpu_layers(layers);
    }
    if let Some(main_gpu) = config.main_gpu {
        params = params.with_main_gpu(main_gpu);
    }
    if let Some(split_mode) = config.split_mode {
        params = params.with_split_mode(split_mode.into());
    }
    if let Some(use_mlock) = config.use_mlock {
        params = params.with_use_mlock(use_mlock);
    }
    if let Some(devices) = config.devices.as_ref() {
        params = params
            .with_devices(devices)
            .map_err(|err| LlamaCppProviderError::Config(err.to_string()))?;
    }

    Ok(params)
}

fn resolve_model_path(
    source: &ModelSource,
    config: &LlamaCppConfig,
) -> Result<String, LlamaCppProviderError> {
    match source {
        ModelSource::Gguf { model_path } => {
            if model_path.is_empty() {
                return Err(LlamaCppProviderError::Config(
                    "Model path is required for llama.cpp".to_string(),
                ));
            }
            Ok(model_path.clone())
        }
        ModelSource::HuggingFace {
            repo_id,
            filename,
            mmproj_filename: _,
        } => crate::huggingface::resolve_hf_model(repo_id, filename.as_deref(), config),
    }
}

fn resolve_n_batch(config: &LlamaCppConfig, n_ctx: u32) -> u32 {
    let n_ctx = n_ctx.max(1);
    let requested = config.n_batch.unwrap_or(DEFAULT_N_BATCH).max(1);
    requested.min(n_ctx)
}

fn build_context_params(
    config: &LlamaCppConfig,
    embeddings: bool,
    n_ctx_override: Option<u32>,
    n_batch_override: Option<u32>,
) -> Result<LlamaContextParams, LlamaCppProviderError> {
    let mut params = LlamaContextParams::default();

    if let Some(n_ctx) = n_ctx_override.or(config.n_ctx) {
        params = params.with_n_ctx(NonZeroU32::new(n_ctx));
    }
    if let Some(n_batch) = n_batch_override.or(config.n_batch) {
        params = params.with_n_batch(n_batch);
    }
    if let Some(n_ubatch) = config.n_ubatch {
        params = params.with_n_ubatch(n_ubatch);
    }
    if let Some(n_threads) = config.n_threads {
        params = params.with_n_threads(n_threads);
    }
    if let Some(n_threads) = config.n_threads_batch {
        params = params.with_n_threads_batch(n_threads);
    }
    params = params.with_embeddings(embeddings);

    Ok(params)
}

fn resolve_context_size(
    model: &LlamaModel,
    config: &LlamaCppConfig,
    required_tokens: u32,
) -> Result<u32, LlamaCppProviderError> {
    if let Some(n_ctx) = config.n_ctx {
        if required_tokens > n_ctx {
            return Err(LlamaCppProviderError::Inference(format!(
                "Prompt length ({required_tokens}) exceeds context size ({n_ctx})",
            )));
        }
        return Ok(n_ctx);
    }

    Ok(model.n_ctx_train().max(required_tokens))
}

fn build_sampler(
    model: &LlamaModel,
    config: &LlamaCppConfig,
    use_json_grammar: bool,
    temperature_override: Option<f32>,
    seed_override: Option<u32>,
) -> Result<LlamaSampler, LlamaCppProviderError> {
    let mut samplers = Vec::new();

    if use_json_grammar {
        let sampler = LlamaSampler::grammar(model, JSON_GRAMMAR, "root")
            .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?;
        samplers.push(sampler);
    }

    let penalty_repeat = config.repeat_penalty.unwrap_or(1.0);
    let penalty_freq = config.frequency_penalty.unwrap_or(0.0);
    let penalty_present = config.presence_penalty.unwrap_or(0.0);
    let penalty_last_n = config.repeat_last_n.unwrap_or(64);
    if penalty_repeat != 1.0 || penalty_freq != 0.0 || penalty_present != 0.0 {
        samplers.push(LlamaSampler::penalties(
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        ));
    }

    if let Some(top_k) = config.top_k {
        samplers.push(LlamaSampler::top_k(top_k as i32));
    }
    if let Some(top_p) = config.top_p {
        samplers.push(LlamaSampler::top_p(top_p, 1));
    }

    let temperature = temperature_override.or(config.temperature);
    if let Some(temp) = temperature {
        if temp > 0.0 {
            samplers.push(LlamaSampler::temp(temp));
            let seed = seed_override.or(config.seed).unwrap_or_else(rand::random);
            samplers.push(LlamaSampler::dist(seed));
        } else {
            samplers.push(LlamaSampler::greedy());
        }
    } else {
        let seed = seed_override.or(config.seed).unwrap_or_else(rand::random);
        samplers.push(LlamaSampler::dist(seed));
    }

    Ok(LlamaSampler::chain_simple(samplers))
}

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '.' | '^' | '$' | '|' | '(' | ')' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '\\' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }
    let mut anchored = String::default();
    if !pattern.starts_with('^') {
        anchored.push('^');
    }
    anchored.push_str(pattern);
    if !pattern.ends_with('$') {
        anchored.push('$');
    }
    anchored
}

fn build_chat_sampler(
    model: &LlamaModel,
    config: &LlamaCppConfig,
    result: &llama_cpp_2::model::ChatTemplateResult,
    temperature_override: Option<f32>,
) -> Result<(LlamaSampler, HashSet<LlamaToken>), LlamaCppProviderError> {
    let mut preserved = HashSet::new();
    for token_str in &result.preserved_tokens {
        let tokens = model
            .str_to_token(token_str, AddBos::Never)
            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
        if tokens.len() == 1 {
            preserved.insert(tokens[0]);
        }
    }

    let grammar_sampler = if let Some(grammar) = result.grammar.as_deref() {
        if result.grammar_lazy {
            if result.grammar_triggers.is_empty() {
                return Err(LlamaCppProviderError::Template(
                    "grammar_lazy enabled but no triggers were provided".to_string(),
                ));
            }
            let mut trigger_patterns = Vec::new();
            let mut trigger_tokens = Vec::new();
            for trigger in &result.grammar_triggers {
                match trigger.trigger_type {
                    GrammarTriggerType::Token => {
                        if let Some(token) = trigger.token {
                            trigger_tokens.push(token);
                        }
                    }
                    GrammarTriggerType::Word => {
                        let tokens = model
                            .str_to_token(&trigger.value, AddBos::Never)
                            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
                        if tokens.len() == 1 {
                            if !preserved.contains(&tokens[0]) {
                                return Err(LlamaCppProviderError::Template(format!(
                                    "grammar trigger word not preserved: {}",
                                    trigger.value
                                )));
                            }
                            trigger_tokens.push(tokens[0]);
                        } else {
                            trigger_patterns.push(regex_escape(&trigger.value));
                        }
                    }
                    GrammarTriggerType::Pattern => {
                        trigger_patterns.push(trigger.value.clone());
                    }
                    GrammarTriggerType::PatternFull => {
                        trigger_patterns.push(anchor_pattern(&trigger.value));
                    }
                }
            }

            Some(
                LlamaSampler::grammar_lazy_patterns(
                    model,
                    grammar,
                    "root",
                    &trigger_patterns,
                    &trigger_tokens,
                )
                .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?,
            )
        } else {
            Some(
                LlamaSampler::grammar(model, grammar, "root")
                    .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?,
            )
        }
    } else {
        None
    };

    let mut samplers = Vec::new();
    if let Some(grammar_sampler) = grammar_sampler {
        samplers.push(grammar_sampler);
    }

    let penalty_repeat = config.repeat_penalty.unwrap_or(1.0);
    let penalty_freq = config.frequency_penalty.unwrap_or(0.0);
    let penalty_present = config.presence_penalty.unwrap_or(0.0);
    let penalty_last_n = config.repeat_last_n.unwrap_or(64);
    if penalty_repeat != 1.0 || penalty_freq != 0.0 || penalty_present != 0.0 {
        samplers.push(LlamaSampler::penalties(
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        ));
    }

    if let Some(top_k) = config.top_k {
        samplers.push(LlamaSampler::top_k(top_k as i32));
    }
    if let Some(top_p) = config.top_p {
        samplers.push(LlamaSampler::top_p(top_p, 1));
    }

    let temperature = temperature_override.or(config.temperature);
    if let Some(temp) = temperature {
        if temp > 0.0 {
            samplers.push(LlamaSampler::temp(temp));
            let seed = config.seed.unwrap_or_else(rand::random);
            samplers.push(LlamaSampler::dist(seed));
        } else {
            samplers.push(LlamaSampler::greedy());
        }
    } else {
        let seed = config.seed.unwrap_or_else(rand::random);
        samplers.push(LlamaSampler::dist(seed));
    }

    Ok((LlamaSampler::chain_simple(samplers), preserved))
}

fn generate_chat_text(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &LlamaCppConfig,
    params: ChatGenerationParams<'_>,
) -> Result<GenerationResult, LlamaCppProviderError> {
    let ChatGenerationParams {
        template_result,
        max_tokens,
        temperature,
        mut on_delta,
    } = params;

    let prompt_tokens = model
        .str_to_token(&template_result.prompt, AddBos::Always)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    if prompt_tokens.is_empty() {
        return Err(LlamaCppProviderError::Inference(
            "Prompt produced no tokens".to_string(),
        ));
    }

    let required_tokens = prompt_tokens.len() as u32 + max_tokens;
    let n_ctx = resolve_context_size(model, config, required_tokens)?;
    let n_batch = resolve_n_batch(config, n_ctx);
    let ctx_params = build_context_params(config, false, Some(n_ctx), Some(n_batch))?;
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|err| LlamaCppProviderError::ContextLoad(err.to_string()))?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let batch_limit = n_batch as usize;
    let mut position = 0_i32;
    for chunk in prompt_tokens.chunks(batch_limit.max(1)) {
        batch.clear();
        let last_index = (chunk.len().saturating_sub(1)) as i32;
        for (idx, token) in (0_i32..).zip(chunk.iter().copied()) {
            let is_last = idx == last_index;
            batch
                .add(token, position + idx, &[0], is_last)
                .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        }

        ctx.decode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        position += chunk.len() as i32;
    }

    let mut n_cur = prompt_tokens.len() as i32;
    let max_tokens_total = n_cur + max_tokens as i32;
    let mut generated_text = String::default();
    let mut completion_tokens = 0u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let (mut sampler, preserved) = build_chat_sampler(model, config, template_result, temperature)?;
    let additional_stops = template_result.additional_stops.clone();
    let mut stream_state = if on_delta.is_some() {
        Some(template_result.streaming_state_oaicompat().map_err(|err| {
            LlamaCppProviderError::Template(format!("Failed to init streaming state: {err}"))
        })?)
    } else {
        None
    };

    let mut finish_reason = "stop".to_string();
    while n_cur < max_tokens_total {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if model.is_eog_token(token) {
            break;
        }

        let decode_special = preserved.contains(&token);
        let output_string = model
            .token_to_piece(token, &mut decoder, decode_special, None)
            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
        generated_text.push_str(&output_string);
        completion_tokens += 1;

        let stop_now = additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop));

        if let (Some(state), Some(ref mut on_delta)) = (stream_state.as_mut(), on_delta.as_mut()) {
            let deltas = state.update(&output_string, !stop_now).map_err(|err| {
                LlamaCppProviderError::Template(format!("Streaming delta update failed: {err}"))
            })?;
            for delta in deltas {
                on_delta(&delta)?;
            }
        }

        if stop_now {
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
    }

    if n_cur >= max_tokens_total {
        finish_reason = "length".to_string();
    }

    let mut text = generated_text;
    for stop in &additional_stops {
        if !stop.is_empty() && text.ends_with(stop) {
            let new_len = text.len().saturating_sub(stop.len());
            text.truncate(new_len);
            break;
        }
    }

    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_tokens.len() as u32,
        completion_tokens,
        finish_reason,
    })
}

#[cfg(feature = "mtmd")]
fn generate_mtmd_text(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &LlamaCppConfig,
    params: MtmdGenerationParams<'_>,
) -> Result<GenerationResult, LlamaCppProviderError> {
    let MtmdGenerationParams {
        prompt,
        marker,
        images,
        max_tokens,
        temperature,
        mut on_token,
    } = params;

    let mmproj_path = config.mmproj_path.as_deref().ok_or_else(|| {
        LlamaCppProviderError::Config("mmproj_path is required for MTMD".to_string())
    })?;

    let mtmd_params = MtmdContextParams {
        use_gpu: config.mmproj_use_gpu.unwrap_or(true),
        print_timings: false,
        n_threads: config.n_threads.unwrap_or(4),
        media_marker: CString::new(marker)
            .map_err(|err| LlamaCppProviderError::Config(err.to_string()))?,
    };

    let mtmd_ctx = MtmdContext::init_from_file(mmproj_path, model, &mtmd_params)
        .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;

    let n_ctx = config
        .n_ctx
        .unwrap_or_else(|| model.n_ctx_train().min(2048));
    let n_batch = resolve_n_batch(config, n_ctx);
    let ctx_params = build_context_params(config, false, Some(n_ctx), Some(n_batch))?;
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|err| LlamaCppProviderError::ContextLoad(format!("{err} (n_ctx={n_ctx})")))?;

    let mut bitmaps = Vec::with_capacity(images.len());
    for image in images {
        let bitmap = MtmdBitmap::from_buffer(&mtmd_ctx, image)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        bitmaps.push(bitmap);
    }

    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    let input_text = MtmdInputText {
        text: prompt.to_string(),
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &bitmap_refs)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    let batch_size = n_batch as i32;
    let n_past = chunks
        .eval_chunks(&mtmd_ctx, &mut ctx, 0, 0, batch_size, true)
        .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;

    let mut sampler = build_sampler(model, config, false, temperature, None)?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut n_cur = n_past;
    let max_tokens_total = n_cur + max_tokens as i32;
    let mut generated_text = String::default();
    let mut completion_tokens = 0u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut finish_reason = "stop".to_string();

    while n_cur < max_tokens_total {
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }

        let output_string = model
            .token_to_piece(token, &mut decoder, false, None)
            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
        generated_text.push_str(&output_string);
        completion_tokens += 1;
        if let Some(ref mut on_token) = on_token
            && !output_string.is_empty()
        {
            on_token(&output_string)?;
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
    }

    if n_cur >= max_tokens_total {
        finish_reason = "length".to_string();
    }

    Ok(GenerationResult {
        text: generated_text,
        prompt_tokens: n_past as u32,
        completion_tokens,
        finish_reason,
    })
}

fn generate_text(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &LlamaCppConfig,
    params: GenerationParams<'_>,
) -> Result<GenerationResult, LlamaCppProviderError> {
    let GenerationParams {
        prompt,
        use_json_grammar,
        max_tokens,
        temperature,
        mut on_token,
    } = params;

    let prompt_tokens = model
        .str_to_token(&prompt.prompt, prompt.add_bos)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    if prompt_tokens.is_empty() {
        return Err(LlamaCppProviderError::Inference(
            "Prompt produced no tokens".to_string(),
        ));
    }

    let required_tokens = prompt_tokens.len() as u32 + max_tokens;
    let n_ctx = resolve_context_size(model, config, required_tokens)?;
    let n_batch = resolve_n_batch(config, n_ctx);
    let ctx_params = build_context_params(config, false, Some(n_ctx), Some(n_batch))?;
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|err| LlamaCppProviderError::ContextLoad(err.to_string()))?;

    let prompt_len = prompt_tokens.len();
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut position = 0;
    let mut last_logits_index = 0_i32;

    for chunk in prompt_tokens.chunks(n_batch as usize) {
        batch.clear();
        for (idx, token) in chunk.iter().enumerate() {
            let is_last = position + idx + 1 == prompt_len;
            if is_last {
                last_logits_index = idx as i32;
            }
            batch
                .add(*token, (position + idx) as i32, &[0], is_last)
                .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        position += chunk.len();
    }

    let mut sampler = build_sampler(model, config, use_json_grammar, temperature, None)?;
    let mut generated_text = String::new();
    let mut completion_tokens = 0_u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let mut next_token = sampler.sample(&ctx, last_logits_index);
    sampler.accept(next_token);

    while completion_tokens < max_tokens {
        if model.is_eog_token(next_token) {
            break;
        }

        completion_tokens += 1;
        let token_str = model
            .token_to_piece(next_token, &mut decoder, true, None)
            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
        generated_text.push_str(&token_str);

        if let Some(ref mut on_token) = on_token
            && !token_str.is_empty()
        {
            on_token(&token_str)?;
        }

        batch.clear();
        batch
            .add(next_token, position as i32, &[0], true)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        ctx.decode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        position += 1;

        if position >= n_ctx as usize {
            break;
        }

        next_token = sampler.sample(&ctx, 0);
        sampler.accept(next_token);
    }

    let finish_reason = if completion_tokens >= max_tokens || position >= n_ctx as usize {
        "length".to_string()
    } else {
        "stop".to_string()
    };

    Ok(GenerationResult {
        text: generated_text,
        prompt_tokens: prompt_len as u32,
        completion_tokens,
        finish_reason,
    })
}

fn generate_embedding(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &LlamaCppConfig,
    text: &str,
) -> Result<Vec<f32>, LlamaCppProviderError> {
    let n_ctx = config.n_ctx.unwrap_or_else(|| model.n_ctx_train());
    let n_batch = resolve_n_batch(config, n_ctx);
    let params = build_context_params(config, true, None, Some(n_batch))?;
    let mut ctx = model
        .new_context(backend, params)
        .map_err(|err| LlamaCppProviderError::ContextLoad(err.to_string()))?;

    let tokens = model
        .str_to_token(text, AddBos::Always)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    if tokens.is_empty() {
        return Err(LlamaCppProviderError::Embedding(
            "Input produced no tokens".to_string(),
        ));
    }

    let batch_size = n_batch as usize;
    let mut batch = LlamaBatch::new(batch_size, 1);
    let mut position = 0;

    for chunk in tokens.chunks(batch_size) {
        batch.clear();
        for (idx, token) in chunk.iter().enumerate() {
            let is_last = position + idx + 1 == tokens.len();
            batch
                .add(*token, (position + idx) as i32, &[0], is_last)
                .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        }
        ctx.encode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        position += chunk.len();
    }

    let embedding = ctx
        .embeddings_seq_ith(0)
        .map_err(|err| LlamaCppProviderError::Embedding(err.to_string()))?;
    Ok(embedding.to_vec())
}

fn extract_json_payload(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    if is_valid_json(trimmed) {
        return Some(trimmed.to_string());
    }

    if let Some(candidate) = extract_from_code_fence(trimmed) {
        return Some(candidate);
    }

    extract_first_json_object(trimmed)
}

fn is_valid_json(candidate: &str) -> bool {
    serde_json::from_str::<Value>(candidate).is_ok()
}

fn extract_from_code_fence(text: &str) -> Option<String> {
    let mut in_fence = false;
    let mut json_fence = false;
    let mut buffer = String::default();

    for line in text.lines() {
        let line_trimmed = line.trim_start();
        if let Some(rest) = line_trimmed.strip_prefix("```") {
            if !in_fence {
                let lang = rest.trim().to_ascii_lowercase();
                json_fence = lang.is_empty() || lang == "json";
                in_fence = true;
                buffer.clear();
            } else {
                if json_fence {
                    let candidate = buffer.trim();
                    if !candidate.is_empty() && is_valid_json(candidate) {
                        return Some(candidate.to_string());
                    }
                }
                in_fence = false;
                json_fence = false;
                buffer.clear();
            }
            continue;
        }

        if in_fence && json_fence {
            buffer.push_str(line);
            buffer.push('\n');
        }
    }

    None
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let mut in_string = false;
    let mut escape = false;
    let mut depth = 0i32;
    let mut start = None;

    for (idx, ch) in text.char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start_idx) = start {
                            let candidate = text[start_idx..=idx].trim();
                            if !candidate.is_empty() && is_valid_json(candidate) {
                                return Some(candidate.to_string());
                            }
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_count(total: usize, batch: usize) -> usize {
        if batch == 0 {
            return 0;
        }
        total.div_ceil(batch)
    }

    #[test]
    fn test_builder_creation() {
        let builder = LlamaCppProvider::builder();
        drop(builder);
    }

    #[test]
    fn test_builder_configuration() {
        let builder = LlamaCppProvider::builder()
            .max_tokens(128)
            .temperature(0.5)
            .repeat_penalty(1.1);

        drop(builder);
    }

    #[test]
    fn test_provider_builder_default() {
        let builder1 = LlamaCppProviderBuilder::default();
        let builder2 = LlamaCppProviderBuilder::new();

        drop(builder1);
        drop(builder2);
    }

    #[test]
    fn test_default_n_batch_smaller_than_context() {
        let config = LlamaCppConfig::default();
        let n_ctx = 4096;
        let n_batch = resolve_n_batch(&config, n_ctx);
        assert_eq!(n_batch, DEFAULT_N_BATCH);
        assert!(n_batch < n_ctx);
    }

    #[test]
    fn test_large_prompt_batches_by_default_n_batch() {
        let config = LlamaCppConfig::default();
        let n_ctx = 4096;
        let n_batch = resolve_n_batch(&config, n_ctx);
        let prompt_len = n_batch as usize + 1;
        assert_eq!(chunk_count(prompt_len, n_batch as usize), 2);
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn test_mtmd_default_marker_smoke() {
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        assert!(!marker.is_empty());
    }
}
