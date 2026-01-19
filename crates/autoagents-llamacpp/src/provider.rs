//! LlamaCppProvider implementation with LLMProvider traits.

use crate::{
    config::{LlamaCppConfig, LlamaCppConfigBuilder},
    conversion::{build_prompt, LlamaCppResponse, PromptData},
    error::LlamaCppProviderError,
    models::ModelSource,
};
use autoagents_llm::{
    async_trait,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, StreamChoice, StreamChunk, StreamDelta,
        StreamResponse, StructuredOutputFormat, Tool, Usage as ChatUsage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    LLMProvider,
};
use futures::{stream::Stream, StreamExt};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::params::LlamaModelParams,
    model::{AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde_json::Value;
use std::{
    num::NonZeroU32,
    path::Path,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
};
use tokio::sync::mpsc;

const JSON_GRAMMAR: &str = include_str!("grammars/json.gbnf");

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
}

enum StreamEvent {
    Token(String),
    Usage(ChatUsage),
    Done,
}

type TokenCallback = Box<dyn FnMut(&str) -> Result<(), LlamaCppProviderError> + Send>;

struct GenerationParams<'a> {
    prompt: &'a PromptData,
    json_schema: Option<&'a StructuredOutputFormat>,
    max_tokens: u32,
    temperature: Option<f32>,
    on_token: Option<TokenCallback>,
}

impl LlamaCppProvider {
    /// Create provider from GGUF model path.
    pub async fn from_gguf(model_path: impl Into<String>) -> Result<Self, LLMError> {
        let config = LlamaCppConfigBuilder::new().model_path(model_path).build();
        Self::from_config(config).await
    }

    /// Create provider from configuration.
    pub async fn from_config(config: LlamaCppConfig) -> Result<Self, LLMError> {
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

    fn prepare_messages(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<&StructuredOutputFormat>,
    ) -> Vec<ChatMessage> {
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

        if let Some(schema) = json_schema {
            let mut schema_hint =
                format!("Return a valid JSON response for schema '{}'.", schema.name);
            if let Some(description) = &schema.description {
                schema_hint.push_str(&format!(" {}", description));
            }
            if let Some(json_schema) = &schema.schema {
                schema_hint.push_str(&format!(" Schema: {}", json_schema));
            }
            all_messages.push(ChatMessage {
                role: autoagents_llm::chat::ChatRole::System,
                message_type: autoagents_llm::chat::MessageType::Text,
                content: schema_hint,
            });
        }

        all_messages.extend_from_slice(messages);
        all_messages
    }

    fn prompt_for_messages(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<&StructuredOutputFormat>,
    ) -> Result<PromptData, LLMError> {
        let messages = self.prepare_messages(messages, json_schema);
        build_prompt(&self.model, &messages, self.config.chat_template.as_deref())
            .map_err(LLMError::from)
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

    async fn generate_response(
        &self,
        prompt: PromptData,
        json_schema: Option<StructuredOutputFormat>,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Result<GenerationResult, LLMError> {
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let has_json_schema = json_schema.is_some();

        let mut result = tokio::task::spawn_blocking(
            move || -> Result<GenerationResult, LlamaCppProviderError> {
                generate_text(
                    &model,
                    &backend,
                    &config,
                    GenerationParams {
                        prompt: &prompt,
                        json_schema: json_schema.as_ref(),
                        max_tokens,
                        temperature,
                        on_token: None,
                    },
                )
            },
        )
        .await
        .map_err(|err| LLMError::ProviderError(format!("Generation task failed: {}", err)))?
        .map_err(LLMError::from)?;

        if has_json_schema || self.config.force_json_grammar {
            if let Some(extracted) = extract_json_payload(&result.text) {
                result.text = extracted;
            }
        }

        Ok(result)
    }

    fn spawn_token_stream(
        &self,
        prompt: PromptData,
        json_schema: Option<StructuredOutputFormat>,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let should_extract = json_schema.is_some() || config.force_json_grammar;
        let emitted_any = Arc::new(AtomicBool::new(false));
        let tx_tokens = tx.clone();

        tokio::spawn(async move {
            let emitted_any = Arc::clone(&emitted_any);
            let emitted_any_for_blocking = Arc::clone(&emitted_any);
            let result = tokio::task::spawn_blocking(
                move || -> Result<GenerationResult, LlamaCppProviderError> {
                    let mut json_started = false;
                    let emitted_any = emitted_any_for_blocking;
                    let on_token: Option<TokenCallback> = Some(Box::new(move |token: &str| {
                        if should_extract && !json_started {
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
                                    emitted_any.store(true, Ordering::Relaxed);
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
                        emitted_any.store(true, Ordering::Relaxed);
                        Ok(())
                    }) as TokenCallback);
                    generate_text(
                        &model,
                        &backend,
                        &config,
                        GenerationParams {
                            prompt: &prompt,
                            json_schema: json_schema.as_ref(),
                            max_tokens,
                            temperature,
                            on_token,
                        },
                    )
                },
            )
            .await;

            match result {
                Ok(Ok(gen)) => {
                    if should_extract && !emitted_any.load(Ordering::Relaxed) {
                        let mut text = gen.text;
                        if let Some(extracted) = extract_json_payload(&text) {
                            text = extracted;
                        }
                        let _ = tx.send(Ok(StreamEvent::Token(text)));
                    }
                    let usage =
                        LlamaCppProvider::build_usage(gen.prompt_tokens, gen.completion_tokens);
                    let _ = tx.send(Ok(StreamEvent::Usage(usage)));
                    let _ = tx.send(Ok(StreamEvent::Done));
                }
                Ok(Err(err)) => {
                    let _ = tx.send(Err(LLMError::from(err)));
                }
                Err(err) => {
                    let _ = tx.send(Err(LLMError::ProviderError(format!(
                        "Generation task failed: {}",
                        err
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

/// Builder for LlamaCppProvider.
pub struct LlamaCppProviderBuilder {
    config_builder: LlamaCppConfigBuilder,
}

impl LlamaCppProviderBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config_builder: LlamaCppConfigBuilder::new(),
        }
    }

    /// Set the model source.
    pub fn model_source(mut self, source: ModelSource) -> Self {
        self.config_builder = self.config_builder.model_source(source);
        self
    }

    /// Set model path for GGUF.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.model_path(path);
        self
    }

    /// Set chat template.
    pub fn chat_template(mut self, template: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.chat_template(template);
        self
    }

    /// Set system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.system_prompt(prompt);
        self
    }

    /// Force JSON grammar enforcement even without a structured output schema.
    pub fn force_json_grammar(mut self, force: bool) -> Self {
        self.config_builder = self.config_builder.force_json_grammar(force);
        self
    }

    /// Set the model cache directory.
    pub fn model_dir(mut self, dir: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.model_dir(dir);
        self
    }

    /// Set the HuggingFace filename (GGUF file).
    pub fn hf_filename(mut self, filename: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.hf_filename(filename);
        self
    }

    /// Set the HuggingFace revision.
    pub fn hf_revision(mut self, revision: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.hf_revision(revision);
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config_builder = self.config_builder.max_tokens(tokens);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config_builder = self.config_builder.temperature(temp);
        self
    }

    /// Set top-p.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config_builder = self.config_builder.top_p(p);
        self
    }

    /// Set top-k.
    pub fn top_k(mut self, k: u32) -> Self {
        self.config_builder = self.config_builder.top_k(k);
        self
    }

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.config_builder = self.config_builder.frequency_penalty(penalty);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.config_builder = self.config_builder.presence_penalty(penalty);
        self
    }

    /// Set repeat last N for penalties.
    pub fn repeat_last_n(mut self, last_n: i32) -> Self {
        self.config_builder = self.config_builder.repeat_last_n(last_n);
        self
    }

    /// Set sampling seed.
    pub fn seed(mut self, seed: u32) -> Self {
        self.config_builder = self.config_builder.seed(seed);
        self
    }

    /// Set context size.
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.config_builder = self.config_builder.n_ctx(n_ctx);
        self
    }

    /// Set batch size.
    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.config_builder = self.config_builder.n_batch(n_batch);
        self
    }

    /// Set micro-batch size.
    pub fn n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.config_builder = self.config_builder.n_ubatch(n_ubatch);
        self
    }

    /// Set number of threads for prompt evaluation.
    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.config_builder = self.config_builder.n_threads(n_threads);
        self
    }

    /// Set number of threads for batch evaluation.
    pub fn n_threads_batch(mut self, n_threads: i32) -> Self {
        self.config_builder = self.config_builder.n_threads_batch(n_threads);
        self
    }

    /// Set number of GPU layers to offload.
    pub fn n_gpu_layers(mut self, layers: u32) -> Self {
        self.config_builder = self.config_builder.n_gpu_layers(layers);
        self
    }

    /// Set main GPU index.
    pub fn main_gpu(mut self, main_gpu: i32) -> Self {
        self.config_builder = self.config_builder.main_gpu(main_gpu);
        self
    }

    /// Set split mode.
    pub fn split_mode(mut self, mode: crate::config::LlamaCppSplitMode) -> Self {
        self.config_builder = self.config_builder.split_mode(mode);
        self
    }

    /// Enable memory lock.
    pub fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.config_builder = self.config_builder.use_mlock(use_mlock);
        self
    }

    /// Set explicit device indices for offload.
    pub fn devices(mut self, devices: Vec<usize>) -> Self {
        self.config_builder = self.config_builder.devices(devices);
        self
    }

    /// Set repeat penalty.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.config_builder = self.config_builder.repeat_penalty(penalty);
        self
    }

    /// Build the provider.
    pub async fn build(self) -> Result<LlamaCppProvider, LLMError> {
        let config = self.config_builder.build();
        LlamaCppProvider::from_config(config).await
    }
}

impl Default for LlamaCppProviderBuilder {
    fn default() -> Self {
        Self::new()
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
        if tools.is_some() {
            return Err(LLMError::NoToolSupport(
                "Tool calls are not supported by llama.cpp backend".to_string(),
            ));
        }

        let prompt = self.prompt_for_messages(messages, json_schema.as_ref())?;
        let result = self
            .generate_response(prompt, json_schema, None, None)
            .await?;

        let usage = Some(Self::build_usage(
            result.prompt_tokens,
            result.completion_tokens,
        ));

        Ok(Box::new(LlamaCppResponse {
            text: result.text,
            usage,
        }))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let prompt = self.prompt_for_messages(messages, json_schema.as_ref())?;
        let response_stream = self.spawn_token_stream(prompt, json_schema, None, None);

        let content_stream = response_stream.filter_map(|event| async move {
            match event {
                Ok(StreamEvent::Token(token)) => Some(Ok(token)),
                Ok(StreamEvent::Usage(_)) | Ok(StreamEvent::Done) => None,
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
        if tools.is_some() {
            return Err(LLMError::NoToolSupport(
                "Tool calls are not supported by llama.cpp backend".to_string(),
            ));
        }

        let prompt = self.prompt_for_messages(messages, json_schema.as_ref())?;
        let response_stream = self.spawn_token_stream(prompt, json_schema, None, None);

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
                Ok(StreamEvent::Usage(usage)) => Some(Ok(StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: None,
                            tool_calls: None,
                        },
                    }],
                    usage: Some(usage),
                })),
                Ok(StreamEvent::Done) => None,
                Err(err) => Some(Err(err)),
            }
        });

        Ok(Box::pin(struct_stream))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        if tools.is_some() {
            return Err(LLMError::NoToolSupport(
                "Tool calls are not supported by llama.cpp backend".to_string(),
            ));
        }

        let prompt = self.prompt_for_messages(messages, json_schema.as_ref())?;
        let response_stream = self.spawn_token_stream(prompt, json_schema, None, None);

        let stream = response_stream.filter_map(|event| async move {
            match event {
                Ok(StreamEvent::Token(token)) => Some(Ok(StreamChunk::Text(token))),
                Ok(StreamEvent::Usage(usage)) => Some(Ok(StreamChunk::Usage(usage))),
                Ok(StreamEvent::Done) => Some(Ok(StreamChunk::Done {
                    stop_reason: "end_turn".to_string(),
                })),
                Err(err) => Some(Err(err)),
            }
        });

        Ok(Box::pin(stream))
    }
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
        let result = self
            .generate_response(prompt, json_schema, req.max_tokens, req.temperature)
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
        .map_err(|err| LLMError::ProviderError(format!("Embedding task failed: {}", err)))?
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
        LlamaCppProviderError::Other(format!("Failed to initialize llama backend: {}", err))
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
    .map_err(|err| LLMError::ProviderError(format!("Model load task failed: {}", err)))?
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
        ModelSource::HuggingFace { repo_id, filename } => {
            crate::huggingface::resolve_hf_model(repo_id, filename.as_deref(), config)
        }
    }
}

fn build_context_params(
    config: &LlamaCppConfig,
    embeddings: bool,
) -> Result<LlamaContextParams, LlamaCppProviderError> {
    let mut params = LlamaContextParams::default();

    if let Some(n_ctx) = config.n_ctx {
        params = params.with_n_ctx(NonZeroU32::new(n_ctx));
    }
    if let Some(n_batch) = config.n_batch {
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

fn build_sampler(
    model: &LlamaModel,
    config: &LlamaCppConfig,
    use_json_grammar: bool,
    temperature_override: Option<f32>,
) -> Result<LlamaSampler, LlamaCppProviderError> {
    let mut samplers = Vec::new();

    if use_json_grammar {
        let trigger_tokens = json_trigger_tokens(model)?;
        let grammar = LlamaSampler::grammar_lazy(
            model,
            JSON_GRAMMAR,
            "root",
            std::iter::empty::<&'static [u8]>(),
            &trigger_tokens,
        )
        .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?;
        samplers.push(grammar);
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

    Ok(LlamaSampler::chain_simple(samplers))
}

fn json_trigger_tokens(model: &LlamaModel) -> Result<Vec<LlamaToken>, LlamaCppProviderError> {
    let mut tokens = Vec::new();

    for (token, piece_result) in model.tokens(Special::Tokenize) {
        let piece = match piece_result {
            Ok(piece) => piece,
            Err(_) => continue,
        };
        let trimmed = piece.trim_start_matches(|c: char| c.is_whitespace());
        if (trimmed == "{" || trimmed == "[")
            && piece[..piece.len() - trimmed.len()]
                .chars()
                .all(|c| c.is_whitespace())
        {
            tokens.push(token);
        }
    }

    if tokens.is_empty() {
        for literal in ["{", "["] {
            let mut literal_tokens = model
                .str_to_token(literal, AddBos::Never)
                .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
            tokens.append(&mut literal_tokens);
        }
    }

    tokens.sort_by_key(|token| token.0);
    tokens.dedup_by_key(|token| token.0);

    if tokens.is_empty() {
        return Err(LlamaCppProviderError::Config(
            "Unable to derive trigger tokens for JSON grammar".to_string(),
        ));
    }

    Ok(tokens)
}

fn generate_text(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &LlamaCppConfig,
    params: GenerationParams<'_>,
) -> Result<GenerationResult, LlamaCppProviderError> {
    let GenerationParams {
        prompt,
        json_schema,
        max_tokens,
        temperature,
        mut on_token,
    } = params;
    let use_json_grammar = json_schema.is_some() || config.force_json_grammar;
    let params = build_context_params(config, false)?;
    let mut ctx = model
        .new_context(backend, params)
        .map_err(|err| LlamaCppProviderError::ContextLoad(err.to_string()))?;

    let prompt_tokens = model
        .str_to_token(&prompt.prompt, prompt.add_bos)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    if prompt_tokens.is_empty() {
        return Err(LlamaCppProviderError::Inference(
            "Prompt produced no tokens".to_string(),
        ));
    }

    let prompt_len = prompt_tokens.len();
    let ctx_limit = ctx.n_ctx() as usize;
    if prompt_len >= ctx_limit {
        return Err(LlamaCppProviderError::Inference(format!(
            "Prompt length ({}) exceeds context size ({})",
            prompt_len, ctx_limit
        )));
    }

    let available = ctx_limit.saturating_sub(prompt_len);
    if available == 0 {
        return Err(LlamaCppProviderError::Inference(
            "No context available for generation".to_string(),
        ));
    }

    let max_tokens = std::cmp::min(max_tokens as usize, available);
    let batch_size = config
        .n_batch
        .map(|n| n as usize)
        .unwrap_or(ctx.n_batch() as usize)
        .max(1);

    let mut batch = LlamaBatch::new(batch_size, 1);
    let mut position = 0;
    let mut last_logits_index = 0_i32;

    for chunk in prompt_tokens.chunks(batch_size) {
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

    let mut sampler = build_sampler(model, config, use_json_grammar, temperature)?;
    if !use_json_grammar {
        sampler.accept_many(&prompt_tokens);
    }

    let mut generated_tokens = Vec::with_capacity(max_tokens);
    let mut completion_tokens = 0_u32;

    let mut next_token = sampler.sample(&ctx, last_logits_index);
    sampler.accept(next_token);

    while completion_tokens < max_tokens as u32 {
        if model.is_eog_token(next_token) {
            break;
        }

        generated_tokens.push(next_token);
        completion_tokens += 1;

        let token_str = model
            .token_to_str(next_token, llama_cpp_2::model::Special::Tokenize)
            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

        if let Some(ref mut on_token) = on_token {
            if !token_str.is_empty() {
                on_token(&token_str)?;
            }
        }

        batch.clear();
        batch
            .add(next_token, position as i32, &[0], true)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        ctx.decode(&mut batch)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        position += 1;

        if position >= ctx_limit {
            break;
        }

        next_token = sampler.sample(&ctx, 0);
        sampler.accept(next_token);
    }

    let text = model
        .tokens_to_str(&generated_tokens, llama_cpp_2::model::Special::Tokenize)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    Ok(GenerationResult {
        text,
        prompt_tokens: prompt_len as u32,
        completion_tokens,
    })
}

fn generate_embedding(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &LlamaCppConfig,
    text: &str,
) -> Result<Vec<f32>, LlamaCppProviderError> {
    let params = build_context_params(config, true)?;
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

    let batch_size = config
        .n_batch
        .map(|n| n as usize)
        .unwrap_or(ctx.n_batch() as usize)
        .max(1);
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
    let mut buffer = String::new();

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
}
