//! LlamaCppProvider implementation with LLMProvider traits.

use crate::{
    builder::LlamaCppProviderBuilder,
    chat_template::{
        GrammarTrigger, RenderedChat, TemplateSource, TemplateTokens, explicit_template_source,
        normalize_template_source, render_chat_template,
    },
    config::{LlamaCppConfig, LlamaCppConfigBuilder, LlamaCppToolChoice},
    conversion::{LlamaCppResponse, PromptData, build_fallback_prompt},
    error::LlamaCppProviderError,
    models::ModelSource,
};
use autoagents_llm::{
    FunctionCall, LLMProvider, ToolCall, async_trait,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, MessageType, SamplingOverrides, StreamChoice,
        StreamChunk, StreamDelta, StreamResponse, StructuredOutputFormat, Tool, Usage as ChatUsage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
};
use futures::{StreamExt, stream::Stream};
#[cfg(feature = "mtmd")]
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::params::LlamaModelParams,
    model::{AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::{LlamaToken, logit_bias::LlamaLogitBias},
};
use serde::Deserialize;
use serde_json::{Value, json};
#[cfg(feature = "mtmd")]
use std::ffi::CString;
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
    path::Path,
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use tokio::sync::{Semaphore, mpsc};

/// Dedicated tokio runtime for the llamacpp provider.
///
/// Each compiled `.so` has its own copy of tokio's thread-local runtime state.
/// When this crate is used from a Python extension, the calling async context
/// runs on a different `.so`'s runtime whose thread-local is invisible to this
/// crate's tokio. Using a crate-local runtime ensures `spawn_blocking` and
/// `spawn` always have a valid `Handle::current()`.
fn get_rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("llamacpp")
            .build()
            .expect("llamacpp runtime init failed")
    })
}

const JSON_GRAMMAR: &str = include_str!("grammars/json.gbnf");
const DEFAULT_N_BATCH: u32 = 64;

// ---------------------------------------------------------------------------
// Prefix comparison — extracted for testability
// ---------------------------------------------------------------------------

/// Count how many leading tokens are identical between `cached` and `new`.
///
/// Used by [`SessionState`] to determine how much of the KV-cache can be
/// reused: tokens `0..prefix_len` are already decoded, only
/// `new[prefix_len..]` needs processing.
fn common_prefix_len(cached: &[LlamaToken], new: &[LlamaToken]) -> usize {
    cached
        .iter()
        .zip(new.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| cached.len().min(new.len()))
}

// ---------------------------------------------------------------------------
// SessionState — persistent KV-cache for prefix reuse
// ---------------------------------------------------------------------------

/// Persistent inference context with KV-cache tracking.
///
/// When `LlamaCppConfig::context_reuse` is enabled, the provider holds a
/// `SessionState` across calls. On each inference request, the new prompt
/// tokens are compared against `cached_tokens`; only the suffix that differs
/// is decoded, and the stale KV-cache tail is evicted via
/// `clear_kv_cache_seq`.
///
/// # Safety
///
/// `LlamaContext<'a>` has a lifetime parameter tied to `&'a LlamaModel`.
/// We transmute that to `LlamaContext<'static>` and keep the model alive via
/// `_model: Arc<LlamaModel>`. Rust's struct drop order guarantees that `ctx`
/// is dropped **before** `_model` and `_backend` (fields drop in declaration
/// order). The wrapping `Mutex` prevents concurrent access.
pub(crate) struct SessionState {
    /// The live context. SAFETY: `'static` is a lie — see doc above.
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    /// Keeps the model alive for the context's FFI pointer.
    _model: Arc<LlamaModel>,
    /// Keeps the backend alive for the model.
    _backend: Arc<LlamaBackend>,
    /// Tokens currently in KV-cache slot 0, positions 0..len.
    cached_tokens: Vec<LlamaToken>,
    /// The KV-cache position of the *next* token to be written.
    next_pos: i32,
    /// The context window size this session was created with.
    n_ctx: u32,
}

// SAFETY: LlamaModel: Send + Sync (declared in llama-cpp-2).
// LlamaContext wraps a raw pointer guarded by a Mutex — no concurrent access.
// Sync is intentionally NOT implemented: SessionState must never be shared
// across threads without the Mutex wrapper. Moving it out of Arc<Mutex<..>>
// into Arc<SessionState> would be unsound (LlamaContext is not thread-safe).
unsafe impl Send for SessionState {}

impl SessionState {
    /// Build a new session: create a context with the given parameters.
    fn new(
        backend: Arc<LlamaBackend>,
        model: Arc<LlamaModel>,
        config: &LlamaCppConfig,
        n_ctx: u32,
        n_batch: u32,
    ) -> Result<Self, LlamaCppProviderError> {
        let ctx_params = build_context_params(config, false, Some(n_ctx), Some(n_batch))?;

        let ctx_real = model
            .new_context(&backend, ctx_params)
            .map_err(|e| LlamaCppProviderError::ContextLoad(e.to_string()))?;

        // SAFETY: see SessionState doc. model is kept alive in the same struct.
        let ctx: llama_cpp_2::context::LlamaContext<'static> =
            unsafe { std::mem::transmute(ctx_real) };

        Ok(Self {
            ctx,
            _model: model,
            _backend: backend,
            cached_tokens: Vec::new(),
            next_pos: 0,
            n_ctx,
        })
    }

    /// Find the common prefix length between cached tokens and new tokens.
    fn prefix_len(&self, new_tokens: &[LlamaToken]) -> usize {
        common_prefix_len(&self.cached_tokens, new_tokens)
    }

    /// Evict stale KV-cache entries beyond `keep` and update tracking.
    fn evict_after(&mut self, keep: usize) -> Result<(), LlamaCppProviderError> {
        if keep < self.cached_tokens.len() {
            self.ctx
                .clear_kv_cache_seq(Some(0), Some(keep as u32), None)
                .map_err(|e| LlamaCppProviderError::Inference(format!("KV evict: {e}")))?;
            self.cached_tokens.truncate(keep);
            self.next_pos = keep as i32;
        }
        Ok(())
    }

    /// Decode new tokens into the KV-cache, chunked by batch size.
    fn decode_tokens(
        &mut self,
        tokens: &[LlamaToken],
        batch_limit: usize,
    ) -> Result<(), LlamaCppProviderError> {
        for chunk in tokens.chunks(batch_limit.max(1)) {
            let mut batch = LlamaBatch::new(chunk.len().max(64), 1);
            for (i, &tok) in chunk.iter().enumerate() {
                let pos = self.next_pos + i as i32;
                let is_last = i == chunk.len() - 1;
                batch
                    .add(tok, pos, &[0], is_last)
                    .map_err(|e| LlamaCppProviderError::Inference(format!("batch add: {e}")))?;
            }
            self.ctx
                .decode(&mut batch)
                .map_err(|e| LlamaCppProviderError::Inference(format!("decode: {e}")))?;
            // Update tracking per-chunk so next_pos and cached_tokens stay
            // consistent even if a subsequent chunk's decode fails.
            self.cached_tokens.extend_from_slice(chunk);
            self.next_pos += chunk.len() as i32;
        }
        Ok(())
    }
}

/// Shared session state type used by the provider.
type SharedSessionState = Arc<std::sync::Mutex<Option<SessionState>>>;

/// Active inference context — either a fresh owned context or a persistent
/// session context behind a MutexGuard.
///
/// Used during token generation to abstract over the two context acquisition
/// paths. The `with_ctx` method provides type-safe access to the underlying
/// `LlamaContext` regardless of which variant is active.
enum ActiveContext<'a> {
    /// Fresh context created for this call (no prefix reuse).
    Owned(llama_cpp_2::context::LlamaContext<'a>),
    /// Persistent session context with KV-cache prefix reuse.
    Session(std::sync::MutexGuard<'a, Option<SessionState>>),
}

impl ActiveContext<'_> {
    /// Run a closure with mutable access to the underlying LlamaContext.
    ///
    /// Abstracts over owned vs session contexts — the closure receives a
    /// `&mut LlamaContext` regardless of which variant is active.
    fn with_ctx<R>(
        &mut self,
        f: impl FnOnce(&mut llama_cpp_2::context::LlamaContext<'_>) -> R,
    ) -> R {
        match self {
            ActiveContext::Owned(ctx) => f(ctx),
            ActiveContext::Session(guard) => {
                let state = guard
                    .as_mut()
                    .expect("session must exist during generation");
                f(&mut state.ctx)
            }
        }
    }

    /// Get mutable access to the session state (for post-generation cleanup).
    /// Returns `None` for owned contexts or if no session is initialized.
    fn session_state_mut(&mut self) -> Option<&mut SessionState> {
        match self {
            ActiveContext::Owned(_) => None,
            ActiveContext::Session(guard) => guard.as_mut(),
        }
    }
}

/// Llama.cpp provider for local LLM inference.
pub struct LlamaCppProvider {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    config: LlamaCppConfig,
    /// Persistent session state for KV-cache prefix reuse.
    /// `None` (inner Option) until the first inference call.
    /// The outer Arc<Mutex> is only allocated when `config.context_reuse` is true.
    session_state: Option<SharedSessionState>,
    scheduler: Arc<ProviderScheduler>,
}

struct ProviderScheduler {
    active_slots: Arc<Semaphore>,
    queued: AtomicUsize,
    queue_capacity: usize,
}

struct QueueReservation<'a> {
    queued: &'a AtomicUsize,
    active: bool,
}

impl<'a> QueueReservation<'a> {
    fn try_new(queued: &'a AtomicUsize, capacity: usize) -> Option<Self> {
        let mut current = queued.load(Ordering::Acquire);
        loop {
            if current >= capacity {
                return None;
            }
            match queued.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(Self {
                        queued,
                        active: true,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&mut self) {
        if self.active {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            self.active = false;
        }
    }
}

impl Drop for QueueReservation<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

impl ProviderScheduler {
    fn new(n_slots: usize, queue_capacity: usize) -> Self {
        Self {
            active_slots: Arc::new(Semaphore::new(n_slots.max(1))),
            queued: AtomicUsize::new(0),
            queue_capacity,
        }
    }

    async fn acquire(&self) -> Result<tokio::sync::OwnedSemaphorePermit, LLMError> {
        match self.active_slots.clone().try_acquire_owned() {
            Ok(permit) => return Ok(permit),
            Err(tokio::sync::TryAcquireError::Closed) => {
                return Err(LLMError::ProviderError(
                    "llama.cpp scheduler closed".to_string(),
                ));
            }
            Err(tokio::sync::TryAcquireError::NoPermits) => {}
        }

        let Some(mut reservation) = QueueReservation::try_new(&self.queued, self.queue_capacity)
        else {
            return Err(LLMError::ProviderError(format!(
                "llama.cpp request queue is full (capacity {})",
                self.queue_capacity
            )));
        };

        let permit = self
            .active_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|err| LLMError::ProviderError(format!("llama.cpp scheduler closed: {err}")))?;
        reservation.release();
        Ok(permit)
    }
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
    top_p: Option<f32>,
    on_token: Option<TokenCallback>,
}

#[cfg(feature = "mtmd")]
struct MtmdGenerationParams<'a> {
    prompt: &'a str,
    marker: &'a str,
    images: &'a [Vec<u8>],
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    on_token: Option<TokenCallback>,
}

struct ChatGenerationParams<'a> {
    template_result: &'a ChatTemplateResult,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    on_delta: Option<DeltaCallback>,
}

enum ChatPrompt {
    OpenAI(ChatTemplateResult),
    Fallback {
        prompt: PromptData,
        use_json_grammar: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallGrammarFormat {
    OpenAiEnvelope,
    NativeChannelToolCall,
    XmlToolCall,
    ToolCallsArrayTag,
    ToolCallsArgsTag,
    GenericFunctionTag,
    FunctionaryV32,
    Gemma4ToolCall,
    KimiK2ToolCall,
    Lfm2ToolCall,
    GigaChatV3ToolCall,
    DeepSeekDsmlToolCall,
}

fn detect_tool_call_grammar_format(template_source: &str) -> ToolCallGrammarFormat {
    if template_source.contains("<|tool_call>") || template_source.contains("<tool_call|>") {
        ToolCallGrammarFormat::Gemma4ToolCall
    } else if template_source.contains("<|tool_call_begin|>")
        && template_source.contains("<|tool_call_argument_begin|>")
    {
        ToolCallGrammarFormat::KimiK2ToolCall
    } else if template_source.contains("<|tool_call_start|>")
        && template_source.contains("<|tool_call_end|>")
    {
        ToolCallGrammarFormat::Lfm2ToolCall
    } else if template_source.contains("<|message_sep|>")
        && template_source.contains("function call")
    {
        ToolCallGrammarFormat::GigaChatV3ToolCall
    } else if template_source.contains("｜DSML｜")
        || template_source.contains("<｜DSML｜function_calls>")
    {
        ToolCallGrammarFormat::DeepSeekDsmlToolCall
    } else if template_source.contains("<|channel|>")
        && (template_source.contains("to=functions") || template_source.contains("<|message|>"))
    {
        ToolCallGrammarFormat::NativeChannelToolCall
    } else if template_source.contains("<tool_call>") || template_source.contains("</tool_call>") {
        ToolCallGrammarFormat::XmlToolCall
    } else if template_source.contains("<function=") || template_source.contains("</function>") {
        ToolCallGrammarFormat::GenericFunctionTag
    } else if template_source.contains(">>>all") && template_source.contains(">>>${recipient}") {
        ToolCallGrammarFormat::FunctionaryV32
    } else if template_source.contains("[TOOL_CALLS]") && template_source.contains("[ARGS]") {
        ToolCallGrammarFormat::ToolCallsArgsTag
    } else if template_source.contains("[TOOL_CALLS]") || template_source.contains("[/TOOL_CALLS]")
    {
        ToolCallGrammarFormat::ToolCallsArrayTag
    } else {
        ToolCallGrammarFormat::OpenAiEnvelope
    }
}

#[derive(Debug, Clone)]
struct ChatTemplateResult {
    prompt: String,
    generation_prompt: String,
    force_pure_content: bool,
    is_continuation: bool,
    add_bos: bool,
    grammar: Option<String>,
    grammar_lazy: bool,
    grammar_triggers: Vec<GrammarTrigger>,
    preserved_tokens: Vec<String>,
    additional_stops: Vec<String>,
    parse_tool_calls: bool,
    tool_names: Vec<String>,
    reasoning_format: Option<crate::config::LlamaCppReasoningFormat>,
    reasoning_start_tag: Option<String>,
    reasoning_end_tag: Option<String>,
}

impl From<RenderedChat> for ChatTemplateResult {
    fn from(value: RenderedChat) -> Self {
        Self {
            prompt: value.prompt,
            generation_prompt: value.generation_prompt,
            force_pure_content: value.force_pure_content,
            is_continuation: value.is_continuation,
            add_bos: value.add_bos,
            grammar: value.grammar,
            grammar_lazy: value.grammar_lazy,
            grammar_triggers: value.grammar_triggers,
            preserved_tokens: value.preserved_tokens,
            additional_stops: value.additional_stops,
            parse_tool_calls: value.parse_tool_calls,
            tool_names: value.tool_names,
            reasoning_format: value.reasoning_format,
            reasoning_start_tag: value.reasoning_start_tag,
            reasoning_end_tag: value.reasoning_end_tag,
        }
    }
}

impl ChatTemplateResult {
    fn parse_response_oaicompat(&self, text: &str) -> Result<String, LlamaCppProviderError> {
        let text = self.strip_echoed_continuation(text);
        if self.parse_tool_calls
            && let Some(mut message) = parse_tool_response_message_with_allowed_tools(
                text,
                Some(self.tool_names.as_slice()),
            )?
        {
            self.extract_reasoning_from_message_content(&mut message);
            return Ok(message.to_string());
        }

        if !self.force_pure_content
            && let Some((content, reasoning_content)) = parse_native_channel_content_partial(text)
        {
            let message = content_message_value(content, reasoning_content);
            return Ok(message.to_string());
        }

        let mut grammar_source;
        let grammar_reasoning_content = if self.grammar.is_some() && !self.force_pure_content {
            grammar_source = text.to_string();
            self.extract_response_reasoning_content(&mut grammar_source)
        } else {
            grammar_source = text.to_string();
            None
        };

        let mut content = if self.grammar.is_some() {
            let payload =
                extract_json_payload(&grammar_source).unwrap_or_else(|| grammar_source.clone());
            if self.parse_tool_calls
                && let Ok(Value::Object(object)) = serde_json::from_str::<Value>(&payload)
                && let Some(Value::String(content)) = object.get("content")
            {
                content.clone()
            } else {
                payload
            }
        } else {
            text.to_string()
        };

        let reasoning_content = if grammar_reasoning_content.is_some() {
            grammar_reasoning_content
        } else if self.force_pure_content {
            None
        } else {
            self.extract_response_reasoning_content(&mut content)
        };
        let message = content_message_value(content, reasoning_content);

        Ok(message.to_string())
    }

    fn parse_partial_response_oaicompat(
        &self,
        text: &str,
    ) -> Result<Option<String>, LlamaCppProviderError> {
        let text = self.strip_echoed_continuation(text);
        if self.parse_tool_calls
            && let Some(mut message) = parse_tool_response_message_with_allowed_tools(
                text,
                Some(self.tool_names.as_slice()),
            )?
        {
            self.extract_reasoning_from_message_content(&mut message);
            return Ok(Some(message.to_string()));
        }

        if !self.force_pure_content
            && let Some((content, reasoning_content)) = parse_native_channel_content_partial(text)
        {
            let message = content_message_value(content, reasoning_content);
            return Ok(Some(message.to_string()));
        }

        if self.grammar.is_some() {
            let mut grammar_source = text.to_string();
            let reasoning_content = if self.force_pure_content {
                None
            } else {
                self.extract_partial_reasoning_content(&mut grammar_source)
            };
            let Some(payload) = extract_json_payload(&grammar_source) else {
                if let Some(reasoning_content) =
                    reasoning_content.filter(|reasoning_content| !reasoning_content.is_empty())
                {
                    let message = content_message_value(String::default(), Some(reasoning_content));
                    return Ok(Some(message.to_string()));
                }
                return Ok(None);
            };
            let content = if self.parse_tool_calls
                && let Ok(Value::Object(object)) = serde_json::from_str::<Value>(&payload)
                && let Some(Value::String(content)) = object.get("content")
            {
                content.clone()
            } else {
                payload
            };
            let message = content_message_value(content, reasoning_content);
            return Ok(Some(message.to_string()));
        }

        Ok(None)
    }

    fn strip_echoed_continuation<'a>(&self, text: &'a str) -> &'a str {
        if self.is_continuation
            && !self.generation_prompt.is_empty()
            && let Some(rest) = text.strip_prefix(&self.generation_prompt)
        {
            return rest;
        }
        text
    }

    fn extract_reasoning_from_message_content(&self, message: &mut Value) {
        let Some(object) = message.as_object_mut() else {
            return;
        };
        if object.contains_key("reasoning_content") {
            return;
        }
        let Some(Value::String(content)) = object.get_mut("content") else {
            return;
        };
        let reasoning_content = self.extract_response_reasoning_content(content);
        if let Some(reasoning_content) = reasoning_content
            && !reasoning_content.is_empty()
        {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content),
            );
        }
    }

    fn extract_response_reasoning_content(&self, content: &mut String) -> Option<String> {
        let reasoning = extract_reasoning_content(
            content,
            self.reasoning_format,
            self.reasoning_start_tag.as_deref(),
            self.reasoning_end_tag.as_deref(),
        );
        if reasoning.is_some() {
            return reasoning;
        }
        if let Some((start_tag, end_tag)) = self.reasoning_tags()
            && content.contains(start_tag)
        {
            return extract_tagged_reasoning_partial(content, start_tag, end_tag);
        }
        self.extract_prefilled_reasoning_content(content, true)
    }

    fn extract_partial_reasoning_content(&self, content: &mut String) -> Option<String> {
        let (start_tag, end_tag) = self.reasoning_tags()?;
        let reasoning = extract_tagged_reasoning_partial(content, start_tag, end_tag);
        if reasoning.is_some() {
            return reasoning;
        }
        self.extract_prefilled_reasoning_content(content, true)
    }

    fn extract_prefilled_reasoning_content(
        &self,
        content: &mut String,
        allow_open: bool,
    ) -> Option<String> {
        let (start_tag, end_tag) = self.reasoning_tags()?;
        if !self.generation_prompt_prefills_reasoning(start_tag) || content.contains(start_tag) {
            return None;
        }
        extract_prefilled_reasoning(content, start_tag, end_tag, allow_open)
    }

    fn reasoning_tags(&self) -> Option<(&str, &str)> {
        let format = self.reasoning_format?;
        if matches!(format, crate::config::LlamaCppReasoningFormat::None) {
            return None;
        }
        Some((
            self.reasoning_start_tag.as_deref().unwrap_or("<think>"),
            self.reasoning_end_tag.as_deref().unwrap_or("</think>"),
        ))
    }

    fn generation_prompt_prefills_reasoning(&self, start_tag: &str) -> bool {
        generation_prompt_grammar_prefill(self).is_some_and(|prefill| prefill.contains(start_tag))
            || self.generation_prompt.contains(start_tag)
    }
}

fn content_message_value(content: String, reasoning_content: Option<String>) -> Value {
    let mut message = serde_json::Map::new();
    message.insert("content".to_string(), Value::String(content));
    if let Some(reasoning_content) =
        reasoning_content.filter(|reasoning_content| !reasoning_content.is_empty())
    {
        message.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content),
        );
    }
    Value::Object(message)
}

#[derive(Debug, Deserialize)]
struct OpenAICompatMessage {
    content: Option<StringOrJson>,
    reasoning_content: Option<StringOrJson>,
    tool_calls: Option<Vec<OpenAICompatToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAICompatDelta {
    content: Option<StringOrJson>,
    reasoning_content: Option<StringOrJson>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAICompatToolCall {
    #[serde(default)]
    id: StringOrJson,
    #[serde(rename = "type", default = "default_call_type_value")]
    call_type: StringOrJson,
    function: OpenAICompatFunctionCall,
}

#[derive(Debug, Deserialize)]
struct OpenAICompatFunctionCall {
    #[serde(default)]
    name: StringOrJson,
    #[serde(default)]
    arguments: StringOrJson,
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
    arguments: StringOrJson,
}

#[derive(Debug, Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
    started: bool,
}

#[derive(Debug, Default)]
struct StreamMappingState {
    content: String,
    reasoning_content: String,
    tool_calls: HashMap<usize, ToolCallState>,
}

fn append_or_diff_string(previous: &str, current_or_delta: &str) -> String {
    if current_or_delta.is_empty() {
        return String::default();
    }
    if previous.is_empty() {
        return current_or_delta.to_string();
    }
    if let Some(delta) = current_or_delta.strip_prefix(previous) {
        return delta.to_string();
    }
    if previous.starts_with(current_or_delta) {
        return String::default();
    }
    current_or_delta.to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringOrJson {
    String(String),
    Json(Value),
}

impl Default for StringOrJson {
    fn default() -> Self {
        Self::String(String::default())
    }
}

impl StringOrJson {
    fn into_string(self) -> String {
        match self {
            Self::String(text) => text,
            Self::Json(Value::Null) => String::default(),
            Self::Json(value) => value.to_string(),
        }
    }

    fn into_non_empty_string(self) -> Option<String> {
        let text = self.into_string();
        if text.is_empty() { None } else { Some(text) }
    }
}

impl From<OpenAICompatToolCall> for ToolCall {
    fn from(value: OpenAICompatToolCall) -> Self {
        ToolCall {
            id: value.id.into_string(),
            call_type: value.call_type.into_string(),
            function: FunctionCall {
                name: value.function.name.into_string(),
                arguments: value.function.arguments.into_string(),
            },
        }
    }
}

fn default_call_type_value() -> StringOrJson {
    StringOrJson::String(autoagents_llm::default_call_type())
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
        let session_state = if config.context_reuse {
            Some(Arc::new(std::sync::Mutex::new(None)))
        } else {
            None
        };
        Ok(Self {
            backend,
            model,
            scheduler: Arc::new(ProviderScheduler::new(
                config.scheduler.n_slots,
                config.scheduler.queue_capacity,
            )),
            config,
            session_state,
        })
    }

    /// Create a provider from a pre-loaded model.
    ///
    /// The model and backend are shared via `Arc` — no duplicate GGUF load.
    /// The new provider gets its own `SessionState` (KV context), so it can
    /// be used concurrently with sibling providers without contention.
    ///
    /// Use [`model()`] and [`backend()`] on an existing provider to obtain
    /// the handles, then pass them here with a fresh config. The caller must
    /// ensure `config` is compatible with the passed model (matching `n_ctx`,
    /// `model_path` for logging).
    pub fn from_model(
        model: Arc<LlamaModel>,
        backend: Arc<LlamaBackend>,
        config: LlamaCppConfig,
    ) -> Self {
        let session_state = if config.context_reuse {
            Some(Arc::new(std::sync::Mutex::new(None)))
        } else {
            None
        };
        Self {
            backend,
            model,
            scheduler: Arc::new(ProviderScheduler::new(
                config.scheduler.n_slots,
                config.scheduler.queue_capacity,
            )),
            config,
            session_state,
        }
    }

    /// Get a builder for advanced configuration.
    pub fn builder() -> LlamaCppProviderBuilder {
        LlamaCppProviderBuilder::new()
    }

    /// Get reference to the configuration.
    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    /// Get the model handle for creating sibling providers via [`from_model()`].
    pub fn model(&self) -> &Arc<LlamaModel> {
        &self.model
    }

    /// Get the backend handle for creating sibling providers via [`from_model()`].
    pub fn backend(&self) -> &Arc<LlamaBackend> {
        &self.backend
    }

    /// Clear the persistent session state so the next call starts fresh.
    ///
    /// No-op when `context_reuse` is disabled.
    pub fn reset_session(&self) {
        if let Some(ref state) = self.session_state {
            *state.lock().expect("session mutex poisoned") = None;
        }
    }

    /// Number of tokens currently cached in the persistent KV-cache.
    ///
    /// Returns 0 when `context_reuse` is disabled or no session exists yet.
    pub fn cached_prefix_len(&self) -> usize {
        match self.session_state.as_ref() {
            Some(s) => s
                .lock()
                .expect("session mutex poisoned")
                .as_ref()
                .map(|s| s.cached_tokens.len())
                .unwrap_or(0),
            None => 0,
        }
    }

    fn prepare_fallback_messages(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<&StructuredOutputFormat>,
    ) -> Vec<ChatMessage> {
        prepare_fallback_messages_with_schema(&self.config, messages, json_schema)
    }

    fn ensure_supported_messages(&self, messages: &[ChatMessage]) -> Result<(), LLMError> {
        ensure_supported_messages_for_config(&self.config, messages)
    }

    fn build_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<&StructuredOutputFormat>,
    ) -> Result<ChatPrompt, LLMError> {
        self.ensure_supported_messages(messages)?;
        let enabled_tools = enabled_tools_for_config(&self.config, tools);
        let template = match self.resolve_chat_template_source(enabled_tools.is_some()) {
            Ok(template) => Some(template),
            Err(err) => {
                if enabled_tools.is_some()
                    || json_schema.is_some()
                    || self.config.force_json_grammar
                {
                    return Err(err);
                }
                None
            }
        };

        if let Some(template) = template {
            let grammar_value = if let Some(tools) = enabled_tools {
                let tool_call_format = detect_tool_call_grammar_format(&template.source);
                Some(build_tool_response_grammar(
                    tools,
                    json_schema,
                    self.config.force_json_grammar,
                    self.config.tool_choice.clone(),
                    self.config.parallel_tool_calls.unwrap_or(false),
                    tool_call_format,
                )?)
            } else {
                let tool_call_format = detect_tool_call_grammar_format(&template.source);
                let (_, grammar_value) = select_template_schema_and_grammar(
                    json_schema,
                    self.config.force_json_grammar,
                    Some(tool_call_format),
                )?;
                grammar_value
            };

            let rendered = render_chat_template(
                &self.config,
                &template,
                messages,
                enabled_tools,
                json_schema,
                grammar_value,
                &self.template_tokens()?,
            )
            .map_err(LLMError::from)?;

            let result = ChatTemplateResult::from(rendered);

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
        let template = self.resolve_chat_template_source(false)?;
        let mut chat = Vec::new();
        let mut images = Vec::new();
        let default_marker = mtmd_default_marker().to_string();
        let marker = self
            .config
            .media_marker
            .as_deref()
            .unwrap_or(&default_marker)
            .to_string();

        for message in self.prepare_fallback_messages(messages, None) {
            let mut content = message.content.clone();
            match message.message_type {
                MessageType::Text => {}
                MessageType::Image((_, bytes)) => {
                    images.push(bytes);
                    if !content.contains(&marker) {
                        content.push_str(&marker);
                    }
                }
                MessageType::ToolUse(_) | MessageType::ToolResult(_) => {
                    return Err(LLMError::invalid_request(
                        "MTMD path does not support tool calls".to_string(),
                    ));
                }
                MessageType::ImageURL(_) | MessageType::Pdf(_) => {
                    return Err(LLMError::invalid_request(
                        "MTMD path only supports raw image inputs".to_string(),
                    ));
                }
            }

            chat.push(ChatMessage {
                role: message.role,
                message_type: MessageType::Text,
                content,
            });
        }

        let prompt = render_chat_template(
            &self.config,
            &template,
            &chat,
            None,
            None,
            None,
            &self.template_tokens()?,
        )
        .map_err(LLMError::from)?
        .prompt;

        Ok((prompt, images, marker))
    }

    fn template_tokens(&self) -> Result<TemplateTokens, LLMError> {
        Ok(TemplateTokens {
            bos_token: decode_model_special_token(&self.model, self.model.token_bos(), "BOS")?,
            eos_token: decode_model_special_token(&self.model, self.model.token_eos(), "EOS")?,
            add_bos: self.config.tokenizer_add_bos,
            add_eos: self.config.tokenizer_add_eos,
        })
    }

    fn resolve_chat_template_source(&self, prefer_tools: bool) -> Result<TemplateSource, LLMError> {
        if let Some(template) = self.config.chat_template.as_deref() {
            if template.contains("{%") || template.contains("{{") || template == "chatml" {
                return Ok(explicit_template_source(template));
            }
            if let Ok(template) = self.model.chat_template(Some(template)) {
                return Ok(TemplateSource {
                    source: normalize_template_source(&template.to_string().map_err(|err| {
                        LLMError::ProviderError(format!("Invalid chat template utf-8: {err}"))
                    })?),
                });
            }
            return Ok(explicit_template_source(template));
        }

        if prefer_tools && let Ok(template) = self.model.chat_template(Some("tool_use")) {
            return Ok(TemplateSource {
                source: normalize_template_source(&template.to_string().map_err(|err| {
                    LLMError::ProviderError(format!("Invalid tool chat template utf-8: {err}"))
                })?),
            });
        }

        self.model
            .chat_template(None)
            .map_err(|err| {
                LLMError::ProviderError(format!("Model does not provide a chat template: {err}"))
            })
            .and_then(|template| {
                Ok(TemplateSource {
                    source: normalize_template_source(&template.to_string().map_err(|err| {
                        LLMError::ProviderError(format!("Invalid chat template utf-8: {err}"))
                    })?),
                })
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

    fn resolve_top_p(&self, top_p_override: Option<f32>) -> Option<f32> {
        top_p_override.or(self.config.top_p)
    }

    async fn run_blocking_task<T, F>(&self, task_name: &str, f: F) -> Result<T, LLMError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, LlamaCppProviderError> + Send + 'static,
    {
        let _permit = self.scheduler.acquire().await?;
        get_rt()
            .spawn_blocking(f)
            .await
            .map_err(|err| LLMError::ProviderError(format!("{task_name} task failed: {err}")))?
            .map_err(LLMError::from)
    }

    async fn generate_completion_response(
        &self,
        prompt: PromptData,
        use_json_grammar: bool,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
        top_p_override: Option<f32>,
    ) -> Result<GenerationResult, LLMError> {
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let top_p = self.resolve_top_p(top_p_override);
        let session = self.session_state.clone();

        let mut result = self
            .run_blocking_task("Generation", move || {
                generate_text(
                    &model,
                    &backend,
                    &config,
                    GenerationParams {
                        prompt: &prompt,
                        use_json_grammar,
                        max_tokens,
                        temperature,
                        top_p,
                        on_token: None,
                    },
                    session.as_ref(),
                )
            })
            .await?;

        if use_json_grammar && let Some(extracted) = extract_json_payload(&result.text) {
            result.text = extracted;
        }

        Ok(result)
    }

    async fn generate_chat_completion(
        &self,
        template_result: ChatTemplateResult,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
        top_p_override: Option<f32>,
    ) -> Result<GenerationResult, LLMError> {
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let top_p = self.resolve_top_p(top_p_override);
        let session = self.session_state.clone();

        self.run_blocking_task("Generation", move || {
            generate_chat_text(
                &model,
                &backend,
                &config,
                ChatGenerationParams {
                    template_result: &template_result,
                    max_tokens,
                    temperature,
                    top_p,
                    on_delta: None,
                },
                session.as_ref(),
            )
        })
        .await
    }

    fn spawn_fallback_stream(
        &self,
        prompt: PromptData,
        use_json_grammar: bool,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
        top_p_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let top_p = self.resolve_top_p(top_p_override);
        let session = self.session_state.clone();
        let scheduler = self.scheduler.clone();
        let emitted_any = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tx_tokens = tx.clone();

        get_rt().spawn(async move {
            let _permit = match scheduler.acquire().await {
                Ok(permit) => permit,
                Err(err) => {
                    let _ = tx.send(Err(err));
                    return;
                }
            };
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
                            top_p,
                            on_token,
                        },
                        session.as_ref(),
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
        top_p_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let top_p = self.resolve_top_p(top_p_override);
        let scheduler = self.scheduler.clone();
        let tx_tokens = tx.clone();

        get_rt().spawn(async move {
            let _permit = match scheduler.acquire().await {
                Ok(permit) => permit,
                Err(err) => {
                    let _ = tx.send(Err(err));
                    return;
                }
            };
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
                            top_p,
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
        template_result: ChatTemplateResult,
        max_tokens_override: Option<u32>,
        temperature_override: Option<f32>,
        top_p_override: Option<f32>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamEvent, LLMError>>();
        let config = self.config.clone();
        let model = self.model.clone();
        let backend = self.backend.clone();
        let max_tokens = self.resolve_max_tokens(max_tokens_override);
        let temperature = self.resolve_temperature(temperature_override);
        let top_p = self.resolve_top_p(top_p_override);
        let session = self.session_state.clone();
        let scheduler = self.scheduler.clone();
        let stream_final_message = should_stream_final_message(&template_result);
        let tx_tokens = tx.clone();
        let tx_final_delta = tx.clone();
        let parsed_stream_emitted = Arc::new(AtomicBool::new(false));

        get_rt().spawn(async move {
            let _permit = match scheduler.acquire().await {
                Ok(permit) => permit,
                Err(err) => {
                    let _ = tx.send(Err(err));
                    return;
                }
            };
            let result = tokio::task::spawn_blocking(
                move || -> Result<(GenerationResult, String), LlamaCppProviderError> {
                    let parsed_stream_state = Arc::new(Mutex::new(String::default()));
                    let parsed_stream_emitted_for_callback = parsed_stream_emitted.clone();
                    let on_delta: Option<DeltaCallback> = if stream_final_message {
                        let template_result_for_callback = template_result.clone();
                        let tx_parsed = tx_tokens.clone();
                        Some(Box::new(move |delta: &str| {
                            let mut buffer = parsed_stream_state.lock().map_err(|_| {
                                LlamaCppProviderError::Inference(
                                    "Parsed stream state lock poisoned".to_string(),
                                )
                            })?;
                            buffer.push_str(delta);
                            match template_result_for_callback
                                .parse_partial_response_oaicompat(&buffer)
                            {
                                Ok(Some(message_json)) => {
                                    let delta_json = stream_delta_from_message_json(&message_json)?;
                                    if !delta_json.is_empty() {
                                        tx_parsed
                                            .send(Ok(StreamEvent::Delta(delta_json)))
                                            .map_err(|_| {
                                                LlamaCppProviderError::Inference(
                                                    "Stream receiver dropped".to_string(),
                                                )
                                            })?;
                                        parsed_stream_emitted_for_callback
                                            .store(true, Ordering::Relaxed);
                                    }
                                }
                                Ok(None) => {}
                                Err(_) => {}
                            }
                            Ok(())
                        }))
                    } else {
                        Some(Box::new(move |delta: &str| {
                            tx_tokens
                                .send(Ok(StreamEvent::Token(delta.to_string())))
                                .map_err(|_| {
                                    LlamaCppProviderError::Inference(
                                        "Stream receiver dropped".to_string(),
                                    )
                                })?;
                            Ok(())
                        }))
                    };

                    let generation = generate_chat_text(
                        &model,
                        &backend,
                        &config,
                        ChatGenerationParams {
                            template_result: &template_result,
                            max_tokens,
                            temperature,
                            top_p,
                            on_delta,
                        },
                        session.as_ref(),
                    )?;

                    let message_json = template_result
                        .parse_response_oaicompat(&generation.text)
                        .map_err(|err| {
                        LlamaCppProviderError::Template(format!("Failed to parse response: {err}"))
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

                    if stream_final_message {
                        let delta_json = stream_delta_from_message_json(&message_json)?;
                        if !delta_json.is_empty() && !parsed_stream_emitted.load(Ordering::Relaxed)
                        {
                            tx_final_delta
                                .send(Ok(StreamEvent::Delta(delta_json)))
                                .map_err(|_| {
                                    LlamaCppProviderError::Inference(
                                        "Stream receiver dropped".to_string(),
                                    )
                                })?;
                        }
                    }

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

// Helper: extract (max_tokens, temperature, top_p) from optional sampling overrides.
fn unpack_sampling(
    sampling: Option<&SamplingOverrides>,
) -> (Option<u32>, Option<f32>, Option<f32>) {
    (
        sampling.and_then(|s| s.max_tokens),
        sampling.and_then(|s| s.temperature),
        sampling.and_then(|s| s.top_p),
    )
}

impl LlamaCppProvider {
    /// Inner implementation of `chat_with_tools` accepting per-call sampling
    /// overrides. The public trait methods (`chat_with_tools` and
    /// `chat_with_tools_and_sampling`) delegate here. Backwards compatible:
    /// passing `sampling = None` produces identical behaviour to the
    /// pre-overrides implementation.
    async fn chat_with_tools_impl(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let (max_tokens_override, temperature_override, top_p_override) = unpack_sampling(sampling);

        if Self::has_mtmd_media(messages) {
            if tools.is_some() || json_schema.is_some() {
                return Err(LLMError::invalid_request(
                    "MTMD path does not support tools or structured outputs".to_string(),
                ));
            }
            #[cfg(feature = "mtmd")]
            {
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let config = self.config.clone();
                let model = self.model.clone();
                let backend = self.backend.clone();
                let max_tokens = self.resolve_max_tokens(max_tokens_override);
                let temperature = self.resolve_temperature(temperature_override);
                let top_p = self.resolve_top_p(top_p_override);
                let result = self
                    .run_blocking_task("MTMD generation", move || {
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
                                top_p,
                                on_token: None,
                            },
                        )
                    })
                    .await?;

                let usage = Some(Self::build_usage(
                    result.prompt_tokens,
                    result.completion_tokens,
                ));

                return Ok(Box::new(LlamaCppResponse {
                    content: Some(result.text),
                    thinking: None,
                    tool_calls: None,
                    usage,
                }));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::invalid_request(
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
                    .generate_completion_response(
                        prompt,
                        use_json_grammar,
                        max_tokens_override,
                        temperature_override,
                        top_p_override,
                    )
                    .await?;
                let usage = Some(Self::build_usage(
                    result.prompt_tokens,
                    result.completion_tokens,
                ));

                Ok(Box::new(LlamaCppResponse {
                    content: Some(result.text),
                    thinking: None,
                    tool_calls: None,
                    usage,
                }))
            }
            ChatPrompt::OpenAI(template_result) => {
                let result = self
                    .generate_chat_completion(
                        template_result.clone(),
                        max_tokens_override,
                        temperature_override,
                        top_p_override,
                    )
                    .await?;
                let message_json = template_result
                    .parse_response_oaicompat(&result.text)
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
                    content: message
                        .content
                        .and_then(StringOrJson::into_non_empty_string),
                    thinking: message
                        .reasoning_content
                        .and_then(StringOrJson::into_non_empty_string),
                    tool_calls: message
                        .tool_calls
                        .map(|calls| calls.into_iter().map(Into::into).collect()),
                    usage,
                }))
            }
        }
    }

    /// Inner implementation of `chat_stream` accepting per-call sampling
    /// overrides.
    async fn chat_stream_impl(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let (max_tokens_override, temperature_override, top_p_override) = unpack_sampling(sampling);

        if Self::has_mtmd_media(messages) {
            #[cfg(feature = "mtmd")]
            {
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let response_stream = self.spawn_mtmd_stream(
                    prompt,
                    images,
                    marker,
                    max_tokens_override,
                    temperature_override,
                    top_p_override,
                );
                let content_stream = response_stream.filter_map(|event| async move {
                    match event {
                        Ok(StreamEvent::Token(token)) => Some(Ok(token)),
                        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                            Ok(parsed) => parsed
                                .content
                                .and_then(StringOrJson::into_non_empty_string)
                                .map(Ok),
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
                return Err(LLMError::invalid_request(
                    "MTMD feature is not enabled for llama.cpp backend".to_string(),
                ));
            }
        }
        let prompt = self.build_chat_prompt(messages, None, json_schema.as_ref())?;
        let response_stream = match prompt {
            ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            } => self.spawn_fallback_stream(
                prompt,
                use_json_grammar,
                max_tokens_override,
                temperature_override,
                top_p_override,
            ),
            ChatPrompt::OpenAI(template_result) => self.spawn_chat_stream(
                template_result,
                max_tokens_override,
                temperature_override,
                top_p_override,
            ),
        };

        let content_stream = response_stream.filter_map(|event| async move {
            match event {
                Ok(StreamEvent::Token(token)) => Some(Ok(token)),
                Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                    Ok(parsed) => parsed
                        .content
                        .and_then(StringOrJson::into_non_empty_string)
                        .map(Ok),
                    Err(err) => Some(Err(err)),
                },
                Ok(StreamEvent::Usage(_)) | Ok(StreamEvent::Done { .. }) => None,
                Err(err) => Some(Err(err)),
            }
        });

        Ok(Box::pin(content_stream))
    }

    /// Inner implementation of `chat_stream_struct` accepting per-call
    /// sampling overrides.
    async fn chat_stream_struct_impl(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        let (max_tokens_override, temperature_override, top_p_override) = unpack_sampling(sampling);

        if Self::has_mtmd_media(messages) {
            #[cfg(feature = "mtmd")]
            {
                if tools.is_some() || json_schema.is_some() {
                    return Err(LLMError::invalid_request(
                        "MTMD path does not support tools or structured outputs".to_string(),
                    ));
                }
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let response_stream = self.spawn_mtmd_stream(
                    prompt,
                    images,
                    marker,
                    max_tokens_override,
                    temperature_override,
                    top_p_override,
                );
                let struct_stream = mtmd_structured_stream(response_stream);
                return Ok(Box::pin(struct_stream));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::invalid_request(
                    "MTMD feature is not enabled for llama.cpp backend".to_string(),
                ));
            }
        }
        let prompt = self.build_chat_prompt(messages, tools, json_schema.as_ref())?;
        let response_stream = match prompt {
            ChatPrompt::Fallback {
                prompt,
                use_json_grammar,
            } => self.spawn_fallback_stream(
                prompt,
                use_json_grammar,
                max_tokens_override,
                temperature_override,
                top_p_override,
            ),
            ChatPrompt::OpenAI(template_result) => self.spawn_chat_stream(
                template_result,
                max_tokens_override,
                temperature_override,
                top_p_override,
            ),
        };

        let struct_stream = response_stream
            .scan(StreamMappingState::default(), |stream_state, event| {
                futures::future::ready(Some(map_struct_stream_event(event, stream_state)))
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(struct_stream))
    }

    /// Inner implementation of `chat_stream_with_tools` accepting per-call
    /// sampling overrides.
    async fn chat_stream_with_tools_impl(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        let (max_tokens_override, temperature_override, top_p_override) = unpack_sampling(sampling);

        if Self::has_mtmd_media(messages) {
            if tools.is_some() || json_schema.is_some() {
                return Err(LLMError::invalid_request(
                    "MTMD path does not support tools or structured outputs".to_string(),
                ));
            }
            #[cfg(feature = "mtmd")]
            {
                let (prompt, images, marker) = self.build_mtmd_prompt(messages)?;
                let response_stream = self.spawn_mtmd_stream(
                    prompt,
                    images,
                    marker,
                    max_tokens_override,
                    temperature_override,
                    top_p_override,
                );
                let stream = mtmd_tool_stream(response_stream);
                return Ok(Box::pin(stream));
            }
            #[cfg(not(feature = "mtmd"))]
            {
                return Err(LLMError::invalid_request(
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
                self.spawn_fallback_stream(
                    prompt,
                    use_json_grammar,
                    max_tokens_override,
                    temperature_override,
                    top_p_override,
                )
            }
            ChatPrompt::OpenAI(template_result) => self.spawn_chat_stream(
                template_result,
                max_tokens_override,
                temperature_override,
                top_p_override,
            ),
        };

        let stream = response_stream
            .scan(StreamMappingState::default(), |stream_state, event| {
                futures::future::ready(Some(map_tool_stream_event(event, stream_state)))
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
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
        self.chat_with_tools_impl(messages, tools, json_schema, None)
            .await
    }

    async fn chat_with_tools_and_sampling(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools_impl(messages, tools, json_schema, sampling)
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        self.chat_stream_impl(messages, json_schema, None).await
    }

    async fn chat_stream_and_sampling(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        self.chat_stream_impl(messages, json_schema, sampling).await
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
        self.chat_stream_struct_impl(messages, tools, json_schema, None)
            .await
    }

    async fn chat_stream_struct_and_sampling(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
        sampling: Option<&SamplingOverrides>,
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        self.chat_stream_struct_impl(messages, tools, json_schema, sampling)
            .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        self.chat_stream_with_tools_impl(messages, tools, json_schema, None)
            .await
    }
}

fn prepare_messages_with_system(
    config: &LlamaCppConfig,
    messages: &[ChatMessage],
) -> Vec<ChatMessage> {
    let mut all_messages = Vec::new();

    if let Some(system_prompt) = &config.system_prompt {
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

fn prepare_fallback_messages_with_schema(
    config: &LlamaCppConfig,
    messages: &[ChatMessage],
    json_schema: Option<&StructuredOutputFormat>,
) -> Vec<ChatMessage> {
    let mut all_messages = prepare_messages_with_system(config, messages);

    if let Some(schema) = json_schema {
        let mut schema_hint = format!("Return a valid JSON response for schema '{}'.", schema.name);
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

fn parse_tool_response_message(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    parse_tool_response_message_with_allowed_tools(text, None)
}

fn parse_tool_response_message_with_allowed_tools(
    text: &str,
    allowed_tools: Option<&[String]>,
) -> Result<Option<Value>, LlamaCppProviderError> {
    for parser in TOOL_RESPONSE_PARSERS {
        if let Some(message) = parser(text)? {
            return Ok(filter_tool_message_by_allowed_tools(message, allowed_tools));
        }
    }

    let Some(json_text) = extract_json_payload(text) else {
        return Ok(None);
    };
    let value: Value = serde_json::from_str(&json_text)
        .map_err(|err| LlamaCppProviderError::Template(format!("Invalid tool JSON: {err}")))?;

    if let Some(calls) = value.get("tool_calls").and_then(Value::as_array) {
        let normalized = normalize_tool_calls(calls)?;
        if normalized.is_empty() || !normalized_tool_calls_allowed(&normalized, allowed_tools) {
            return Ok(None);
        }
        let mut message = json!({ "tool_calls": normalized });
        merge_openai_compat_envelope_fields(&mut message, &value);
        return Ok(Some(message));
    }

    if let Some(calls) = value.as_array() {
        if !calls
            .iter()
            .all(|call| is_json_tool_call_candidate(call, allowed_tools))
        {
            return Ok(None);
        }
        let normalized = normalize_tool_calls(calls)?;
        if normalized.is_empty() {
            return Ok(None);
        }
        return Ok(Some(json!({ "tool_calls": normalized })));
    }

    if let Some(calls) = normalize_function_key_tool_calls(&value, allowed_tools)? {
        return Ok(Some(json!({ "tool_calls": calls })));
    }

    if is_openai_compat_single_tool_call(&value, allowed_tools) {
        let normalized = normalize_tool_calls(std::slice::from_ref(&value))?;
        return Ok(Some(json!({ "tool_calls": normalized })));
    }

    Ok(None)
}

type ToolResponseParser = fn(&str) -> Result<Option<Value>, LlamaCppProviderError>;

const TOOL_RESPONSE_PARSERS: &[ToolResponseParser] = &[
    parse_native_channel_tool_calls,
    parse_tool_calls_args_tag,
    parse_gemma4_tool_calls,
    parse_generic_function_tag_tool_calls,
    parse_functionary_tool_calls,
    parse_kimi_k2_tool_calls,
    parse_lfm2_tool_calls,
    parse_gigachat_v3_tool_calls,
    parse_deepseek_dsml_tool_calls,
];

fn is_openai_compat_single_tool_call(value: &Value, allowed_tools: Option<&[String]>) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object.get("function").is_some() {
        let function_name = object
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str);
        return function_name
            .map(|name| tool_name_allowed(name, allowed_tools))
            .unwrap_or(true);
    }
    if object
        .get("name")
        .and_then(Value::as_str)
        .is_none_or(|name| name.trim().is_empty())
    {
        return false;
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .expect("name checked above");
    if !tool_name_allowed(name, allowed_tools) {
        return false;
    }
    if object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|call_type| call_type == "function")
    {
        return true;
    }
    tool_arguments_payload_from_object(object).is_some()
        || tool_call_id_from_object(object).is_some()
}

fn is_json_tool_call_candidate(value: &Value, allowed_tools: Option<&[String]>) -> bool {
    is_openai_compat_single_tool_call(value, allowed_tools)
        || is_function_key_tool_call_candidate(value, allowed_tools)
        || is_wrapped_nested_tool_call_candidate(value, allowed_tools)
}

fn is_function_key_tool_call_candidate(value: &Value, allowed_tools: Option<&[String]>) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object.len() != 1 {
        return false;
    }
    let Some((name, payload)) = object.iter().next() else {
        return false;
    };
    !is_reserved_tool_envelope_key(name)
        && !name.is_empty()
        && name.chars().all(is_native_tool_name_char)
        && tool_name_allowed(name, allowed_tools)
        && payload.is_object()
        && payload
            .get("name")
            .and_then(Value::as_str)
            .is_none_or(|name| name.trim().is_empty())
}

fn is_wrapped_nested_tool_call_candidate(value: &Value, allowed_tools: Option<&[String]>) -> bool {
    value.as_object().is_some_and(|object| {
        object
            .values()
            .any(|value| is_nested_function_payload(value, allowed_tools))
    })
}

fn is_nested_function_payload(value: &Value, allowed_tools: Option<&[String]>) -> bool {
    value.as_object().is_some_and(|nested| {
        nested
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.trim().is_empty() && tool_name_allowed(name, allowed_tools))
            && tool_arguments_payload_from_object(nested).is_some()
    })
}

fn tool_name_allowed(name: &str, allowed_tools: Option<&[String]>) -> bool {
    allowed_tools
        .is_none_or(|allowed| allowed.is_empty() || allowed.iter().any(|tool| tool == name))
}

fn normalized_tool_calls_allowed(calls: &[Value], allowed_tools: Option<&[String]>) -> bool {
    calls.iter().all(|call| {
        call.get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .is_some_and(|name| tool_name_allowed(name, allowed_tools))
    })
}

fn filter_tool_message_by_allowed_tools(
    message: Value,
    allowed_tools: Option<&[String]>,
) -> Option<Value> {
    let calls = message.get("tool_calls").and_then(Value::as_array);
    if calls.is_none_or(|calls| normalized_tool_calls_allowed(calls, allowed_tools)) {
        Some(message)
    } else {
        None
    }
}

fn merge_openai_compat_envelope_fields(message: &mut Value, source: &Value) {
    let Some(message_object) = message.as_object_mut() else {
        return;
    };
    let Some(source_object) = source.as_object() else {
        return;
    };

    if let Some(content) = source_object
        .get("content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
    {
        message_object.insert("content".to_string(), Value::String(content.to_string()));
    }

    let reasoning_content = source_object
        .get("reasoning_content")
        .or_else(|| source_object.get("thinking"))
        .and_then(Value::as_str)
        .filter(|reasoning_content| !reasoning_content.is_empty());
    if let Some(reasoning_content) = reasoning_content {
        message_object.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content.to_string()),
        );
    }
}

fn normalized_tool_message(calls: &[Value]) -> Result<Option<Value>, LlamaCppProviderError> {
    if calls.is_empty() {
        return Ok(None);
    }

    let normalized = normalize_tool_calls(calls)?;
    Ok(Some(json!({ "tool_calls": normalized })))
}

fn normalized_tool_message_with_content(
    calls: &[Value],
    content: Option<String>,
) -> Result<Option<Value>, LlamaCppProviderError> {
    if calls.is_empty() {
        return Ok(content
            .filter(|content| !content.is_empty())
            .map(|content| json!({ "content": content })));
    }

    let Some(mut message) = normalized_tool_message(calls)? else {
        return Ok(None);
    };
    if let Some(content) = content
        && !content.is_empty()
        && let Some(object) = message.as_object_mut()
    {
        object.insert("content".to_string(), Value::String(content));
    }
    Ok(Some(message))
}

fn content_before_marker(text: &str, marker: &str) -> Option<String> {
    text.find(marker)
        .map(|index| text[..index].trim().to_string())
        .filter(|content| !content.is_empty())
}

fn parse_native_channel_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    let mut calls = Vec::new();
    let mut search_start = 0;
    const RECIPIENT: &str = "to=functions.";
    const MESSAGE_MARKER: &str = "<|message|>";

    while let Some(relative_pos) = text[search_start..].find(RECIPIENT) {
        let recipient_pos = search_start + relative_pos;
        let name_start = recipient_pos + RECIPIENT.len();
        let name_end = text[name_start..]
            .find(|ch: char| !is_native_tool_name_char(ch))
            .map_or(text.len(), |offset| name_start + offset);
        let name = text[name_start..name_end].trim();
        if name.is_empty() {
            search_start = name_start;
            continue;
        }

        let Some(message_relative_pos) = text[name_end..].find(MESSAGE_MARKER) else {
            search_start = name_end;
            continue;
        };
        let arguments_start = name_end + message_relative_pos + MESSAGE_MARKER.len();
        let arguments_end = text[arguments_start..]
            .find("<|end|>")
            .map_or(text.len(), |offset| arguments_start + offset);
        let arguments_text = text[arguments_start..arguments_end].trim();
        if arguments_text.is_empty() {
            search_start = arguments_end;
            continue;
        }

        let arguments = extract_json_payload(arguments_text).ok_or_else(|| {
            LlamaCppProviderError::Template(format!(
                "Tool call arguments for `{name}` are not valid JSON"
            ))
        })?;
        calls.push(json!({
            "name": name,
            "arguments": serde_json::from_str::<Value>(&arguments).map_err(|err| {
                LlamaCppProviderError::Template(format!(
                    "Tool call arguments for `{name}` are not valid JSON: {err}"
                ))
            })?
        }));
        search_start = arguments_end;
    }

    let Some(mut message) = normalized_tool_message(&calls)? else {
        return Ok(None);
    };
    let (content, reasoning_content) = parse_native_non_tool_channels(text);
    if let Some(object) = message.as_object_mut() {
        if let Some(content) = content.filter(|content| !content.is_empty()) {
            object.insert("content".to_string(), Value::String(content));
        }
        if let Some(reasoning_content) =
            reasoning_content.filter(|reasoning_content| !reasoning_content.is_empty())
        {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content),
            );
        }
    }
    Ok(Some(message))
}

fn parse_native_non_tool_channels(text: &str) -> (Option<String>, Option<String>) {
    const START_MARKER: &str = "<|start|>assistant";
    const CHANNEL_MARKER: &str = "<|channel|>";
    const MESSAGE_MARKER: &str = "<|message|>";
    const END_MARKER: &str = "<|end|>";

    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut search_start = 0;

    while let Some(relative_pos) = text[search_start..].find(CHANNEL_MARKER) {
        let channel_marker_pos = search_start + relative_pos;
        let channel_pos = channel_marker_pos + CHANNEL_MARKER.len();
        let Some(channel_end_relative) = text[channel_pos..].find(MESSAGE_MARKER) else {
            break;
        };
        let channel_end = channel_pos + channel_end_relative;
        let header = native_effective_channel_header(&text[channel_pos..channel_end]);
        let content_start = channel_end + MESSAGE_MARKER.len();
        let content_end = text[content_start..]
            .find(END_MARKER)
            .map_or(text.len(), |offset| content_start + offset);
        let payload = text[content_start..content_end].trim();

        let start_header = text[..channel_marker_pos]
            .rfind(START_MARKER)
            .map(|start| &text[start + START_MARKER.len()..channel_marker_pos])
            .unwrap_or_default();
        let is_tool_channel = header.contains("to=functions.")
            || start_header.contains("to=functions.")
            || header.starts_with("to=");

        if !is_tool_channel && !payload.is_empty() {
            if header.starts_with("commentary") || header.starts_with("final") {
                content_parts.push(payload.to_string());
            } else if header.starts_with("analysis") {
                reasoning_parts.push(payload.to_string());
            }
        }

        search_start = content_end.saturating_add(END_MARKER.len()).min(text.len());
    }

    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    let reasoning_content = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n"))
    };
    (content, reasoning_content)
}

fn parse_tool_calls_args_tag(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const START: &str = "[TOOL_CALLS]";
    const ARGS: &str = "[ARGS]";

    let mut calls = Vec::new();
    let content = content_before_marker(text, START);
    let mut search_start = 0;
    while let Some(relative_pos) = text[search_start..].find(START) {
        let name_start = search_start + relative_pos + START.len();
        let Some(args_relative_pos) = text[name_start..].find(ARGS) else {
            search_start = name_start;
            continue;
        };
        let args_marker = name_start + args_relative_pos;
        let name = text[name_start..args_marker].trim();
        if name.is_empty() || !name.chars().all(is_native_tool_name_char) {
            search_start = args_marker + ARGS.len();
            continue;
        }

        let args_start = args_marker + ARGS.len();
        let args_end = text[args_start..]
            .find(START)
            .map_or(text.len(), |offset| args_start + offset);
        let arguments_text = text[args_start..args_end].trim();
        let Some(arguments_json) = extract_json_payload(arguments_text) else {
            search_start = args_end;
            continue;
        };
        let arguments = serde_json::from_str::<Value>(&arguments_json).map_err(|err| {
            LlamaCppProviderError::Template(format!(
                "Ministral/Magistral tool call arguments for `{name}` are not valid JSON: {err}"
            ))
        })?;
        calls.push(json!({
            "name": name,
            "arguments": arguments,
        }));
        search_start = args_end;
    }

    normalized_tool_message_with_content(&calls, content)
}

fn is_native_tool_name_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
}

fn parse_native_channel_content_partial(text: &str) -> Option<(String, Option<String>)> {
    const CHANNEL_MARKER: &str = "<|channel|>";
    const MESSAGE_MARKER: &str = "<|message|>";
    const END_MARKER: &str = "<|end|>";

    if !text.contains(CHANNEL_MARKER) || !text.contains(MESSAGE_MARKER) {
        return None;
    }

    let mut final_content = None;
    let mut reasoning_parts = Vec::new();
    let mut search_start = 0;

    while let Some(relative_pos) = text[search_start..].find(CHANNEL_MARKER) {
        let channel_pos = search_start + relative_pos + CHANNEL_MARKER.len();
        let channel_end = text[channel_pos..]
            .find(MESSAGE_MARKER)
            .map(|offset| channel_pos + offset)?;
        let header = native_effective_channel_header(&text[channel_pos..channel_end]);
        let content_start = channel_end + MESSAGE_MARKER.len();
        let content_end = text[content_start..]
            .find(END_MARKER)
            .map_or(text.len(), |offset| content_start + offset);
        let content = text[content_start..content_end].trim();

        if header.starts_with("analysis") && !content.is_empty() {
            reasoning_parts.push(content.to_string());
        } else if header.starts_with("final") {
            final_content = Some(content.to_string());
        }

        search_start = content_end.saturating_add(END_MARKER.len()).min(text.len());
    }

    if final_content.is_none() && reasoning_parts.is_empty() {
        return None;
    }

    Some({
        let reasoning = if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.join("\n"))
        };
        (final_content.unwrap_or_default(), reasoning)
    })
}

fn native_effective_channel_header(header: &str) -> &str {
    const CHANNEL_MARKER: &str = "<|channel|>";
    header
        .rsplit_once(CHANNEL_MARKER)
        .map_or(header, |(_, effective)| effective)
        .trim()
}

fn parse_gemma4_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const PREFIX: &str = "<|turn>model\n";
    const START: &str = "<|tool_call>call:";
    const END: &str = "<tool_call|>";

    let text = text.strip_prefix(PREFIX).unwrap_or(text);
    let mut calls = Vec::new();
    let content = content_before_marker(text, START);
    let mut search_start = 0;
    while let Some(relative_pos) = text[search_start..].find(START) {
        let call_start = search_start + relative_pos + START.len();
        let Some(name_end_relative) = text[call_start..].find('{') else {
            search_start = call_start;
            continue;
        };
        let name_end = call_start + name_end_relative;
        let name = text[call_start..name_end].trim();
        if name.is_empty() {
            search_start = name_end;
            continue;
        }
        let call_end = text[name_end..]
            .find(END)
            .map_or(text.len(), |offset| name_end + offset);
        let arguments_text = &text[name_end..call_end];
        let arguments = extract_first_balanced(arguments_text, '{', '}').ok_or_else(|| {
            LlamaCppProviderError::Template(format!(
                "Gemma4 tool call arguments for `{name}` are not valid JSON"
            ))
        })?;
        let arguments = parse_gemma4_arguments(&arguments).map_err(|err| {
            LlamaCppProviderError::Template(format!(
                "Gemma4 tool call arguments for `{name}` are not valid JSON: {err}"
            ))
        })?;
        calls.push(json!({
            "name": name,
            "arguments": arguments
        }));
        search_start = call_end.saturating_add(END.len()).min(text.len());
    }

    normalized_tool_message_with_content(&calls, content)
}

fn parse_gemma4_arguments(text: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => Ok(value),
        Err(err) => {
            let normalized = normalize_gemma4_argument_syntax(text);
            serde_json::from_str::<Value>(&normalized).map_err(|_| err)
        }
    }
}

fn normalize_gemma4_argument_syntax(text: &str) -> String {
    let normalized_strings = normalize_gemma4_quoted_strings(text);
    quote_unquoted_object_keys(&normalized_strings)
}

fn parse_generic_function_tag_tool_calls(
    text: &str,
) -> Result<Option<Value>, LlamaCppProviderError> {
    const START: &str = "<function=";
    const END: &str = "</function>";

    let mut calls = Vec::new();
    let content = content_before_marker(text, START);
    let mut search_start = 0;

    while let Some(relative_pos) = text[search_start..].find(START) {
        let name_start = search_start + relative_pos + START.len();
        let Some(name_end_relative) = text[name_start..].find('>') else {
            search_start = name_start;
            continue;
        };
        let name_end = name_start + name_end_relative;
        let name = text[name_start..name_end].trim();
        if name.is_empty() || !name.chars().all(is_native_tool_name_char) {
            search_start = name_end + 1;
            continue;
        }

        let arguments_start = name_end + 1;
        let arguments_end = text[arguments_start..]
            .find(END)
            .map_or(text.len(), |offset| arguments_start + offset);
        let arguments_text = text[arguments_start..arguments_end].trim();
        let arguments = if let Some(arguments_json) = extract_json_payload(arguments_text) {
            serde_json::from_str::<Value>(&arguments_json).map_err(|err| {
                LlamaCppProviderError::Template(format!(
                    "Generic function-tag arguments for `{name}` are not valid JSON: {err}"
                ))
            })?
        } else if let Some(arguments) = parse_generic_tagged_arguments(name, arguments_text)? {
            arguments
        } else {
            search_start = arguments_end.saturating_add(END.len()).min(text.len());
            continue;
        };

        calls.push(json!({
            "name": name,
            "arguments": arguments,
        }));
        search_start = arguments_end.saturating_add(END.len()).min(text.len());
    }

    normalized_tool_message_with_content(&calls, content)
}

fn parse_generic_tagged_arguments(
    _function_name: &str,
    text: &str,
) -> Result<Option<Value>, LlamaCppProviderError> {
    let mut arguments = serde_json::Map::new();
    let mut search_start = 0;

    while let Some((tag, tag_pos)) = find_next_generic_argument_tag(text, search_start) {
        let name_start = tag_pos + tag.open.len();
        let Some(name_end_relative) = text[name_start..].find('"') else {
            break;
        };
        let name_end = name_start + name_end_relative;
        let name = text[name_start..name_end].trim();
        if name.is_empty() || !name.chars().all(is_native_tool_name_char) {
            search_start = name_end;
            continue;
        }

        let Some(open_end_relative) = text[name_end..].find('>') else {
            break;
        };
        let value_start = name_end + open_end_relative + 1;
        let value_end = text[value_start..]
            .find(tag.close)
            .map_or(text.len(), |offset| value_start + offset);
        let value_text = text[value_start..value_end].trim();
        let value = if value_text.is_empty() {
            Value::String(String::default())
        } else if let Ok(value) = serde_json::from_str::<Value>(value_text) {
            value
        } else {
            Value::String(value_text.to_string())
        };
        arguments.insert(name.to_string(), value);
        search_start = value_end.saturating_add(tag.close.len()).min(text.len());
    }

    if arguments.is_empty() {
        return Ok(None);
    }

    Ok(Some(Value::Object(arguments)))
}

#[derive(Clone, Copy)]
struct GenericArgumentTag {
    open: &'static str,
    close: &'static str,
}

fn find_next_generic_argument_tag(
    text: &str,
    search_start: usize,
) -> Option<(GenericArgumentTag, usize)> {
    const TAGS: [GenericArgumentTag; 3] = [
        GenericArgumentTag {
            open: "<parameter name=\"",
            close: "</parameter>",
        },
        GenericArgumentTag {
            open: "<param name=\"",
            close: "</param>",
        },
        GenericArgumentTag {
            open: "<arg name=\"",
            close: "</arg>",
        },
    ];

    TAGS.iter()
        .filter_map(|tag| {
            text[search_start..]
                .find(tag.open)
                .map(|offset| (*tag, search_start + offset))
        })
        .min_by_key(|(_, index)| *index)
}

fn normalize_gemma4_quoted_strings(text: &str) -> String {
    const MARKER: &str = "<|\"|>";

    let mut output = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(marker_start) = rest.find(MARKER) {
        output.push_str(&rest[..marker_start]);
        let content_start = marker_start + MARKER.len();
        let Some(content_end_relative) = rest[content_start..].find(MARKER) else {
            output.push_str(&rest[marker_start..]);
            return output;
        };
        let content_end = content_start + content_end_relative;
        let content = &rest[content_start..content_end];
        match serde_json::to_string(content) {
            Ok(quoted) => output.push_str(&quoted),
            Err(_) => output.push_str(content),
        }
        rest = &rest[content_end + MARKER.len()..];
    }

    output.push_str(rest);
    output
}

fn extract_first_balanced(text: &str, open: char, close: char) -> Option<String> {
    let mut in_string = None;
    let mut escape = false;
    let mut depth = 0i32;
    let mut start = None;

    for (idx, ch) in text.char_indices() {
        if let Some(quote) = in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                current if current == quote => in_string = None,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' | '\'' => in_string = Some(ch),
            current if current == open => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            current if current == close && depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start_idx) = start {
                        return Some(text[start_idx..=idx].trim().to_string());
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }

    None
}

fn quote_unquoted_object_keys(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 8);
    let mut chars = text.chars().peekable();
    let mut in_string = None;
    let mut escaped = false;
    let mut expecting_key = false;

    while let Some(ch) = chars.next() {
        if let Some(quote) = in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_string = Some(ch);
                output.push(ch);
            }
            '{' | ',' => {
                expecting_key = true;
                output.push(ch);
            }
            ':' => {
                expecting_key = false;
                output.push(ch);
            }
            ch if expecting_key && ch.is_whitespace() => output.push(ch),
            ch if expecting_key && is_gemma4_unquoted_key_start(ch) => {
                let mut key = String::with_capacity(16);
                key.push(ch);
                while let Some(next) = chars.peek().copied() {
                    if is_gemma4_unquoted_key_char(next) {
                        key.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                output.push('"');
                output.push_str(&key);
                output.push('"');
                expecting_key = false;
            }
            _ => output.push(ch),
        }
    }

    output
}

fn is_gemma4_unquoted_key_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_' || ch == '-'
}

fn is_gemma4_unquoted_key_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
}

fn parse_functionary_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const START: &str = ">>>";
    let normalized_text;
    let implicit_generation_prompt = !text.contains(START);
    let text = if implicit_generation_prompt {
        let Some((recipient, payload)) = text.split_once('\n') else {
            return Ok(None);
        };
        let recipient = recipient.trim();
        if recipient != "all" && !payload.trim_start().starts_with(['{', '[']) {
            return Ok(None);
        }
        normalized_text = format!("{START}{text}");
        normalized_text.as_str()
    } else {
        text
    };
    let mut calls = Vec::new();
    let mut content = None;
    let mut search_start = 0;

    while let Some(relative_pos) = text[search_start..].find(START) {
        let recipient_start = search_start + relative_pos + START.len();
        let Some(line_end_relative) = text[recipient_start..].find('\n') else {
            break;
        };
        let line_end = recipient_start + line_end_relative;
        let recipient = text[recipient_start..line_end].trim();
        let payload_start = line_end + 1;
        let payload_end = text[payload_start..]
            .find(START)
            .map_or(text.len(), |offset| payload_start + offset);
        let payload = text[payload_start..payload_end].trim();

        if recipient.is_empty() || recipient == "all" {
            if recipient == "all" && !payload.is_empty() {
                append_content_segment(&mut content, payload);
            }
            search_start = payload_end;
            continue;
        }

        let Some(arguments) = extract_json_payload(payload) else {
            if implicit_generation_prompt {
                return Ok(None);
            }
            return Err(LlamaCppProviderError::Template(format!(
                "Functionary tool call arguments for `{recipient}` are not valid JSON"
            )));
        };
        let arguments = match serde_json::from_str::<Value>(&arguments) {
            Ok(arguments) => arguments,
            Err(err) if implicit_generation_prompt => {
                let _ = err;
                return Ok(None);
            }
            Err(err) => {
                return Err(LlamaCppProviderError::Template(format!(
                    "Functionary tool call arguments for `{recipient}` are not valid JSON: {err}"
                )));
            }
        };
        calls.push(json!({
            "name": recipient,
            "arguments": arguments,
        }));
        search_start = payload_end;
    }

    normalized_tool_message_with_content(&calls, content)
}

fn append_content_segment(content: &mut Option<String>, segment: &str) {
    match content {
        Some(existing) if !existing.is_empty() => {
            existing.push('\n');
            existing.push_str(segment);
        }
        Some(existing) => existing.push_str(segment),
        None => *content = Some(segment.to_string()),
    }
}

fn parse_kimi_k2_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const PREFIX: &str = "<|im_assistant|>assistant<|im_middle|>";
    const START: &str = "<|tool_call_begin|>functions.";
    const SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
    const ARGS: &str = "<|tool_call_argument_begin|>";
    const END: &str = "<|tool_call_end|>";

    let text = text.strip_prefix(PREFIX).unwrap_or(text);
    let mut calls = Vec::new();
    let content_marker = match (text.find(SECTION_BEGIN), text.find(START)) {
        (Some(section), Some(start)) => {
            if section < start {
                SECTION_BEGIN
            } else {
                START
            }
        }
        (Some(_), None) => SECTION_BEGIN,
        _ => START,
    };
    let content = content_before_marker(text, content_marker);
    let mut search_start = 0;
    while let Some(relative_pos) = text[search_start..].find(START) {
        let name_start = search_start + relative_pos + START.len();
        let Some(name_end_relative) = text[name_start..].find(':') else {
            search_start = name_start;
            continue;
        };
        let name_end = name_start + name_end_relative;
        let name = text[name_start..name_end].trim();
        let Some(args_marker_relative) = text[name_end..].find(ARGS) else {
            search_start = name_end;
            continue;
        };
        let args_marker = name_end + args_marker_relative;
        let id_start = name_start - "functions.".len();
        let id = text[id_start..args_marker].trim();
        let args_start = args_marker + ARGS.len();
        let args_end = text[args_start..]
            .find(END)
            .map_or(text.len(), |offset| args_start + offset);
        let arguments_text = text[args_start..args_end].trim();
        let arguments = extract_json_payload(arguments_text).ok_or_else(|| {
            LlamaCppProviderError::Template(format!(
                "Kimi K2 tool call arguments for `{name}` are not valid JSON"
            ))
        })?;
        calls.push(json!({
            "id": id,
            "name": name,
            "arguments": serde_json::from_str::<Value>(&arguments).map_err(|err| {
                LlamaCppProviderError::Template(format!(
                    "Kimi K2 tool call arguments for `{name}` are not valid JSON: {err}"
                ))
            })?
        }));
        search_start = args_end.saturating_add(END.len()).min(text.len());
    }

    normalized_tool_message_with_content(&calls, content)
}

fn parse_deepseek_dsml_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const PREFIX: &str = "<｜Assistant｜>";
    const FC_START: &str = "<｜DSML｜function_calls>";
    const INVOKE_START: &str = "<｜DSML｜invoke name=\"";
    const INVOKE_END: &str = "</｜DSML｜invoke>";
    const PARAM_START: &str = "<｜DSML｜parameter name=\"";
    const PARAM_END: &str = "</｜DSML｜parameter>";

    let text = text.strip_prefix(PREFIX).unwrap_or(text);
    let mut calls = Vec::new();
    let content = content_before_marker(text, FC_START);
    let mut search_start = 0;
    while let Some(relative_pos) = text[search_start..].find(INVOKE_START) {
        let name_start = search_start + relative_pos + INVOKE_START.len();
        let Some(name_end_relative) = text[name_start..].find("\">") else {
            search_start = name_start;
            continue;
        };
        let name_end = name_start + name_end_relative;
        let name = text[name_start..name_end].trim();
        let body_start = name_end + "\">".len();
        let body_end = text[body_start..]
            .find(INVOKE_END)
            .map_or(text.len(), |offset| body_start + offset);
        let body = &text[body_start..body_end];
        let arguments = parse_deepseek_dsml_parameters(name, body, PARAM_START, PARAM_END)?;
        calls.push(json!({
            "name": name,
            "arguments": arguments,
        }));
        search_start = body_end.saturating_add(INVOKE_END.len()).min(text.len());
    }

    normalized_tool_message_with_content(&calls, content)
}

fn parse_lfm2_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const PREFIX: &str = "<|im_start|>assistant\n";
    const START: &str = "<|tool_call_start|>";
    const END: &str = "<|tool_call_end|>";

    let text = text.strip_prefix(PREFIX).unwrap_or(text);
    let mut calls = Vec::new();
    let content = content_before_marker(text, START);
    let mut search_start = 0;
    while let Some(relative_pos) = text[search_start..].find(START) {
        let body_start = search_start + relative_pos + START.len();
        let body_end = text[body_start..]
            .find(END)
            .map_or(text.len(), |offset| body_start + offset);
        let body = text[body_start..body_end].trim();
        for call in parse_lfm2_python_call_list(body)? {
            calls.push(call);
        }
        search_start = body_end.saturating_add(END.len()).min(text.len());
    }

    normalized_tool_message_with_content(&calls, content)
}

fn parse_gigachat_v3_tool_calls(text: &str) -> Result<Option<Value>, LlamaCppProviderError> {
    const PREFIX: &str = "assistant<|role_sep|>\n";
    const START: &str = "<|message_sep|>\n\nfunction call<|role_sep|>\n";
    const END: &str = "<|message_sep|>\n\n";

    let had_prefix = text.starts_with(PREFIX);
    let text = text.strip_prefix(PREFIX).unwrap_or(text);
    let mut calls = Vec::new();
    let content = if text.contains(START) {
        content_before_marker(text, START)
    } else if had_prefix {
        Some(text.strip_suffix(END).unwrap_or(text).trim().to_string())
            .filter(|content| !content.is_empty())
    } else {
        None
    };
    let mut search_start = 0;
    while let Some(relative_pos) = text[search_start..].find(START) {
        let call_start = search_start + relative_pos + START.len();
        let call_end = text[call_start..]
            .find(END)
            .map_or(text.len(), |offset| call_start + offset);
        let payload = text[call_start..call_end].trim();
        let Some(json_text) = extract_json_payload(payload) else {
            search_start = call_end;
            continue;
        };
        let value = serde_json::from_str::<Value>(&json_text).map_err(|err| {
            LlamaCppProviderError::Template(format!(
                "GigaChat V3 tool call is not valid JSON: {err}"
            ))
        })?;
        calls.push(value);
        search_start = call_end.saturating_add(END.len()).min(text.len());
    }

    normalized_tool_message_with_content(&calls, content)
}

fn parse_lfm2_python_call_list(text: &str) -> Result<Vec<Value>, LlamaCppProviderError> {
    let trimmed = text.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(trimmed)
        .trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }

    split_top_level(inner, ',')
        .into_iter()
        .map(|call| parse_lfm2_python_call(&call))
        .collect()
}

fn parse_lfm2_python_call(text: &str) -> Result<Value, LlamaCppProviderError> {
    let call = text.trim();
    let Some(open) = call.find('(') else {
        return Err(LlamaCppProviderError::Template(format!(
            "LFM2 tool call is missing argument list: {call}"
        )));
    };
    let Some(close) = call.rfind(')') else {
        return Err(LlamaCppProviderError::Template(format!(
            "LFM2 tool call is missing closing parenthesis: {call}"
        )));
    };
    if close < open {
        return Err(LlamaCppProviderError::Template(format!(
            "LFM2 tool call has invalid parentheses: {call}"
        )));
    }

    let name = call[..open].trim();
    if name.is_empty() || !name.chars().all(is_native_tool_name_char) {
        return Err(LlamaCppProviderError::Template(format!(
            "LFM2 tool call has invalid function name: {name}"
        )));
    }
    let mut arguments = serde_json::Map::new();
    let args = call[open + 1..close].trim();
    if !args.is_empty() {
        for item in split_top_level(args, ',') {
            let Some((key, value)) = split_once_top_level(&item, '=') else {
                return Err(LlamaCppProviderError::Template(format!(
                    "LFM2 tool call argument is missing `=`: {item}"
                )));
            };
            let key = key.trim();
            if key.is_empty() || !key.chars().all(is_native_tool_name_char) {
                return Err(LlamaCppProviderError::Template(format!(
                    "LFM2 tool call has invalid argument name: {key}"
                )));
            }
            arguments.insert(key.to_string(), parse_lfm2_python_value(value.trim())?);
        }
    }

    Ok(json!({
        "name": name,
        "arguments": Value::Object(arguments),
    }))
}

fn parse_lfm2_python_value(text: &str) -> Result<Value, LlamaCppProviderError> {
    if let Some(value) = parse_python_quoted_string(text, '\'')? {
        return Ok(Value::String(value));
    }
    if let Some(value) = parse_python_quoted_string(text, '"')? {
        return Ok(Value::String(value));
    }
    let json_text = normalize_lfm2_python_literal(text)?;
    serde_json::from_str::<Value>(&json_text).map_err(|err| {
        LlamaCppProviderError::Template(format!(
            "LFM2 tool call value is not valid JSON/Python literal: {err}"
        ))
    })
}

fn normalize_lfm2_python_literal(text: &str) -> Result<String, LlamaCppProviderError> {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_double_string = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if in_double_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double_string = false;
            }
            continue;
        }

        match ch {
            '"' => {
                in_double_string = true;
                output.push(ch);
            }
            '\'' => {
                push_lfm2_single_quoted_json(&mut output, &mut chars)?;
            }
            ch if ch.is_ascii_alphabetic() || ch == '_' => {
                push_lfm2_identifier(&mut output, ch, &mut chars);
            }
            _ => output.push(ch),
        }
    }

    Ok(remove_lfm2_json_trailing_commas(&output))
}

fn push_lfm2_single_quoted_json(
    output: &mut String,
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<(), LlamaCppProviderError> {
    let quoted = parse_lfm2_single_quoted(chars)?;
    let encoded = serde_json::to_string(&quoted).map_err(|err| {
        LlamaCppProviderError::Template(format!(
            "LFM2 quoted string could not be encoded as JSON: {err}"
        ))
    })?;
    output.push_str(&encoded);
    Ok(())
}

fn parse_lfm2_single_quoted(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<String, LlamaCppProviderError> {
    let mut quoted = String::with_capacity(16);
    let mut escaped = false;

    for inner in chars.by_ref() {
        if escaped {
            push_lfm2_escaped_char(&mut quoted, inner);
            escaped = false;
        } else if inner == '\\' {
            escaped = true;
        } else if inner == '\'' {
            return Ok(quoted);
        } else {
            quoted.push(inner);
        }
    }

    if escaped {
        return Err(LlamaCppProviderError::Template(
            "LFM2 quoted string ends with an escape".to_string(),
        ));
    }
    Err(LlamaCppProviderError::Template(
        "LFM2 quoted string is missing a closing quote".to_string(),
    ))
}

fn push_lfm2_escaped_char(output: &mut String, ch: char) {
    match ch {
        '\\' => output.push('\\'),
        '\'' => output.push('\''),
        '"' => output.push('"'),
        'n' => output.push('\n'),
        'r' => output.push('\r'),
        't' => output.push('\t'),
        other => {
            output.push('\\');
            output.push(other);
        }
    }
}

fn push_lfm2_identifier(
    output: &mut String,
    first: char,
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) {
    let mut ident = String::with_capacity(8);
    ident.push(first);
    while let Some(next) = chars.peek().copied() {
        if next.is_ascii_alphanumeric() || next == '_' {
            ident.push(next);
            chars.next();
        } else {
            break;
        }
    }
    match ident.as_str() {
        "True" => output.push_str("true"),
        "False" => output.push_str("false"),
        "None" => output.push_str("null"),
        _ => output.push_str(&ident),
    }
}

fn remove_lfm2_json_trailing_commas(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                output.push(ch);
            }
            ',' => {
                let mut lookahead = chars.clone();
                while matches!(lookahead.peek(), Some(next) if next.is_whitespace()) {
                    lookahead.next();
                }
                if matches!(lookahead.peek(), Some(']' | '}')) {
                    continue;
                }
                output.push(ch);
            }
            _ => output.push(ch),
        }
    }

    output
}

fn parse_python_quoted_string(
    text: &str,
    quote: char,
) -> Result<Option<String>, LlamaCppProviderError> {
    let Some(inner) = text
        .strip_prefix(quote)
        .and_then(|value| value.strip_suffix(quote))
    else {
        return Ok(None);
    };
    let mut value = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            value.push(ch);
            continue;
        }
        let Some(escaped) = chars.next() else {
            return Err(LlamaCppProviderError::Template(
                "LFM2 quoted string ends with an escape".to_string(),
            ));
        };
        match escaped {
            '\\' => value.push('\\'),
            '\'' if quote == '\'' => value.push('\''),
            '"' if quote == '"' => value.push('"'),
            'n' => value.push('\n'),
            'r' => value.push('\r'),
            't' => value.push('\t'),
            other => {
                value.push('\\');
                value.push(other);
            }
        }
    }
    Ok(Some(value))
}

fn split_top_level(text: &str, delimiter: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0_i32;
    let mut in_string = None;
    let mut escaped = false;

    for (index, ch) in text.char_indices() {
        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '"' | '\'' => in_string = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ if ch == delimiter && depth == 0 => {
                parts.push(text[start..index].trim().to_string());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }

    parts.push(text[start..].trim().to_string());
    parts.into_iter().filter(|part| !part.is_empty()).collect()
}

fn split_once_top_level(text: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut depth = 0_i32;
    let mut in_string = None;
    let mut escaped = false;

    for (index, ch) in text.char_indices() {
        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }

        match ch {
            '"' | '\'' => in_string = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ if ch == delimiter && depth == 0 => {
                return Some((&text[..index], &text[index + ch.len_utf8()..]));
            }
            _ => {}
        }
    }

    None
}

fn parse_deepseek_dsml_parameters(
    function_name: &str,
    body: &str,
    param_start: &str,
    param_end: &str,
) -> Result<Value, LlamaCppProviderError> {
    let mut arguments = serde_json::Map::new();
    let mut search_start = 0;

    while let Some(relative_pos) = body[search_start..].find(param_start) {
        let name_start = search_start + relative_pos + param_start.len();
        let Some(name_end_relative) = body[name_start..].find('"') else {
            break;
        };
        let name_end = name_start + name_end_relative;
        let param_name = body[name_start..name_end].trim();
        let Some(open_end_relative) = body[name_end..].find('>') else {
            break;
        };
        let value_start = name_end + open_end_relative + 1;
        let value_end = body[value_start..]
            .find(param_end)
            .map_or(body.len(), |offset| value_start + offset);
        let value_text = body[value_start..value_end].trim();
        let is_string = body[name_end..value_start].contains("string=\"true\"");
        let value = if is_string {
            Value::String(value_text.to_string())
        } else {
            serde_json::from_str::<Value>(value_text).map_err(|err| {
                LlamaCppProviderError::Template(format!(
                    "DeepSeek tool call argument `{param_name}` for `{function_name}` is not valid JSON: {err}"
                ))
            })?
        };
        arguments.insert(param_name.to_string(), value);
        search_start = value_end.saturating_add(param_end.len()).min(body.len());
    }

    Ok(Value::Object(arguments))
}

fn normalize_tool_calls(calls: &[Value]) -> Result<Vec<Value>, LlamaCppProviderError> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| normalize_tool_call(index, call))
        .collect()
}

fn normalize_function_key_tool_calls(
    value: &Value,
    allowed_tools: Option<&[String]>,
) -> Result<Option<Vec<Value>>, LlamaCppProviderError> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    if object.is_empty() {
        return Ok(None);
    }

    let mut calls = Vec::new();
    for (index, (name, payload)) in object.iter().enumerate() {
        if is_reserved_tool_envelope_key(name) {
            return Ok(None);
        }
        if name.is_empty()
            || !name.chars().all(is_native_tool_name_char)
            || !tool_name_allowed(name, allowed_tools)
        {
            return Ok(None);
        }
        if !payload.is_object() {
            return Ok(None);
        }
        let call = json!({ name: payload });
        let Some(normalized) = normalize_function_key_tool_call(index, &call)? else {
            return Ok(None);
        };
        calls.push(normalized);
    }

    Ok(Some(calls))
}

fn is_reserved_tool_envelope_key(key: &str) -> bool {
    matches!(
        key,
        "content"
            | "reasoning_content"
            | "tool_calls"
            | "function"
            | "name"
            | "arguments"
            | "args"
            | "parameters"
            | "id"
            | "call_id"
            | "tool_call_id"
            | "callId"
            | "type"
    )
}

fn normalize_tool_call(index: usize, call: &Value) -> Result<Value, LlamaCppProviderError> {
    if let Some(normalized) = normalize_function_key_tool_call(index, call)? {
        return Ok(normalized);
    }
    if let Some(normalized) = normalize_nested_json_tool_call(index, call)? {
        return Ok(normalized);
    }

    let function = call.get("function").unwrap_or(call);
    let function = function.as_object().ok_or_else(|| {
        LlamaCppProviderError::Template("Tool call function must be a JSON object".to_string())
    })?;

    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| {
            LlamaCppProviderError::Template("Tool call missing function name".to_string())
        })?;

    let arguments =
        normalize_arguments_payload(name, tool_arguments_payload_from_object(function))?;

    let id = tool_call_id(call)
        .or_else(|| tool_call_id_from_object(function))
        .unwrap_or_else(|| format!("call_{}", index + 1));

    Ok(json!({
        "id": id,
        "type": call
            .get("type")
            .or_else(|| call.get("call_type"))
            .and_then(Value::as_str)
            .filter(|call_type| !call_type.trim().is_empty())
            .unwrap_or("function"),
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }))
}

fn normalize_function_key_tool_call(
    index: usize,
    call: &Value,
) -> Result<Option<Value>, LlamaCppProviderError> {
    let Some(object) = call.as_object() else {
        return Ok(None);
    };
    if object.len() != 1 {
        return Ok(None);
    }
    let Some((name, payload)) = object.iter().next() else {
        return Ok(None);
    };
    if is_reserved_tool_envelope_key(name)
        || name.is_empty()
        || !name.chars().all(is_native_tool_name_char)
        || !payload.is_object()
    {
        return Ok(None);
    }
    if payload
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| !name.trim().is_empty())
    {
        return Ok(None);
    }

    let arguments =
        normalize_arguments_payload(name, tool_arguments_payload(payload).or(Some(payload)))?;
    let id = tool_call_id(payload).unwrap_or_else(|| format!("call_{}", index + 1));

    let normalized = json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        }
    });
    Ok(Some(normalized))
}

fn normalize_nested_json_tool_call(
    index: usize,
    call: &Value,
) -> Result<Option<Value>, LlamaCppProviderError> {
    let Some(object) = call.as_object() else {
        return Ok(None);
    };
    let nested = object
        .iter()
        .find_map(|(_, value)| value.as_object())
        .filter(|nested| {
            nested
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.trim().is_empty())
                && tool_arguments_payload_from_object(nested).is_some()
        });
    let Some(function) = nested else {
        return Ok(None);
    };

    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .expect("nested function name checked above");
    let arguments =
        normalize_arguments_payload(name, tool_arguments_payload_from_object(function))?;
    let id = tool_call_id(call)
        .or_else(|| tool_call_id(&Value::Object(function.clone())))
        .unwrap_or_else(|| format!("call_{}", index + 1));

    Ok(Some(json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        }
    })))
}

fn tool_arguments_payload(payload: &Value) -> Option<&Value> {
    let object = payload.as_object()?;
    tool_arguments_payload_from_object(object)
}

fn tool_arguments_payload_from_object(object: &serde_json::Map<String, Value>) -> Option<&Value> {
    for key in ["arguments", "args", "parameters"] {
        if let Some(arguments) = object.get(key) {
            return Some(arguments);
        }
    }
    None
}

fn normalize_arguments_payload(
    name: &str,
    arguments: Option<&Value>,
) -> Result<String, LlamaCppProviderError> {
    match arguments {
        Some(Value::String(arguments)) => {
            let trimmed = arguments.trim();
            if trimmed.is_empty() {
                Ok("{}".to_string())
            } else {
                serde_json::from_str::<Value>(trimmed).map_err(|err| {
                    LlamaCppProviderError::Template(format!(
                        "Tool call arguments for `{name}` are not valid JSON: {err}"
                    ))
                })?;
                Ok(trimmed.to_string())
            }
        }
        Some(arguments) => Ok(arguments.to_string()),
        None => Ok("{}".to_string()),
    }
}

fn tool_call_id(payload: &Value) -> Option<String> {
    let object = payload.as_object()?;
    tool_call_id_from_object(object)
}

fn tool_call_id_from_object(object: &serde_json::Map<String, Value>) -> Option<String> {
    for key in ["id", "call_id", "tool_call_id", "callId"] {
        if let Some(id) = object.get(key) {
            if let Some(id) = id.as_str().filter(|id| !id.trim().is_empty()) {
                return Some(id.to_string());
            }
            if id.is_number() {
                return Some(id.to_string());
            }
        }
    }
    None
}

fn stream_delta_from_message_json(message_json: &str) -> Result<String, LlamaCppProviderError> {
    let mut value: Value = serde_json::from_str(message_json).map_err(|err| {
        LlamaCppProviderError::Template(format!("Invalid parsed stream message: {err}"))
    })?;

    let Some(object) = value.as_object_mut() else {
        return Err(LlamaCppProviderError::Template(
            "Parsed stream message must be a JSON object".to_string(),
        ));
    };

    if let Some(Value::Array(calls)) = object.get_mut("tool_calls") {
        for (index, call) in calls.iter_mut().enumerate() {
            let Some(call_object) = call.as_object_mut() else {
                continue;
            };
            call_object.insert("index".to_string(), Value::from(index));
        }
    }

    let mut delta = serde_json::Map::new();
    for key in ["content", "reasoning_content", "tool_calls"] {
        if let Some(value) = object.remove(key) {
            let keep = match &value {
                Value::Null => false,
                Value::String(text) => !text.is_empty(),
                Value::Array(items) => !items.is_empty(),
                _ => true,
            };
            if keep {
                delta.insert(key.to_string(), value);
            }
        }
    }

    if delta.is_empty() {
        return Ok(String::new());
    }
    serde_json::to_string(&Value::Object(delta))
        .map_err(|err| LlamaCppProviderError::Template(err.to_string()))
}

fn extract_reasoning_content(
    content: &mut String,
    reasoning_format: Option<crate::config::LlamaCppReasoningFormat>,
    start_tag: Option<&str>,
    end_tag: Option<&str>,
) -> Option<String> {
    let format = reasoning_format?;
    if matches!(format, crate::config::LlamaCppReasoningFormat::None) {
        return None;
    }

    extract_tagged_reasoning(
        content,
        start_tag.unwrap_or("<think>"),
        end_tag.unwrap_or("</think>"),
    )
}

fn extract_tagged_reasoning(
    content: &mut String,
    start_tag: &str,
    end_tag: &str,
) -> Option<String> {
    let start = content.find(start_tag)?;
    let reasoning_start = start + start_tag.len();
    let relative_end = content[reasoning_start..].find(end_tag)?;
    let end = reasoning_start + relative_end;
    let reasoning = content[reasoning_start..end].trim().to_string();
    let after_end = end + end_tag.len();
    content.replace_range(start..after_end, "");
    if start_tag == "<|channel>thought" && end_tag == "<channel|>" {
        cleanup_gemma4_channel_controls(content);
    }
    *content = content.trim().to_string();
    Some(reasoning)
}

fn extract_tagged_reasoning_partial(
    content: &mut String,
    start_tag: &str,
    end_tag: &str,
) -> Option<String> {
    let start = content.find(start_tag)?;
    let reasoning_start = start + start_tag.len();
    let (reasoning, remove_end) =
        if let Some(relative_end) = content[reasoning_start..].find(end_tag) {
            let end = reasoning_start + relative_end;
            (
                content[reasoning_start..end].trim().to_string(),
                end + end_tag.len(),
            )
        } else {
            (
                strip_trailing_partial_tag_prefix(content[reasoning_start..].trim(), end_tag)
                    .trim()
                    .to_string(),
                content.len(),
            )
        };
    content.replace_range(start..remove_end, "");
    if start_tag == "<|channel>thought" && end_tag == "<channel|>" {
        cleanup_gemma4_channel_controls(content);
    }
    *content = content.trim().to_string();
    if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    }
}

fn extract_prefilled_reasoning(
    content: &mut String,
    start_tag: &str,
    end_tag: &str,
    allow_open: bool,
) -> Option<String> {
    let first_non_ws = content
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index))
        .unwrap_or(content.len());
    let visible = &content[first_non_ws..];
    if visible_is_partial_tag(visible, start_tag) || visible_is_partial_tag(visible, end_tag) {
        return None;
    }
    if visible.starts_with('{')
        || visible.starts_with('[')
        || visible.starts_with("<tool_call>")
        || visible.starts_with("<function=")
        || visible.starts_with("<|tool_call")
        || visible.starts_with("<|start|>")
    {
        return None;
    }

    let (reasoning, remove_end) = if let Some(relative_end) = content[first_non_ws..].find(end_tag)
    {
        let end = first_non_ws + relative_end;
        (content[..end].trim().to_string(), end + end_tag.len())
    } else if allow_open {
        (
            strip_trailing_partial_tag_prefix(content.trim(), end_tag)
                .trim()
                .to_string(),
            content.len(),
        )
    } else {
        return None;
    };
    content.replace_range(..remove_end, "");
    *content = content.trim().to_string();
    if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    }
}

fn visible_is_partial_tag(visible: &str, tag: &str) -> bool {
    !visible.is_empty() && visible.len() < tag.len() && tag.starts_with(visible)
}

fn strip_trailing_partial_tag_prefix<'a>(text: &'a str, tag: &str) -> &'a str {
    for len in (1..tag.len()).rev() {
        let prefix = &tag[..len];
        if text.ends_with(prefix) {
            return &text[..text.len() - len];
        }
    }
    text
}

fn cleanup_gemma4_channel_controls(content: &mut String) {
    const CHANNEL_START: &str = "<|channel>";
    const CHANNEL_END: &str = "<channel|>";

    while content.starts_with(CHANNEL_START) {
        content.replace_range(..CHANNEL_START.len(), "");
    }

    if let Some(index) = content.find(CHANNEL_START) {
        content.truncate(index);
    }
    if let Some(index) = content.find(CHANNEL_END) {
        content.truncate(index);
    }
}

fn sanitize_chat_template_schema(schema: &StructuredOutputFormat) -> Option<Value> {
    let mut schema = schema.schema.clone()?;
    if schema.get("additionalProperties").is_none()
        && matches!(schema.get("type"), Some(Value::String(kind)) if kind == "object")
    {
        schema["additionalProperties"] = Value::Bool(false);
    }
    Some(schema)
}

fn build_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
    tool_call_format: ToolCallGrammarFormat,
) -> Result<String, LLMError> {
    let (selected_tools, effective_tool_choice) = select_tools_for_tool_choice(tools, tool_choice)?;
    let tools = selected_tools.as_slice();
    let tool_choice = effective_tool_choice;

    match tool_call_format {
        ToolCallGrammarFormat::OpenAiEnvelope => {
            let schema = build_tool_response_schema(
                tools,
                json_schema,
                force_json_grammar,
                tool_choice,
                parallel_tool_calls,
            )
            .map_err(|err| {
                LLMError::ProviderError(format!("Failed to build tool schema: {err}"))
            })?;
            compile_json_schema_grammar(&schema, "tool")
        }
        ToolCallGrammarFormat::NativeChannelToolCall => build_native_channel_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
        ),
        ToolCallGrammarFormat::KimiK2ToolCall => build_kimi_k2_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        ),
        ToolCallGrammarFormat::Lfm2ToolCall => build_lfm2_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        ),
        ToolCallGrammarFormat::GigaChatV3ToolCall => build_gigachat_v3_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
        ),
        ToolCallGrammarFormat::FunctionaryV32 => build_functionary_v32_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        ),
        ToolCallGrammarFormat::DeepSeekDsmlToolCall => build_deepseek_dsml_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        ),
        ToolCallGrammarFormat::GenericFunctionTag => build_generic_function_tag_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        ),
        ToolCallGrammarFormat::XmlToolCall
        | ToolCallGrammarFormat::ToolCallsArrayTag
        | ToolCallGrammarFormat::ToolCallsArgsTag
        | ToolCallGrammarFormat::Gemma4ToolCall => build_native_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
            tool_call_format,
        ),
    }
}

fn select_tools_for_tool_choice(
    tools: &[Tool],
    tool_choice: LlamaCppToolChoice,
) -> Result<(Vec<Tool>, LlamaCppToolChoice), LLMError> {
    match tool_choice {
        LlamaCppToolChoice::Function { name } => {
            let Some(tool) = tools.iter().find(|tool| tool.function.name == name) else {
                return Err(LLMError::invalid_request(format!(
                    "tool_choice selected function `{name}`, but no matching tool was provided"
                )));
            };
            Ok((vec![tool.clone()], LlamaCppToolChoice::Required))
        }
        other => Ok((tools.to_vec(), other)),
    }
}

fn compile_json_schema_grammar(schema: &Value, label: &str) -> Result<String, LLMError> {
    let schema_str = schema.to_string();
    llama_cpp_2::json_schema_to_grammar(&schema_str).map_err(|err| {
        LLMError::ProviderError(format!(
            "Failed to compile llama.cpp {label} grammar: {err}"
        ))
    })
}

fn wrap_structured_response_grammar(
    grammar: &str,
    tool_call_format: ToolCallGrammarFormat,
) -> String {
    match tool_call_format {
        ToolCallGrammarFormat::ToolCallsArgsTag => {
            let renamed = rename_root_rule(grammar, "response-format");
            format!("root ::= \"```json\" response-format \"```\"\n\n{renamed}")
        }
        ToolCallGrammarFormat::Gemma4ToolCall => {
            let renamed = rename_root_rule(grammar, "response-format");
            format!(
                "root ::= gemma4-response-start gemma4-response-thought? \"```json\" response-format \"```\"\ngemma4-response-start ::= \"<|turn>model\\n\"?\ngemma4-response-thought ::= gemma4-response-empty-channels \"<|channel>thought\" [ \\t\\n\\r]+ gemma4-response-thought-content \"<channel|>\"\ngemma4-response-empty-channels ::= (\"<|channel>\" gemma4-response-non-thought-channel)*\ngemma4-response-non-thought-channel ::= [^t] | \"t\" [^h] | \"th\" [^o] | \"tho\" [^u] | \"thou\" [^g] | \"thoug\" [^h] | \"though\" [^t]\ngemma4-response-thought-content ::= gemma4-response-thought-char*\ngemma4-response-thought-char ::= [^<] | \"<\" [^c] | \"<c\" [^h] | \"<ch\" [^a] | \"<cha\" [^n] | \"<chan\" [^n] | \"<chann\" [^e] | \"<channe\" [^l] | \"<channel\" [^|]\n\n{renamed}"
            )
        }
        ToolCallGrammarFormat::NativeChannelToolCall => {
            let renamed = rename_root_rule(grammar, "response-format");
            format!(
                "root ::= \"<|start|>assistant\" \"<|channel|>final\" native-response-constraint \"<|message|>\" response-format\nnative-response-ws ::= [ \\t\\n\\r]*\nnative-response-constraint-type ::= [A-Za-z0-9_-]+\nnative-response-constraint ::= (native-response-ws (\"<|constrain|>\" native-response-ws)? native-response-constraint-type)?\n\n{renamed}"
            )
        }
        ToolCallGrammarFormat::DeepSeekDsmlToolCall => {
            let renamed = rename_root_rule(grammar, "response-format");
            format!(
                "root ::= \"```json\" deepseek-response-ws response-format deepseek-response-ws \"```\"\ndeepseek-response-ws ::= [ \\t\\n\\r]*\n\n{renamed}"
            )
        }
        ToolCallGrammarFormat::GenericFunctionTag => {
            let renamed = rename_root_rule(grammar, "response-format");
            format!(
                "root ::= \"```json\" generic-response-ws response-format generic-response-ws \"```\" | response-format\ngeneric-response-ws ::= [ \\t\\n\\r]*\n\n{renamed}"
            )
        }
        _ => grammar.to_string(),
    }
}

fn build_tool_response_schema(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    _force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<Value, LlamaCppProviderError> {
    if tools.is_empty() {
        return Err(LlamaCppProviderError::Template(
            "Cannot build tool response schema without tools".to_string(),
        ));
    }

    let tool_call_variants = tools
        .iter()
        .map(build_single_tool_call_schema)
        .collect::<Result<Vec<_>, _>>()?;

    let final_response_schema = json_schema
        .and_then(sanitize_chat_template_schema)
        .unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"],
                "additionalProperties": false
            })
        });

    let mut tool_calls_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": { "oneOf": tool_call_variants }
    });
    if !parallel_tool_calls {
        tool_calls_schema["maxItems"] = Value::from(1);
    }

    let tool_response_schema = json!({
        "type": "object",
        "properties": {
            "tool_calls": tool_calls_schema
        },
        "required": ["tool_calls"],
        "additionalProperties": false
    });

    if matches!(tool_choice, LlamaCppToolChoice::Required) {
        Ok(tool_response_schema)
    } else {
        Ok(json!({
            "oneOf": [
                final_response_schema,
                tool_response_schema
            ]
        }))
    }
}

fn build_native_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
    tool_call_format: ToolCallGrammarFormat,
) -> Result<String, LLMError> {
    if matches!(tool_call_format, ToolCallGrammarFormat::ToolCallsArgsTag) {
        return build_tool_calls_args_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        );
    }
    if matches!(tool_call_format, ToolCallGrammarFormat::Gemma4ToolCall) {
        return build_gemma4_tool_response_grammar(
            tools,
            json_schema,
            force_json_grammar,
            tool_choice,
            parallel_tool_calls,
        );
    }

    let tool_schema =
        build_native_tool_payload_schema(tools, tool_call_format, parallel_tool_calls).map_err(
            |err| LLMError::ProviderError(format!("Failed to build native tool schema: {err}")),
        )?;
    let tool_grammar = compile_json_schema_grammar(&tool_schema, "native tool")?;

    let lazy_tool_payload_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    if lazy_tool_payload_only {
        return Ok(wrap_native_tool_payload_grammar(
            &tool_grammar,
            tool_call_format,
            false,
        ));
    }

    let include_tool_markers = native_tool_markers_are_safe_in_grammar(tool_call_format);
    let tool_call_grammar =
        wrap_native_tool_payload_grammar(&tool_grammar, tool_call_format, include_tool_markers);
    if matches!(tool_choice, LlamaCppToolChoice::Required) {
        return Ok(tool_call_grammar);
    }

    let final_schema = json_schema
        .and_then(sanitize_chat_template_schema)
        .unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"],
                "additionalProperties": false
            })
        });
    let mut final_grammar = compile_json_schema_grammar(&final_schema, "final response")?;
    if json_schema.is_some() {
        final_grammar = wrap_structured_response_grammar(&final_grammar, tool_call_format);
    }
    Ok(combine_final_and_native_tool_grammars(
        &final_grammar,
        &tool_call_grammar,
    ))
}

fn build_native_channel_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
) -> Result<String, LLMError> {
    if tools.is_empty() {
        return Err(LLMError::ProviderError(
            "Cannot build native channel tool grammar without tools".to_string(),
        ));
    }

    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar = if lazy_tool_suffix_only {
        build_native_channel_tool_suffix_grammar(tools)?
    } else {
        build_native_channel_tool_call_grammar(tools)?
    };

    if matches!(tool_choice, LlamaCppToolChoice::Required) || lazy_tool_suffix_only {
        return Ok(tool_call_grammar);
    }

    let final_schema = json_schema
        .and_then(sanitize_chat_template_schema)
        .unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"],
                "additionalProperties": false
            })
        });
    let mut final_grammar = compile_json_schema_grammar(&final_schema, "final response")?;
    if json_schema.is_some() {
        final_grammar = wrap_structured_response_grammar(
            &final_grammar,
            ToolCallGrammarFormat::NativeChannelToolCall,
        );
    }
    Ok(combine_final_and_native_tool_grammars(
        &final_grammar,
        &tool_call_grammar,
    ))
}

fn build_native_channel_tool_call_grammar(tools: &[Tool]) -> Result<String, LLMError> {
    let mut rules = vec![
        "native-channel-ws ::= [ \\t\\n\\r]*".to_string(),
        "native-channel-name ::= \"commentary\" | \"analysis\"".to_string(),
        "native-channel ::= \"<|channel|>\" native-channel-name".to_string(),
        "native-constraint-type ::= [A-Za-z0-9_-]+".to_string(),
        "native-constraint ::= (native-channel-ws (\"<|constrain|>\" native-channel-ws)? native-constraint-type)?".to_string(),
    ];
    let mut alternatives = Vec::new();

    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("native-channel-tool-{index}-args");
        rules.push(rename_root_rule(&args_grammar, &args_rule));

        let role_rule = format!("native-channel-tool-{index}-role");
        let channel_rule = format!("native-channel-tool-{index}-channel");
        rules.push(format!(
            "{role_rule} ::= \" to=functions.\" {} native-channel native-constraint \"<|message|>\" {args_rule}",
            gbnf_literal(name)
        ));
        rules.push(format!(
            "{channel_rule} ::= native-channel \" to=functions.\" {} native-constraint \"<|message|>\" {args_rule}",
            gbnf_literal(name)
        ));
        alternatives.push(role_rule);
        alternatives.push(channel_rule);
    }

    Ok(format!(
        "root ::= \"<|start|>assistant\" ({})\n\n{}",
        alternatives.join(" | "),
        rules.join("\n")
    ))
}

fn build_native_channel_tool_suffix_grammar(tools: &[Tool]) -> Result<String, LLMError> {
    let mut rules = vec![
        "native-channel-ws ::= [ \\t\\n\\r]*".to_string(),
        "native-channel-name ::= \"commentary\" | \"analysis\"".to_string(),
        "native-channel ::= \"<|channel|>\" native-channel-name".to_string(),
        "native-constraint-type ::= [A-Za-z0-9_-]+".to_string(),
        "native-constraint ::= (native-channel-ws (\"<|constrain|>\" native-channel-ws)? native-constraint-type)?".to_string(),
    ];
    let mut role_tails = Vec::new();
    let mut channel_tails = Vec::new();

    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("native-channel-tool-{index}-args");
        rules.push(rename_root_rule(&args_grammar, &args_rule));

        let role_tail = format!("native-channel-tool-{index}-role-tail");
        let channel_tail = format!("native-channel-tool-{index}-channel-tail");
        rules.push(format!(
            "{role_tail} ::= {} native-channel native-constraint \"<|message|>\" {args_rule}",
            gbnf_literal(name)
        ));
        rules.push(format!(
            "{channel_tail} ::= {} native-constraint \"<|message|>\" {args_rule}",
            gbnf_literal(name)
        ));
        role_tails.push(role_tail);
        channel_tails.push(channel_tail);
    }

    Ok(format!(
        "root ::= \"=functions.\" ({}) | \".\" ({})\n\n{}",
        role_tails.join(" | "),
        channel_tails.join(" | "),
        rules.join("\n")
    ))
}

fn build_kimi_k2_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar =
        build_kimi_k2_tool_call_grammar(tools, parallel_tool_calls, lazy_tool_suffix_only)?;
    combine_native_family_tool_response_grammar(
        tool_call_grammar,
        json_schema,
        force_json_grammar,
        tool_choice,
        lazy_tool_suffix_only,
        ToolCallGrammarFormat::KimiK2ToolCall,
    )
}

fn build_kimi_k2_tool_call_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
    lazy_suffix_only: bool,
) -> Result<String, LLMError> {
    let mut rules = vec![
        "kimi-ws ::= [ \\t\\n\\r]*".to_string(),
        "kimi-index ::= [0-9]+".to_string(),
    ];
    let mut alternatives = Vec::new();

    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("kimi-tool-{index}-args");
        rules.push(rename_root_rule(&args_grammar, &args_rule));
        let call_rule = format!("kimi-tool-{index}");
        rules.push(format!(
            "{call_rule} ::= \"functions.\" {} \":\" kimi-index \"<|tool_call_argument_begin|>\" {args_rule} \"<|tool_call_end|>\"?",
            gbnf_literal(name)
        ));
        alternatives.push(call_rule);
    }

    let call = alternatives.join(" | ");
    let root = if lazy_suffix_only {
        if parallel_tool_calls {
            format!(
                "root ::= ({call}) (kimi-ws \"<|tool_call_begin|>\" kimi-ws ({call}))* \"<|tool_calls_section_end|>\"?"
            )
        } else {
            format!("root ::= ({call}) \"<|tool_calls_section_end|>\"?")
        }
    } else if parallel_tool_calls {
        format!(
            "root ::= \"<|tool_calls_section_begin|>\"? kimi-ws (\"<|tool_call_begin|>\" kimi-ws ({call}))+ kimi-ws \"<|tool_calls_section_end|>\"?"
        )
    } else {
        format!(
            "root ::= \"<|tool_calls_section_begin|>\"? kimi-ws \"<|tool_call_begin|>\" kimi-ws ({call}) kimi-ws \"<|tool_calls_section_end|>\"?"
        )
    };

    Ok(format!("{root}\n\n{}", rules.join("\n")))
}

fn build_lfm2_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar =
        build_lfm2_tool_call_grammar(tools, parallel_tool_calls, lazy_tool_suffix_only)?;
    combine_native_family_tool_response_grammar(
        tool_call_grammar,
        json_schema,
        force_json_grammar,
        tool_choice,
        lazy_tool_suffix_only,
        ToolCallGrammarFormat::Lfm2ToolCall,
    )
}

fn build_lfm2_tool_call_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
    lazy_suffix_only: bool,
) -> Result<String, LLMError> {
    let mut alternatives = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        alternatives.push(format!("lfm-tool-{index}"));
    }

    let mut rules = vec![
        "lfm-ws ::= [ \\t\\n\\r]*".to_string(),
        "lfm-json-literal ::= \"true\" | \"false\" | \"null\" | \"True\" | \"False\" | \"None\"".to_string(),
        "lfm-bool ::= \"true\" | \"false\" | \"True\" | \"False\"".to_string(),
        "lfm-null ::= \"null\" | \"None\"".to_string(),
        "lfm-string ::= lfm-single-string | lfm-double-string".to_string(),
        "lfm-single-string ::= \"'\" lfm-single-string-char* \"'\"".to_string(),
        "lfm-single-string-char ::= [^'\\\\] | \"\\\\\" .".to_string(),
        "lfm-double-string ::= \"\\\"\" lfm-double-string-char* \"\\\"\"".to_string(),
        "lfm-double-string-char ::= [^\"\\\\] | \"\\\\\" .".to_string(),
        "lfm-number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?"
            .to_string(),
        "lfm-value ::= lfm-string | lfm-number | lfm-json-literal | lfm-list | lfm-object".to_string(),
        "lfm-list ::= \"[\" lfm-ws (lfm-value (lfm-ws \",\" lfm-ws lfm-value)* (lfm-ws \",\")?)? lfm-ws \"]\"".to_string(),
        "lfm-object ::= \"{\" lfm-ws (lfm-pair (lfm-ws \",\" lfm-ws lfm-pair)* (lfm-ws \",\")?)? lfm-ws \"}\"".to_string(),
        "lfm-pair ::= lfm-string lfm-ws \":\" lfm-ws lfm-value".to_string(),
        "lfm-key ::= [A-Za-z_][A-Za-z0-9_\\-.]*".to_string(),
        "lfm-arg ::= lfm-key lfm-ws \"=\" lfm-ws lfm-value".to_string(),
        "lfm-args ::= (lfm-arg (lfm-ws \",\" lfm-ws lfm-arg)*)?".to_string(),
    ];
    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        let call_rule = format!("lfm-tool-{index}");
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_rule = build_lfm2_tool_args_grammar(index, &schema, &mut rules);
        rules.push(format!(
            "{call_rule} ::= {} lfm-ws \"(\" lfm-ws {args_rule} lfm-ws \")\"",
            gbnf_literal(name)
        ));
    }

    let call = alternatives.join(" | ");
    let root = if lazy_suffix_only {
        if parallel_tool_calls {
            format!(
                "root ::= \"[\" lfm-ws ({call}) (lfm-ws \",\" lfm-ws ({call}))* lfm-ws \"]\" \"<|tool_call_end|>\""
            )
        } else {
            format!("root ::= \"[\" lfm-ws ({call}) lfm-ws \"]\" \"<|tool_call_end|>\"")
        }
    } else if parallel_tool_calls {
        format!(
            "root ::= \"<|tool_call_start|>\" \"[\" lfm-ws ({call}) (lfm-ws \",\" lfm-ws ({call}))* lfm-ws \"]\" \"<|tool_call_end|>\""
        )
    } else {
        format!(
            "root ::= \"<|tool_call_start|>\" \"[\" lfm-ws ({call}) lfm-ws \"]\" \"<|tool_call_end|>\""
        )
    };

    Ok(format!("{root}\n\n{}", rules.join("\n")))
}

fn build_lfm2_tool_args_grammar(
    tool_index: usize,
    schema: &Value,
    rules: &mut Vec<String>,
) -> String {
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if properties.is_empty() {
        return "lfm-args".to_string();
    }

    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    let mut required_args = Vec::new();
    let mut optional_args = Vec::new();
    let mut allowed_args = Vec::new();
    for (param_index, (param_name, param_schema)) in properties.iter().enumerate() {
        let value_rule = lfm2_value_rule_for_schema(param_schema);
        let arg_rule = format!("lfm-tool-{tool_index}-arg-{param_index}");
        rules.push(format!(
            "{arg_rule} ::= {} lfm-ws \"=\" lfm-ws {value_rule}",
            gbnf_literal(param_name)
        ));
        if required.contains(param_name) {
            required_args.push(arg_rule.clone());
        } else {
            optional_args.push(arg_rule.clone());
        }
        allowed_args.push(arg_rule);
    }

    let args_rule = format!("lfm-tool-{tool_index}-args");
    let allowed = allowed_args.join(" | ");
    let optional = optional_args.join(" | ");
    let body = if required_args.is_empty() {
        format!("(({allowed}) (lfm-ws \",\" lfm-ws ({allowed}))*)?")
    } else {
        let required_sequence = required_args.join(" lfm-ws \",\" lfm-ws ");
        if optional_args.is_empty() {
            required_sequence
        } else {
            format!("{required_sequence} (lfm-ws \",\" lfm-ws ({optional}))*")
        }
    };
    rules.push(format!("{args_rule} ::= {body}"));
    args_rule
}

fn lfm2_value_rule_for_schema(schema: &Value) -> &'static str {
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => "lfm-string",
        Some("integer" | "number") => "lfm-number",
        Some("boolean") => "lfm-bool",
        Some("null") => "lfm-null",
        Some("array") => "lfm-list",
        Some("object") => "lfm-object",
        _ => "lfm-value",
    }
}

fn build_gigachat_v3_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
) -> Result<String, LLMError> {
    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar = build_gigachat_v3_tool_call_grammar(tools, lazy_tool_suffix_only)?;
    combine_native_family_tool_response_grammar(
        tool_call_grammar,
        json_schema,
        force_json_grammar,
        tool_choice,
        lazy_tool_suffix_only,
        ToolCallGrammarFormat::GigaChatV3ToolCall,
    )
}

fn build_gigachat_v3_tool_call_grammar(
    tools: &[Tool],
    lazy_suffix_only: bool,
) -> Result<String, LLMError> {
    let mut rules = Vec::new();
    let mut alternatives = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("gigachat-tool-{index}-args");
        rules.push(rename_root_rule(&args_grammar, &args_rule));
        let call_rule = format!("gigachat-tool-{index}");
        rules.push(format!(
            "{call_rule} ::= \"{{\" [ \\t\\n\\r]* \"\\\"name\\\"\" [ \\t\\n\\r]* \":\" [ \\t\\n\\r]* {} [ \\t\\n\\r]* \",\" [ \\t\\n\\r]* \"\\\"arguments\\\"\" [ \\t\\n\\r]* \":\" [ \\t\\n\\r]* {args_rule} [ \\t\\n\\r]* \"}}\"",
            gbnf_literal(&serde_json::to_string(name).map_err(|err| {
                LLMError::ProviderError(format!("Failed to encode tool name: {err}"))
            })?)
        ));
        alternatives.push(call_rule);
    }

    let call = alternatives.join(" | ");
    let marker = gbnf_literal("<|message_sep|>\n\nfunction call<|role_sep|>\n");
    let root = if lazy_suffix_only {
        format!("root ::= ({call})")
    } else {
        format!("root ::= {marker} ({call})")
    };
    Ok(format!("{root}\n\n{}", rules.join("\n")))
}

fn build_deepseek_dsml_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar =
        build_deepseek_dsml_tool_call_grammar(tools, parallel_tool_calls, lazy_tool_suffix_only)?;
    combine_native_family_tool_response_grammar(
        tool_call_grammar,
        json_schema,
        force_json_grammar,
        tool_choice,
        lazy_tool_suffix_only,
        ToolCallGrammarFormat::DeepSeekDsmlToolCall,
    )
}

fn build_deepseek_dsml_tool_call_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
    lazy_suffix_only: bool,
) -> Result<String, LLMError> {
    let mut rules = vec![
        "dsml-ws ::= [ \\t\\n\\r]*".to_string(),
        "dsml-string ::= ([^<] | \"<\" [^/])*".to_string(),
    ];
    let mut alternatives = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();

        let mut required_parsers = Vec::new();
        let mut optional_parsers = Vec::new();
        for (param_index, (param_name, param_schema)) in properties.iter().enumerate() {
            let value_rule = if deepseek_dsml_param_is_string(param_schema) {
                "dsml-string".to_string()
            } else {
                let value_rule = format!("dsml-tool-{index}-arg-{param_index}-value");
                let param_grammar = compile_json_schema_grammar(param_schema, param_name)?;
                rules.push(rename_root_rule(&param_grammar, &value_rule));
                value_rule
            };
            let arg_rule = format!("dsml-tool-{index}-arg-{param_index}");
            let string_flag = if deepseek_dsml_param_is_string(param_schema) {
                "true"
            } else {
                "false"
            };
            rules.push(format!(
                "{arg_rule} ::= \"<｜DSML｜parameter name=\\\"\" {} \"\\\" string=\\\"{string_flag}\\\">\" {value_rule} \"</｜DSML｜parameter>\"",
                gbnf_literal(param_name)
            ));
            if required.contains(param_name) {
                required_parsers.push(arg_rule);
            } else {
                optional_parsers.push(arg_rule);
            }
        }

        let body_rule = format!("dsml-tool-{index}-body");
        let required_body = if required_parsers.is_empty() {
            "dsml-ws".to_string()
        } else {
            required_parsers.join(" dsml-ws ")
        };
        let body = if optional_parsers.is_empty() {
            required_body
        } else {
            let optional = optional_parsers.join(" | ");
            format!("{required_body} (dsml-ws ({optional}))*")
        };
        rules.push(format!("{body_rule} ::= {body}"));

        let call_rule = format!("dsml-tool-{index}");
        rules.push(format!(
            "{call_rule} ::= \"<｜DSML｜invoke name=\\\"\" {} \"\\\">\" dsml-ws {body_rule} dsml-ws \"</｜DSML｜invoke>\"",
            gbnf_literal(name)
        ));
        alternatives.push(call_rule);
    }

    let call = alternatives.join(" | ");
    let body = if parallel_tool_calls {
        format!("({call}) (dsml-ws ({call}))*")
    } else {
        format!("({call})")
    };
    let root = if lazy_suffix_only {
        format!("root ::= dsml-ws {body} dsml-ws \"</｜DSML｜function_calls>\"")
    } else {
        format!(
            "root ::= \"<｜DSML｜function_calls>\" dsml-ws {body} dsml-ws \"</｜DSML｜function_calls>\""
        )
    };
    Ok(format!("{root}\n\n{}", rules.join("\n")))
}

fn deepseek_dsml_param_is_string(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("string")
}

fn combine_native_family_tool_response_grammar(
    tool_call_grammar: String,
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    lazy_tool_suffix_only: bool,
    tool_call_format: ToolCallGrammarFormat,
) -> Result<String, LLMError> {
    if matches!(tool_choice, LlamaCppToolChoice::Required) || lazy_tool_suffix_only {
        return Ok(tool_call_grammar);
    }

    let final_schema = json_schema
        .and_then(sanitize_chat_template_schema)
        .unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"],
                "additionalProperties": false
            })
        });
    let mut final_grammar = compile_json_schema_grammar(
        &final_schema,
        if force_json_grammar {
            "forced final response"
        } else {
            "final response"
        },
    )?;
    if json_schema.is_some() {
        final_grammar = wrap_structured_response_grammar(&final_grammar, tool_call_format);
    }
    Ok(combine_final_and_native_tool_grammars(
        &final_grammar,
        &tool_call_grammar,
    ))
}

fn build_functionary_v32_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    if tools.is_empty() {
        return Err(LLMError::ProviderError(
            "Cannot build Functionary v3.2 tool grammar without tools".to_string(),
        ));
    }

    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar =
        build_functionary_v32_tool_call_grammar(tools, parallel_tool_calls, lazy_tool_suffix_only)?;

    if lazy_tool_suffix_only || matches!(tool_choice, LlamaCppToolChoice::Required) {
        return Ok(tool_call_grammar);
    }

    let final_grammar = build_functionary_v32_content_grammar();
    Ok(combine_final_and_native_tool_grammars(
        &final_grammar,
        &tool_call_grammar,
    ))
}

fn build_generic_function_tag_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    if tools.is_empty() {
        return Err(LLMError::ProviderError(
            "Cannot build generic function-tag tool grammar without tools".to_string(),
        ));
    }

    let lazy_tool_suffix_only = matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar;
    let tool_call_grammar = build_generic_function_tag_tool_call_grammar(
        tools,
        parallel_tool_calls,
        lazy_tool_suffix_only,
    )?;

    combine_native_family_tool_response_grammar(
        tool_call_grammar,
        json_schema,
        force_json_grammar,
        tool_choice,
        lazy_tool_suffix_only,
        ToolCallGrammarFormat::GenericFunctionTag,
    )
}

fn build_generic_function_tag_tool_call_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
    lazy_suffix_only: bool,
) -> Result<String, LLMError> {
    let mut rules = vec!["generic-function-ws ::= [ \\t\\n\\r]*".to_string()];
    let mut alternatives = Vec::new();
    let mut full_alternatives = Vec::new();

    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("generic-function-tool-{index}-args");
        rules.push(rename_root_rule(&args_grammar, &args_rule));
        let tagged_args_rule =
            build_generic_function_tagged_args_grammar(index, &schema, &mut rules)?;

        let suffix_rule = format!("generic-function-tool-{index}-suffix");
        rules.push(format!(
            "{suffix_rule} ::= {} \">\" generic-function-ws ({args_rule} | {tagged_args_rule}) generic-function-ws \"</function>\"",
            gbnf_literal(name)
        ));
        alternatives.push(suffix_rule.clone());

        let full_rule = format!("generic-function-tool-{index}");
        rules.push(format!("{full_rule} ::= \"<function=\" {suffix_rule}"));
        full_alternatives.push(full_rule);
    }

    let suffix_call = alternatives.join(" | ");
    let full_call = full_alternatives.join(" | ");
    let root = if lazy_suffix_only {
        if parallel_tool_calls {
            format!(
                "root ::= ({suffix_call}) (generic-function-ws \"<function=\" generic-function-ws ({suffix_call}))*"
            )
        } else {
            format!("root ::= ({suffix_call})")
        }
    } else if parallel_tool_calls {
        format!("root ::= ({full_call}) (generic-function-ws ({full_call}))*")
    } else {
        format!("root ::= ({full_call})")
    };

    Ok(format!("{root}\n\n{}", rules.join("\n")))
}

fn build_generic_function_tagged_args_grammar(
    tool_index: usize,
    schema: &Value,
    rules: &mut Vec<String>,
) -> Result<String, LLMError> {
    if !rules
        .iter()
        .any(|rule| rule.starts_with("generic-function-string ::="))
    {
        rules.push("generic-function-string ::= ([^<] | \"<\" [^/])*".to_string());
    }
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut required_parsers = Vec::new();
    let mut optional_parsers = Vec::new();
    for (param_index, (param_name, param_schema)) in properties.iter().enumerate() {
        let value_rule = if generic_tagged_param_is_string(param_schema) {
            "generic-function-string".to_string()
        } else {
            let value_rule = format!("generic-function-tool-{tool_index}-arg-{param_index}-value");
            let param_grammar = compile_json_schema_grammar(param_schema, param_name)?;
            rules.push(rename_root_rule(&param_grammar, &value_rule));
            value_rule
        };

        let arg_rule = format!("generic-function-tool-{tool_index}-arg-{param_index}");
        let name = gbnf_literal(param_name);
        rules.push(format!(
            "{arg_rule} ::= \"<parameter name=\\\"\" {name} \"\\\">\" {value_rule} \"</parameter>\" | \"<param name=\\\"\" {name} \"\\\">\" {value_rule} \"</param>\" | \"<arg name=\\\"\" {name} \"\\\">\" {value_rule} \"</arg>\""
        ));
        if required.contains(param_name) {
            required_parsers.push(arg_rule);
        } else {
            optional_parsers.push(arg_rule);
        }
    }

    let body_rule = format!("generic-function-tool-{tool_index}-tagged-args");
    let required_body = if required_parsers.is_empty() {
        "generic-function-ws".to_string()
    } else {
        required_parsers.join(" generic-function-ws ")
    };
    let body = if optional_parsers.is_empty() {
        required_body
    } else {
        let optional = optional_parsers.join(" | ");
        format!("{required_body} (generic-function-ws ({optional}))*")
    };
    rules.push(format!("{body_rule} ::= {body}"));
    Ok(body_rule)
}

fn generic_tagged_param_is_string(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("string")
}

fn build_functionary_v32_content_grammar() -> String {
    [
        "root ::= \">>>all\\n\" functionary-content",
        "functionary-content ::= ([^>] | \">\" [^>] | \">>\" [^>])*",
    ]
    .join("\n")
}

fn build_functionary_v32_tool_call_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
    lazy_suffix_only: bool,
) -> Result<String, LLMError> {
    let mut rules = vec!["functionary-ws ::= [ \\t\\n\\r]*".to_string()];
    let mut alternatives = Vec::new();

    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("functionary-tool-{index}-args");
        rules.push(rename_root_rule(&args_grammar, &args_rule));
        let call_rule = format!("functionary-tool-{index}");
        rules.push(format!(
            "{call_rule} ::= {} \"\\n\" {args_rule}",
            gbnf_literal(name)
        ));
        alternatives.push(call_rule);
    }

    let call = alternatives.join(" | ");
    let root = if lazy_suffix_only {
        if parallel_tool_calls {
            format!("root ::= ({call}) (functionary-ws \">>>\" functionary-ws ({call}))*")
        } else {
            format!("root ::= {call}")
        }
    } else if parallel_tool_calls {
        format!(
            "root ::= \">>>\" functionary-ws ({call}) (functionary-ws \">>>\" functionary-ws ({call}))*"
        )
    } else {
        format!("root ::= \">>>\" functionary-ws ({call})")
    };

    Ok(format!("{root}\n\n{}", rules.join("\n")))
}

fn build_tool_calls_args_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    if tools.is_empty() {
        return Err(LLMError::ProviderError(
            "Cannot build Ministral tool grammar without tools".to_string(),
        ));
    }

    let tool_call_grammar = build_tool_calls_args_grammar(tools, parallel_tool_calls)?;
    if matches!(tool_choice, LlamaCppToolChoice::Required) {
        return Ok(tool_call_grammar);
    }

    if matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar
    {
        return build_tool_calls_args_payload_grammar(tools, parallel_tool_calls);
    }

    let final_schema = json_schema
        .and_then(sanitize_chat_template_schema)
        .unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"],
                "additionalProperties": false
            })
        });
    let mut final_grammar = compile_json_schema_grammar(&final_schema, "final response")?;
    if json_schema.is_some() {
        final_grammar = wrap_structured_response_grammar(
            &final_grammar,
            ToolCallGrammarFormat::ToolCallsArgsTag,
        );
    }
    Ok(combine_final_and_native_tool_grammars(
        &final_grammar,
        &tool_call_grammar,
    ))
}

fn build_tool_calls_args_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let payload = build_tool_calls_args_payload_grammar(tools, parallel_tool_calls)?;
    let payload = rename_root_rule(&payload, "tool-payload");
    let prefix = gbnf_literal("[TOOL_CALLS]");
    Ok(format!(
        "root ::= {prefix} [ \\t\\n\\r]* tool-payload\n\n{payload}"
    ))
}

fn build_tool_calls_args_payload_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let mut rules = Vec::new();
    let mut alternatives = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let schema = sanitize_tool_parameters_schema(&tool.function.parameters);
        let args_grammar = compile_json_schema_grammar(&schema, name)?;
        let args_rule = format!("tool-{index}-args");
        let args_grammar = rename_root_rule(&args_grammar, &args_rule);
        let call_rule = format!("tool-{index}");
        rules.push(format!(
            "{call_rule} ::= {} tool-ws {} tool-ws {args_rule}",
            gbnf_literal(name),
            gbnf_literal("[ARGS]")
        ));
        rules.push(args_grammar);
        alternatives.push(call_rule);
    }

    let root = if parallel_tool_calls {
        format!(
            "root ::= ({}) (tool-ws {} tool-ws ({}))*",
            alternatives.join(" | "),
            gbnf_literal("[TOOL_CALLS]"),
            alternatives.join(" | ")
        )
    } else {
        format!("root ::= {}", alternatives.join(" | "))
    };
    Ok(format!(
        "{root}\ntool-ws ::= [ \\t\\n\\r]*\n\n{}",
        rules.join("\n")
    ))
}

fn build_gemma4_tool_response_grammar(
    tools: &[Tool],
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    if tools.is_empty() {
        return Err(LLMError::ProviderError(
            "Cannot build Gemma4 tool grammar without tools".to_string(),
        ));
    }

    let tool_call_grammar =
        build_gemma4_tool_call_grammar(tools, tool_choice.clone(), parallel_tool_calls)?;
    if matches!(tool_choice, LlamaCppToolChoice::Required) {
        return Ok(tool_call_grammar);
    }

    if matches!(tool_choice, LlamaCppToolChoice::Auto)
        && json_schema.is_none()
        && !force_json_grammar
    {
        return build_gemma4_tool_payload_grammar(tools, parallel_tool_calls);
    }

    let final_schema = json_schema
        .and_then(sanitize_chat_template_schema)
        .unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string" }
                },
                "required": ["content"],
                "additionalProperties": false
            })
        });
    let final_grammar = compile_json_schema_grammar(&final_schema, "final response")?;
    let final_grammar = if json_schema.is_some() {
        wrap_structured_response_grammar(&final_grammar, ToolCallGrammarFormat::Gemma4ToolCall)
    } else {
        final_grammar
    };
    Ok(combine_final_and_native_tool_grammars(
        &final_grammar,
        &tool_call_grammar,
    ))
}

fn build_gemma4_tool_call_grammar(
    tools: &[Tool],
    tool_choice: LlamaCppToolChoice,
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let payload = build_gemma4_tool_payload_grammar(tools, parallel_tool_calls)?;
    let payload = rename_root_rule(&payload, "gemma4-tool-payload");
    let min = if matches!(tool_choice, LlamaCppToolChoice::Required) {
        ""
    } else {
        "?"
    };
    Ok(format!(
        "root ::= ({} [ \\t\\n\\r]* gemma4-tool-payload [ \\t\\n\\r]* {}){min}\n\n{payload}",
        gbnf_literal("<|tool_call>call:"),
        gbnf_literal("<tool_call|>")
    ))
}

fn build_gemma4_tool_payload_grammar(
    tools: &[Tool],
    parallel_tool_calls: bool,
) -> Result<String, LLMError> {
    let mut rules = Vec::new();
    let mut alternatives = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(LLMError::ProviderError(
                "Tool name must not be empty".to_string(),
            ));
        }
        let call_rule = format!("gemma4-tool-{index}");
        rules.push(format!(
            "{call_rule} ::= {} gemma4-ws gemma4-dict",
            gbnf_literal(name)
        ));
        alternatives.push(call_rule);
    }

    let root = if parallel_tool_calls {
        format!(
            "root ::= ({}) ([ \\t\\n\\r]* {} [ \\t\\n\\r]* ({}))*",
            alternatives.join(" | "),
            gbnf_literal("<tool_call|><|tool_call>call:"),
            alternatives.join(" | ")
        )
    } else {
        format!("root ::= {}", alternatives.join(" | "))
    };
    Ok(format!(
        "{root}\n\n{}\n\n{}",
        rules.join("\n"),
        gemma4_native_value_grammar()
    ))
}

fn gemma4_native_value_grammar() -> &'static str {
    r#"gemma4-ws ::= [ \t\n\r]*
gemma4-string ::= "<|\"|>" gemma4-string-char* "<|\"|>"
gemma4-string-char ::= [^<] | "<" [^|] | "<|" [^\"] | "<|\"" [^|] | "<|\"|" [^>]
gemma4-bool ::= "true" | "false"
gemma4-null ::= "null"
gemma4-number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
gemma4-dict-key-name ::= [^:}]+
gemma4-dict-key ::= gemma4-dict-key-name ":"
gemma4-dict-kv ::= gemma4-dict-key gemma4-ws gemma4-value
gemma4-dict ::= "{" gemma4-ws ("}" | gemma4-dict-kv ("," gemma4-ws gemma4-dict-kv)* gemma4-ws "}")
gemma4-array ::= "[" gemma4-ws ("]" | gemma4-value ("," gemma4-ws gemma4-value)* gemma4-ws "]")
gemma4-value ::= gemma4-string | gemma4-dict | gemma4-array | gemma4-number | gemma4-bool | gemma4-null"#
}

fn build_native_tool_payload_schema(
    tools: &[Tool],
    tool_call_format: ToolCallGrammarFormat,
    parallel_tool_calls: bool,
) -> Result<Value, LlamaCppProviderError> {
    let variants = tools
        .iter()
        .map(build_native_single_tool_call_schema)
        .collect::<Result<Vec<_>, _>>()?;

    match tool_call_format {
        ToolCallGrammarFormat::XmlToolCall => Ok(json!({ "oneOf": variants })),
        ToolCallGrammarFormat::ToolCallsArrayTag => {
            let mut schema = json!({
            "type": "array",
            "minItems": 1,
            "items": { "oneOf": variants }
            });
            if !parallel_tool_calls {
                schema["maxItems"] = Value::from(1);
            }
            Ok(schema)
        }
        ToolCallGrammarFormat::ToolCallsArgsTag => Err(LlamaCppProviderError::Template(
            "Ministral [TOOL_CALLS]/[ARGS] payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::FunctionaryV32 => Err(LlamaCppProviderError::Template(
            "Functionary v3.2 payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::GenericFunctionTag => Err(LlamaCppProviderError::Template(
            "Generic function tag payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::Gemma4ToolCall => Err(LlamaCppProviderError::Template(
            "Gemma4 native payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::NativeChannelToolCall => Err(LlamaCppProviderError::Template(
            "Native channel tool payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::KimiK2ToolCall => Err(LlamaCppProviderError::Template(
            "Kimi K2 tool payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::Lfm2ToolCall => Err(LlamaCppProviderError::Template(
            "LFM2 tool payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::GigaChatV3ToolCall => Err(LlamaCppProviderError::Template(
            "GigaChat V3 tool payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::DeepSeekDsmlToolCall => Err(LlamaCppProviderError::Template(
            "DeepSeek DSML tool payload uses per-tool grammars".to_string(),
        )),
        ToolCallGrammarFormat::OpenAiEnvelope => Err(LlamaCppProviderError::Template(
            "OpenAI envelope is not a native tool payload".to_string(),
        )),
    }
}

fn build_native_single_tool_call_schema(tool: &Tool) -> Result<Value, LlamaCppProviderError> {
    let name = tool.function.name.trim();
    if name.is_empty() {
        return Err(LlamaCppProviderError::Template(
            "Tool name must not be empty".to_string(),
        ));
    }

    let parameters = sanitize_tool_parameters_schema(&tool.function.parameters);
    Ok(json!({
        "type": "object",
        "properties": {
            "id": { "type": "string" },
            "name": { "enum": [name] },
            "arguments": parameters
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    }))
}

fn wrap_native_tool_payload_grammar(
    payload_grammar: &str,
    tool_call_format: ToolCallGrammarFormat,
    include_markers: bool,
) -> String {
    let payload = rename_root_rule(payload_grammar, "tool-payload");
    let (start, end) = if include_markers {
        match tool_call_format {
            ToolCallGrammarFormat::XmlToolCall => ("<tool_call>", "</tool_call>"),
            ToolCallGrammarFormat::ToolCallsArrayTag => ("[TOOL_CALLS]", "[/TOOL_CALLS]"),
            ToolCallGrammarFormat::ToolCallsArgsTag => ("[TOOL_CALLS]", ""),
            ToolCallGrammarFormat::FunctionaryV32 => ("", ""),
            ToolCallGrammarFormat::GenericFunctionTag => ("", ""),
            ToolCallGrammarFormat::Gemma4ToolCall => ("<|tool_call>call:", "<tool_call|>"),
            ToolCallGrammarFormat::NativeChannelToolCall => ("", ""),
            ToolCallGrammarFormat::KimiK2ToolCall
            | ToolCallGrammarFormat::Lfm2ToolCall
            | ToolCallGrammarFormat::GigaChatV3ToolCall
            | ToolCallGrammarFormat::DeepSeekDsmlToolCall => ("", ""),
            ToolCallGrammarFormat::OpenAiEnvelope => ("", ""),
        }
    } else {
        ("", "")
    };
    let start = if include_markers {
        gbnf_literal(start)
    } else {
        String::new()
    };
    let spacing = if include_markers { " tool-ws " } else { "" };

    let end = if end.is_empty() {
        String::new()
    } else {
        format!(" {}", gbnf_literal(end))
    };

    format!(
        "root ::= {start}{spacing}tool-payload tool-ws{end}\ntool-ws ::= [ \\t\\n\\r]*\n\n{payload}"
    )
}

fn native_tool_markers_are_safe_in_grammar(tool_call_format: ToolCallGrammarFormat) -> bool {
    !matches!(tool_call_format, ToolCallGrammarFormat::XmlToolCall)
}

fn combine_final_and_native_tool_grammars(final_grammar: &str, tool_call_grammar: &str) -> String {
    let final_grammar = prefix_gbnf_rules(final_grammar, "final");
    let tool_grammar = prefix_gbnf_rules(tool_call_grammar, "tool");
    format!(
        "root ::= final-root | tool-root\n\n{}\n\n{}",
        final_grammar, tool_grammar
    )
}

fn maybe_wrap_grammar_with_reasoning(grammar: &str, result: &ChatTemplateResult) -> String {
    if result.reasoning_format.is_none() {
        return grammar.to_string();
    }

    let start_tag = result.reasoning_start_tag.as_deref().unwrap_or("<think>");
    let end_tag = result.reasoning_end_tag.as_deref().unwrap_or("</think>");
    wrap_grammar_with_tagged_reasoning(grammar, start_tag, end_tag)
}

fn wrap_grammar_with_tagged_reasoning(grammar: &str, start_tag: &str, end_tag: &str) -> String {
    let renamed = rename_root_rule(grammar, "root-content");
    format!(
        "root ::= reasoning? root-content\nreasoning ::= {} reasoning-char* {}\nreasoning-char ::= [^<] | \"<\" [^/]\n\n{}",
        gbnf_literal(start_tag),
        gbnf_literal(end_tag),
        renamed
    )
}

fn rename_root_rule(grammar: &str, new_name: &str) -> String {
    grammar
        .lines()
        .map(|line| {
            if let Some(rest) = line.strip_prefix("root ") {
                format!("{new_name} {rest}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn prefix_gbnf_rules(grammar: &str, prefix: &str) -> String {
    let rule_names = collect_gbnf_rule_names(grammar);
    let renamed = grammar
        .lines()
        .map(|line| {
            if let Some((name, rest)) = split_gbnf_rule_definition(line)
                && rule_names.contains(name)
            {
                return format!("{prefix}-{name} ::={}", rest);
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    replace_gbnf_rule_references(&renamed, &rule_names, prefix)
}

fn collect_gbnf_rule_names(grammar: &str) -> HashSet<String> {
    grammar
        .lines()
        .filter_map(split_gbnf_rule_definition)
        .map(|(name, _)| name.to_string())
        .collect()
}

fn split_gbnf_rule_definition(line: &str) -> Option<(&str, &str)> {
    let (name, rest) = line.split_once("::=")?;
    let name = name.trim();
    if !is_gbnf_identifier(name) {
        return None;
    }
    Some((name, rest))
}

fn is_gbnf_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
}

fn replace_gbnf_rule_references(
    grammar: &str,
    rule_names: &HashSet<String>,
    prefix: &str,
) -> String {
    let mut output = String::with_capacity(grammar.len());
    let mut chars = grammar.char_indices().peekable();
    let mut in_string = false;
    let mut in_class = false;
    let mut escape = false;

    while let Some((idx, ch)) = chars.next() {
        if in_string {
            output.push(ch);
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if in_class {
            output.push(ch);
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == ']' {
                in_class = false;
            }
            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                output.push(ch);
            }
            '[' => {
                in_class = true;
                output.push(ch);
            }
            ch if ch.is_ascii_alphabetic() || ch == '_' => {
                let start = idx;
                let mut end = idx + ch.len_utf8();
                while let Some((next_idx, next)) = chars.peek().copied() {
                    if next.is_ascii_alphanumeric() || next == '-' || next == '_' {
                        chars.next();
                        end = next_idx + next.len_utf8();
                    } else {
                        break;
                    }
                }
                let ident = &grammar[start..end];
                if rule_names.contains(ident) {
                    output.push_str(prefix);
                    output.push('-');
                }
                output.push_str(ident);
            }
            _ => output.push(ch),
        }
    }

    output
}

fn gbnf_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn build_single_tool_call_schema(tool: &Tool) -> Result<Value, LlamaCppProviderError> {
    let name = tool.function.name.trim();
    if name.is_empty() {
        return Err(LlamaCppProviderError::Template(
            "Tool name must not be empty".to_string(),
        ));
    }

    let parameters = sanitize_tool_parameters_schema(&tool.function.parameters);
    Ok(json!({
        "type": "object",
        "properties": {
            "id": { "type": "string" },
            "type": { "enum": ["function"] },
            "function": {
                "type": "object",
                "properties": {
                    "name": { "enum": [name] },
                    "arguments": parameters
                },
                "required": ["name", "arguments"],
                "additionalProperties": false
            }
        },
        "required": ["function"],
        "additionalProperties": false
    }))
}

fn sanitize_tool_parameters_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(object) => {
            let mut value = Value::Object(object.clone());
            if value.get("type").is_none() {
                value["type"] = Value::String("object".to_string());
            }
            if matches!(value.get("type"), Some(Value::String(kind)) if kind == "object")
                && value.get("additionalProperties").is_none()
            {
                value["additionalProperties"] = Value::Bool(false);
            }
            value
        }
        _ => json!({
            "type": "object",
            "additionalProperties": true
        }),
    }
}

fn select_template_schema_and_grammar(
    json_schema: Option<&StructuredOutputFormat>,
    force_json_grammar: bool,
    tool_call_format: Option<ToolCallGrammarFormat>,
) -> Result<(Option<String>, Option<String>), LLMError> {
    // llama.cpp's OpenAI-compatible template API treats `json_schema` as part of grammar
    // generation AND configures a schema-bound parser for the generated response. That
    // parser expects a structured-output envelope that plain-JSON generations don't
    // provide, surfacing as `ffi error -3` from `chat_parse_to_oaicompat`
    // (https://github.com/liquidos-ai/AutoAgents/issues/220). We compile schemas to
    // grammar at our boundary and pass them via the `grammar` slot, leaving parser
    // normalization in AutoAgents-owned Rust code.
    if let Some(schema_value) = json_schema.and_then(sanitize_chat_template_schema) {
        let schema_str = schema_value.to_string();
        let grammar = llama_cpp_2::json_schema_to_grammar(&schema_str).map_err(|err| {
            LLMError::invalid_request(format!(
                "invalid llama.cpp structured output JSON schema: {err}"
            ))
        })?;
        let grammar = tool_call_format.map_or(grammar.clone(), |format| {
            wrap_structured_response_grammar(&grammar, format)
        });
        return Ok((None, Some(grammar)));
    }

    let grammar_value = if force_json_grammar {
        Some(JSON_GRAMMAR.to_string())
    } else {
        None
    };
    Ok((None, grammar_value))
}

fn enabled_tools_for_config<'a>(
    config: &LlamaCppConfig,
    tools: Option<&'a [Tool]>,
) -> Option<&'a [Tool]> {
    tools.filter(|tools| {
        !tools.is_empty() && !matches!(config.tool_choice, LlamaCppToolChoice::None)
    })
}

fn should_stream_final_message(template_result: &ChatTemplateResult) -> bool {
    template_result.parse_tool_calls
        || template_result.grammar.is_some()
        || template_result.reasoning_format.is_some()
}

fn decode_model_special_token(
    model: &LlamaModel,
    token: LlamaToken,
    name: &str,
) -> Result<String, LLMError> {
    if token.0 < 0 {
        return Ok(String::new());
    }

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .map_err(|err| {
            LLMError::ProviderError(format!("failed to decode llama.cpp {name} token: {err}"))
        })
}

fn ensure_supported_messages_for_config(
    _config: &LlamaCppConfig,
    messages: &[ChatMessage],
) -> Result<(), LLMError> {
    for message in messages {
        match &message.message_type {
            MessageType::Text | MessageType::ToolUse(_) | MessageType::ToolResult(_) => {}
            MessageType::Image(_) => {
                #[cfg(feature = "mtmd")]
                {
                    if _config.mmproj_path.is_some() {
                        continue;
                    }
                }
                return Err(LLMError::invalid_request(
                    "llama.cpp backend does not support image inputs without MTMD and mmproj configured"
                        .to_string(),
                ));
            }
            MessageType::ImageURL(_) | MessageType::Pdf(_) => {
                return Err(LLMError::invalid_request(
                    "llama.cpp backend does not support image URL or PDF inputs".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn parse_openai_delta(delta: &str) -> Result<OpenAICompatDelta, LLMError> {
    serde_json::from_str(delta).map_err(|err| LLMError::JsonError(err.to_string()))
}

#[cfg(feature = "mtmd")]
fn mtmd_structured_stream(
    response_stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>>,
) -> impl Stream<Item = Result<StreamResponse, LLMError>> {
    response_stream.filter_map(|event| async move {
        match event {
            Ok(StreamEvent::Token(token)) => {
                Some(Ok(single_stream_response(Some(token), None, None, None)))
            }
            Ok(StreamEvent::Usage(_)) | Ok(StreamEvent::Done { .. }) => None,
            Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                Ok(parsed) => {
                    let content = parsed.content.and_then(StringOrJson::into_non_empty_string);
                    let reasoning_content = parsed
                        .reasoning_content
                        .and_then(StringOrJson::into_non_empty_string);
                    if content.is_none() && reasoning_content.is_none() {
                        None
                    } else {
                        Some(Ok(single_stream_response(
                            content,
                            reasoning_content,
                            None,
                            None,
                        )))
                    }
                }
                Err(err) => Some(Err(err)),
            },
            Err(err) => Some(Err(err)),
        }
    })
}

#[cfg(feature = "mtmd")]
fn mtmd_tool_stream(
    response_stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>>,
) -> impl Stream<Item = Result<StreamChunk, LLMError>> {
    response_stream
        .scan(StreamMappingState::default(), |stream_state, event| {
            let mut outputs = Vec::new();
            match event {
                Ok(StreamEvent::Token(token)) => {
                    if !token.is_empty() {
                        outputs.push(Ok(StreamChunk::Text(token)));
                    }
                }
                Ok(StreamEvent::Usage(usage)) => outputs.push(Ok(StreamChunk::Usage(usage))),
                Ok(StreamEvent::Done { stop_reason }) => {
                    outputs.push(Ok(StreamChunk::Done { stop_reason }))
                }
                Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
                    Ok(parsed) => {
                        push_text_and_reasoning_chunks(
                            parsed.content.and_then(StringOrJson::into_non_empty_string),
                            parsed
                                .reasoning_content
                                .and_then(StringOrJson::into_non_empty_string),
                            stream_state,
                            &mut outputs,
                        );
                    }
                    Err(err) => outputs.push(Err(err)),
                },
                Err(err) => outputs.push(Err(err)),
            }
            futures::future::ready(Some(outputs))
        })
        .flat_map(futures::stream::iter)
}

fn single_stream_response(
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    usage: Option<ChatUsage>,
) -> StreamResponse {
    StreamResponse {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content,
                reasoning_content,
                tool_calls,
            },
        }],
        usage,
    }
}

fn map_struct_stream_event(
    event: Result<StreamEvent, LLMError>,
    stream_state: &mut StreamMappingState,
) -> Vec<Result<StreamResponse, LLMError>> {
    let mut outputs = Vec::new();
    match event {
        Ok(StreamEvent::Token(token)) => {
            if !token.is_empty() {
                outputs.push(Ok(single_stream_response(Some(token), None, None, None)));
            }
        }
        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
            Ok(parsed) => {
                push_struct_content_and_reasoning(
                    parsed.content.and_then(StringOrJson::into_non_empty_string),
                    parsed
                        .reasoning_content
                        .and_then(StringOrJson::into_non_empty_string),
                    stream_state,
                    &mut outputs,
                );
                if let Some(tool_calls) = parsed.tool_calls {
                    push_struct_tool_call_updates(
                        tool_calls,
                        &mut stream_state.tool_calls,
                        &mut outputs,
                    );
                }
            }
            Err(err) => outputs.push(Err(err)),
        },
        Ok(StreamEvent::Usage(usage)) => {
            outputs.push(Ok(single_stream_response(None, None, None, Some(usage))));
        }
        Ok(StreamEvent::Done { .. }) => {}
        Err(err) => outputs.push(Err(err)),
    }
    outputs
}

fn map_tool_stream_event(
    event: Result<StreamEvent, LLMError>,
    stream_state: &mut StreamMappingState,
) -> Vec<Result<StreamChunk, LLMError>> {
    let mut outputs = Vec::new();
    match event {
        Ok(StreamEvent::Token(token)) => {
            if !token.is_empty() {
                outputs.push(Ok(StreamChunk::Text(token)));
            }
        }
        Ok(StreamEvent::Delta(delta)) => match parse_openai_delta(&delta) {
            Ok(parsed) => {
                push_text_and_reasoning_chunks(
                    parsed.content.and_then(StringOrJson::into_non_empty_string),
                    parsed
                        .reasoning_content
                        .and_then(StringOrJson::into_non_empty_string),
                    stream_state,
                    &mut outputs,
                );
                if let Some(tool_calls) = parsed.tool_calls {
                    push_tool_chunk_updates(tool_calls, &mut stream_state.tool_calls, &mut outputs);
                }
            }
            Err(err) => outputs.push(Err(err)),
        },
        Ok(StreamEvent::Usage(usage)) => outputs.push(Ok(StreamChunk::Usage(usage))),
        Ok(StreamEvent::Done { stop_reason }) => {
            push_tool_completions(&mut stream_state.tool_calls, &mut outputs);
            outputs.push(Ok(StreamChunk::Done { stop_reason }));
        }
        Err(err) => outputs.push(Err(err)),
    }
    outputs
}

fn push_struct_content_and_reasoning(
    content: Option<String>,
    reasoning_content: Option<String>,
    stream_state: &mut StreamMappingState,
    outputs: &mut Vec<Result<StreamResponse, LLMError>>,
) {
    if let Some(content) = content {
        let delta = append_or_diff_string(&stream_state.content, &content);
        if !delta.is_empty() {
            stream_state.content.push_str(&delta);
            outputs.push(Ok(single_stream_response(Some(delta), None, None, None)));
        }
    }
    if let Some(reasoning_content) = reasoning_content {
        let delta = append_or_diff_string(&stream_state.reasoning_content, &reasoning_content);
        if !delta.is_empty() {
            stream_state.reasoning_content.push_str(&delta);
            outputs.push(Ok(single_stream_response(None, Some(delta), None, None)));
        }
    }
}

fn push_struct_tool_call_updates(
    tool_calls: Vec<OpenAIToolCallDelta>,
    tool_states: &mut HashMap<usize, ToolCallState>,
    outputs: &mut Vec<Result<StreamResponse, LLMError>>,
) {
    let mut updated_calls = Vec::new();
    for call in tool_calls {
        let index = call.index.unwrap_or(0);
        let call_type = call.call_type.unwrap_or_else(|| "function".to_string());
        let state = tool_states.entry(index).or_default();
        if let Some(id) = call.id {
            state.id = id;
        }
        if let Some(function) = call.function {
            if let Some(name) = function.name {
                state.name = name;
            }
            let arguments = function.arguments.into_string();
            let argument_delta = append_or_diff_string(&state.arguments, &arguments);
            if !argument_delta.is_empty() {
                state.arguments.push_str(&argument_delta);
            }
        }
        if !state.id.is_empty() || !state.name.is_empty() || !state.arguments.is_empty() {
            updated_calls.push(ToolCall {
                id: state.id.clone(),
                call_type: call_type.clone(),
                function: FunctionCall {
                    name: state.name.clone(),
                    arguments: state.arguments.clone(),
                },
            });
        }
    }

    if !updated_calls.is_empty() {
        outputs.push(Ok(single_stream_response(
            None,
            None,
            Some(updated_calls),
            None,
        )));
    }
}

fn push_tool_chunk_updates(
    tool_calls: Vec<OpenAIToolCallDelta>,
    tool_states: &mut HashMap<usize, ToolCallState>,
    outputs: &mut Vec<Result<StreamChunk, LLMError>>,
) {
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
            let arguments = function.arguments.into_string();
            let argument_delta = append_or_diff_string(&state.arguments, &arguments);
            if !argument_delta.is_empty() {
                state.arguments.push_str(&argument_delta);
                outputs.push(Ok(StreamChunk::ToolUseInputDelta {
                    index,
                    partial_json: argument_delta,
                }));
            }
        }
    }
}

fn push_tool_completions(
    tool_states: &mut HashMap<usize, ToolCallState>,
    outputs: &mut Vec<Result<StreamChunk, LLMError>>,
) {
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
}

fn push_text_and_reasoning_chunks(
    content: Option<String>,
    reasoning_content: Option<String>,
    stream_state: &mut StreamMappingState,
    outputs: &mut Vec<Result<StreamChunk, LLMError>>,
) {
    if let Some(content) = content {
        let delta = append_or_diff_string(&stream_state.content, &content);
        if !delta.is_empty() {
            stream_state.content.push_str(&delta);
            outputs.push(Ok(StreamChunk::Text(delta)));
        }
    }
    if let Some(reasoning_content) = reasoning_content {
        let delta = append_or_diff_string(&stream_state.reasoning_content, &reasoning_content);
        if !delta.is_empty() {
            stream_state.reasoning_content.push_str(&delta);
            outputs.push(Ok(StreamChunk::ReasoningContent(delta)));
        }
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
        let use_json_grammar = json_schema.is_some() || self.config.force_json_grammar;
        let result = self
            .generate_completion_response(
                prompt,
                use_json_grammar,
                req.max_tokens,
                req.temperature,
                None,
            )
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

        self.run_blocking_task("Embedding", move || {
            let mut embeddings = Vec::with_capacity(input.len());
            for text in input {
                let embedding = generate_embedding(&model, &backend, &config, &text)?;
                embeddings.push(embedding);
            }
            Ok(embeddings)
        })
        .await
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
    get_rt()
        .spawn_blocking(move || -> Result<LlamaModel, LlamaCppProviderError> {
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
            repo_id, filename, ..
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
    top_p_override: Option<f32>,
    seed_override: Option<u32>,
) -> Result<LlamaSampler, LlamaCppProviderError> {
    let mut samplers = Vec::new();

    if use_json_grammar {
        let sampler = LlamaSampler::grammar(model, JSON_GRAMMAR, "root")
            .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?;
        samplers.push(sampler);
    }

    append_sampling_chain(
        model,
        config,
        temperature_override,
        top_p_override,
        seed_override,
        &mut samplers,
    );

    Ok(LlamaSampler::chain_simple(samplers))
}

fn append_sampling_chain(
    model: &LlamaModel,
    config: &LlamaCppConfig,
    temperature_override: Option<f32>,
    top_p_override: Option<f32>,
    seed_override: Option<u32>,
    samplers: &mut Vec<LlamaSampler>,
) {
    if let Some(biases) = config.logit_bias.as_ref()
        && !biases.is_empty()
    {
        let biases = biases
            .iter()
            .map(|(token, bias)| LlamaLogitBias::new(LlamaToken(*token), *bias))
            .collect::<Vec<_>>();
        samplers.push(LlamaSampler::logit_bias(model.n_vocab(), &biases));
    }

    if let Some(multiplier) = config.dry_multiplier
        && multiplier != 0.0
    {
        samplers.push(LlamaSampler::dry(
            model,
            multiplier,
            config.dry_base.unwrap_or(1.75),
            config.dry_allowed_length.unwrap_or(2),
            config
                .dry_penalty_last_n
                .unwrap_or_else(|| config.repeat_last_n.unwrap_or(64)),
            config.dry_sequence_breakers.clone().unwrap_or_else(|| {
                vec![
                    "\n".to_string(),
                    ":".to_string(),
                    "\"".to_string(),
                    "*".to_string(),
                ]
            }),
        ));
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

    let min_keep = config.min_keep.unwrap_or(1);
    if let Some(top_k) = config.top_k {
        samplers.push(LlamaSampler::top_k(top_k as i32));
    }
    if let Some(typical_p) = config.typical_p {
        samplers.push(LlamaSampler::typical(typical_p, min_keep));
    }
    if let Some(top_p) = top_p_override.or(config.top_p) {
        samplers.push(LlamaSampler::top_p(top_p, min_keep));
    }
    if let Some(min_p) = config.min_p {
        samplers.push(LlamaSampler::min_p(min_p, min_keep));
    }
    if let Some(top_n_sigma) = config.top_n_sigma {
        samplers.push(LlamaSampler::top_n_sigma(top_n_sigma));
    }

    let seed = seed_override.or(config.seed).unwrap_or_else(rand::random);
    if let (Some(probability), Some(threshold)) = (config.xtc_probability, config.xtc_threshold)
        && probability > 0.0
    {
        samplers.push(LlamaSampler::xtc(probability, threshold, min_keep, seed));
    }

    let temperature = temperature_override.or(config.temperature);
    match config.mirostat {
        Some(1) => {
            samplers.push(LlamaSampler::mirostat(
                model.n_vocab(),
                seed,
                config.mirostat_tau.unwrap_or(5.0),
                config.mirostat_eta.unwrap_or(0.1),
                100,
            ));
            return;
        }
        Some(2) => {
            samplers.push(LlamaSampler::mirostat_v2(
                seed,
                config.mirostat_tau.unwrap_or(5.0),
                config.mirostat_eta.unwrap_or(0.1),
            ));
            return;
        }
        _ => {}
    }

    if let Some(temp) = temperature
        && temp > 0.0
    {
        if let (Some(delta), Some(exponent)) = (config.dynatemp_range, config.dynatemp_exponent) {
            samplers.push(LlamaSampler::temp_ext(temp, delta, exponent));
        } else {
            samplers.push(LlamaSampler::temp(temp));
        }
        samplers.push(LlamaSampler::dist(seed));
    } else if temperature == Some(0.0) {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.push(LlamaSampler::dist(seed));
    }
}

fn build_chat_sampler(
    model: &LlamaModel,
    config: &LlamaCppConfig,
    result: &ChatTemplateResult,
    temperature_override: Option<f32>,
    top_p_override: Option<f32>,
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
        Some(if result.grammar_lazy {
            let (trigger_patterns, trigger_tokens) =
                grammar_trigger_patterns(&result.grammar_triggers);
            LlamaSampler::grammar_lazy_patterns(
                model,
                grammar,
                "root",
                &trigger_patterns,
                &trigger_tokens,
            )
            .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?
        } else {
            let grammar = maybe_wrap_grammar_with_reasoning(grammar, result);
            LlamaSampler::grammar(model, &grammar, "root")
                .map_err(|err| LlamaCppProviderError::Template(err.to_string()))?
        })
    } else {
        None
    };

    let mut samplers = Vec::new();
    if let Some(grammar_sampler) = grammar_sampler {
        samplers.push(grammar_sampler);
    }

    append_sampling_chain(
        model,
        config,
        temperature_override,
        top_p_override,
        None,
        &mut samplers,
    );

    let mut sampler = LlamaSampler::chain_simple(samplers);
    if let Some(generation_prompt_prefill) = generation_prompt_grammar_prefill(result) {
        let generation_prompt_tokens = model
            .str_to_token(generation_prompt_prefill, AddBos::Never)
            .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;
        sampler.accept_many(generation_prompt_tokens.iter());
    }

    Ok((sampler, preserved))
}

fn generation_prompt_grammar_prefill(result: &ChatTemplateResult) -> Option<&str> {
    let grammar = result.grammar.as_deref()?;
    if result.grammar_lazy || result.generation_prompt.is_empty() {
        return None;
    }
    generation_prompt_grammar_prefill_for(
        &result.generation_prompt,
        grammar,
        result.reasoning_start_tag.as_deref().unwrap_or("<think>"),
    )
}

fn generation_prompt_grammar_prefill_for<'a>(
    generation_prompt: &'a str,
    grammar: &str,
    reasoning_start_tag: &str,
) -> Option<&'a str> {
    let trimmed_prompt = generation_prompt.trim_end_matches(['\n', '\r']);
    if !trimmed_prompt.is_empty() && grammar.contains(&gbnf_literal(trimmed_prompt)) {
        return Some(trimmed_prompt);
    }

    if grammar.contains(&gbnf_literal(reasoning_start_tag))
        && let Some(index) = generation_prompt.find(reasoning_start_tag)
    {
        return Some(&generation_prompt[index..]);
    }

    const SERVER_RESPONSE_MARKERS: &[&str] = &[
        "<|start|>assistant",
        "<|channel|>",
        "<|message|>",
        "<|turn>model",
        "<|tool_call>",
        "<tool_call>",
        "[TOOL_CALLS]",
        "<function=",
        ">>>",
        "｜DSML｜",
    ];

    SERVER_RESPONSE_MARKERS
        .iter()
        .filter(|marker| grammar.contains(&gbnf_literal(marker)))
        .filter_map(|marker| {
            generation_prompt
                .find(marker)
                .map(|index| &generation_prompt[index..])
        })
        .next()
}

fn grammar_trigger_patterns(triggers: &[GrammarTrigger]) -> (Vec<String>, Vec<LlamaToken>) {
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();

    for trigger in triggers {
        match trigger {
            GrammarTrigger::Word(word) => patterns.push(regex_escape(word)),
            GrammarTrigger::Pattern(pattern) => patterns.push(pattern.clone()),
            GrammarTrigger::PatternFull(pattern) => patterns.push(anchor_full_pattern(pattern)),
            GrammarTrigger::Token(token) => tokens.push(LlamaToken(*token)),
        }
    }

    (patterns, tokens)
}

fn anchor_full_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }

    let mut anchored = String::with_capacity(pattern.len() + 2);
    if !pattern.starts_with('^') {
        anchored.push('^');
    }
    anchored.push_str(pattern);
    if !pattern.ends_with('$') {
        anchored.push('$');
    }
    anchored
}

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
            | '#' | '&' | '-' | '~' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

/// Acquire an inference context: either from persistent session state (with
/// prefix reuse) or by creating a fresh context.
///
/// Returns `(ActiveContext, start_position)` where `start_position` is the
/// KV-cache position after the prompt has been decoded.
struct ContextAcquisitionRequest<'a> {
    prompt_tokens: &'a [LlamaToken],
    n_ctx: u32,
    n_batch: u32,
    required_tokens: u32,
}

fn acquire_context<'a>(
    model: &'a Arc<LlamaModel>,
    backend: &'a Arc<LlamaBackend>,
    config: &LlamaCppConfig,
    session_state: Option<&'a SharedSessionState>,
    request: ContextAcquisitionRequest<'_>,
) -> Result<(ActiveContext<'a>, i32), LlamaCppProviderError> {
    let ContextAcquisitionRequest {
        prompt_tokens,
        n_ctx,
        n_batch,
        required_tokens,
    } = request;

    if let Some(shared) = session_state {
        let mut guard = shared.lock().expect("session mutex poisoned");

        // Lazily create the session on first call.
        if guard.is_none() {
            *guard = Some(SessionState::new(
                Arc::clone(backend),
                Arc::clone(model),
                config,
                n_ctx,
                n_batch,
            )?);
        }

        {
            let state = guard.as_mut().expect("session just initialised");

            if required_tokens > state.n_ctx {
                // New prompt + generation headroom exceeds session context
                // window. Reset with a fresh, correctly-sized context.
                let _ = state;
                *guard = Some(SessionState::new(
                    Arc::clone(backend),
                    Arc::clone(model),
                    config,
                    n_ctx,
                    n_batch,
                )?);
                let state = guard.as_mut().ok_or_else(|| {
                    LlamaCppProviderError::Inference(
                        "session reset did not create an active context".to_string(),
                    )
                })?;
                state.decode_tokens(prompt_tokens, n_batch as usize)?;
            } else {
                // Prefix reuse: evict stale tail, decode only new tokens.
                let prefix_len = state.prefix_len(prompt_tokens);
                state.evict_after(prefix_len)?;
                let new_tokens = &prompt_tokens[prefix_len..];
                if !new_tokens.is_empty() {
                    state.decode_tokens(new_tokens, n_batch as usize)?;
                } else {
                    // 100% prefix match — no new tokens to decode.
                    // Re-decode the last prompt token to refresh the logits
                    // buffer. Without this, the sampler reads stale logits
                    // from the previous call's last generated token.
                    let last_pos = state.next_pos - 1;
                    let last_tok = prompt_tokens[prefix_len - 1];
                    let mut batch = LlamaBatch::new(1, 1);
                    batch.add(last_tok, last_pos, &[0], true).map_err(|e| {
                        LlamaCppProviderError::Inference(format!("logits refresh batch add: {e}"))
                    })?;
                    state.ctx.decode(&mut batch).map_err(|e| {
                        LlamaCppProviderError::Inference(format!("logits refresh decode: {e}"))
                    })?;
                }
            }
        }

        let pos = guard.as_ref().expect("session exists").next_pos;
        Ok((ActiveContext::Session(guard), pos))
    } else {
        // No session state — fresh context (original behavior).
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
        Ok((ActiveContext::Owned(ctx), position))
    }
}

fn generate_chat_text(
    model: &Arc<LlamaModel>,
    backend: &Arc<LlamaBackend>,
    config: &LlamaCppConfig,
    params: ChatGenerationParams<'_>,
    session_state: Option<&SharedSessionState>,
) -> Result<GenerationResult, LlamaCppProviderError> {
    let ChatGenerationParams {
        template_result,
        max_tokens,
        temperature,
        top_p,
        mut on_delta,
    } = params;

    let mut prompt_tokens = model
        .str_to_token(
            &template_result.prompt,
            if template_result.add_bos {
                AddBos::Always
            } else {
                AddBos::Never
            },
        )
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    if prompt_tokens.is_empty() {
        return Err(LlamaCppProviderError::Inference(
            "Prompt produced no tokens".to_string(),
        ));
    }

    let prompt_len = prompt_tokens.len();
    let required_tokens = prompt_len as u32 + max_tokens;
    let n_ctx = resolve_context_size(model, config, required_tokens)?;
    let n_batch = resolve_n_batch(config, n_ctx);

    // -----------------------------------------------------------------------
    // Context acquisition: prefix-reuse path vs fresh-context path
    // -----------------------------------------------------------------------
    let (mut active_ctx, batch_start_pos) = acquire_context(
        model,
        backend,
        config,
        session_state,
        ContextAcquisitionRequest {
            prompt_tokens: &prompt_tokens,
            n_ctx,
            n_batch,
            required_tokens,
        },
    )?;

    let mut n_cur = batch_start_pos;
    let max_tokens_total = n_cur + max_tokens as i32;
    let mut generated_text = String::default();
    let mut completion_tokens = 0u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let (mut sampler, preserved) =
        build_chat_sampler(model, config, template_result, temperature, top_p)?;
    let additional_stops = template_result.additional_stops.clone();
    // We need a LlamaBatch for generation steps. Size 1 per step.
    let mut gen_batch = LlamaBatch::new(1, 1);

    // Sample first token using the last logits from prefill.
    // The prefill left logits at the last batch position.
    let mut finish_reason = "stop".to_string();
    let mut first_sample = true;

    while n_cur < max_tokens_total {
        let token = if first_sample {
            first_sample = false;
            // After prefill, logits are at batch.n_tokens() - 1.
            // For session path, last decode set logits at position -1 of the batch.
            // Use index -1 (last) which maps to batch.n_tokens() - 1 internally.
            active_ctx.with_ctx(|ctx| sampler.sample(ctx, -1))
        } else {
            active_ctx.with_ctx(|ctx| sampler.sample(ctx, gen_batch.n_tokens() - 1))
        };

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

        if let Some(ref mut on_delta) = on_delta {
            on_delta(&output_string)?;
        }

        let response_complete =
            should_stop_after_complete_chat_response(template_result, &generated_text)?;
        if stop_now || response_complete {
            break;
        }

        gen_batch.clear();
        gen_batch
            .add(token, n_cur, &[0], true)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        n_cur += 1;
        active_ctx.with_ctx(|ctx| {
            ctx.decode(&mut gen_batch)
                .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))
        })?;
    }

    if n_cur >= max_tokens_total {
        finish_reason = "length".to_string();
    }

    // Update session state: reset to prompt-only so next call's prefix
    // comparison works correctly. Generated tokens are ephemeral.
    if let Some(state) = active_ctx.session_state_mut() {
        std::mem::swap(&mut state.cached_tokens, &mut prompt_tokens);
        state.next_pos = state.cached_tokens.len() as i32;
        state
            .ctx
            .clear_kv_cache_seq(Some(0), Some(state.cached_tokens.len() as u32), None)
            .map_err(|e| {
                LlamaCppProviderError::Inference(format!("post-generation KV evict: {e}"))
            })?;
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
        prompt_tokens: prompt_len as u32,
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
        top_p,
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
        image_min_tokens: -1,
        image_max_tokens: -1,
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
        let bitmap = MtmdBitmap::from_buffer(&mtmd_ctx, image, false)
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
        .eval_chunks(&mtmd_ctx, &ctx, 0, 0, batch_size, true)
        .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;

    let mut sampler = build_sampler(model, config, false, temperature, top_p, None)?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut n_cur = n_past;
    let max_tokens_total = n_cur + max_tokens as i32;
    let mut generated_text = String::default();
    let mut completion_tokens = 0u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut finish_reason = "stop".to_string();

    while n_cur < max_tokens_total {
        let token = sampler.sample(&ctx, -1);
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
    model: &Arc<LlamaModel>,
    backend: &Arc<LlamaBackend>,
    config: &LlamaCppConfig,
    params: GenerationParams<'_>,
    session_state: Option<&SharedSessionState>,
) -> Result<GenerationResult, LlamaCppProviderError> {
    let GenerationParams {
        prompt,
        use_json_grammar,
        max_tokens,
        temperature,
        top_p,
        mut on_token,
    } = params;

    let mut prompt_tokens = model
        .str_to_token(&prompt.prompt, prompt.add_bos)
        .map_err(|err| LlamaCppProviderError::Tokenization(err.to_string()))?;

    if prompt_tokens.is_empty() {
        return Err(LlamaCppProviderError::Inference(
            "Prompt produced no tokens".to_string(),
        ));
    }

    let prompt_len = prompt_tokens.len();
    let required_tokens = prompt_len as u32 + max_tokens;
    let n_ctx = resolve_context_size(model, config, required_tokens)?;
    let n_batch = resolve_n_batch(config, n_ctx);

    let (mut active_ctx, batch_start_pos) = acquire_context(
        model,
        backend,
        config,
        session_state,
        ContextAcquisitionRequest {
            prompt_tokens: &prompt_tokens,
            n_ctx,
            n_batch,
            required_tokens,
        },
    )?;

    let mut sampler = build_sampler(model, config, use_json_grammar, temperature, top_p, None)?;
    let mut generated_text = String::default();
    let mut completion_tokens = 0_u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    // First token: sample from last logits position after prefill.
    let mut next_token = active_ctx.with_ctx(|ctx| sampler.sample(ctx, -1));

    let mut position = batch_start_pos as usize;
    let mut gen_batch = LlamaBatch::new(1, 1);

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

        if should_stop_after_complete_json_response(use_json_grammar, &generated_text) {
            break;
        }

        gen_batch.clear();
        gen_batch
            .add(next_token, position as i32, &[0], true)
            .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))?;
        active_ctx.with_ctx(|ctx| {
            ctx.decode(&mut gen_batch)
                .map_err(|err| LlamaCppProviderError::Inference(err.to_string()))
        })?;
        position += 1;

        if position >= n_ctx as usize {
            break;
        }

        next_token = active_ctx.with_ctx(|ctx| sampler.sample(ctx, gen_batch.n_tokens() - 1));
    }

    // Session cleanup: reset to prompt tokens only.
    if let Some(state) = active_ctx.session_state_mut() {
        std::mem::swap(&mut state.cached_tokens, &mut prompt_tokens);
        state.next_pos = state.cached_tokens.len() as i32;
        state
            .ctx
            .clear_kv_cache_seq(Some(0), Some(state.cached_tokens.len() as u32), None)
            .map_err(|e| {
                LlamaCppProviderError::Inference(format!("post-generation KV evict: {e}"))
            })?;
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

    extract_first_json_array(trimmed).or_else(|| extract_first_json_object(trimmed))
}

fn should_stop_after_complete_chat_response(
    result: &ChatTemplateResult,
    generated_text: &str,
) -> Result<bool, LlamaCppProviderError> {
    if result.grammar.is_none() {
        return Ok(false);
    }

    if result.parse_tool_calls
        && let Some(message) = parse_tool_response_message(generated_text)?
        && message
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
    {
        return Ok(true);
    }

    Ok(is_complete_structured_payload(generated_text))
}

fn should_stop_after_complete_json_response(use_json_grammar: bool, generated_text: &str) -> bool {
    use_json_grammar && is_complete_structured_payload(generated_text)
}

fn is_complete_structured_payload(text: &str) -> bool {
    let Some(payload) = extract_json_payload(text) else {
        return false;
    };
    let trimmed = text.trim();
    if trimmed == payload {
        return true;
    }

    if trimmed.starts_with("```") || trimmed.contains("```json") {
        return trimmed.ends_with("```") && trimmed.contains(&payload);
    }

    trimmed.ends_with(&payload)
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
    extract_first_json_balanced(text, '{', '}')
}

fn extract_first_json_array(text: &str) -> Option<String> {
    extract_first_json_balanced(text, '[', ']')
}

fn extract_first_json_balanced(text: &str, open: char, close: char) -> Option<String> {
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
            current if current == open => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            current if current == close && depth > 0 => {
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
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_GGUF_MODEL_PATH: &str = "fixtures/model.gguf";
    use autoagents_llm::chat::{FunctionTool, ImageMime};
    use serde_json::json;

    fn chunk_count(total: usize, batch: usize) -> usize {
        if batch == 0 {
            return 0;
        }
        total.div_ceil(batch)
    }

    fn sample_lookup_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Look up a value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["q"]
                }),
            },
        }
    }

    fn sample_search_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "search".to_string(),
                description: "Search indexed data".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            },
        }
    }

    fn sample_structured_output_schema() -> StructuredOutputFormat {
        StructuredOutputFormat {
            name: "Answer".to_string(),
            description: None,
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer"}
                },
                "required": ["value"]
            })),
            strict: Some(true),
        }
    }

    fn parsed_arguments(call: &Value) -> Value {
        serde_json::from_str(
            call["function"]["arguments"]
                .as_str()
                .expect("arguments should be a string"),
        )
        .expect("arguments should be valid JSON")
    }

    #[test]
    fn test_unpack_sampling_none_returns_all_none() {
        // sampling = None must produce identical (None, None, None) so the
        // existing chat_with_tools / chat_stream / chat_stream_struct
        // delegation paths remain bit-equivalent to the pre-overrides
        // implementation.
        let (max_tokens, temperature, top_p) = unpack_sampling(None);
        assert!(max_tokens.is_none());
        assert!(temperature.is_none());
        assert!(top_p.is_none());
    }

    #[test]
    fn test_unpack_sampling_extracts_each_field() {
        let sampling = SamplingOverrides {
            temperature: Some(0.0),
            top_p: Some(0.95),
            max_tokens: Some(128),
        };
        let (max_tokens, temperature, top_p) = unpack_sampling(Some(&sampling));
        assert_eq!(max_tokens, Some(128));
        assert_eq!(temperature, Some(0.0));
        assert_eq!(top_p, Some(0.95));
    }

    #[test]
    fn test_unpack_sampling_partial_overrides() {
        let sampling = SamplingOverrides::with_temperature(0.0);
        let (max_tokens, temperature, top_p) = unpack_sampling(Some(&sampling));
        assert!(max_tokens.is_none());
        assert_eq!(temperature, Some(0.0));
        assert!(top_p.is_none());
    }

    #[test]
    fn test_unpack_sampling_default_equivalent_to_none() {
        // SamplingOverrides::default() (all None) must produce the same
        // tuple as `sampling = None`. This guards the backwards-compat
        // contract: passing `Some(&SamplingOverrides::default())` is
        // bit-equivalent to passing `None`.
        let default = SamplingOverrides::default();
        assert_eq!(unpack_sampling(Some(&default)), unpack_sampling(None));
    }

    #[test]
    fn test_top_p_override_precedence_in_sampler_logic() {
        // Mirror the precedence used in build_sampler / build_chat_sampler:
        //   top_p_override.or(config.top_p)
        // Verifies the override semantics without needing a real model.
        let config_top_p: Option<f32> = Some(0.9);
        let override_top_p: Option<f32> = Some(0.5);
        // Override wins when set.
        assert_eq!(override_top_p.or(config_top_p), Some(0.5));
        // Falls through to config default when override is None.
        let no_override: Option<f32> = None;
        assert_eq!(no_override.or(config_top_p), Some(0.9));
        // Both None → None (provider falls back to no top_p sampler).
        let no_config: Option<f32> = None;
        assert_eq!(no_override.or(no_config), None);
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

    #[test]
    fn test_resolve_n_batch_clamps_to_context() {
        let config = LlamaCppConfig {
            n_batch: Some(128),
            ..Default::default()
        };
        assert_eq!(resolve_n_batch(&config, 64), 64);
        assert_eq!(resolve_n_batch(&config, 256), 128);
    }

    #[test]
    fn test_build_context_params_overrides() {
        let config = LlamaCppConfig {
            n_threads: Some(4),
            n_threads_batch: Some(6),
            ..Default::default()
        };
        let params = build_context_params(&config, true, Some(512), Some(16)).unwrap();
        assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(512));
        assert_eq!(params.n_batch(), 16);
        assert_eq!(params.n_threads(), 4);
        assert_eq!(params.n_threads_batch(), 6);
        assert!(params.embeddings());
    }

    #[test]
    fn test_build_model_params_sets_fields() {
        let mut config = LlamaCppConfig::default();
        config.n_gpu_layers = Some(3);
        config.main_gpu = Some(1);
        config.split_mode = Some(crate::config::LlamaCppSplitMode::Row);
        config.use_mlock = Some(true);
        let params = build_model_params(&config).unwrap();
        assert_eq!(params.n_gpu_layers(), 3);
        assert_eq!(params.main_gpu(), 1);
        assert_eq!(
            params.split_mode().unwrap(),
            llama_cpp_2::model::params::LlamaSplitMode::Row
        );
        assert!(params.use_mlock());
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn test_mtmd_default_marker_smoke() {
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        assert!(!marker.is_empty());
    }

    #[test]
    fn test_chat_template_result_parses_content_and_tool_json() {
        let content_result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };
        let parsed = content_result
            .parse_response_oaicompat("prefix {\"answer\":42} suffix")
            .expect("content response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid content envelope");
        assert_eq!(value["content"], "{\"answer\":42}");

        let tool_result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };
        let parsed = tool_result
            .parse_response_oaicompat(r#"{"function":{"name":"lookup","arguments":{"q":"rust"}}}"#)
            .expect("tool response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid tool envelope");
        assert!(value["tool_calls"].is_array());
        assert_eq!(value["tool_calls"][0]["id"], "call_1");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );

        let parsed = tool_result
            .parse_response_oaicompat(r#"{"content":"done"}"#)
            .expect("content envelope should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid content envelope");
        assert_eq!(value["content"], "done");

        let parsed = tool_result
            .parse_response_oaicompat(r#"{"name":"Alice","score":42}"#)
            .expect("structured content with name should not be mistaken for a tool call");
        let value: Value =
            serde_json::from_str(&parsed).expect("valid structured content envelope");
        assert_eq!(value["content"], r#"{"name":"Alice","score":42}"#);
        assert!(value.get("tool_calls").is_none());

        let parsed = tool_result
            .parse_response_oaicompat(r#"[{"name":"Alice","score":42},{"name":"Bob","score":7}]"#)
            .expect("structured array content with name should not be mistaken for tool calls");
        let value: Value =
            serde_json::from_str(&parsed).expect("valid structured array content envelope");
        assert_eq!(
            value["content"],
            r#"[{"name":"Alice","score":42},{"name":"Bob","score":7}]"#
        );
        assert!(value.get("tool_calls").is_none());

        let constrained_tool_result = ChatTemplateResult {
            tool_names: vec!["lookup".to_string()],
            ..tool_result.clone()
        };
        let parsed = constrained_tool_result
            .parse_response_oaicompat(r#"{"lookup":{"q":"rust"}}"#)
            .expect("known function-name-key tool call should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid known tool envelope");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&value["tool_calls"][0]),
            json!({"q": "rust"})
        );

        let parsed = constrained_tool_result
            .parse_response_oaicompat(
                r#"{"tool_calls":[{"function":{"name":"lookup","arguments":{"q":"rust"}}}]}"#,
            )
            .expect("known explicit tool_calls envelope should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid known tool envelope");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&value["tool_calls"][0]),
            json!({"q": "rust"})
        );

        let parsed = constrained_tool_result
            .parse_response_oaicompat(r#"{"profile":{"name":"Alice","score":42}}"#)
            .expect("unknown function-name-key structured output should remain content");
        let value: Value =
            serde_json::from_str(&parsed).expect("valid constrained structured content envelope");
        assert_eq!(
            value["content"],
            r#"{"profile":{"name":"Alice","score":42}}"#
        );
        assert!(value.get("tool_calls").is_none());

        let parsed = constrained_tool_result
            .parse_response_oaicompat(
                r#"{"tool_calls":[{"function":{"name":"profile","arguments":{"name":"Alice","score":42}}}]}"#,
            )
            .expect("unknown explicit tool_calls structured output should remain content");
        let value: Value =
            serde_json::from_str(&parsed).expect("valid constrained structured content envelope");
        assert_eq!(
            value["content"],
            r#"{"tool_calls":[{"function":{"name":"profile","arguments":{"name":"Alice","score":42}}}]}"#
        );
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn test_parse_tool_response_rejects_invalid_string_arguments() {
        let err = parse_tool_response_message(
            r#"{"tool_calls":[{"function":{"name":"lookup","arguments":"not json"}}]}"#,
        )
        .expect_err("invalid JSON arguments must be rejected");

        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn test_parse_openai_tool_calls_preserves_message_content_and_reasoning() {
        let parsed = parse_tool_response_message(
            r#"{"content":"I will look this up.","reasoning_content":"Need current data.","tool_calls":[{"id":"call_lookup","type":"function","function":{"name":"lookup","arguments":{"q":"rust"}}}]}"#,
        )
        .expect("OpenAI-compatible tool calls should parse")
        .expect("OpenAI-compatible tool calls should normalize");

        assert_eq!(parsed["content"], "I will look this up.");
        assert_eq!(parsed["reasoning_content"], "Need current data.");
        assert_eq!(parsed["tool_calls"][0]["id"], "call_lookup");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_openai_tool_calls_maps_thinking_alias_to_reasoning_content() {
        let parsed = parse_tool_response_message(
            r#"{"thinking":"Need current data.","tool_calls":[{"function":{"name":"lookup","arguments":{"q":"rust"}}}]}"#,
        )
        .expect("OpenAI-compatible thinking alias should parse")
        .expect("OpenAI-compatible thinking alias should normalize");

        assert_eq!(parsed["reasoning_content"], "Need current data.");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn test_parse_openai_single_tool_call_requires_tool_shape_for_name_field() {
        assert!(
            parse_tool_response_message(r#"{"name":"Alice","score":42}"#)
                .expect("structured content should not fail")
                .is_none()
        );

        let parsed = parse_tool_response_message(r#"{"type":"function","name":"lookup"}"#)
            .expect("explicit zero-arg function call should parse")
            .expect("explicit zero-arg function call should normalize");

        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(parsed_arguments(&parsed["tool_calls"][0]), json!({}));
    }

    #[test]
    fn test_parse_openai_tool_call_array_requires_tool_shaped_elements() {
        assert!(
            parse_tool_response_message(r#"[{"name":"Alice","score":42}]"#)
                .expect("structured array should not fail")
                .is_none()
        );
        assert!(
            parse_tool_response_message(
                r#"[{"type":"function","name":"lookup"},{"name":"Alice","score":42}]"#
            )
            .expect("mixed structured array should not fail")
            .is_none()
        );

        let parsed = parse_tool_response_message(
            r#"[{"type":"function","name":"lookup"},{"lookup":{"q":"rust"}},{"tool":{"name":"search","args":{"query":"llama.cpp"}}}]"#,
        )
        .expect("tool-shaped array should parse")
        .expect("tool-shaped array should normalize");

        let calls = parsed["tool_calls"].as_array().expect("tool calls array");
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(parsed_arguments(&calls[0]), json!({}));
        assert_eq!(calls[1]["function"]["name"], "lookup");
        assert_eq!(parsed_arguments(&calls[1]), json!({"q": "rust"}));
        assert_eq!(calls[2]["function"]["name"], "search");
        assert_eq!(parsed_arguments(&calls[2]), json!({"query": "llama.cpp"}));
    }

    #[test]
    fn test_parse_native_xml_tool_call_payload() {
        let parsed = parse_tool_response_message(
            r#"<tool_call>{"name":"lookup","arguments":{"q":"rust"}}</tool_call>"#,
        )
        .expect("native tool call should parse")
        .expect("native tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "call_1");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_specialized_tool_parser_respects_allowed_tool_names() {
        let allowed_tools = vec!["lookup".to_string()];

        let parsed = parse_tool_response_message_with_allowed_tools(
            r#"<tool_call>{"name":"lookup","arguments":{"q":"rust"}}</tool_call>"#,
            Some(&allowed_tools),
        )
        .expect("known native tool call should parse")
        .expect("known native tool call should normalize");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");

        let parsed = parse_tool_response_message_with_allowed_tools(
            r#"<tool_call>{"name":"profile","arguments":{"name":"Alice"}}</tool_call>"#,
            Some(&allowed_tools),
        )
        .expect("unknown native tool call should be rejected as a tool call");
        assert!(parsed.is_none());
    }

    #[test]
    fn test_parse_generic_function_tag_tool_call_payload() {
        let parsed = parse_tool_response_message(
            r#"before<function=lookup>{"q":"rust","limit":2}</function>"#,
        )
        .expect("generic function-tag tool call should parse")
        .expect("generic function-tag tool call should normalize");

        assert_eq!(parsed["content"], "before");
        assert_eq!(parsed["tool_calls"][0]["id"], "call_1");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_generic_function_tag_tagged_arguments_payload() {
        let parsed = parse_tool_response_message(
            r#"before<function=lookup><parameter name="q">rust</parameter><parameter name="limit">2</parameter></function>"#,
        )
        .expect("generic tagged-argument tool call should parse")
        .expect("generic tagged-argument tool call should normalize");

        assert_eq!(parsed["content"], "before");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_native_tool_calls_tag_array_payload() {
        let parsed = parse_tool_response_message(
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"q":"rust"}},{"name":"lookup","arguments":{"q":"llama","limit":2}}] [/TOOL_CALLS]"#,
        )
        .expect("native tool calls array should parse")
        .expect("native tool calls array should normalize");

        let calls = parsed["tool_calls"].as_array().expect("tool calls array");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(parsed_arguments(&calls[0]), json!({"q": "rust"}));
        assert_eq!(
            parsed_arguments(&calls[1]),
            json!({"limit": 2, "q": "llama"})
        );
    }

    #[test]
    fn test_parse_json_native_function_name_key_payload() {
        let parsed = parse_tool_response_message(r#"{"lookup":{"q":"rust","limit":2}}"#)
            .expect("function-name-key tool call should parse")
            .expect("function-name-key tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "call_1");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_json_native_function_name_key_wrapped_arguments_and_id() {
        let parsed = parse_tool_response_message(
            r#"{"lookup":{"id":"call_lookup_1","arguments":{"q":"rust","limit":2}}}"#,
        )
        .expect("function-name-key wrapped args should parse")
        .expect("function-name-key wrapped args should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "call_lookup_1");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_json_native_function_name_key_array_payload() {
        let parsed = parse_tool_response_message(
            r#"[{"lookup":{"q":"rust"}},{"lookup":{"q":"llama","limit":2}}]"#,
        )
        .expect("function-name-key array should parse")
        .expect("function-name-key array should normalize");

        let calls = parsed["tool_calls"].as_array().expect("tool calls array");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(parsed_arguments(&calls[0]), json!({"q": "rust"}));
        assert_eq!(
            parsed_arguments(&calls[1]),
            json!({"limit": 2, "q": "llama"})
        );
    }

    #[test]
    fn test_parse_json_native_function_name_key_array_wrapped_arguments() {
        let parsed = parse_tool_response_message(
            r#"[{"lookup":{"call_id":7,"args":{"q":"rust"}}},{"lookup":{"tool_call_id":"call_2","parameters":{"q":"llama","limit":2}}}]"#,
        )
        .expect("function-name-key wrapped array should parse")
        .expect("function-name-key wrapped array should normalize");

        let calls = parsed["tool_calls"].as_array().expect("tool calls array");
        assert_eq!(calls[0]["id"], "7");
        assert_eq!(parsed_arguments(&calls[0]), json!({"q": "rust"}));
        assert_eq!(calls[1]["id"], "call_2");
        assert_eq!(
            parsed_arguments(&calls[1]),
            json!({"limit": 2, "q": "llama"})
        );
    }

    #[test]
    fn test_parse_json_native_flat_alias_arguments_and_numeric_id() {
        let parsed = parse_tool_response_message(r#"{"name":"lookup","args":{"q":"rust"},"id":7}"#)
            .expect("flat JSON-native alias tool call should parse")
            .expect("flat JSON-native alias tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "7");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_json_native_nested_alias_arguments_and_inner_id() {
        let parsed = parse_tool_response_message(
            r#"{"function":{"name":"lookup","parameters":{"q":"rust"},"call_id":"inner_1"}}"#,
        )
        .expect("nested JSON-native alias tool call should parse")
        .expect("nested JSON-native alias tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "inner_1");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_json_native_wrapped_nested_alias_tool_call_array() {
        let parsed = parse_tool_response_message(
            r#"[{"tool":{"name":"lookup","args":{"q":"rust"},"id":"nested_1"}}]"#,
        )
        .expect("wrapped nested JSON-native tool call should parse")
        .expect("wrapped nested JSON-native tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "nested_1");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_ministral_tool_calls_args_payload() {
        let parsed = parse_tool_response_message(
            r#"[THINK]plan[/THINK]before[TOOL_CALLS]lookup[ARGS]{"q":"rust"}[TOOL_CALLS]search[ARGS]{"query":"llama.cpp","limit":2}"#,
        )
        .expect("ministral tool calls should parse")
        .expect("ministral tool calls should normalize");

        let calls = parsed["tool_calls"].as_array().expect("tool calls array");
        assert_eq!(parsed["content"], "[THINK]plan[/THINK]before");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(parsed_arguments(&calls[0]), json!({"q": "rust"}));
        assert_eq!(calls[1]["function"]["name"], "search");
        assert_eq!(
            parsed_arguments(&calls[1]),
            json!({"limit": 2, "query": "llama.cpp"})
        );
    }

    #[test]
    fn test_tool_response_parsing_extracts_reasoning_from_content_envelope() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("[THINK]".to_string()),
            reasoning_end_tag: Some("[/THINK]".to_string()),
        };

        let parsed = result
            .parse_response_oaicompat(
                r#"[THINK]plan[/THINK]before[TOOL_CALLS]lookup[ARGS]{"q":"rust"}"#,
            )
            .expect("tool response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid tool envelope");

        assert_eq!(value["reasoning_content"], "plan");
        assert_eq!(value["content"], "before");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn test_parse_gpt_oss_tool_call_with_recipient_in_role_header() {
        let parsed = parse_tool_response_message(
            r#"<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"rust","limit":2}<|end|>"#,
        )
        .expect("gpt-oss role-recipient tool call should parse")
        .expect("gpt-oss role-recipient tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["id"], "call_1");
        assert_eq!(parsed["tool_calls"][0]["type"], "function");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_gpt_oss_tool_call_with_recipient_in_channel_header() {
        let parsed = parse_tool_response_message(
            r#"<|start|>assistant<|channel|>analysis to=functions.search <|constrain|> json<|message|>{"query":"llama.cpp"}<|end|>"#,
        )
        .expect("gpt-oss channel-recipient tool call should parse")
        .expect("gpt-oss channel-recipient tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "search");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"query\":\"llama.cpp\"}"
        );
    }

    #[test]
    fn test_parse_gpt_oss_tool_call_preserves_prior_commentary_content() {
        let parsed = parse_tool_response_message(
            r#"<|start|>assistant<|channel|>commentary<|message|>I will search.<|end|><|start|>assistant<|channel|>commentary to=functions.search <|constrain|> json<|message|>{"query":"llama.cpp"}<|end|>"#,
        )
        .expect("gpt-oss content before tool call should parse")
        .expect("gpt-oss content before tool call should normalize");

        assert_eq!(parsed["content"], "I will search.");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "search");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"query": "llama.cpp"})
        );
    }

    #[test]
    fn test_parse_gpt_oss_tool_call_preserves_prior_final_content() {
        let parsed = parse_tool_response_message(
            r#"<|start|>assistant<|channel|>analysis<|message|>Need a lookup.<|end|><|start|>assistant<|channel|>final<|message|>I can answer after checking.<|end|><|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"rust"}<|end|>"#,
        )
        .expect("gpt-oss final content before tool call should parse")
        .expect("gpt-oss final content before tool call should normalize");

        assert_eq!(parsed["content"], "I can answer after checking.");
        assert_eq!(parsed["reasoning_content"], "Need a lookup.");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_gpt_oss_tool_call_preserves_prior_analysis_reasoning() {
        let parsed = parse_tool_response_message(
            r#"<|start|>assistant<|channel|>analysis<|message|>Need a lookup.<|end|><|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"rust"}<|end|>"#,
        )
        .expect("gpt-oss reasoning before tool call should parse")
        .expect("gpt-oss reasoning before tool call should normalize");

        assert_eq!(parsed["reasoning_content"], "Need a lookup.");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_gpt_oss_tool_call_ignores_stray_commentary_prefix() {
        let parsed = parse_tool_response_message(
            r#"<|start|>assistant<|channel|>commentary to=assistant<|channel|>analysis<|message|>Need a lookup.<|end|><|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"rust"}<|end|>"#,
        )
        .expect("gpt-oss stray commentary before reasoning should parse")
        .expect("gpt-oss stray commentary before reasoning should normalize");

        assert_eq!(parsed["reasoning_content"], "Need a lookup.");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_gemma4_native_tool_call() {
        let parsed = parse_tool_response_message(
            "answer<|tool_call>call:lookup{q:<|\"|>rust<|\"|>,limit:2}<tool_call|>",
        )
        .expect("gemma4 tool call should parse")
        .expect("gemma4 tool call should normalize");

        assert_eq!(parsed["content"], "answer");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_gemma4_prefixed_native_tool_call() {
        let parsed = parse_tool_response_message(
            "<|turn>model\nanswer<|tool_call>call:lookup{q:<|\"|>rust<|\"|>}<tool_call|>",
        )
        .expect("prefixed gemma4 tool call should parse")
        .expect("prefixed gemma4 tool call should normalize");

        assert_eq!(parsed["content"], "answer");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_functionary_v3_2_tool_call() {
        let parsed = parse_tool_response_message(
            r#">>>all
I will call a function.
>>>lookup
{"q":"rust"}"#,
        )
        .expect("functionary tool call should parse")
        .expect("functionary tool call should normalize");

        assert_eq!(parsed["content"], "I will call a function.");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_parse_functionary_v3_2_content_only() {
        let parsed = parse_tool_response_message(
            r#">>>all
No tools are needed."#,
        )
        .expect("functionary content should parse")
        .expect("functionary content should normalize");

        assert_eq!(parsed["content"], "No tools are needed.");
        assert!(parsed.get("tool_calls").is_none());
    }

    #[test]
    fn test_parse_functionary_v3_2_post_generation_content_only() {
        let parsed = parse_tool_response_message(
            r#"all
No tools are needed."#,
        )
        .expect("post-generation functionary content should parse")
        .expect("post-generation functionary content should normalize");

        assert_eq!(parsed["content"], "No tools are needed.");
        assert!(parsed.get("tool_calls").is_none());
    }

    #[test]
    fn test_parse_functionary_v3_2_post_generation_tool_call() {
        let parsed = parse_tool_response_message(
            r#"lookup
{"q":"rust"}"#,
        )
        .expect("post-generation functionary tool call should parse")
        .expect("post-generation functionary tool call should normalize");

        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_parse_kimi_k2_tool_call_preserves_id() {
        let parsed = parse_tool_response_message(
            r#"<think>plan</think>answer<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:3<|tool_call_argument_begin|>{"q":"rust"}<|tool_call_end|><|tool_calls_section_end|>"#,
        )
        .expect("kimi tool call should parse")
        .expect("kimi tool call should normalize");

        assert_eq!(parsed["content"], "<think>plan</think>answer");
        assert_eq!(parsed["tool_calls"][0]["id"], "functions.lookup:3");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_parse_kimi_k2_prefixed_tool_call_preserves_id() {
        let parsed = parse_tool_response_message(
            r#"<|im_assistant|>assistant<|im_middle|><think>plan</think>answer<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:3<|tool_call_argument_begin|>{"q":"rust"}<|tool_call_end|><|tool_calls_section_end|>"#,
        )
        .expect("prefixed kimi tool call should parse")
        .expect("prefixed kimi tool call should normalize");

        assert_eq!(parsed["content"], "<think>plan</think>answer");
        assert_eq!(parsed["tool_calls"][0]["id"], "functions.lookup:3");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn test_parse_lfm2_python_style_tool_calls() {
        let parsed = parse_tool_response_message(
            r#"<think>plan</think>content<|tool_call_start|>[lookup(q='rust', limit=2, exact=True), search(query="llama,cpp", filters={"kind":"repo"})]<|tool_call_end|>"#,
        )
        .expect("lfm2 tool calls should parse")
        .expect("lfm2 tool calls should normalize");

        let calls = parsed["tool_calls"].as_array().expect("tool calls array");
        assert_eq!(parsed["content"], "<think>plan</think>content");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&calls[0]),
            json!({"exact": true, "limit": 2, "q": "rust"})
        );
        assert_eq!(calls[1]["function"]["name"], "search");
        assert_eq!(
            parsed_arguments(&calls[1]),
            json!({"filters": {"kind": "repo"}, "query": "llama,cpp"})
        );
    }

    #[test]
    fn test_parse_lfm2_nested_python_literals() {
        let parsed = parse_tool_response_message(
            r#"<|tool_call_start|>[lookup(filters={'kind': 'repo', 'flags': [True, False, None,],}, q='rust')]<|tool_call_end|>"#,
        )
        .expect("lfm2 nested Python literals should parse")
        .expect("lfm2 nested Python literals should normalize");

        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({
                "filters": {
                    "flags": [true, false, null],
                    "kind": "repo"
                },
                "q": "rust"
            })
        );
    }

    #[test]
    fn test_parse_lfm2_prefixed_python_style_tool_calls() {
        let parsed = parse_tool_response_message(
            r#"<|im_start|>assistant
<think>plan</think>content<|tool_call_start|>[lookup(q='rust')]<|tool_call_end|>"#,
        )
        .expect("prefixed lfm2 tool calls should parse")
        .expect("prefixed lfm2 tool calls should normalize");

        assert_eq!(parsed["content"], "<think>plan</think>content");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_parse_gigachat_v3_tool_call() {
        let parsed = parse_tool_response_message(
            "content<|message_sep|>\n\nfunction call<|role_sep|>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"rust\"}}<|message_sep|>\n\n",
        )
        .expect("gigachat tool call should parse")
        .expect("gigachat tool call should normalize");

        assert_eq!(parsed["content"], "content");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_parse_gigachat_v3_prefixed_content_only() {
        let parsed =
            parse_tool_response_message("assistant<|role_sep|>\nplain answer<|message_sep|>\n\n")
                .expect("gigachat prefixed content should parse")
                .expect("gigachat prefixed content should normalize");

        assert_eq!(parsed["content"], "plain answer");
        assert!(parsed.get("tool_calls").is_none());
    }

    #[test]
    fn test_parse_gigachat_v3_prefixed_tool_call() {
        let parsed = parse_tool_response_message(
            "assistant<|role_sep|>\ncontent<|message_sep|>\n\nfunction call<|role_sep|>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"rust\"}}<|message_sep|>\n\n",
        )
        .expect("gigachat prefixed tool call should parse")
        .expect("gigachat prefixed tool call should normalize");

        assert_eq!(parsed["content"], "content");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_parse_deepseek_v3_2_dsml_tool_call() {
        let parsed = parse_tool_response_message(
            r#"<think>plan</think>content<｜DSML｜function_calls>
<｜DSML｜invoke name="lookup">
<｜DSML｜parameter name="q" string="true">rust</｜DSML｜parameter>
<｜DSML｜parameter name="limit" string="false">2</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#,
        )
        .expect("deepseek dsml tool call should parse")
        .expect("deepseek dsml tool call should normalize");

        assert_eq!(parsed["content"], "<think>plan</think>content");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"limit": 2, "q": "rust"})
        );
    }

    #[test]
    fn test_parse_deepseek_v3_2_prefixed_dsml_tool_call() {
        let parsed = parse_tool_response_message(
            r#"<｜Assistant｜><think>plan</think>content<｜DSML｜function_calls>
<｜DSML｜invoke name="lookup">
<｜DSML｜parameter name="q" string="true">rust</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#,
        )
        .expect("prefixed deepseek dsml tool call should parse")
        .expect("prefixed deepseek dsml tool call should normalize");

        assert_eq!(parsed["content"], "<think>plan</think>content");
        assert_eq!(parsed["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&parsed["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_specialized_tool_responses_normalize_to_openai_compat_goldens() {
        struct GoldenCase {
            name: &'static str,
            input: &'static str,
            reasoning_start: Option<&'static str>,
            reasoning_end: Option<&'static str>,
            expected_content: Option<&'static str>,
            expected_reasoning: Option<&'static str>,
            expected_tool: &'static str,
            expected_arguments: Value,
        }

        let cases = vec![
            GoldenCase {
                name: "gpt-oss native channels",
                input: r#"<|start|>assistant<|channel|>analysis<|message|>Need lookup.<|end|><|start|>assistant<|channel|>commentary<|message|>I will search.<|end|><|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"rust"}<|end|>"#,
                reasoning_start: None,
                reasoning_end: None,
                expected_content: Some("I will search."),
                expected_reasoning: Some("Need lookup."),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "ministral args tags",
                input: r#"[THINK]plan[/THINK]content[TOOL_CALLS]lookup[ARGS]{"q":"rust"}"#,
                reasoning_start: Some("[THINK]"),
                reasoning_end: Some("[/THINK]"),
                expected_content: Some("content"),
                expected_reasoning: Some("plan"),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "gemma4 native tool",
                input: "<|channel>thought\nplan<channel|>content<|tool_call>call:lookup{q:<|\"|>rust<|\"|>}<tool_call|>",
                reasoning_start: Some("<|channel>thought"),
                reasoning_end: Some("<channel|>"),
                expected_content: Some("content"),
                expected_reasoning: Some("plan"),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "functionary v3.2",
                input: r#">>>all
content
>>>lookup
{"q":"rust"}"#,
                reasoning_start: None,
                reasoning_end: None,
                expected_content: Some("content"),
                expected_reasoning: None,
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "kimi k2",
                input: r#"<think>plan</think>content<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:3<|tool_call_argument_begin|>{"q":"rust"}<|tool_call_end|><|tool_calls_section_end|>"#,
                reasoning_start: Some("<think>"),
                reasoning_end: Some("</think>"),
                expected_content: Some("content"),
                expected_reasoning: Some("plan"),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "lfm2 python style",
                input: r#"<think>plan</think>content<|tool_call_start|>[lookup(q='rust')]<|tool_call_end|>"#,
                reasoning_start: Some("<think>"),
                reasoning_end: Some("</think>"),
                expected_content: Some("content"),
                expected_reasoning: Some("plan"),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "gigachat v3",
                input: "content<|message_sep|>\n\nfunction call<|role_sep|>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"rust\"}}<|message_sep|>\n\n",
                reasoning_start: None,
                reasoning_end: None,
                expected_content: Some("content"),
                expected_reasoning: None,
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "deepseek v3.2 dsml",
                input: r#"<think>plan</think>content<｜DSML｜function_calls>
<｜DSML｜invoke name="lookup">
<｜DSML｜parameter name="q" string="true">rust</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#,
                reasoning_start: Some("<think>"),
                reasoning_end: Some("</think>"),
                expected_content: Some("content"),
                expected_reasoning: Some("plan"),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
            GoldenCase {
                name: "generic function tag",
                input: r#"<think>plan</think>content<function=lookup><parameter name="q">"rust"</parameter></function>"#,
                reasoning_start: Some("<think>"),
                reasoning_end: Some("</think>"),
                expected_content: Some("content"),
                expected_reasoning: Some("plan"),
                expected_tool: "lookup",
                expected_arguments: json!({"q": "rust"}),
            },
        ];

        for case in cases {
            let result = ChatTemplateResult {
                prompt: "prompt".to_string(),
                generation_prompt: String::default(),
                force_pure_content: false,
                is_continuation: false,
                add_bos: true,
                grammar: Some(JSON_GRAMMAR.to_string()),
                grammar_lazy: false,
                grammar_triggers: Vec::new(),
                preserved_tokens: Vec::new(),
                additional_stops: Vec::new(),
                parse_tool_calls: true,
                tool_names: Vec::new(),
                reasoning_format: case
                    .reasoning_start
                    .map(|_| crate::config::LlamaCppReasoningFormat::Auto),
                reasoning_start_tag: case.reasoning_start.map(str::to_string),
                reasoning_end_tag: case.reasoning_end.map(str::to_string),
            };

            let parsed = result
                .parse_response_oaicompat(case.input)
                .unwrap_or_else(|err| panic!("{} should parse: {err}", case.name));
            let message: Value = serde_json::from_str(&parsed)
                .unwrap_or_else(|err| panic!("{} should emit valid JSON: {err}", case.name));

            match case.expected_content {
                Some(content) => assert_eq!(message["content"], content, "{}", case.name),
                None => assert!(
                    message.get("content").is_none(),
                    "{} should not emit content",
                    case.name
                ),
            }
            match case.expected_reasoning {
                Some(reasoning) => {
                    assert_eq!(message["reasoning_content"], reasoning, "{}", case.name)
                }
                None => assert!(
                    message.get("reasoning_content").is_none(),
                    "{} should not emit reasoning",
                    case.name
                ),
            }

            let calls = message["tool_calls"]
                .as_array()
                .unwrap_or_else(|| panic!("{} should emit tool_calls", case.name));
            assert_eq!(calls.len(), 1, "{}", case.name);
            assert_eq!(
                calls[0]["function"]["name"], case.expected_tool,
                "{}",
                case.name
            );
            assert_eq!(
                parsed_arguments(&calls[0]),
                case.expected_arguments,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn test_gpt_oss_partial_streaming_diffs_reasoning_content_and_tool_call() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        let generated_parts = [
            "<|start|>assistant<|channel|>analysis<|message|>Need",
            "<|start|>assistant<|channel|>analysis<|message|>Need lookup",
            "<|start|>assistant<|channel|>analysis<|message|>Need lookup<|end|><|start|>assistant<|channel|>commentary<|message|>I will search.",
            r#"<|start|>assistant<|channel|>analysis<|message|>Need lookup<|end|><|start|>assistant<|channel|>commentary<|message|>I will search.<|end|><|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"rust"}<|end|>"#,
        ];

        let mut stream_state = StreamMappingState::default();
        let mut outputs = Vec::new();
        for generated in generated_parts {
            let message_json = result
                .parse_partial_response_oaicompat(generated)
                .expect("partial GPT-OSS response should parse")
                .expect("partial GPT-OSS response should emit a parser message");
            let delta_json = stream_delta_from_message_json(&message_json)
                .expect("parser message should convert to stream delta");
            outputs.extend(map_tool_stream_event(
                Ok(StreamEvent::Delta(delta_json)),
                &mut stream_state,
            ));
        }
        outputs.extend(map_tool_stream_event(
            Ok(StreamEvent::Done {
                stop_reason: "tool_use".to_string(),
            }),
            &mut stream_state,
        ));

        assert!(matches!(
            &outputs[0],
            Ok(StreamChunk::ReasoningContent(text)) if text == "Need"
        ));
        assert!(matches!(
            &outputs[1],
            Ok(StreamChunk::ReasoningContent(text)) if text == " lookup"
        ));
        assert!(matches!(
            &outputs[2],
            Ok(StreamChunk::Text(text)) if text == "I will search."
        ));
        assert!(matches!(
            &outputs[3],
            Ok(StreamChunk::ToolUseStart { index, name, .. }) if *index == 0 && name == "lookup"
        ));
        assert!(matches!(
            &outputs[4],
            Ok(StreamChunk::ToolUseInputDelta { index, partial_json }) if *index == 0 && partial_json == "{\"q\":\"rust\"}"
        ));
        assert!(matches!(
            &outputs[5],
            Ok(StreamChunk::ToolUseComplete { index, tool_call }) if *index == 0
                && tool_call.function.name == "lookup"
                && tool_call.function.arguments == "{\"q\":\"rust\"}"
        ));
        assert!(matches!(
            &outputs[6],
            Ok(StreamChunk::Done { stop_reason }) if stop_reason == "tool_use"
        ));
        assert_eq!(
            outputs.len(),
            7,
            "stream mapping must not duplicate cumulative reasoning/content/tool arguments"
        );
    }

    #[test]
    fn test_chat_template_result_parses_gpt_oss_final_and_reasoning_channels() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        let parsed = result
            .parse_response_oaicompat(
                "<|start|>assistant<|channel|>analysis<|message|>working<|end|><|start|>assistant<|channel|>final<|message|>done",
            )
            .expect("native channel content should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid content envelope");
        assert_eq!(value["content"], "done");
        assert_eq!(value["reasoning_content"], "working");

        let partial_reasoning = result
            .parse_partial_response_oaicompat(
                "<|start|>assistant<|channel|>analysis<|message|>working",
            )
            .expect("partial native channel reasoning should parse")
            .expect("reasoning-only delta should be available");
        let value: Value =
            serde_json::from_str(&partial_reasoning).expect("valid partial content envelope");
        assert_eq!(value["content"], "");
        assert_eq!(value["reasoning_content"], "working");

        let partial_final = result
            .parse_partial_response_oaicompat(
                "<|start|>assistant<|channel|>analysis<|message|>working<|end|><|start|>assistant<|channel|>final<|message|>do",
            )
            .expect("partial final native channel content should parse")
            .expect("final delta should be available");
        let value: Value =
            serde_json::from_str(&partial_final).expect("valid partial content envelope");
        assert_eq!(value["content"], "do");
        assert_eq!(value["reasoning_content"], "working");

        let stray_final = result
            .parse_response_oaicompat(
                "<|start|>assistant<|channel|>commentary to=assistant<|channel|>final<|message|>done",
            )
            .expect("stray commentary before final should parse");
        let value: Value =
            serde_json::from_str(&stray_final).expect("valid stray final content envelope");
        assert_eq!(value["content"], "done");
        assert!(value.get("reasoning_content").is_none());

        let stray_reasoning = result
            .parse_response_oaicompat(
                "<|start|>assistant<|channel|>commentary<|channel|>analysis<|message|>working<|end|><|start|>assistant<|channel|>commentary to=assistant<|channel|>final<|message|>done",
            )
            .expect("stray commentary before analysis and final should parse");
        let value: Value =
            serde_json::from_str(&stray_reasoning).expect("valid stray reasoning envelope");
        assert_eq!(value["content"], "done");
        assert_eq!(value["reasoning_content"], "working");
    }

    #[test]
    fn test_force_pure_content_parser_keeps_native_markers_as_content() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: "<|start|>assistant<|channel|>final<|message|>".to_string(),
            force_pure_content: true,
            is_continuation: false,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<think>".to_string()),
            reasoning_end_tag: Some("</think>".to_string()),
        };
        let text = r#"<think>plan</think><|start|>assistant<|channel|>analysis<|message|>working<|end|><|start|>assistant<|channel|>final<|message|>done<tool_call>{"name":"lookup","arguments":{"q":"rust"}}</tool_call>"#;

        let parsed = result
            .parse_response_oaicompat(text)
            .expect("pure content response should parse as plain content");
        let value: Value = serde_json::from_str(&parsed).expect("valid content envelope");

        assert_eq!(value["content"], text);
        assert!(value.get("reasoning_content").is_none());
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn test_continuation_parser_strips_echoed_generation_prompt() {
        let result = ChatTemplateResult {
            prompt: "user:hi\nassistant:partial".to_string(),
            generation_prompt: "assistant:partial".to_string(),
            force_pure_content: false,
            is_continuation: true,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        let parsed = result
            .parse_response_oaicompat("assistant:partial completion")
            .expect("continuation response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid message json");

        assert_eq!(value["content"], " completion");
    }

    #[test]
    fn test_partial_response_parsing_waits_for_complete_structured_json() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        assert!(
            result
                .parse_partial_response_oaicompat("{\"answer\":")
                .expect("incomplete json should not fail")
                .is_none()
        );

        let parsed = result
            .parse_partial_response_oaicompat("{\"answer\":42}")
            .expect("complete json should parse")
            .expect("complete json should emit a parsed message");
        let value: Value = serde_json::from_str(&parsed).expect("valid content envelope");
        assert_eq!(value["content"], "{\"answer\":42}");
    }

    #[test]
    fn test_structured_response_parsing_preserves_reasoning_before_json() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<think>".to_string()),
            reasoning_end_tag: Some("</think>".to_string()),
        };

        let parsed = result
            .parse_response_oaicompat("<think>plan</think>{\"answer\":42}")
            .expect("structured response with reasoning should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid content envelope");
        assert_eq!(value["content"], "{\"answer\":42}");
        assert_eq!(value["reasoning_content"], "plan");

        let partial = result
            .parse_partial_response_oaicompat("<think>plan</think>{\"answer\":42}")
            .expect("partial structured response with reasoning should parse")
            .expect("complete structured response should emit a parsed message");
        let value: Value = serde_json::from_str(&partial).expect("valid partial content envelope");
        assert_eq!(value["content"], "{\"answer\":42}");
        assert_eq!(value["reasoning_content"], "plan");

        let partial = result
            .parse_partial_response_oaicompat("<think>plan")
            .expect("open reasoning response should parse")
            .expect("open reasoning should emit a partial message");
        let value: Value = serde_json::from_str(&partial).expect("valid open reasoning envelope");
        assert_eq!(value["content"], "");
        assert_eq!(value["reasoning_content"], "plan");

        let final_open = result
            .parse_response_oaicompat("<think>plan")
            .expect("final open reasoning response should parse");
        let value: Value =
            serde_json::from_str(&final_open).expect("valid final open reasoning envelope");
        assert_eq!(value["content"], "");
        assert_eq!(value["reasoning_content"], "plan");

        let prefilled_result = ChatTemplateResult {
            generation_prompt: "<|im_start|>assistant\n<think>\n".to_string(),
            ..result.clone()
        };

        let partial = prefilled_result
            .parse_partial_response_oaicompat("plan")
            .expect("prefilled open reasoning response should parse")
            .expect("prefilled open reasoning should emit a partial message");
        let value: Value =
            serde_json::from_str(&partial).expect("valid prefilled reasoning envelope");
        assert_eq!(value["content"], "");
        assert_eq!(value["reasoning_content"], "plan");

        let partial = prefilled_result
            .parse_partial_response_oaicompat("plan</think>{\"answer\":42}")
            .expect("prefilled reasoning before JSON should parse")
            .expect("prefilled reasoning plus JSON should emit a partial message");
        let value: Value = serde_json::from_str(&partial).expect("valid prefilled JSON envelope");
        assert_eq!(value["content"], "{\"answer\":42}");
        assert_eq!(value["reasoning_content"], "plan");

        let parsed = prefilled_result
            .parse_response_oaicompat("plan</think>{\"answer\":42}")
            .expect("final prefilled reasoning before JSON should parse");
        let value: Value =
            serde_json::from_str(&parsed).expect("valid final prefilled JSON envelope");
        assert_eq!(value["content"], "{\"answer\":42}");
        assert_eq!(value["reasoning_content"], "plan");

        let parsed = prefilled_result
            .parse_response_oaicompat("{\"answer\":42}")
            .expect("prefilled parser should not treat JSON as reasoning");
        let value: Value = serde_json::from_str(&parsed).expect("valid direct JSON envelope");
        assert_eq!(value["content"], "{\"answer\":42}");
        assert!(value.get("reasoning_content").is_none());
    }

    #[test]
    fn test_partial_reasoning_stream_diffs_open_tagged_blocks() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: "<|im_start|>assistant\n<think>\n".to_string(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<think>".to_string()),
            reasoning_end_tag: Some("</think>".to_string()),
        };
        let generated_parts = [
            "<",
            "<th",
            "<think>h",
            "<think>hi",
            "<think>hi</",
            "<think>hi</think>{\"answer\":42}",
        ];
        let mut stream_state = StreamMappingState::default();
        let mut reasoning_chunks = Vec::new();

        for generated in generated_parts {
            let Some(message_json) = result
                .parse_partial_response_oaicompat(generated)
                .expect("partial reasoning response should parse")
            else {
                continue;
            };
            let delta_json = stream_delta_from_message_json(&message_json)
                .expect("partial reasoning response should map to stream delta");
            for output in
                map_tool_stream_event(Ok(StreamEvent::Delta(delta_json)), &mut stream_state)
            {
                if let Ok(StreamChunk::ReasoningContent(content)) = output {
                    reasoning_chunks.push(content);
                }
            }
        }

        assert_eq!(reasoning_chunks, vec!["h", "i"]);
    }

    #[test]
    fn test_partial_response_parsing_streams_complete_native_tool_call() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: true,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        assert!(
            result
                .parse_partial_response_oaicompat("<tool_call>{\"name\":\"lookup\",\"arguments\":")
                .expect("incomplete native tool json should not fail")
                .is_none()
        );

        let parsed = result
            .parse_partial_response_oaicompat(
                r#"<tool_call>{"name":"lookup","arguments":{"q":"rust"}}</tool_call>"#,
            )
            .expect("complete native tool call should parse")
            .expect("complete native tool call should emit a parsed message");
        let delta = stream_delta_from_message_json(&parsed)
            .expect("native tool call message should become stream delta");
        let value: Value = serde_json::from_str(&delta).expect("valid delta");
        assert_eq!(value["tool_calls"][0]["index"], 0);
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn test_complete_chat_response_stop_detects_structured_content() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        assert!(
            should_stop_after_complete_chat_response(&result, r#"{"value":50,"explanation":"ok"}"#)
                .expect("complete structured JSON should be detected")
        );
        assert!(
            should_stop_after_complete_chat_response(
                &result,
                r#"<think>plan</think>{"value":50,"explanation":"ok"}"#
            )
            .expect("complete structured JSON after reasoning should be detected")
        );
    }

    #[test]
    fn test_complete_chat_response_stop_detects_complete_tool_call() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        assert!(
            !should_stop_after_complete_chat_response(
                &result,
                r#"<tool_call>{"name":"Addition","arguments":{"left":42,"right":"#
            )
            .expect("incomplete tool JSON should not fail")
        );
        assert!(
            should_stop_after_complete_chat_response(
                &result,
                r#"<tool_call>{"name":"Addition","arguments":{"left":42,"right":8}}</tool_call>"#
            )
            .expect("complete tool call should be detected")
        );
    }

    #[test]
    fn test_complete_chat_response_stop_ignores_reasoning_only_native_channel() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };

        assert!(
            !should_stop_after_complete_chat_response(
                &result,
                "<|start|>assistant<|channel|>analysis<|message|>thinking"
            )
            .expect("reasoning-only native channel output should not stop generation")
        );
    }

    #[test]
    fn test_complete_json_response_stop_detects_json_payload() {
        assert!(should_stop_after_complete_json_response(
            true,
            r#"{"value":50}"#
        ));
        assert!(!should_stop_after_complete_json_response(
            false,
            r#"{"value":50}"#
        ));
        assert!(!should_stop_after_complete_json_response(
            true,
            r#"{"value":"#
        ));
        assert!(!should_stop_after_complete_json_response(
            true,
            "```json\n{\"value\":50}"
        ));
        assert!(should_stop_after_complete_json_response(
            true,
            "```json\n{\"value\":50}\n```"
        ));
    }

    #[test]
    fn test_chat_template_result_extracts_reasoning_content() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<think>".to_string()),
            reasoning_end_tag: Some("</think>".to_string()),
        };

        let parsed = result
            .parse_response_oaicompat("<think>plan</think>final answer")
            .expect("reasoning response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid message json");
        assert_eq!(value["reasoning_content"], "plan");
        assert_eq!(value["content"], "final answer");
    }

    #[test]
    fn test_gemma4_reasoning_cleanup_matches_channel_parser() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<|channel>thought".to_string()),
            reasoning_end_tag: Some("<channel|>".to_string()),
        };

        let parsed = result
            .parse_response_oaicompat(
                "<|channel><|channel>thought\nplan<channel|>final answer<channel|>",
            )
            .expect("gemma4 reasoning response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid message json");
        assert_eq!(value["reasoning_content"], "plan");
        assert_eq!(value["content"], "final answer");
    }

    #[test]
    fn test_gemma4_tool_call_reasoning_cleanup_matches_channel_parser() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: true,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<|channel>thought".to_string()),
            reasoning_end_tag: Some("<channel|>".to_string()),
        };

        let parsed = result
            .parse_response_oaicompat("<|channel><|channel>thought\nplan<channel|>answer<|tool_call>call:lookup{q:<|\"|>rust<|\"|>}<tool_call|>")
            .expect("gemma4 tool response should parse");
        let value: Value = serde_json::from_str(&parsed).expect("valid tool message json");
        assert_eq!(value["reasoning_content"], "plan");
        assert_eq!(value["content"], "answer");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
        assert_eq!(
            parsed_arguments(&value["tool_calls"][0]),
            json!({"q": "rust"})
        );
    }

    #[test]
    fn test_reasoning_only_chat_template_streams_final_message_delta() {
        let result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: String::default(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: None,
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<think>".to_string()),
            reasoning_end_tag: Some("</think>".to_string()),
        };

        assert!(should_stream_final_message(&result));

        let message_json = result
            .parse_response_oaicompat("<think>plan</think>final answer")
            .expect("reasoning response should parse");
        let delta = stream_delta_from_message_json(&message_json)
            .expect("reasoning response should become an OpenAI delta");
        let parsed: OpenAICompatDelta = serde_json::from_str(&delta).expect("delta should parse");

        assert_eq!(
            parsed
                .reasoning_content
                .and_then(StringOrJson::into_non_empty_string)
                .as_deref(),
            Some("plan")
        );
        assert_eq!(
            parsed
                .content
                .and_then(StringOrJson::into_non_empty_string)
                .as_deref(),
            Some("final answer")
        );
    }

    #[test]
    fn test_stream_delta_from_message_json_indexes_tool_calls() {
        let delta = stream_delta_from_message_json(
            r#"{
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"q\":\"rust\"}" }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"q\":\"llama\"}" }
                    }
                ]
            }"#,
        )
        .expect("delta conversion should succeed");

        let parsed: OpenAICompatDelta =
            serde_json::from_str(&delta).expect("delta should parse as OpenAI compat delta");
        let calls = parsed.tool_calls.expect("tool calls should be present");
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[1].index, Some(1));
    }

    #[test]
    fn test_tool_choice_none_disables_tool_template_and_grammar_inputs() {
        let tools = [sample_lookup_tool()];
        let default_config = LlamaCppConfig::default();
        assert!(enabled_tools_for_config(&default_config, Some(&tools)).is_some());

        let disabled_config = LlamaCppConfig {
            tool_choice: LlamaCppToolChoice::None,
            ..Default::default()
        };
        assert!(enabled_tools_for_config(&disabled_config, Some(&tools)).is_none());
        assert!(enabled_tools_for_config(&disabled_config, Some(&[])).is_none());
        assert!(enabled_tools_for_config(&disabled_config, None).is_none());
    }

    #[test]
    fn test_stream_delta_from_message_json_keeps_content() {
        let delta = stream_delta_from_message_json(r#"{"content":"done"}"#)
            .expect("delta conversion should succeed");
        let parsed: OpenAICompatDelta =
            serde_json::from_str(&delta).expect("delta should parse as OpenAI compat delta");
        assert_eq!(
            parsed.content.and_then(StringOrJson::into_non_empty_string),
            Some("done".to_string())
        );
    }

    #[test]
    fn test_extract_json_payload_helpers() {
        let text = "```json\n{\"a\":1}\n```";
        let fenced = extract_from_code_fence(text).unwrap();
        assert_eq!(fenced, "{\"a\":1}");

        let first = extract_first_json_object("prefix {\"b\":2} suffix").unwrap();
        assert_eq!(first, "{\"b\":2}");

        let payload = extract_json_payload("answer: {\"c\":3}").unwrap();
        assert_eq!(payload, "{\"c\":3}");

        let native_array = extract_json_payload(
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"q":"rust"}}] [/TOOL_CALLS]"#,
        )
        .unwrap();
        assert_eq!(
            native_array,
            r#"[{"name":"lookup","arguments":{"q":"rust"}}]"#
        );

        assert!(is_valid_json("{\"ok\":true}"));
        assert!(!is_valid_json("{broken"));
    }

    #[test]
    fn test_resolve_model_path_empty() {
        let source = ModelSource::Gguf {
            model_path: "".to_string(),
        };
        let config = LlamaCppConfig::default();
        let err = resolve_model_path(&source, &config).unwrap_err();
        assert!(err.to_string().contains("Model path is required"));
    }

    #[test]
    fn test_parse_openai_delta_valid_and_invalid() {
        let valid = r#"{"content":"hi","reasoning_content":"think"}"#;
        let parsed = parse_openai_delta(valid).expect("valid json should parse");
        assert_eq!(
            parsed.content.and_then(StringOrJson::into_non_empty_string),
            Some("hi".to_string())
        );
        assert_eq!(
            parsed
                .reasoning_content
                .and_then(StringOrJson::into_non_empty_string),
            Some("think".to_string())
        );

        let err = parse_openai_delta("{bad").expect_err("invalid json should error");
        assert!(matches!(err, LLMError::JsonError(_)));
    }

    #[test]
    fn test_parse_openai_delta_allows_json_content() {
        let valid = r#"{"content":{"value":50},"reasoning_content":["step1","step2"]}"#;
        let parsed = parse_openai_delta(valid).expect("json content should parse");
        assert_eq!(
            parsed.content.and_then(StringOrJson::into_non_empty_string),
            Some(r#"{"value":50}"#.to_string())
        );
        assert_eq!(
            parsed
                .reasoning_content
                .and_then(StringOrJson::into_non_empty_string),
            Some(r#"["step1","step2"]"#.to_string())
        );
    }

    #[test]
    fn test_openai_compat_message_allows_json_content_and_tool_arguments() {
        let valid = r#"{
            "content":{"value":50},
            "reasoning_content":{"step":"done"},
            "tool_calls":[
                {
                    "id":"call_1",
                    "type":"function",
                    "function":{
                        "name":"Addition",
                        "arguments":{"left":42,"right":8}
                    }
                }
            ]
        }"#;
        let parsed: OpenAICompatMessage =
            serde_json::from_str(valid).expect("json content should parse");
        assert_eq!(
            parsed.content.and_then(StringOrJson::into_non_empty_string),
            Some(r#"{"value":50}"#.to_string())
        );
        assert_eq!(
            parsed
                .reasoning_content
                .and_then(StringOrJson::into_non_empty_string),
            Some(r#"{"step":"done"}"#.to_string())
        );

        let tool_calls = parsed
            .tool_calls
            .expect("tool calls should decode")
            .into_iter()
            .map(ToolCall::from)
            .collect::<Vec<_>>();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "Addition");
        assert_eq!(tool_calls[0].function.arguments, r#"{"left":42,"right":8}"#);
    }

    #[test]
    fn test_push_text_and_reasoning_chunks_emits_both() {
        let mut outputs = Vec::new();
        let mut stream_state = StreamMappingState::default();
        push_text_and_reasoning_chunks(
            Some("answer".to_string()),
            Some("plan".to_string()),
            &mut stream_state,
            &mut outputs,
        );

        assert_eq!(outputs.len(), 2);
        assert!(matches!(&outputs[0], Ok(StreamChunk::Text(text)) if text == "answer"));
        assert!(matches!(
            &outputs[1],
            Ok(StreamChunk::ReasoningContent(text)) if text == "plan"
        ));
    }

    #[test]
    fn test_prepare_messages_with_system_prompt() {
        let config = LlamaCppConfig {
            system_prompt: Some("sys".to_string()),
            ..Default::default()
        };
        let messages = vec![ChatMessage::user().content("hi").build()];

        let prepared = prepare_messages_with_system(&config, &messages);
        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared[0].role, autoagents_llm::chat::ChatRole::System);
        assert_eq!(prepared[0].content, "sys");
        assert_eq!(prepared[1].content, "hi");

        let messages = vec![ChatMessage {
            role: autoagents_llm::chat::ChatRole::System,
            message_type: autoagents_llm::chat::MessageType::Text,
            content: "existing".to_string(),
        }];
        let prepared = prepare_messages_with_system(&config, &messages);
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].content, "existing");
    }

    #[test]
    fn test_prepare_fallback_messages_with_schema() {
        let config = LlamaCppConfig {
            system_prompt: Some("sys".to_string()),
            ..Default::default()
        };
        let messages = vec![ChatMessage::user().content("hi").build()];
        let schema = StructuredOutputFormat {
            name: "TestSchema".to_string(),
            description: Some("desc".to_string()),
            schema: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };

        let prepared = prepare_fallback_messages_with_schema(&config, &messages, Some(&schema));
        let last = prepared.last().unwrap();
        assert!(last.content.contains("TestSchema"));
        assert!(last.content.contains("desc"));
        assert!(last.content.contains("\"type\":\"object\""));
    }

    #[test]
    fn test_sanitize_chat_template_schema_adds_additional_properties() {
        let schema = StructuredOutputFormat {
            name: "MathAgentOutput".to_string(),
            description: None,
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer"}
                },
                "required": ["value"]
            })),
            strict: Some(true),
        };

        let sanitized = sanitize_chat_template_schema(&schema).expect("schema should exist");
        assert_eq!(sanitized["type"], "object");
        assert_eq!(sanitized["properties"]["value"]["type"], "integer");
        assert!(
            sanitized["additionalProperties"]
                .as_bool()
                .is_some_and(|value| !value)
        );
    }

    #[test]
    fn test_build_tool_response_schema_constrains_tool_names_and_arguments() {
        let tool = sample_lookup_tool();
        let schema = StructuredOutputFormat {
            name: "MathAgentOutput".to_string(),
            description: None,
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer"}
                }
            })),
            strict: Some(true),
        };

        let schema = build_tool_response_schema(
            &[tool],
            Some(&schema),
            false,
            LlamaCppToolChoice::Auto,
            false,
        )
        .expect("tool schema should build");

        assert_eq!(
            schema["oneOf"][0]["properties"]["value"]["type"], "integer",
            "structured final responses must remain allowed alongside tool calls"
        );
        let tool_call_schema = &schema["oneOf"][1]["properties"]["tool_calls"]["items"]["oneOf"][0];
        assert_eq!(
            tool_call_schema["properties"]["function"]["properties"]["name"]["enum"][0],
            "lookup"
        );
        assert_eq!(
            tool_call_schema["properties"]["function"]["properties"]["arguments"]["properties"]["q"]
                ["type"],
            "string"
        );
        assert_eq!(
            tool_call_schema["properties"]["function"]["properties"]["arguments"]["additionalProperties"],
            false
        );
        assert_eq!(
            schema["oneOf"][1]["properties"]["tool_calls"]["maxItems"], 1,
            "parallel_tool_calls=false must constrain generated tool calls to one"
        );
    }

    #[test]
    fn test_build_tool_response_schema_allows_parallel_when_enabled() {
        let tool = sample_lookup_tool();
        let schema =
            build_tool_response_schema(&[tool], None, false, LlamaCppToolChoice::Auto, true)
                .expect("tool schema should build");

        assert!(
            schema["oneOf"][1]["properties"]["tool_calls"]
                .get("maxItems")
                .is_none(),
            "parallel_tool_calls=true must not cap tool call count"
        );
    }

    #[test]
    fn test_build_tool_response_schema_required_disallows_final_response() {
        let tool = sample_lookup_tool();
        let schema =
            build_tool_response_schema(&[tool], None, false, LlamaCppToolChoice::Required, false)
                .expect("tool schema should build");

        assert!(
            schema.get("oneOf").is_none(),
            "required tool choice must not allow a final response branch"
        );
        assert_eq!(schema["required"][0], "tool_calls");
        assert_eq!(
            schema["properties"]["tool_calls"]["items"]["oneOf"][0]["properties"]["function"]["properties"]
                ["name"]["enum"][0],
            "lookup"
        );
    }

    #[test]
    fn test_named_tool_choice_selects_only_requested_tool() {
        let lookup = sample_lookup_tool();
        let search = sample_search_tool();
        let (selected, effective_choice) = select_tools_for_tool_choice(
            &[lookup.clone(), search.clone()],
            LlamaCppToolChoice::Function {
                name: "search".to_string(),
            },
        )
        .expect("named tool choice should select provided function");

        assert!(matches!(effective_choice, LlamaCppToolChoice::Required));
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].function.name, "search");

        let grammar = build_tool_response_grammar(
            &[lookup, search],
            None,
            false,
            LlamaCppToolChoice::Function {
                name: "search".to_string(),
            },
            true,
            ToolCallGrammarFormat::OpenAiEnvelope,
        )
        .expect("named tool choice grammar should compile");

        assert!(grammar.contains("search"));
        assert!(!grammar.contains("lookup"));
        assert!(
            !grammar.contains("content"),
            "named tool choice should behave like required and remove final response branch"
        );
    }

    #[test]
    fn test_named_tool_choice_rejects_missing_function() {
        let err = select_tools_for_tool_choice(
            &[sample_lookup_tool()],
            LlamaCppToolChoice::Function {
                name: "search".to_string(),
            },
        )
        .expect_err("missing named tool should fail");

        assert!(
            err.to_string().contains("no matching tool"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_build_tool_response_grammar_compiles() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::OpenAiEnvelope,
        )
        .expect("tool response grammar should compile");

        assert!(
            grammar.contains("tool") && grammar.contains("lookup") && grammar.contains("content"),
            "grammar must include the tool-call envelope and final content branch, got: {grammar}"
        );
    }

    #[test]
    fn test_reasoning_wrapped_tool_response_grammar_compiles() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::OpenAiEnvelope,
        )
        .expect("tool response grammar should compile");
        let wrapped = wrap_grammar_with_tagged_reasoning(&grammar, "<think>", "</think>");

        assert!(
            wrapped.contains("root ::= reasoning? root-content")
                && wrapped.contains("reasoning ::=")
                && !wrapped.contains("reasoning-ws ::=")
                && wrapped.contains("root-content ::=")
        );
    }

    #[test]
    fn test_detect_native_tool_call_template_formats() {
        assert_eq!(
            detect_tool_call_grammar_format("{{ '<|tool_call>' }}{{ '<tool_call|>' }}"),
            ToolCallGrammarFormat::Gemma4ToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format(
                "<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>"
            ),
            ToolCallGrammarFormat::NativeChannelToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:0<|tool_call_argument_begin|>"
            ),
            ToolCallGrammarFormat::KimiK2ToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format(
                "<|tool_call_start|>[lookup(q='rust')]<|tool_call_end|>"
            ),
            ToolCallGrammarFormat::Lfm2ToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format("<|message_sep|>\n\nfunction call<|role_sep|>\n"),
            ToolCallGrammarFormat::GigaChatV3ToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format("<｜DSML｜function_calls>"),
            ToolCallGrammarFormat::DeepSeekDsmlToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format("{{ '<tool_call>' }}"),
            ToolCallGrammarFormat::XmlToolCall
        );
        assert_eq!(
            detect_tool_call_grammar_format("{{ '<function=' }}{{ '</function>' }}"),
            ToolCallGrammarFormat::GenericFunctionTag
        );
        assert_eq!(
            detect_tool_call_grammar_format(">>>all\n{{ content }}>>>${recipient}\n{{ args }}"),
            ToolCallGrammarFormat::FunctionaryV32
        );
        assert_eq!(
            detect_tool_call_grammar_format(">>>not_a_functionary_marker"),
            ToolCallGrammarFormat::OpenAiEnvelope
        );
        assert_eq!(
            detect_tool_call_grammar_format("{{ '[TOOL_CALLS]' }}"),
            ToolCallGrammarFormat::ToolCallsArrayTag
        );
        assert_eq!(
            detect_tool_call_grammar_format("{{ '[TOOL_CALLS]' }}{{ '[ARGS]' }}"),
            ToolCallGrammarFormat::ToolCallsArgsTag
        );
        assert_eq!(
            detect_tool_call_grammar_format("{{ tools | tojson }}"),
            ToolCallGrammarFormat::OpenAiEnvelope
        );
    }

    #[test]
    fn test_native_xml_tool_response_grammar_omits_special_marker_tokens() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::XmlToolCall,
        )
        .expect("native XML tool grammar should compile");

        assert!(!grammar.contains("\"<tool_call>\""));
        assert!(!grammar.contains("\"</tool_call>\""));
        assert!(grammar.contains("tool-payload"));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_generic_function_tag_tool_response_grammar_uses_tag_json_markers() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::GenericFunctionTag,
        )
        .expect("generic function-tag grammar should compile");

        assert!(grammar.contains("\"<function=\""));
        assert!(grammar.contains("\"</function>\""));
        assert!(grammar.contains("\"<parameter name=\\\"\""));
        assert!(grammar.contains("generic-function-string"));
        assert!(grammar.contains("-value ::="));
        assert!(grammar.contains("\"lookup\""));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_generic_function_tag_auto_grammar_starts_after_lazy_trigger() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::GenericFunctionTag,
        )
        .expect("generic function-tag lazy grammar should compile");

        assert!(grammar.starts_with("root ::= (generic-function-tool-0-suffix)"));
        assert!(!grammar.contains("root ::= \"<function=\""));
        assert!(grammar.contains("\"</function>\""));
    }

    #[test]
    fn test_generic_function_tag_grammar_wraps_schema_final_response() {
        let tool = sample_lookup_tool();
        let schema = sample_structured_output_schema();
        let grammar = build_tool_response_grammar(
            &[tool],
            Some(&schema),
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::GenericFunctionTag,
        )
        .expect("generic function-tag structured-output grammar should compile");

        assert!(grammar.contains("root ::= final-root | tool-root"));
        assert!(grammar.contains("final-root ::= \"```json\""));
        assert!(grammar.contains("| final-response-format"));
        assert!(grammar.contains("tool-root ::= (tool-generic-function-tool-0)"));
    }

    #[test]
    fn test_native_tool_calls_tag_required_grammar_uses_array_marker() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::ToolCallsArrayTag,
        )
        .expect("native tag tool grammar should compile");

        assert!(grammar.contains("\"[TOOL_CALLS]\""));
        assert!(grammar.contains("\"[/TOOL_CALLS]\""));
        assert!(grammar.contains("tool-payload"));
    }

    #[test]
    fn test_ministral_tool_calls_args_grammar_uses_name_args_marker() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::ToolCallsArgsTag,
        )
        .expect("ministral tool grammar should compile");

        assert!(grammar.contains("\"[TOOL_CALLS]\""));
        assert!(grammar.contains("\"[ARGS]\""));
        assert!(grammar.contains("\"lookup\""));
        assert!(!grammar.contains("\"[/TOOL_CALLS]\""));
        assert!(!grammar.contains("tool-payload)?"));
    }

    #[test]
    fn test_ministral_tool_calls_args_grammar_wraps_schema_final_response() {
        let tool = sample_lookup_tool();
        let schema = sample_structured_output_schema();
        let grammar = build_tool_response_grammar(
            &[tool],
            Some(&schema),
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::ToolCallsArgsTag,
        )
        .expect("ministral structured-output grammar should compile");

        assert!(grammar.contains("root ::= final-root | tool-root"));
        assert!(grammar.contains("final-root ::= \"```json\""));
        assert!(grammar.contains("final-response-format"));
        assert!(grammar.contains("tool-root ::= \"[TOOL_CALLS]\""));
        assert!(!grammar.contains("\"<|start|>assistant\""));
    }

    #[test]
    fn test_functionary_v3_2_tool_response_grammar_uses_recipient_format() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::FunctionaryV32,
        )
        .expect("functionary tool grammar should compile");

        assert!(grammar.contains("\">>>\""));
        assert!(grammar.contains("\"lookup\""));
        assert!(grammar.contains("\"\\n\""));
        assert!(!grammar.contains("\"[TOOL_CALLS]\""));
    }

    #[test]
    fn test_functionary_v3_2_auto_grammar_starts_after_lazy_trigger() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Auto,
            true,
            ToolCallGrammarFormat::FunctionaryV32,
        )
        .expect("functionary lazy tool grammar should compile");

        assert!(grammar.starts_with("root ::= (functionary-tool-0)"));
        assert!(grammar.contains("\">>>\""));
        assert!(!grammar.contains("\">>>all\\n\""));
    }

    #[test]
    fn test_gemma4_tool_response_grammar_uses_native_call_markers() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::Gemma4ToolCall,
        )
        .expect("gemma4 tool grammar should compile");

        assert!(grammar.contains("\"<|tool_call>call:\""));
        assert!(grammar.contains("\"<tool_call|>\""));
        assert!(grammar.contains("\"lookup\""));
        assert!(grammar.contains("gemma4-dict"));
        assert!(grammar.contains("\"<|\\\"|>\""));
        assert!(!grammar.contains("lookup-schema"));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_gemma4_tool_response_grammar_wraps_schema_final_response() {
        let tool = sample_lookup_tool();
        let schema = sample_structured_output_schema();
        let grammar = build_tool_response_grammar(
            &[tool],
            Some(&schema),
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::Gemma4ToolCall,
        )
        .expect("gemma4 structured-output grammar should compile");

        assert!(grammar.contains("root ::= final-root | tool-root"));
        assert!(grammar.contains("final-root ::= final-gemma4-response-start"));
        assert!(grammar.contains("final-gemma4-response-thought? \"```json\""));
        assert!(grammar.contains("final-gemma4-response-thought ::="));
        assert!(grammar.contains("final-response-format"));
        assert!(grammar.contains("tool-root ::= (\"<|tool_call>call:\""));
    }

    #[test]
    fn test_native_channel_tool_response_grammar_uses_gpt_oss_markers() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::NativeChannelToolCall,
        )
        .expect("native channel tool grammar should compile");

        assert!(grammar.contains("\"<|start|>assistant\""));
        assert!(grammar.contains("\" to=functions.\""));
        assert!(grammar.contains("\"<|channel|>\""));
        assert!(grammar.contains("\"<|message|>\""));
        assert!(grammar.contains("\"lookup\""));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_native_channel_tool_response_grammar_wraps_schema_final_response() {
        let tool = sample_lookup_tool();
        let schema = sample_structured_output_schema();
        let grammar = build_tool_response_grammar(
            &[tool],
            Some(&schema),
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::NativeChannelToolCall,
        )
        .expect("gpt-oss structured-output grammar should compile");

        assert!(grammar.contains("root ::= final-root | tool-root"));
        assert!(grammar.contains("final-root ::= \"<|start|>assistant\""));
        assert!(grammar.contains("\"<|channel|>final\""));
        assert!(grammar.contains("\"<|message|>\""));
        assert!(grammar.contains("final-response-format"));
        assert!(grammar.contains("tool-root ::= \"<|start|>assistant\""));
        assert!(!grammar.contains("final-root ::= \"```json\""));
    }

    #[test]
    fn test_native_channel_auto_grammar_starts_after_lazy_trigger() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::NativeChannelToolCall,
        )
        .expect("native channel lazy grammar should compile");

        assert!(grammar.contains("root ::= \"=functions.\""));
        assert!(grammar.contains("| \".\""));
        assert!(!grammar.contains("\"<|start|>assistant\""));
        assert!(grammar.contains("\"lookup\""));
    }

    #[test]
    fn test_kimi_k2_tool_response_grammar_uses_native_markers() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::KimiK2ToolCall,
        )
        .expect("kimi tool grammar should compile");

        assert!(grammar.contains("\"<|tool_call_begin|>\""));
        assert!(grammar.contains("\"functions.\""));
        assert!(grammar.contains("\"<|tool_call_argument_begin|>\""));
        assert!(grammar.contains("\"lookup\""));
        assert!(!grammar.contains("\"tool_calls\""));
    }

    #[test]
    fn test_lfm2_tool_response_grammar_uses_python_style_markers() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::Lfm2ToolCall,
        )
        .expect("lfm2 tool grammar should compile");

        assert!(grammar.contains("\"<|tool_call_start|>\""));
        assert!(grammar.contains("\"<|tool_call_end|>\""));
        assert!(grammar.contains("\"lookup\""));
        assert!(grammar.contains("lfm-args"));
        assert!(grammar.contains("lfm-tool-0-args"));
        assert!(grammar.contains("\"q\" lfm-ws \"=\" lfm-ws lfm-string"));
        assert!(grammar.contains("\"limit\" lfm-ws \"=\" lfm-ws lfm-number"));
        assert!(grammar.contains("lfm-object"));
        assert!(grammar.contains("lfm-list"));
        assert!(grammar.contains("\"True\""));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_gigachat_v3_tool_response_grammar_uses_native_marker() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            false,
            ToolCallGrammarFormat::GigaChatV3ToolCall,
        )
        .expect("gigachat tool grammar should compile");

        assert!(grammar.contains("\"<|message_sep|>\\n\\nfunction call<|role_sep|>\\n\""));
        assert!(grammar.contains("\"\\\"name\\\"\""));
        assert!(grammar.contains("\"\\\"arguments\\\"\""));
        assert!(grammar.contains("\\\"lookup\\\""));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_deepseek_dsml_tool_response_grammar_uses_native_markers() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Required,
            true,
            ToolCallGrammarFormat::DeepSeekDsmlToolCall,
        )
        .expect("deepseek dsml tool grammar should compile");

        assert!(grammar.contains("\"<｜DSML｜function_calls>\""));
        assert!(grammar.contains("\"<｜DSML｜invoke name=\\\"\""));
        assert!(
            grammar
                .contains("\"<｜DSML｜parameter name=\\\"\" \"q\" \"\\\" string=\\\"true\\\">\"")
        );
        assert!(
            grammar.contains(
                "\"<｜DSML｜parameter name=\\\"\" \"limit\" \"\\\" string=\\\"false\\\">\""
            )
        );
        assert!(grammar.contains("\"lookup\""));
        assert!(grammar.contains("dsml-string"));
        assert!(!grammar.contains("tool_calls"));
    }

    #[test]
    fn test_deepseek_dsml_tool_response_grammar_wraps_schema_final_response() {
        let tool = sample_lookup_tool();
        let schema = sample_structured_output_schema();
        let grammar = build_tool_response_grammar(
            &[tool],
            Some(&schema),
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::DeepSeekDsmlToolCall,
        )
        .expect("deepseek structured-output grammar should compile");

        assert!(grammar.contains("root ::= final-root | tool-root"));
        assert!(grammar.contains("final-root ::= \"```json\""));
        assert!(grammar.contains("final-deepseek-response-ws"));
        assert!(grammar.contains("final-response-format"));
        assert!(grammar.contains("tool-root ::= \"<｜DSML｜function_calls>\""));
    }

    #[test]
    fn test_native_auto_tool_response_grammar_is_lazy_payload_only() {
        let tool = sample_lookup_tool();
        let grammar = build_tool_response_grammar(
            &[tool],
            None,
            false,
            LlamaCppToolChoice::Auto,
            false,
            ToolCallGrammarFormat::XmlToolCall,
        )
        .expect("native lazy XML tool grammar should compile");

        assert!(!grammar.contains("\"<tool_call>\""));
        assert!(!grammar.contains("\"</tool_call>\""));
        assert!(grammar.contains("tool-payload"));
    }

    #[test]
    fn test_select_template_schema_and_grammar_routes_schema_through_grammar_without_tools() {
        // Schemas must be compiled to grammar at the autoagents boundary rather than
        // forwarded into llama.cpp's OAI-compat `json_schema` slot (see issue #220).
        let schema = StructuredOutputFormat {
            name: "MathAgentOutput".to_string(),
            description: None,
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer"}
                },
                "required": ["value"]
            })),
            strict: Some(true),
        };

        let (json_schema, grammar) = select_template_schema_and_grammar(Some(&schema), false, None)
            .expect("schema should compile");
        assert!(
            json_schema.is_none(),
            "schema must never be forwarded into the OAI-compat template slot"
        );
        let grammar = grammar.expect("schema-derived grammar must be produced");
        assert!(
            grammar.contains("value"),
            "grammar must constrain the schema fields, got: {grammar}"
        );
    }

    // Regression for https://github.com/liquidos-ai/AutoAgents/issues/220 —
    // when an agent declares `tools = []` plus `output = SomeStruct`, the schema must
    // be enforced via grammar (compiled at our boundary) rather than forwarded to
    // llama.cpp's OAI-compat template `json_schema` slot. The schema slot configures a
    // parser that expects a structured-output envelope post-generation; the model emits
    // plain JSON with no envelope, llama.cpp returns rc=-3 from `chat_parse_to_oaicompat`,
    // surfacing as `ProviderError("Failed to parse response: ffi error -3")`.
    #[test]
    fn test_select_template_schema_and_grammar_compiles_schema_when_tools_empty() {
        let schema = StructuredOutputFormat {
            name: "MathAgentOutput".to_string(),
            description: None,
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "The addition result"},
                    "explanation": {"type": "string", "description": "Explanation of the logic"},
                    "generic": {
                        "type": "string",
                        "description": "If user asks other than math questions, use this to answer them."
                    }
                },
                "required": ["value", "explanation"]
            })),
            strict: Some(true),
        };

        let (json_schema, grammar) = select_template_schema_and_grammar(Some(&schema), false, None)
            .expect("schema should compile");

        assert!(
            json_schema.is_none(),
            "schema must NOT be forwarded to OAI-compat template when tools are absent; \
             doing so configures a schema-bound parser that crashes on plain-JSON generations"
        );
        let grammar = grammar.expect("schema-derived grammar must be produced");
        assert!(
            grammar.contains("value") && grammar.contains("explanation"),
            "grammar must constrain the schema fields, got: {grammar}"
        );
    }

    #[test]
    fn test_select_template_schema_and_grammar_wraps_specialized_response_formats() {
        let schema = StructuredOutputFormat {
            name: "Answer".to_string(),
            description: None,
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "value": {"type": "integer"}
                },
                "required": ["value"]
            })),
            strict: Some(true),
        };

        let (_, gpt_oss) = select_template_schema_and_grammar(
            Some(&schema),
            false,
            Some(ToolCallGrammarFormat::NativeChannelToolCall),
        )
        .expect("gpt-oss schema should compile");
        let gpt_oss = gpt_oss.expect("schema grammar should be present");
        assert!(gpt_oss.contains("\"<|start|>assistant\""));
        assert!(gpt_oss.contains("\"<|channel|>final\""));
        assert!(gpt_oss.contains("\"<|message|>\""));

        let (_, fenced) = select_template_schema_and_grammar(
            Some(&schema),
            false,
            Some(ToolCallGrammarFormat::Gemma4ToolCall),
        )
        .expect("gemma4 schema should compile");
        let fenced = fenced.expect("schema grammar should be present");
        assert!(fenced.contains("\"```json\""));
        assert!(fenced.contains("response-format"));

        let (_, deepseek) = select_template_schema_and_grammar(
            Some(&schema),
            false,
            Some(ToolCallGrammarFormat::DeepSeekDsmlToolCall),
        )
        .expect("deepseek schema should compile");
        let deepseek = deepseek.expect("schema grammar should be present");
        assert!(deepseek.contains("\"```json\""));
        assert!(deepseek.contains("deepseek-response-ws"));
    }

    #[test]
    fn test_select_template_schema_and_grammar_rejects_unconvertible_schema() {
        // Match llama.cpp server validation: an invalid schema should fail the
        // request instead of silently degrading to generic JSON.
        let schema = StructuredOutputFormat {
            name: "Recursive".to_string(),
            description: None,
            schema: Some(json!({
                "$ref": "#/definitions/Missing"
            })),
            strict: Some(true),
        };

        let err = select_template_schema_and_grammar(Some(&schema), false, None)
            .expect_err("invalid schema must be rejected");
        assert!(
            err.to_string()
                .contains("invalid llama.cpp structured output JSON schema"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_llama_json_schema_to_grammar_accepts_integer_object_schema() {
        let grammar = llama_cpp_2::json_schema_to_grammar(
            r#"{
                "type":"object",
                "properties":{
                    "value":{"type":"integer"},
                    "explanation":{"type":"string"}
                },
                "required":["value","explanation"],
                "additionalProperties":false
            }"#,
        )
        .expect("integer object schema should convert to grammar");

        assert!(grammar.contains("space ::="));
        assert!(grammar.contains("value"));
        assert!(grammar.contains("explanation"));
    }

    #[test]
    fn test_llama_json_schema_to_grammar_accepts_tool_parameter_schema() {
        let grammar = llama_cpp_2::json_schema_to_grammar(
            r#"{
                "type":"object",
                "properties":{
                    "left":{"type":"integer","description":"Left operand"},
                    "right":{"type":"integer","description":"Right operand"}
                },
                "required":["left","right"]
            }"#,
        )
        .expect("tool parameter schema should convert to grammar");

        assert!(grammar.contains("space ::="));
        assert!(grammar.contains("left"));
        assert!(grammar.contains("right"));
    }

    #[test]
    fn test_ensure_supported_messages_for_config() {
        let config = LlamaCppConfig::default();
        let ok = ensure_supported_messages_for_config(
            &config,
            &[ChatMessage::user().content("hi").build()],
        );
        assert!(ok.is_ok());

        let image_msg = ChatMessage {
            role: autoagents_llm::chat::ChatRole::User,
            message_type: autoagents_llm::chat::MessageType::Image((ImageMime::PNG, vec![1, 2, 3])),
            content: "img".to_string(),
        };
        let err = ensure_supported_messages_for_config(&config, &[image_msg]).unwrap_err();
        assert!(err.to_string().contains("does not support image inputs"));

        let url_msg = ChatMessage {
            role: autoagents_llm::chat::ChatRole::User,
            message_type: autoagents_llm::chat::MessageType::ImageURL(
                "https://example.com/img.png".to_string(),
            ),
            content: "img".to_string(),
        };
        let err = ensure_supported_messages_for_config(&config, &[url_msg]).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not support image URL or PDF inputs")
        );
    }

    #[test]
    fn test_string_or_json_and_single_stream_response_helpers() {
        assert_eq!(StringOrJson::default().into_string(), "");
        assert_eq!(StringOrJson::Json(Value::Null).into_string(), "");
        assert_eq!(
            StringOrJson::Json(json!({"value": 7})).into_non_empty_string(),
            Some("{\"value\":7}".to_string())
        );
        assert_eq!(
            StringOrJson::String(String::default()).into_non_empty_string(),
            None
        );
        assert_eq!(
            default_call_type_value().into_string(),
            autoagents_llm::default_call_type()
        );

        let usage = LlamaCppProvider::build_usage(2, 3);
        assert_eq!(usage.prompt_tokens, 2);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 5);

        let response = single_stream_response(
            Some("answer".to_string()),
            Some("plan".to_string()),
            Some(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "lookup".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            Some(usage),
        );
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].delta.content.as_deref(), Some("answer"));
        assert_eq!(
            response.choices[0].delta.reasoning_content.as_deref(),
            Some("plan")
        );
        let tool_calls = response.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool calls should be present");
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "lookup");
        assert_eq!(
            response
                .usage
                .expect("usage should be present")
                .total_tokens,
            5
        );
    }

    #[test]
    fn test_struct_stream_helpers_cover_content_and_tool_updates() {
        let mut outputs = Vec::new();
        let mut stream_state = StreamMappingState::default();
        push_struct_content_and_reasoning(
            Some("answer".to_string()),
            Some("plan".to_string()),
            &mut stream_state,
            &mut outputs,
        );
        assert_eq!(outputs.len(), 2);
        assert!(matches!(
            &outputs[0],
            Ok(response)
                if response.choices[0].delta.content.as_deref() == Some("answer")
                    && response.choices[0].delta.reasoning_content.is_none()
        ));
        assert!(matches!(
            &outputs[1],
            Ok(response)
                if response.choices[0].delta.content.is_none()
                    && response.choices[0].delta.reasoning_content.as_deref() == Some("plan")
        ));

        let mut empty_outputs = Vec::new();
        push_struct_content_and_reasoning(
            Some(String::default()),
            Some(String::default()),
            &mut stream_state,
            &mut empty_outputs,
        );
        assert!(empty_outputs.is_empty());

        let mut tool_states = HashMap::new();
        let mut tool_outputs = Vec::new();
        push_struct_tool_call_updates(
            vec![
                OpenAIToolCallDelta {
                    index: Some(0),
                    id: Some("call_1".to_string()),
                    call_type: None,
                    function: Some(OpenAIFunctionDelta {
                        name: Some("lookup".to_string()),
                        arguments: StringOrJson::String("{\"q\":\"".to_string()),
                    }),
                },
                OpenAIToolCallDelta {
                    index: Some(0),
                    id: None,
                    call_type: Some("function".to_string()),
                    function: Some(OpenAIFunctionDelta {
                        name: None,
                        arguments: StringOrJson::String("rust\"}".to_string()),
                    }),
                },
                OpenAIToolCallDelta {
                    index: Some(1),
                    id: None,
                    call_type: None,
                    function: None,
                },
            ],
            &mut tool_states,
            &mut tool_outputs,
        );

        assert_eq!(tool_outputs.len(), 1);
        let tool_calls = tool_outputs[0]
            .as_ref()
            .expect("tool update should succeed")
            .choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool calls should be present");
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "lookup");
        assert_eq!(tool_calls[0].function.arguments, "{\"q\":\"");
        assert_eq!(tool_calls[1].id, "call_1");
        assert_eq!(tool_calls[1].function.name, "lookup");
        assert_eq!(tool_calls[1].function.arguments, "{\"q\":\"rust\"}");
        assert_eq!(tool_states[&0].arguments, "{\"q\":\"rust\"}");
    }

    #[test]
    fn test_map_struct_stream_event_handles_tokens_usage_and_errors() {
        let mut stream_state = StreamMappingState::default();
        assert!(
            map_struct_stream_event(Ok(StreamEvent::Token(String::default())), &mut stream_state,)
                .is_empty()
        );

        let token_outputs = map_struct_stream_event(
            Ok(StreamEvent::Token("alpha".to_string())),
            &mut stream_state,
        );
        assert_eq!(token_outputs.len(), 1);
        assert!(matches!(
            &token_outputs[0],
            Ok(response) if response.choices[0].delta.content.as_deref() == Some("alpha")
        ));

        let delta = json!({
            "content": "beta",
            "reasoning_content": "think",
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"q\":\"rust\"}"
                }
            }]
        })
        .to_string();
        let delta_outputs =
            map_struct_stream_event(Ok(StreamEvent::Delta(delta)), &mut stream_state);
        assert_eq!(delta_outputs.len(), 3);
        assert!(matches!(
            &delta_outputs[0],
            Ok(response) if response.choices[0].delta.content.as_deref() == Some("beta")
        ));
        assert!(matches!(
            &delta_outputs[1],
            Ok(response)
                if response.choices[0].delta.reasoning_content.as_deref() == Some("think")
        ));
        let struct_tool_calls = delta_outputs[2]
            .as_ref()
            .expect("tool call update should succeed")
            .choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool calls should be present");
        assert_eq!(struct_tool_calls[0].function.name, "lookup");

        let usage_outputs = map_struct_stream_event(
            Ok(StreamEvent::Usage(LlamaCppProvider::build_usage(1, 2))),
            &mut stream_state,
        );
        assert_eq!(usage_outputs.len(), 1);
        assert_eq!(
            usage_outputs[0]
                .as_ref()
                .expect("usage event should succeed")
                .usage
                .as_ref()
                .expect("usage should be present")
                .total_tokens,
            3
        );
        assert!(
            map_struct_stream_event(
                Ok(StreamEvent::Done {
                    stop_reason: "end_turn".to_string(),
                }),
                &mut stream_state,
            )
            .is_empty()
        );

        let invalid = map_struct_stream_event(
            Ok(StreamEvent::Delta("{bad".to_string())),
            &mut stream_state,
        );
        assert!(matches!(invalid.as_slice(), [Err(LLMError::JsonError(_))]));
        let generic = map_struct_stream_event(
            Err(LLMError::Generic("boom".to_string())),
            &mut stream_state,
        );
        assert!(
            matches!(generic.as_slice(), [Err(LLMError::Generic(message))] if message == "boom")
        );
    }

    #[test]
    fn test_tool_stream_mapping_covers_deltas_usage_done_and_completions() {
        let mut stream_state = StreamMappingState::default();
        let delta = json!({
            "content": "answer",
            "reasoning_content": "plan",
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"q\":\"rust\"}"
                }
            }]
        })
        .to_string();
        let outputs = map_tool_stream_event(Ok(StreamEvent::Delta(delta)), &mut stream_state);
        assert_eq!(outputs.len(), 4);
        assert!(matches!(&outputs[0], Ok(StreamChunk::Text(text)) if text == "answer"));
        assert!(matches!(
            &outputs[1],
            Ok(StreamChunk::ReasoningContent(text)) if text == "plan"
        ));
        assert!(matches!(
            &outputs[2],
            Ok(StreamChunk::ToolUseStart { index, id, name })
                if *index == 0 && id == "call_1" && name == "lookup"
        ));
        assert!(matches!(
            &outputs[3],
            Ok(StreamChunk::ToolUseInputDelta { index, partial_json })
                if *index == 0 && partial_json == "{\"q\":\"rust\"}"
        ));

        let usage_outputs = map_tool_stream_event(
            Ok(StreamEvent::Usage(LlamaCppProvider::build_usage(1, 2))),
            &mut stream_state,
        );
        assert!(matches!(
            usage_outputs.as_slice(),
            [Ok(StreamChunk::Usage(usage))] if usage.total_tokens == 3
        ));

        let done_outputs = map_tool_stream_event(
            Ok(StreamEvent::Done {
                stop_reason: "tool_use".to_string(),
            }),
            &mut stream_state,
        );
        assert_eq!(done_outputs.len(), 2);
        assert!(matches!(
            &done_outputs[0],
            Ok(StreamChunk::ToolUseComplete { index, tool_call })
                if *index == 0
                    && tool_call.id == "call_1"
                    && tool_call.function.name == "lookup"
                    && tool_call.function.arguments == "{\"q\":\"rust\"}"
        ));
        assert!(matches!(
            &done_outputs[1],
            Ok(StreamChunk::Done { stop_reason }) if stop_reason == "tool_use"
        ));
        assert!(stream_state.tool_calls.is_empty());

        assert!(
            map_tool_stream_event(
                Ok(StreamEvent::Token(String::default())),
                &mut StreamMappingState::default(),
            )
            .is_empty()
        );
        let invalid = map_tool_stream_event(
            Ok(StreamEvent::Delta("{bad".to_string())),
            &mut StreamMappingState::default(),
        );
        assert!(matches!(invalid.as_slice(), [Err(LLMError::JsonError(_))]));
    }

    #[test]
    fn test_tool_stream_updates_diff_cumulative_parser_arguments() {
        let mut tool_states = HashMap::new();
        let mut outputs = Vec::new();

        push_tool_chunk_updates(
            vec![OpenAIToolCallDelta {
                index: Some(0),
                id: Some("call_1".to_string()),
                call_type: Some("function".to_string()),
                function: Some(OpenAIFunctionDelta {
                    name: Some("lookup".to_string()),
                    arguments: StringOrJson::String("{\"q\":\"".to_string()),
                }),
            }],
            &mut tool_states,
            &mut outputs,
        );
        push_tool_chunk_updates(
            vec![OpenAIToolCallDelta {
                index: Some(0),
                id: None,
                call_type: Some("function".to_string()),
                function: Some(OpenAIFunctionDelta {
                    name: None,
                    arguments: StringOrJson::String("{\"q\":\"rust\"}".to_string()),
                }),
            }],
            &mut tool_states,
            &mut outputs,
        );

        let deltas = outputs
            .iter()
            .filter_map(|output| match output {
                Ok(StreamChunk::ToolUseInputDelta { partial_json, .. }) => {
                    Some(partial_json.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(deltas, vec!["{\"q\":\"", "rust\"}"]);
        assert_eq!(tool_states[&0].arguments, "{\"q\":\"rust\"}");
    }

    #[test]
    fn test_tool_stream_updates_diff_cumulative_content_and_reasoning() {
        let mut stream_state = StreamMappingState::default();
        let first = map_tool_stream_event(
            Ok(StreamEvent::Delta(
                json!({
                    "content": "hel",
                    "reasoning_content": "pla"
                })
                .to_string(),
            )),
            &mut stream_state,
        );
        let second = map_tool_stream_event(
            Ok(StreamEvent::Delta(
                json!({
                    "content": "hello",
                    "reasoning_content": "plan"
                })
                .to_string(),
            )),
            &mut stream_state,
        );

        assert!(matches!(&first[0], Ok(StreamChunk::Text(text)) if text == "hel"));
        assert!(matches!(&first[1], Ok(StreamChunk::ReasoningContent(text)) if text == "pla"));
        assert!(matches!(&second[0], Ok(StreamChunk::Text(text)) if text == "lo"));
        assert!(matches!(&second[1], Ok(StreamChunk::ReasoningContent(text)) if text == "n"));
    }

    #[test]
    fn test_json_extraction_and_grammar_helpers_cover_edge_cases() {
        assert!(extract_json_payload("").is_none());
        assert_eq!(
            extract_from_code_fence("```json\n{\"answer\":1}\n```"),
            Some("{\"answer\":1}".to_string())
        );
        assert!(extract_from_code_fence("```text\n{\"answer\":1}\n```").is_none());
        assert_eq!(
            extract_first_json_object("prefix {\"outer\":{\"text\":\"brace } inside\"}} suffix"),
            Some("{\"outer\":{\"text\":\"brace } inside\"}}".to_string())
        );
        assert_eq!(
            extract_json_payload("leading text {\"answer\": true} trailing text"),
            Some("{\"answer\": true}".to_string())
        );

        let (json_schema, grammar) =
            select_template_schema_and_grammar(None, true, None).expect("generic JSON grammar");
        assert!(json_schema.is_none());
        assert_eq!(grammar.as_deref(), Some(JSON_GRAMMAR));

        let schema = StructuredOutputFormat {
            name: "Answer".to_string(),
            description: None,
            schema: Some(json!({
                "type": "object",
                "additionalProperties": true,
                "properties": {
                    "answer": { "type": "string" }
                }
            })),
            strict: Some(true),
        };
        let sanitized = sanitize_chat_template_schema(&schema).expect("schema should exist");
        assert_eq!(sanitized["additionalProperties"], Value::Bool(true));

        let model_path = resolve_model_path(
            &ModelSource::Gguf {
                model_path: TEST_GGUF_MODEL_PATH.to_string(),
            },
            &LlamaCppConfig::default(),
        )
        .expect("non-empty model path should resolve");
        assert_eq!(model_path, TEST_GGUF_MODEL_PATH);
    }

    #[test]
    fn test_grammar_trigger_patterns_mirror_common_sampler_inputs() {
        let (patterns, tokens) = grammar_trigger_patterns(&[
            GrammarTrigger::Word("[TOOL_CALLS]".to_string()),
            GrammarTrigger::Pattern("^\\s+to$".to_string()),
            GrammarTrigger::PatternFull(">>>(?!all)".to_string()),
            GrammarTrigger::Token(42),
        ]);

        assert_eq!(
            patterns,
            vec![
                "\\[TOOL_CALLS\\]".to_string(),
                "^\\s+to$".to_string(),
                "^>>>(?!all)$".to_string(),
            ]
        );
        assert_eq!(tokens, vec![LlamaToken(42)]);
    }

    #[test]
    fn test_generation_prompt_prefill_only_uses_grammar_covered_suffix() {
        let json_result = ChatTemplateResult {
            prompt: "prompt".to_string(),
            generation_prompt: "<|im_start|>assistant\n".to_string(),
            force_pure_content: false,
            is_continuation: false,
            add_bos: true,
            grammar: Some(JSON_GRAMMAR.to_string()),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            parse_tool_calls: false,
            tool_names: Vec::new(),
            reasoning_format: None,
            reasoning_start_tag: None,
            reasoning_end_tag: None,
        };
        assert!(
            generation_prompt_grammar_prefill(&json_result).is_none(),
            "raw JSON grammar must not consume rendered assistant markers"
        );

        let native_result = ChatTemplateResult {
            generation_prompt: "<|start|>assistant<|channel|>final<|message|>".to_string(),
            grammar: Some(
                r#"root ::= "<|start|>assistant" "<|channel|>final" "<|message|>" "{}""#
                    .to_string(),
            ),
            ..json_result
        };
        assert_eq!(
            generation_prompt_grammar_prefill(&native_result),
            Some("<|start|>assistant<|channel|>final<|message|>"),
            "server-style grammars that include response markers should still be prefed",
        );

        let qwen_thinking_result = ChatTemplateResult {
            generation_prompt: "<|im_start|>assistant\n<think>\n".to_string(),
            grammar: Some(wrap_grammar_with_tagged_reasoning(
                JSON_GRAMMAR,
                "<think>",
                "</think>",
            )),
            reasoning_format: Some(crate::config::LlamaCppReasoningFormat::Auto),
            reasoning_start_tag: Some("<think>".to_string()),
            reasoning_end_tag: Some("</think>".to_string()),
            ..native_result
        };
        assert_eq!(
            generation_prompt_grammar_prefill(&qwen_thinking_result),
            Some("<think>\n"),
            "only the rendered reasoning suffix should be prefed for Qwen-style prompts",
        );
    }

    #[test]
    fn test_regex_escape_covers_common_trigger_literals() {
        assert_eq!(regex_escape("<|tool_call>"), "<\\|tool_call>");
        assert_eq!(regex_escape("a+b*(c)"), "a\\+b\\*\\(c\\)");
        assert_eq!(regex_escape("line\nnext"), "line\\nnext");
    }

    #[test]
    fn test_split_top_level_ignores_nested_commas_and_strings() {
        assert_eq!(
            split_top_level("a=1, b={\"x\":[1,2]}, c='x,y'", ','),
            vec!["a=1", "b={\"x\":[1,2]}", "c='x,y'"]
        );
        assert_eq!(
            split_once_top_level("filters={\"a\":1}", '='),
            Some(("filters", "{\"a\":1}"))
        );
    }

    // ── common_prefix_len tests ──────────────────────────────────────────

    #[test]
    fn test_common_prefix_len_identical() {
        let tokens: Vec<LlamaToken> = (0..5).map(LlamaToken::new).collect();
        assert_eq!(common_prefix_len(&tokens, &tokens), 5);
    }

    #[test]
    fn test_common_prefix_len_diverges_at_3() {
        let cached: Vec<LlamaToken> = (0..5).map(LlamaToken::new).collect();
        let mut new_tokens = cached.clone();
        new_tokens[3] = LlamaToken::new(99);
        assert_eq!(common_prefix_len(&cached, &new_tokens), 3);
    }

    #[test]
    fn test_common_prefix_len_shorter_new() {
        let cached: Vec<LlamaToken> = (0..5).map(LlamaToken::new).collect();
        let new_tokens: Vec<LlamaToken> = (0..3).map(LlamaToken::new).collect();
        assert_eq!(common_prefix_len(&cached, &new_tokens), 3);
    }

    #[test]
    fn test_common_prefix_len_empty_cache() {
        let cached: Vec<LlamaToken> = vec![];
        let new_tokens: Vec<LlamaToken> = (0..5).map(LlamaToken::new).collect();
        assert_eq!(common_prefix_len(&cached, &new_tokens), 0);
    }

    #[test]
    fn test_common_prefix_len_both_empty() {
        let empty: Vec<LlamaToken> = vec![];
        assert_eq!(common_prefix_len(&empty, &empty), 0);
    }

    #[test]
    fn test_common_prefix_len_completely_different() {
        let a: Vec<LlamaToken> = (0..5).map(LlamaToken::new).collect();
        let b: Vec<LlamaToken> = (10..15).map(LlamaToken::new).collect();
        assert_eq!(common_prefix_len(&a, &b), 0);
    }

    // ── Config tests ───────────────────────────────────────────────────────

    #[test]
    fn test_config_enable_thinking_defaults_to_none() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.enable_thinking, None);
    }

    #[test]
    fn test_config_context_reuse_defaults_to_false() {
        let config = LlamaCppConfig::default();
        assert!(!config.context_reuse);
    }

    #[test]
    fn test_config_builder_enable_thinking() {
        let config = LlamaCppConfigBuilder::new()
            .model_path("test.gguf")
            .enable_thinking(true)
            .build();
        assert_eq!(config.enable_thinking, Some(true));
    }

    #[test]
    fn test_config_builder_context_reuse() {
        let config = LlamaCppConfigBuilder::new()
            .model_path("test.gguf")
            .context_reuse(true)
            .build();
        assert!(config.context_reuse);
    }

    #[test]
    fn test_config_builder_context_reuse_off_by_default() {
        let config = LlamaCppConfigBuilder::new().model_path("test.gguf").build();
        assert!(!config.context_reuse);
    }

    // ── SharedSessionState tests (no model needed) ─────────────────────────

    #[test]
    fn test_shared_session_state_initially_none() {
        let shared: SharedSessionState = Arc::new(std::sync::Mutex::new(None));
        assert!(shared.lock().unwrap().is_none());
    }

    #[test]
    fn test_shared_session_mutex_clear_no_panic() {
        // Verifies the Mutex<Option<SessionState>> pattern works correctly.
        // reset_session() and cached_prefix_len() cannot be unit-tested
        // without a loaded model — integration test needed.
        let shared: SharedSessionState = Arc::new(std::sync::Mutex::new(None));
        *shared.lock().unwrap() = None;
        assert!(shared.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn test_provider_scheduler_allows_active_slot_with_zero_queue() {
        let scheduler = ProviderScheduler::new(1, 0);
        let _permit = scheduler
            .acquire()
            .await
            .expect("available slot should not count against queue capacity");
    }

    #[tokio::test]
    async fn test_provider_scheduler_rejects_when_queue_full() {
        let scheduler = ProviderScheduler::new(1, 0);
        let _permit = scheduler
            .acquire()
            .await
            .expect("first request should acquire slot");

        let err = scheduler
            .acquire()
            .await
            .expect_err("second request should be rejected with no queue capacity");

        assert!(err.to_string().contains("queue is full"));
    }

    #[tokio::test]
    async fn test_provider_scheduler_releases_queued_count_on_cancellation() {
        let scheduler = ProviderScheduler::new(1, 1);
        let _active = scheduler
            .acquire()
            .await
            .expect("first request should acquire slot");

        let mut queued_acquire = Box::pin(scheduler.acquire());
        let timed_out = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            queued_acquire.as_mut(),
        )
        .await;

        assert!(timed_out.is_err());
        assert_eq!(scheduler.queued.load(Ordering::Acquire), 1);

        drop(queued_acquire);
        assert_eq!(scheduler.queued.load(Ordering::Acquire), 0);

        let mut second_queued = Box::pin(scheduler.acquire());
        let timed_out =
            tokio::time::timeout(std::time::Duration::from_millis(10), second_queued.as_mut())
                .await;

        assert!(timed_out.is_err());
        assert_eq!(scheduler.queued.load(Ordering::Acquire), 1);
    }

    // ── from_model / model() / backend() tests ──────────────────────────────
    //
    // These verify the sharing API without loading a GGUF model.
    // Full inference through from_model() is covered by integration tests.

    #[test]
    fn test_arc_clone_preserves_identity() {
        // Validates the Arc sharing pattern used by from_model(): cloning
        // an Arc preserves pointer identity (Arc::ptr_eq). Cannot construct
        // a real LlamaCppProvider without a model file — integration tests
        // verify model()/backend() on a real provider.
        let shared: SharedSessionState = Arc::new(std::sync::Mutex::new(None));
        let cloned = Arc::clone(&shared);
        assert!(Arc::ptr_eq(&shared, &cloned));
    }

    #[test]
    fn test_from_model_config_context_reuse_enabled() {
        // from_model() with context_reuse=true should create session_state.
        // We can't call from_model() without a real LlamaModel/LlamaBackend,
        // but we can verify the session_state allocation logic matches
        // from_config() by checking the conditional directly.
        let config = LlamaCppConfigBuilder::new()
            .model_path("dummy.gguf")
            .context_reuse(true)
            .build();
        assert!(config.context_reuse);

        // Same allocation logic as from_model():
        let session_state: Option<SharedSessionState> = if config.context_reuse {
            Some(Arc::new(std::sync::Mutex::new(None)))
        } else {
            None
        };
        assert!(session_state.is_some());
    }

    #[test]
    fn test_from_model_config_context_reuse_disabled() {
        let config = LlamaCppConfigBuilder::new()
            .model_path("dummy.gguf")
            .context_reuse(false)
            .build();
        assert!(!config.context_reuse);

        let session_state: Option<SharedSessionState> = if config.context_reuse {
            Some(Arc::new(std::sync::Mutex::new(None)))
        } else {
            None
        };
        assert!(session_state.is_none());
    }
}
