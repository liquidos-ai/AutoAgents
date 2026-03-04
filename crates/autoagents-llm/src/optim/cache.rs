//! In-memory LLM response cache.
//!
//! # Overview
//! [`CacheLayer`] is an [`LLMLayer`] that caches chat, completion, embedding, and
//! **streaming** responses in `HashMap` buckets protected by `std::sync::RwLock`.
//!
//! ## Non-streaming methods
//! The read-lock is acquired, checked, and dropped **before** the inner provider
//! is awaited; the write-lock is acquired only after the inner call completes.
//! No cache lock is ever held across an `.await` point.
//! Concurrent misses for the same non-streaming key are coalesced via a
//! per-key async mutex (single-flight), preventing duplicate upstream calls.
//!
//! ## Streaming methods (`chat_stream`, `chat_stream_with_tools`, `chat_stream_struct`)
//! On a **cache miss** the inner stream is wrapped in a [`TeeStream`] and returned
//! to the caller immediately — chunks flow through in real time exactly as if no
//! cache were present. When the stream terminates successfully the accumulated
//! buffer is written to cache in a single synchronous lock acquisition. On error
//! the buffer is discarded and nothing is cached.
//!
//! On a **cache hit** the cached chunks are replayed as a `futures::stream::iter`
//! stream — the caller sees the same `Stream` interface and receives all chunks
//! instantly without any network round-trip.
//!
//! `chat_with_web_search` always delegates to the inner provider (web results are
//! time-sensitive and must not be cached).

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
    pin::Pin,
    sync::{Arc, RwLock, Weak},
    task::{Context, Poll},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::Stream;
use serde::Serialize;

use crate::{
    LLMProvider, ToolCall,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, StreamChunk, StreamResponse,
        StructuredOutputFormat, Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider},
    pipeline::LLMLayer,
};

// ---------------------------------------------------------------------------
// Shared cache type alias — reduces repetition for all six buckets
// ---------------------------------------------------------------------------

type Cache<T> = Arc<RwLock<HashMap<u64, CacheEntry<T>>>>;
type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, LLMError>> + Send>>;
type InFlight = Arc<tokio::sync::Mutex<HashMap<u64, Weak<tokio::sync::Mutex<()>>>>>;

fn new_cache<T>() -> Cache<T> {
    Arc::new(RwLock::new(HashMap::new()))
}

fn new_inflight() -> InFlight {
    Arc::new(tokio::sync::Mutex::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for [`CacheLayer`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Strategy for deriving chat cache keys.
    pub chat_key_mode: ChatCacheKeyMode,
    /// How long a cached response is considered fresh. `None` means no expiry.
    pub ttl: Option<Duration>,
    /// Maximum number of entries **per cache bucket** (chat, completion, embedding,
    /// stream_chat, stream_tools, stream_struct each maintain their own bucket).
    /// `None` means unbounded.
    pub max_size: Option<usize>,
    /// Whether to cache non-streaming completion responses (default `true`).
    pub cache_completions: bool,
    /// Whether to cache embedding responses (default `true`).
    pub cache_embeddings: bool,
    /// Whether to cache streaming responses (default `true`).
    ///
    /// When `true`, the first streaming request for a given key is served live from
    /// the inner provider and the chunks are simultaneously buffered. Subsequent
    /// identical requests replay the buffer as a stream with zero network overhead.
    pub cache_streaming: bool,
}

/// Cache key strategy for chat requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatCacheKeyMode {
    /// Cache key includes only latest user prompt (plus `tools + schema`).
    UserPromptOnly,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            chat_key_mode: ChatCacheKeyMode::UserPromptOnly,
            ttl: Some(Duration::from_secs(3600)),
            max_size: Some(1000),
            cache_completions: true,
            cache_embeddings: true,
            cache_streaming: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Public layer handle
// ---------------------------------------------------------------------------

/// An [`LLMLayer`] that inserts in-memory caching into the pipeline.
///
/// # Example
///
/// ```rust,ignore
/// use autoagents_llm::pipeline::PipelineBuilder;
/// use autoagents_llm::optim::{CacheLayer, CacheConfig};
/// use std::time::Duration;
///
/// let llm = PipelineBuilder::new(base)
///     .add_layer(CacheLayer::new(CacheConfig {
///         ttl: Some(Duration::from_secs(3600)),
///         max_size: Some(500),
///         ..CacheConfig::default()
///     }))
///     .build();
/// ```
pub struct CacheLayer {
    config: CacheConfig,
}

impl CacheLayer {
    /// Create a cache layer with the provided configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }

    /// Create a cache layer using [`CacheConfig::default`].
    pub fn with_defaults() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }
}

impl LLMLayer for CacheLayer {
    fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
        Arc::new(InMemoryCacheLayer::new(next, self.config))
    }
}

// ---------------------------------------------------------------------------
// Internal cache entry
// ---------------------------------------------------------------------------

struct CacheEntry<T> {
    value: T,
    created_at: Instant,
}

// ---------------------------------------------------------------------------
// Materialised (cloneable) chat response — avoids storing `dyn ChatResponse`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CachedChatResponse {
    text: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    thinking: Option<String>,
    usage: Option<crate::chat::Usage>,
}

impl ChatResponse for CachedChatResponse {
    fn text(&self) -> Option<String> {
        self.text.clone()
    }
    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }
    fn thinking(&self) -> Option<String> {
        self.thinking.clone()
    }
    fn usage(&self) -> Option<crate::chat::Usage> {
        self.usage.clone()
    }
}

impl fmt::Display for CachedChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text.as_deref().unwrap_or(""))
    }
}

// ---------------------------------------------------------------------------
// TeeStream — passthrough stream that writes to cache on completion
//
// Design invariants:
//   • Items are forwarded to the caller as they arrive — no buffering delay.
//   • The accumulated buffer is written to the cache in a single sync lock
//     acquisition only when `Poll::Ready(None)` is reached (successful end).
//   • On any `Err` item the buffer is discarded; nothing is cached.
//   • Once `done` is true, subsequent `poll_next` calls always return `None`.
//   • No lock is held across any await point (poll_next is synchronous).
//
// TeeStream<T>: Unpin because:
//   • Pin<Box<dyn Stream + Send>>: Unpin  (Box<T>: Unpin ∀T)
//   • Vec<T>: Unpin ∀T
//   • Cache<Vec<T>> (= Arc<RwLock<...>>): Unpin
//   • u64, bool, Option<usize>: Unpin
// Therefore self.get_mut() is sound inside poll_next.
// ---------------------------------------------------------------------------

struct TeeStream<T> {
    inner: Pin<Box<dyn Stream<Item = Result<T, LLMError>> + Send>>,
    buffer: Vec<T>,
    cache: Cache<Vec<T>>,
    key: u64,
    ttl: Option<Duration>,
    max_size: Option<usize>,
    done: bool,
}

impl<T: Clone + Send + Unpin> Stream for TeeStream<T> {
    type Item = Result<T, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut(); // safe: TeeStream<T>: Unpin
        if this.done {
            return Poll::Ready(None);
        }
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => {
                this.buffer.push(item.clone());
                Poll::Ready(Some(Ok(item)))
            }
            Poll::Ready(Some(Err(e))) => {
                // Error: discard buffer, do not cache partial data.
                this.done = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                // Stream ended successfully — write buffer to cache.
                this.done = true;
                let chunks = std::mem::take(&mut this.buffer);
                match this.cache.write() {
                    Ok(mut cache) => {
                        evict_expired(&mut cache, this.ttl);
                        if let Some(max) = this.max_size {
                            evict_oldest(&mut cache, max);
                        }
                        cache.insert(
                            this.key,
                            CacheEntry {
                                value: chunks,
                                created_at: Instant::now(),
                            },
                        );
                    }
                    Err(e) => {
                        log::warn!("stream cache write lock poisoned: {e}");
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// The actual provider wrapper
// ---------------------------------------------------------------------------

struct InMemoryCacheLayer {
    inner: Arc<dyn LLMProvider>,
    // Non-streaming caches
    chat_cache: Cache<CachedChatResponse>,
    completion_cache: Cache<String>,
    embedding_cache: Cache<Vec<Vec<f32>>>,
    // Streaming caches (one bucket per stream flavour)
    stream_chat_cache: Cache<Vec<String>>,
    stream_tools_cache: Cache<Vec<StreamChunk>>,
    stream_struct_cache: Cache<Vec<StreamResponse>>,
    // Coalesces concurrent misses for the same non-streaming key.
    chat_inflight: InFlight,
    completion_inflight: InFlight,
    embedding_inflight: InFlight,
    config: CacheConfig,
}

impl InMemoryCacheLayer {
    fn new(inner: Arc<dyn LLMProvider>, config: CacheConfig) -> Self {
        Self {
            inner,
            chat_cache: new_cache(),
            completion_cache: new_cache(),
            embedding_cache: new_cache(),
            stream_chat_cache: new_cache(),
            stream_tools_cache: new_cache(),
            stream_struct_cache: new_cache(),
            chat_inflight: new_inflight(),
            completion_inflight: new_inflight(),
            embedding_inflight: new_inflight(),
            config,
        }
    }
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

fn hash_val<T: Serialize>(v: &T) -> u64 {
    let s = serde_json::to_string(v).unwrap_or_default();
    let mut h = DefaultHasher::default();
    s.hash(&mut h);
    h.finish()
}

fn hash_chat_user_prompt(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    json_schema: Option<&StructuredOutputFormat>,
) -> u64 {
    let prompt = messages
        .iter()
        .rev()
        .find(|message| message.role == ChatRole::User)
        .map(|message| message.content.as_str())
        .unwrap_or_default();
    hash_val(&(prompt, tools, json_schema))
}

fn hash_chat_with_mode(
    mode: ChatCacheKeyMode,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    json_schema: Option<&StructuredOutputFormat>,
) -> u64 {
    match mode {
        ChatCacheKeyMode::UserPromptOnly => hash_chat_user_prompt(messages, tools, json_schema),
    }
}

fn hash_completion(req: &CompletionRequest, json_schema: Option<&StructuredOutputFormat>) -> u64 {
    let mut h = DefaultHasher::default();
    req.prompt.hash(&mut h);
    req.max_tokens.hash(&mut h);
    req.temperature.map(f32::to_bits).hash(&mut h);
    hash_val(&json_schema).hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Eviction / TTL helpers
// ---------------------------------------------------------------------------

fn is_expired(config: &CacheConfig, created_at: Instant) -> bool {
    config.ttl.is_some_and(|ttl| created_at.elapsed() > ttl)
}

fn evict_oldest<V>(cache: &mut HashMap<u64, CacheEntry<V>>, max_size: usize) {
    if cache.len() >= max_size
        && let Some(oldest_key) = cache
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| *k)
    {
        cache.remove(&oldest_key);
    }
}

fn evict_expired<V>(cache: &mut HashMap<u64, CacheEntry<V>>, ttl: Option<Duration>) {
    if let Some(ttl) = ttl {
        cache.retain(|_, entry| entry.created_at.elapsed() <= ttl);
    }
}

async fn acquire_inflight_lock(inflight: &InFlight, key: u64) -> Arc<tokio::sync::Mutex<()>> {
    let mut map = inflight.lock().await;
    if let Some(lock) = map.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    map.insert(key, Arc::downgrade(&lock));
    lock
}

// ---------------------------------------------------------------------------
// Helper: look up a cached Vec<T> and replay it as a stream
// ---------------------------------------------------------------------------

fn lookup_stream<T: Clone + Send + 'static>(
    cache: &Cache<Vec<T>>,
    key: u64,
    config: &CacheConfig,
) -> Option<BoxStream<T>> {
    let mut stale = false;
    match cache.read() {
        Ok(guard) => {
            if let Some(entry) = guard.get(&key)
                && !is_expired(config, entry.created_at)
            {
                let chunks = entry.value.clone();
                return Some(Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))));
            } else if guard.get(&key).is_some() {
                stale = true;
            }
        }
        Err(e) => {
            log::warn!("stream cache read lock poisoned: {e}");
            return None;
        }
    }

    if stale {
        match cache.write() {
            Ok(mut guard) => {
                guard.remove(&key);
            }
            Err(e) => {
                log::warn!("stream cache write lock poisoned while removing stale key: {e}");
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// ChatProvider — non-streaming cached; streaming via TeeStream
// ---------------------------------------------------------------------------

#[async_trait]
impl ChatProvider for InMemoryCacheLayer {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let key = hash_chat_with_mode(
            self.config.chat_key_mode,
            messages,
            tools,
            json_schema.as_ref(),
        );

        {
            match self.chat_cache.read() {
                Ok(cache) => {
                    if let Some(entry) = cache.get(&key)
                        && !is_expired(&self.config, entry.created_at)
                    {
                        return Ok(Box::new(entry.value.clone()));
                    }
                }
                Err(e) => {
                    log::warn!("chat cache read lock poisoned: {e}");
                }
            }
        }

        let key_lock = acquire_inflight_lock(&self.chat_inflight, key).await;
        let _guard = key_lock.lock().await;

        // Re-check after entering single-flight critical section.
        {
            match self.chat_cache.read() {
                Ok(cache) => {
                    if let Some(entry) = cache.get(&key)
                        && !is_expired(&self.config, entry.created_at)
                    {
                        return Ok(Box::new(entry.value.clone()));
                    }
                }
                Err(e) => {
                    log::warn!("chat cache read lock poisoned after single-flight wait: {e}");
                }
            }
        }

        let response = self
            .inner
            .chat_with_tools(messages, tools, json_schema)
            .await?;

        let cached = CachedChatResponse {
            text: response.text(),
            tool_calls: response.tool_calls(),
            thinking: response.thinking(),
            usage: response.usage(),
        };

        {
            match self.chat_cache.write() {
                Ok(mut cache) => {
                    evict_expired(&mut cache, self.config.ttl);
                    if let Some(max) = self.config.max_size {
                        evict_oldest(&mut cache, max);
                    }
                    cache.insert(
                        key,
                        CacheEntry {
                            value: cached.clone(),
                            created_at: Instant::now(),
                        },
                    );
                }
                Err(e) => {
                    log::warn!("chat cache write lock poisoned: {e}");
                }
            }
        }

        Ok(Box::new(cached))
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        if !self.config.cache_streaming {
            return self.inner.chat_stream(messages, json_schema).await;
        }

        let key = hash_chat_with_mode(
            self.config.chat_key_mode,
            messages,
            None,
            json_schema.as_ref(),
        );

        if let Some(stream) = lookup_stream(&self.stream_chat_cache, key, &self.config) {
            return Ok(stream);
        }

        // Cache miss — tee the inner stream so it writes to cache on completion.
        let inner_stream = self.inner.chat_stream(messages, json_schema).await?;
        Ok(Box::pin(TeeStream {
            inner: inner_stream,
            buffer: Vec::new(),
            cache: self.stream_chat_cache.clone(),
            key,
            ttl: self.config.ttl,
            max_size: self.config.max_size,
            done: false,
        }))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        if !self.config.cache_streaming {
            return self
                .inner
                .chat_stream_with_tools(messages, tools, json_schema)
                .await;
        }

        let key = hash_chat_with_mode(
            self.config.chat_key_mode,
            messages,
            tools,
            json_schema.as_ref(),
        );

        if let Some(stream) = lookup_stream(&self.stream_tools_cache, key, &self.config) {
            return Ok(stream);
        }

        let inner_stream = self
            .inner
            .chat_stream_with_tools(messages, tools, json_schema)
            .await?;
        Ok(Box::pin(TeeStream {
            inner: inner_stream,
            buffer: Vec::new(),
            cache: self.stream_tools_cache.clone(),
            key,
            ttl: self.config.ttl,
            max_size: self.config.max_size,
            done: false,
        }))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        if !self.config.cache_streaming {
            return self
                .inner
                .chat_stream_struct(messages, tools, json_schema)
                .await;
        }

        let key = hash_chat_with_mode(
            self.config.chat_key_mode,
            messages,
            tools,
            json_schema.as_ref(),
        );

        if let Some(stream) = lookup_stream(&self.stream_struct_cache, key, &self.config) {
            return Ok(stream);
        }

        let inner_stream = self
            .inner
            .chat_stream_struct(messages, tools, json_schema)
            .await?;
        Ok(Box::pin(TeeStream {
            inner: inner_stream,
            buffer: Vec::new(),
            cache: self.stream_struct_cache.clone(),
            key,
            ttl: self.config.ttl,
            max_size: self.config.max_size,
            done: false,
        }))
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        // Web results are time-sensitive — always delegate.
        self.inner.chat_with_web_search(input).await
    }
}

// ---------------------------------------------------------------------------
// CompletionProvider — optionally cached
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionProvider for InMemoryCacheLayer {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        if !self.config.cache_completions {
            return self.inner.complete(req, json_schema).await;
        }

        let key = hash_completion(req, json_schema.as_ref());

        {
            match self.completion_cache.read() {
                Ok(cache) => {
                    if let Some(entry) = cache.get(&key)
                        && !is_expired(&self.config, entry.created_at)
                    {
                        return Ok(CompletionResponse {
                            text: entry.value.clone(),
                        });
                    }
                }
                Err(e) => {
                    log::warn!("completion cache read lock poisoned: {e}");
                }
            }
        }

        let key_lock = acquire_inflight_lock(&self.completion_inflight, key).await;
        let _guard = key_lock.lock().await;

        {
            match self.completion_cache.read() {
                Ok(cache) => {
                    if let Some(entry) = cache.get(&key)
                        && !is_expired(&self.config, entry.created_at)
                    {
                        return Ok(CompletionResponse {
                            text: entry.value.clone(),
                        });
                    }
                }
                Err(e) => {
                    log::warn!("completion cache read lock poisoned after single-flight wait: {e}");
                }
            }
        }

        let response = self.inner.complete(req, json_schema).await?;

        {
            match self.completion_cache.write() {
                Ok(mut cache) => {
                    evict_expired(&mut cache, self.config.ttl);
                    if let Some(max) = self.config.max_size {
                        evict_oldest(&mut cache, max);
                    }
                    cache.insert(
                        key,
                        CacheEntry {
                            value: response.text.clone(),
                            created_at: Instant::now(),
                        },
                    );
                }
                Err(e) => {
                    log::warn!("completion cache write lock poisoned: {e}");
                }
            }
        }

        Ok(response)
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProvider — optionally cached
// ---------------------------------------------------------------------------

#[async_trait]
impl EmbeddingProvider for InMemoryCacheLayer {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if !self.config.cache_embeddings {
            return self.inner.embed(input).await;
        }

        let key = hash_val(&input);

        {
            match self.embedding_cache.read() {
                Ok(cache) => {
                    if let Some(entry) = cache.get(&key)
                        && !is_expired(&self.config, entry.created_at)
                    {
                        return Ok(entry.value.clone());
                    }
                }
                Err(e) => {
                    log::warn!("embedding cache read lock poisoned: {e}");
                }
            }
        }

        let key_lock = acquire_inflight_lock(&self.embedding_inflight, key).await;
        let _guard = key_lock.lock().await;

        {
            match self.embedding_cache.read() {
                Ok(cache) => {
                    if let Some(entry) = cache.get(&key)
                        && !is_expired(&self.config, entry.created_at)
                    {
                        return Ok(entry.value.clone());
                    }
                }
                Err(e) => {
                    log::warn!("embedding cache read lock poisoned after single-flight wait: {e}");
                }
            }
        }

        let result = self.inner.embed(input).await?;

        {
            match self.embedding_cache.write() {
                Ok(mut cache) => {
                    evict_expired(&mut cache, self.config.ttl);
                    if let Some(max) = self.config.max_size {
                        evict_oldest(&mut cache, max);
                    }
                    cache.insert(
                        key,
                        CacheEntry {
                            value: result.clone(),
                            created_at: Instant::now(),
                        },
                    );
                }
                Err(e) => {
                    log::warn!("embedding cache write lock poisoned: {e}");
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// ModelsProvider — always delegate (model lists are not cached)
// ---------------------------------------------------------------------------

#[async_trait]
impl ModelsProvider for InMemoryCacheLayer {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        self.inner.list_models(request).await
    }
}

// ---------------------------------------------------------------------------
// LLMProvider marker
// ---------------------------------------------------------------------------

impl LLMProvider for InMemoryCacheLayer {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HasConfig, LLMProvider, NoConfig,
        chat::{
            ChatMessage, ChatProvider, ChatResponse, StreamChoice, StreamChunk, StreamDelta,
            StreamResponse, StructuredOutputFormat, Tool,
        },
        completion::{CompletionProvider, CompletionRequest, CompletionResponse},
        embedding::EmbeddingProvider,
        error::LLMError,
        models::ModelsProvider,
        optim::CacheLayer,
        pipeline::PipelineBuilder,
    };
    use async_trait::async_trait;
    use futures::StreamExt;
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    // ------------------------------------------------------------------
    // Mock inner provider
    // ------------------------------------------------------------------

    #[derive(Debug)]
    struct MockChatResp(String);

    impl ChatResponse for MockChatResp {
        fn text(&self) -> Option<String> {
            Some(self.0.clone())
        }
        fn tool_calls(&self) -> Option<Vec<crate::ToolCall>> {
            None
        }
    }
    impl fmt::Display for MockChatResp {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    struct MockInner {
        chat_calls: Arc<AtomicU32>,
        completion_calls: Arc<AtomicU32>,
        embedding_calls: Arc<AtomicU32>,
        stream_chat_calls: Arc<AtomicU32>,
        stream_tools_calls: Arc<AtomicU32>,
        stream_struct_calls: Arc<AtomicU32>,
    }

    impl MockInner {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                chat_calls: Arc::new(AtomicU32::new(0)),
                completion_calls: Arc::new(AtomicU32::new(0)),
                embedding_calls: Arc::new(AtomicU32::new(0)),
                stream_chat_calls: Arc::new(AtomicU32::new(0)),
                stream_tools_calls: Arc::new(AtomicU32::new(0)),
                stream_struct_calls: Arc::new(AtomicU32::new(0)),
            })
        }
    }

    #[async_trait]
    impl ChatProvider for MockInner {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.chat_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(MockChatResp("chat-response".to_string())))
        }

        async fn chat_stream(
            &self,
            _messages: &[ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
        {
            self.stream_chat_calls.fetch_add(1, Ordering::SeqCst);
            let chunks: Vec<Result<String, LLMError>> = vec![
                Ok("chunk1".into()),
                Ok("chunk2".into()),
                Ok("chunk3".into()),
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }

        async fn chat_stream_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
        {
            self.stream_tools_calls.fetch_add(1, Ordering::SeqCst);
            let chunks: Vec<Result<StreamChunk, LLMError>> = vec![
                Ok(StreamChunk::Text("hello".into())),
                Ok(StreamChunk::Text(" world".into())),
                Ok(StreamChunk::Done {
                    stop_reason: "end_turn".into(),
                }),
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }

        async fn chat_stream_struct(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
        {
            self.stream_struct_calls.fetch_add(1, Ordering::SeqCst);
            let chunk = StreamResponse {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: Some("hi".into()),
                        tool_calls: None,
                    },
                }],
                usage: None,
            };
            Ok(Box::pin(futures::stream::iter(vec![Ok(chunk)])))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockInner {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            self.completion_calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse {
                text: "completion-response".to_string(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockInner {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            self.embedding_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![vec![1.0, 2.0, 3.0]])
        }
    }

    #[async_trait]
    impl ModelsProvider for MockInner {}

    impl LLMProvider for MockInner {}

    impl HasConfig for MockInner {
        type Config = NoConfig;
    }

    // ------------------------------------------------------------------
    // Helper: build a cached pipeline around MockInner
    // ------------------------------------------------------------------

    fn make_pipeline(config: CacheConfig) -> (Arc<MockInner>, Arc<dyn LLMProvider>) {
        let inner = MockInner::new();
        let provider = PipelineBuilder::new(inner.clone() as Arc<dyn LLMProvider>)
            .add_layer(CacheLayer::new(config))
            .build();
        (inner, provider)
    }

    /// Consume a stream and collect all `Ok` items into a `Vec`.
    async fn collect_stream<T: Clone>(
        stream: Pin<Box<dyn Stream<Item = Result<T, LLMError>> + Send>>,
    ) -> Vec<T> {
        let mut out = Vec::new();
        let mut s = stream;
        while let Some(item) = s.next().await {
            out.push(item.unwrap());
        }
        out
    }

    // ------------------------------------------------------------------
    // Non-streaming chat tests (unchanged behaviour)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_chat_cache_hit_does_not_call_inner() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("hello").build()];

        let r1 = provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(r1.text(), Some("chat-response".to_string()));

        let r2 = provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(r2.text(), Some("chat-response".to_string()));
    }

    #[tokio::test]
    async fn test_chat_cache_different_messages_are_separate_entries() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let msg_a = vec![ChatMessage::user().content("question A").build()];
        let msg_b = vec![ChatMessage::user().content("question B").build()];

        provider.chat_with_tools(&msg_a, None, None).await.unwrap();
        provider.chat_with_tools(&msg_b, None, None).await.unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 2);

        provider.chat_with_tools(&msg_a, None, None).await.unwrap();
        provider.chat_with_tools(&msg_b, None, None).await.unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_chat_cache_reuses_same_prompt_across_different_history() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let msg_a = vec![ChatMessage::user().content("repeat prompt").build()];
        let msg_b = vec![
            ChatMessage::assistant().content("older context").build(),
            ChatMessage::user().content("repeat prompt").build(),
        ];

        provider.chat_with_tools(&msg_a, None, None).await.unwrap();
        provider.chat_with_tools(&msg_b, None, None).await.unwrap();

        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_chat_cache_ttl_expiry() {
        let (inner, provider) = make_pipeline(CacheConfig {
            ttl: Some(Duration::from_millis(5)),
            ..CacheConfig::default()
        });
        let messages = vec![ChatMessage::user().content("ttl test").build()];

        provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);

        provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(15)).await;

        provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_chat_cache_max_size_evicts_oldest() {
        let (inner, provider) = make_pipeline(CacheConfig {
            max_size: Some(2),
            ..CacheConfig::default()
        });
        let msg_a = vec![ChatMessage::user().content("A").build()];
        let msg_b = vec![ChatMessage::user().content("B").build()];
        let msg_c = vec![ChatMessage::user().content("C").build()];

        provider.chat_with_tools(&msg_a, None, None).await.unwrap();
        provider.chat_with_tools(&msg_b, None, None).await.unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 2);

        provider.chat_with_tools(&msg_c, None, None).await.unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 3);

        // B and C still cached
        provider.chat_with_tools(&msg_b, None, None).await.unwrap();
        provider.chat_with_tools(&msg_c, None, None).await.unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 3);

        // A was evicted — miss
        provider.chat_with_tools(&msg_a, None, None).await.unwrap();
        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 4);
    }

    // ------------------------------------------------------------------
    // chat_stream — cache hit / miss / disabled / TTL
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_chat_stream_first_call_hits_inner() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("stream test").build()];

        let stream = provider.chat_stream(&messages, None).await.unwrap();
        let chunks = collect_stream(stream).await;

        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(chunks, vec!["chunk1", "chunk2", "chunk3"]);
    }

    #[tokio::test]
    async fn test_chat_stream_cache_hit_does_not_call_inner() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("stream test").build()];

        // First call — miss, populates cache
        let chunks1 = collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);

        // Second call — hit, inner not called
        let chunks2 = collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);

        assert_eq!(chunks1, chunks2);
    }

    #[tokio::test]
    async fn test_chat_stream_cache_disabled_always_calls_inner() {
        let (inner, provider) = make_pipeline(CacheConfig {
            cache_streaming: false,
            ..CacheConfig::default()
        });
        let messages = vec![ChatMessage::user().content("no cache").build()];

        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_chat_stream_ttl_expiry() {
        let (inner, provider) = make_pipeline(CacheConfig {
            ttl: Some(Duration::from_millis(5)),
            ..CacheConfig::default()
        });
        let messages = vec![ChatMessage::user().content("ttl stream").build()];

        // Miss
        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);

        // Hit (before TTL)
        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(15)).await;

        // Miss (after TTL)
        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_chat_stream_max_size_evicts_oldest() {
        let (inner, provider) = make_pipeline(CacheConfig {
            max_size: Some(2),
            ..CacheConfig::default()
        });
        let msg_a = vec![ChatMessage::user().content("SA").build()];
        let msg_b = vec![ChatMessage::user().content("SB").build()];
        let msg_c = vec![ChatMessage::user().content("SC").build()];

        // Populate A and B
        collect_stream(provider.chat_stream(&msg_a, None).await.unwrap()).await;
        collect_stream(provider.chat_stream(&msg_b, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 2);

        // C causes A to be evicted
        collect_stream(provider.chat_stream(&msg_c, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 3);

        // B and C hit
        collect_stream(provider.chat_stream(&msg_b, None).await.unwrap()).await;
        collect_stream(provider.chat_stream(&msg_c, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 3);

        // A is evicted — miss
        collect_stream(provider.chat_stream(&msg_a, None).await.unwrap()).await;
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 4);
    }

    // ------------------------------------------------------------------
    // chat_stream_with_tools — cache hit / miss / disabled
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_chat_stream_with_tools_first_call_hits_inner() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("tool stream").build()];

        let chunks = collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(inner.stream_tools_calls.load(Ordering::SeqCst), 1);
        assert_eq!(chunks.len(), 3); // Text, Text, Done
    }

    #[tokio::test]
    async fn test_chat_stream_with_tools_cache_hit() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("tool stream cached").build()];

        let c1 = collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        let c2 = collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;

        assert_eq!(inner.stream_tools_calls.load(Ordering::SeqCst), 1);
        assert_eq!(c1.len(), c2.len());
    }

    #[tokio::test]
    async fn test_chat_stream_with_tools_disabled() {
        let (inner, provider) = make_pipeline(CacheConfig {
            cache_streaming: false,
            ..CacheConfig::default()
        });
        let messages = vec![ChatMessage::user().content("no cache tools").build()];

        collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(inner.stream_tools_calls.load(Ordering::SeqCst), 2);
    }

    // ------------------------------------------------------------------
    // chat_stream_struct — cache hit / miss / disabled
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_chat_stream_struct_first_call_hits_inner() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("struct stream").build()];

        let chunks = collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(inner.stream_struct_calls.load(Ordering::SeqCst), 1);
        assert_eq!(chunks.len(), 1);
    }

    #[tokio::test]
    async fn test_chat_stream_struct_cache_hit() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let messages = vec![ChatMessage::user().content("struct cached").build()];

        let c1 = collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        let c2 = collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;

        assert_eq!(inner.stream_struct_calls.load(Ordering::SeqCst), 1);
        assert_eq!(c1.len(), c2.len());
        assert_eq!(
            c1[0].choices[0].delta.content,
            c2[0].choices[0].delta.content
        );
    }

    #[tokio::test]
    async fn test_chat_stream_struct_disabled() {
        let (inner, provider) = make_pipeline(CacheConfig {
            cache_streaming: false,
            ..CacheConfig::default()
        });
        let messages = vec![ChatMessage::user().content("no cache struct").build()];

        collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(inner.stream_struct_calls.load(Ordering::SeqCst), 2);
    }

    // ------------------------------------------------------------------
    // TeeStream: partial stream error must not cache anything
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_chat_stream_error_not_cached() {
        // Build a pipeline with a custom inner that errors on first call,
        // succeeds on second.
        use std::sync::atomic::AtomicBool;
        struct ErrorOnFirst {
            failed: Arc<AtomicBool>,
            calls: Arc<AtomicU32>,
        }
        #[async_trait]
        impl ChatProvider for ErrorOnFirst {
            async fn chat_with_tools(
                &self,
                _: &[ChatMessage],
                _: Option<&[Tool]>,
                _: Option<StructuredOutputFormat>,
            ) -> Result<Box<dyn ChatResponse>, LLMError> {
                Err(LLMError::Generic("not impl".into()))
            }
            async fn chat_stream(
                &self,
                _: &[ChatMessage],
                _: Option<StructuredOutputFormat>,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
            {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if !self.failed.swap(true, Ordering::SeqCst) {
                    // First call: stream that errors mid-way
                    let s: Vec<Result<String, LLMError>> = vec![
                        Ok("ok".into()),
                        Err(LLMError::Generic("mid-stream error".into())),
                    ];
                    Ok(Box::pin(futures::stream::iter(s)))
                } else {
                    // Second call: clean stream
                    let s: Vec<Result<String, LLMError>> =
                        vec![Ok("clean1".into()), Ok("clean2".into())];
                    Ok(Box::pin(futures::stream::iter(s)))
                }
            }
        }
        #[async_trait]
        impl CompletionProvider for ErrorOnFirst {
            async fn complete(
                &self,
                _: &CompletionRequest,
                _: Option<StructuredOutputFormat>,
            ) -> Result<CompletionResponse, LLMError> {
                Err(LLMError::Generic("not impl".into()))
            }
        }
        #[async_trait]
        impl EmbeddingProvider for ErrorOnFirst {
            async fn embed(&self, _: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
                Err(LLMError::Generic("not impl".into()))
            }
        }
        #[async_trait]
        impl ModelsProvider for ErrorOnFirst {}
        impl LLMProvider for ErrorOnFirst {}

        let calls = Arc::new(AtomicU32::new(0));
        let inner = Arc::new(ErrorOnFirst {
            failed: Arc::new(AtomicBool::new(false)),
            calls: calls.clone(),
        });
        let provider = PipelineBuilder::new(inner as Arc<dyn LLMProvider>)
            .add_layer(CacheLayer::with_defaults())
            .build();

        let messages = vec![ChatMessage::user().content("error test").build()];

        // First call: consume until error
        let mut stream1 = provider.chat_stream(&messages, None).await.unwrap();
        let mut saw_error = false;
        while let Some(item) = stream1.next().await {
            if item.is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error, "expected an error chunk");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Second call: inner must be called again (nothing was cached)
        let chunks2 = collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(chunks2, vec!["clean1", "clean2"]);

        // Third call: now it IS cached (second call succeeded)
        let chunks3 = collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2); // no new call
        assert_eq!(chunks3, vec!["clean1", "clean2"]);
    }

    // ------------------------------------------------------------------
    // Completion cache tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_completion_cache_hit() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let req = CompletionRequest::new("prompt");

        provider.complete(&req, None).await.unwrap();
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 1);

        provider.complete(&req, None).await.unwrap();
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_completion_cache_key_includes_json_schema() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let req = CompletionRequest::new("prompt");
        let schema = StructuredOutputFormat {
            name: "Answer".into(),
            description: Some("Structured answer".into()),
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } },
                "required": ["answer"]
            })),
            strict: Some(true),
        };

        provider.complete(&req, Some(schema.clone())).await.unwrap();
        provider.complete(&req, None).await.unwrap();
        provider.complete(&req, Some(schema)).await.unwrap();

        // Schema and non-schema requests are distinct cache keys.
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_completion_cache_disabled() {
        let (inner, provider) = make_pipeline(CacheConfig {
            cache_completions: false,
            ..CacheConfig::default()
        });
        let req = CompletionRequest::new("prompt");

        provider.complete(&req, None).await.unwrap();
        provider.complete(&req, None).await.unwrap();
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_completion_cache_ttl_expiry() {
        let (inner, provider) = make_pipeline(CacheConfig {
            ttl: Some(Duration::from_millis(5)),
            ..CacheConfig::default()
        });
        let req = CompletionRequest::new("ttl-prompt");

        provider.complete(&req, None).await.unwrap();
        provider.complete(&req, None).await.unwrap();
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(15)).await;

        provider.complete(&req, None).await.unwrap();
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 2);
    }

    // ------------------------------------------------------------------
    // Embedding cache tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_embedding_cache_hit() {
        let (inner, provider) = make_pipeline(CacheConfig::default());
        let input = vec!["hello world".to_string()];

        let r1 = provider.embed(input.clone()).await.unwrap();
        assert_eq!(inner.embedding_calls.load(Ordering::SeqCst), 1);
        assert_eq!(r1, vec![vec![1.0, 2.0, 3.0]]);

        let r2 = provider.embed(input).await.unwrap();
        assert_eq!(inner.embedding_calls.load(Ordering::SeqCst), 1);
        assert_eq!(r2, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[tokio::test]
    async fn test_embedding_cache_disabled() {
        let (inner, provider) = make_pipeline(CacheConfig {
            cache_embeddings: false,
            ..CacheConfig::default()
        });
        let input = vec!["hello".to_string()];

        provider.embed(input.clone()).await.unwrap();
        provider.embed(input).await.unwrap();
        assert_eq!(inner.embedding_calls.load(Ordering::SeqCst), 2);
    }

    // ------------------------------------------------------------------
    // All caches are independent from each other
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_all_caches_are_independent() {
        let (inner, provider) = make_pipeline(CacheConfig::default());

        let messages = vec![ChatMessage::user().content("x").build()];
        let req = CompletionRequest::new("x");
        let emb_input = vec!["x".to_string()];

        // All miss on first call
        provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        provider.complete(&req, None).await.unwrap();
        provider.embed(emb_input.clone()).await.unwrap();

        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.stream_tools_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.stream_struct_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.embedding_calls.load(Ordering::SeqCst), 1);

        // All hit on second call
        provider
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();
        collect_stream(provider.chat_stream(&messages, None).await.unwrap()).await;
        collect_stream(
            provider
                .chat_stream_with_tools(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        collect_stream(
            provider
                .chat_stream_struct(&messages, None, None)
                .await
                .unwrap(),
        )
        .await;
        provider.complete(&req, None).await.unwrap();
        provider.embed(emb_input).await.unwrap();

        assert_eq!(inner.chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.stream_chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.stream_tools_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.stream_struct_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inner.embedding_calls.load(Ordering::SeqCst), 1);
    }
}
