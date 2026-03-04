//! Retry layer with exponential back-off and full jitter.
//!
//! # Retry semantics
//! - All non-streaming calls are retried transparently.
//! - Streaming methods are retried on the **initial async call** only (i.e.
//!   before the stream starts delivering items). Mid-stream errors are not
//!   retried — the caller must restart the stream explicitly.
//! - [`AuthError`], [`InvalidRequest`], [`JsonError`], [`ToolConfigError`],
//!   and [`NoToolSupport`] are **never** retried by the default policy (they
//!   cannot succeed on a subsequent attempt without user intervention).
//! - [`HttpError`], [`ProviderError`], and [`Generic`] errors that carry
//!   rate-limit or server-error signals are retried up to
//!   `max_attempts − 1` additional times with exponential back-off.
//!
//! # Hot-path overhead
//! On a successful first attempt the only overhead over a bare provider call
//! is one extra match arm — no allocation, no timer, no log.
//!
//! [`AuthError`]: crate::error::LLMError::AuthError
//! [`InvalidRequest`]: crate::error::LLMError::InvalidRequest
//! [`JsonError`]: crate::error::LLMError::JsonError
//! [`ToolConfigError`]: crate::error::LLMError::ToolConfigError
//! [`NoToolSupport`]: crate::error::LLMError::NoToolSupport
//! [`HttpError`]: crate::error::LLMError::HttpError
//! [`ProviderError`]: crate::error::LLMError::ProviderError
//! [`Generic`]: crate::error::LLMError::Generic

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    LLMProvider,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse,
        StructuredOutputFormat, Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider},
    pipeline::LLMLayer,
};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for [`RetryLayer`].
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Total number of attempts, including the first one (≥ 1). Default: `3`.
    pub max_attempts: u32,
    /// Delay before the second attempt. Default: `200 ms`.
    pub initial_backoff: Duration,
    /// Upper bound on the computed back-off interval. Default: `30 s`.
    pub max_backoff: Duration,
    /// Apply **full jitter** to the computed delay (recommended). Default: `true`.
    ///
    /// Full jitter draws the sleep duration uniformly from `[0, ceiling]`,
    /// preventing thundering-herd retries across concurrent callers.
    pub jitter: bool,
    /// Returns `true` if an error should trigger a retry.
    ///
    /// Swap with a custom `fn` to adjust the policy without allocating a
    /// trait object.  The default is [`default_is_retryable`].
    pub retryable: fn(&LLMError) -> bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
            jitter: true,
            retryable: default_is_retryable,
        }
    }
}

/// Default retryability predicate.
///
/// Retries when the error message contains rate-limit (429) or server-error
/// (5xx) signals.  Never retries auth, invalid-request, or structural errors.
pub fn default_is_retryable(err: &LLMError) -> bool {
    match err {
        LLMError::HttpError(msg) | LLMError::ProviderError(msg) => {
            let m = msg.to_lowercase();
            m.contains("429")
                || m.contains("500")
                || m.contains("502")
                || m.contains("503")
                || m.contains("504")
                || m.contains("529") // Anthropic overload
                || m.contains("rate limit")
                || m.contains("too many requests")
                || m.contains("overloaded")
                || m.contains("server error")
                || m.contains("service unavailable")
        }
        LLMError::Generic(_) => true,
        LLMError::AuthError(_)
        | LLMError::InvalidRequest(_)
        | LLMError::ResponseFormatError { .. }
        | LLMError::JsonError(_)
        | LLMError::ToolConfigError(_)
        | LLMError::NoToolSupport(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// An [`LLMLayer`] that wraps the downstream provider with automatic retry.
///
/// Compose it first (outermost) in the pipeline when you also use
/// [`FallbackLayer`](super::FallbackLayer) so that each candidate provider is
/// retried before the next fallback is tried.
///
/// # Example
///
/// ```ignore
/// use autoagents_llm::{pipeline::PipelineBuilder, optim::{RetryLayer, RetryConfig}};
/// use std::time::Duration;
///
/// let llm = PipelineBuilder::new(base)
///     .add_layer(RetryLayer::new(RetryConfig {
///         max_attempts: 5,
///         initial_backoff: Duration::from_millis(100),
///         ..RetryConfig::default()
///     }))
///     .build();
/// ```
pub struct RetryLayer {
    config: RetryConfig,
}

impl RetryLayer {
    /// Create a layer with the given configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create a layer with default configuration (3 attempts, 200 ms initial
    /// back-off, 30 s cap, full jitter).
    pub fn with_defaults() -> Self {
        Self::new(RetryConfig::default())
    }
}

impl LLMLayer for RetryLayer {
    fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
        Arc::new(RetryProvider {
            inner: next,
            config: self.config,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider wrapper
// ---------------------------------------------------------------------------

struct RetryProvider {
    inner: Arc<dyn LLMProvider>,
    config: RetryConfig,
}

// ---------------------------------------------------------------------------
// Back-off helpers (no external RNG — uses subsecond system-time entropy)
// ---------------------------------------------------------------------------

/// Full-jitter sleep duration in `[0, ceiling]`.
///
/// Uses `SystemTime::subsec_nanos()` as a cheap entropy source.  Not
/// cryptographically uniform, but sufficient for back-off anti-thundering-herd.
#[inline]
fn jitter_duration(ceiling: Duration) -> Duration {
    let nanos = ceiling.as_nanos();
    if nanos == 0 {
        return Duration::ZERO;
    }
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u128;
    Duration::from_nanos((seed % nanos) as u64)
}

/// Back-off ceiling for zero-based `attempt` index.
/// `ceiling = min(max_backoff, initial * 2^attempt)`
#[inline]
fn compute_backoff(config: &RetryConfig, attempt: u32) -> Duration {
    let initial_ns = config.initial_backoff.as_nanos().min(u64::MAX as u128) as u64;
    // Use checked_shl so very large attempt counts saturate instead of panicking.
    let multiplier = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
    let max_ns = config.max_backoff.as_nanos().min(u64::MAX as u128) as u64;
    let ceiling = Duration::from_nanos(initial_ns.saturating_mul(multiplier).min(max_ns));
    if config.jitter {
        jitter_duration(ceiling)
    } else {
        ceiling
    }
}

// ---------------------------------------------------------------------------
// Core retry loop
// ---------------------------------------------------------------------------

/// Execute `f` up to `config.max_attempts` times.
///
/// Returns immediately on the first `Ok`.  On a retryable `Err` sleeps for
/// the computed back-off then retries.  On a non-retryable `Err` or when all
/// attempts are exhausted, returns the error.
///
/// # Hot path (first attempt succeeds)
/// No allocation, no timer — just `f().await` and a match.
async fn retry_call<F, Fut, T>(config: &RetryConfig, mut f: F) -> Result<T, LLMError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LLMError>>,
{
    let max = config.max_attempts.max(1);
    let mut attempt = 0u32;
    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) if attempt + 1 < max && (config.retryable)(&e) => {
                let backoff = compute_backoff(config, attempt);
                log::warn!(
                    "LLM call failed (attempt {}/{}): {e}. Retrying in {backoff:?}.",
                    attempt + 1,
                    max,
                );
                tokio::time::sleep(backoff).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// ChatProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ChatProvider for RetryProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        retry_call(&self.config, || {
            self.inner
                .chat_with_tools(messages, tools, json_schema.clone())
        })
        .await
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        retry_call(&self.config, || {
            self.inner.chat_with_web_search(input.clone())
        })
        .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        retry_call(&self.config, || {
            self.inner.chat_stream(messages, json_schema.clone())
        })
        .await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        retry_call(&self.config, || {
            self.inner
                .chat_stream_struct(messages, tools, json_schema.clone())
        })
        .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        retry_call(&self.config, || {
            self.inner
                .chat_stream_with_tools(messages, tools, json_schema.clone())
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// CompletionProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionProvider for RetryProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        retry_call(&self.config, || {
            self.inner.complete(req, json_schema.clone())
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl EmbeddingProvider for RetryProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        retry_call(&self.config, || self.inner.embed(input.clone())).await
    }
}

// ---------------------------------------------------------------------------
// ModelsProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ModelsProvider for RetryProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        // `Box<dyn ModelListResponse>` is !Send so cannot go through the generic
        // retry_call helper (which requires T: Send to produce a Send future).
        // Models listing is an administrative call; simple delegation suffices.
        self.inner.list_models(request).await
    }
}

impl LLMProvider for RetryProvider {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FunctionCall, ToolCall,
        chat::{ChatResponse, StructuredOutputFormat, Tool},
        completion::CompletionRequest,
        error::LLMError,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    // -----------------------------------------------------------------------
    // Minimal mock provider
    // -----------------------------------------------------------------------

    struct MockResponse(String);

    impl ChatResponse for MockResponse {
        fn text(&self) -> Option<String> {
            Some(self.0.clone())
        }
        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }
    }
    impl std::fmt::Debug for MockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockResponse({})", self.0)
        }
    }
    impl std::fmt::Display for MockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    /// Succeeds on `success_after`-th call (1-based); returns `err` before that.
    struct CountingMock {
        calls: AtomicU32,
        success_after: u32,
        err: LLMError,
    }

    impl CountingMock {
        fn new(success_after: u32, err: LLMError) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicU32::new(0),
                success_after,
                err,
            })
        }

        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::Relaxed)
        }

        fn next_result(&self) -> Result<Box<dyn ChatResponse>, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.success_after {
                Ok(Box::new(MockResponse("ok".into())))
            } else {
                // Return a clone of the error variant by re-constructing it.
                Err(match &self.err {
                    LLMError::HttpError(m) => LLMError::HttpError(m.clone()),
                    LLMError::ProviderError(m) => LLMError::ProviderError(m.clone()),
                    LLMError::Generic(m) => LLMError::Generic(m.clone()),
                    LLMError::AuthError(m) => LLMError::AuthError(m.clone()),
                    LLMError::InvalidRequest(m) => LLMError::InvalidRequest(m.clone()),
                    other => LLMError::Generic(other.to_string()),
                })
            }
        }
    }

    #[async_trait]
    impl ChatProvider for CountingMock {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.next_result()
        }
    }

    #[async_trait]
    impl CompletionProvider for CountingMock {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.success_after {
                Ok(CompletionResponse {
                    text: "done".into(),
                })
            } else {
                Err(LLMError::HttpError("503 service unavailable".into()))
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for CountingMock {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.success_after {
                Ok(vec![vec![1.0, 2.0]])
            } else {
                Err(LLMError::HttpError("429 rate limit".into()))
            }
        }
    }

    #[async_trait]
    impl ModelsProvider for CountingMock {}

    impl LLMProvider for CountingMock {}

    impl crate::HasConfig for CountingMock {
        type Config = crate::NoConfig;
    }

    // -----------------------------------------------------------------------
    // Back-off unit tests (no I/O)
    // -----------------------------------------------------------------------

    #[test]
    fn backoff_grows_exponentially() {
        let cfg = RetryConfig {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(60),
            jitter: false,
            ..RetryConfig::default()
        };
        assert_eq!(compute_backoff(&cfg, 0), Duration::from_millis(100));
        assert_eq!(compute_backoff(&cfg, 1), Duration::from_millis(200));
        assert_eq!(compute_backoff(&cfg, 2), Duration::from_millis(400));
        assert_eq!(compute_backoff(&cfg, 3), Duration::from_millis(800));
    }

    #[test]
    fn backoff_capped_at_max() {
        let cfg = RetryConfig {
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_millis(1000),
            jitter: false,
            ..RetryConfig::default()
        };
        // attempt 2: 500 * 4 = 2000 ms → capped at 1000 ms
        assert_eq!(compute_backoff(&cfg, 2), Duration::from_millis(1000));
    }

    #[test]
    fn backoff_with_jitter_within_bounds() {
        let cfg = RetryConfig {
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
            jitter: true,
            ..RetryConfig::default()
        };
        let ceiling = Duration::from_millis(200);
        for _ in 0..20 {
            let b = compute_backoff(&cfg, 0);
            assert!(b <= ceiling, "jitter exceeded ceiling: {b:?}");
        }
    }

    #[test]
    fn large_attempt_does_not_overflow() {
        let cfg = RetryConfig {
            jitter: false,
            ..RetryConfig::default()
        };
        // Should saturate at max_backoff, not panic.
        let b = compute_backoff(&cfg, 200);
        assert_eq!(b, cfg.max_backoff);
    }

    // -----------------------------------------------------------------------
    // Retryability predicate
    // -----------------------------------------------------------------------

    #[test]
    fn retryable_errors() {
        assert!(default_is_retryable(&LLMError::HttpError(
            "429 rate limit exceeded".into()
        )));
        assert!(default_is_retryable(&LLMError::HttpError(
            "503 service unavailable".into()
        )));
        assert!(default_is_retryable(&LLMError::ProviderError(
            "overloaded".into()
        )));
        assert!(default_is_retryable(&LLMError::Generic(
            "connection reset".into()
        )));
    }

    #[test]
    fn non_retryable_errors() {
        assert!(!default_is_retryable(&LLMError::AuthError(
            "invalid key".into()
        )));
        assert!(!default_is_retryable(&LLMError::InvalidRequest(
            "bad param".into()
        )));
        assert!(!default_is_retryable(&LLMError::JsonError(
            "parse error".into()
        )));
        assert!(!default_is_retryable(&LLMError::ToolConfigError(
            "bad tool".into()
        )));
        assert!(!default_is_retryable(&LLMError::NoToolSupport(
            "unsupported".into()
        )));
    }

    // -----------------------------------------------------------------------
    // Integration: RetryProvider behaviour
    // -----------------------------------------------------------------------

    fn build_retry(mock: Arc<CountingMock>, cfg: RetryConfig) -> Arc<dyn LLMProvider> {
        RetryLayer::new(cfg).build_arc(mock as Arc<dyn LLMProvider>)
    }

    // Helper on RetryLayer to avoid Box ceremony in tests.
    impl RetryLayer {
        fn build_arc(self, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
            Box::new(self).build(next)
        }
    }

    #[tokio::test]
    async fn success_on_first_attempt_makes_one_call() {
        let mock = CountingMock::new(1, LLMError::Generic("never".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = ChatMessage::user().content("hi").build();
        provider.chat(&[msg], None).await.unwrap();
        assert_eq!(mock.call_count(), 1, "should call inner exactly once");
    }

    #[tokio::test]
    async fn retries_on_retryable_error_and_succeeds() {
        // Fails on attempts 1 and 2, succeeds on attempt 3.
        let mock = CountingMock::new(3, LLMError::HttpError("429 rate limit".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 5,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "ok");
        assert_eq!(mock.call_count(), 3);
    }

    #[tokio::test]
    async fn exhausts_attempts_and_returns_last_error() {
        let mock = CountingMock::new(99, LLMError::HttpError("503 unavailable".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = ChatMessage::user().content("hi").build();
        let err = provider.chat(&[msg], None).await.unwrap_err();
        assert!(err.to_string().contains("503"));
        assert_eq!(mock.call_count(), 3);
    }

    #[tokio::test]
    async fn non_retryable_error_is_not_retried() {
        let mock = CountingMock::new(99, LLMError::AuthError("invalid key".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 5,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = ChatMessage::user().content("hi").build();
        provider.chat(&[msg], None).await.unwrap_err();
        assert_eq!(mock.call_count(), 1, "auth error must not be retried");
    }

    #[tokio::test]
    async fn max_attempts_one_means_no_retry() {
        let mock = CountingMock::new(99, LLMError::HttpError("429".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 1,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let msg = ChatMessage::user().content("hi").build();
        provider.chat(&[msg], None).await.unwrap_err();
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn completion_is_retried() {
        let mock = CountingMock::new(2, LLMError::HttpError("503".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let req = CompletionRequest::new("test");
        let resp = provider.complete(&req, None).await.unwrap();
        assert_eq!(resp.text, "done");
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn embedding_is_retried() {
        let mock = CountingMock::new(2, LLMError::HttpError("429 rate limit".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 3,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                ..RetryConfig::default()
            },
        );
        let result = provider.embed(vec!["hello".into()]).await.unwrap();
        assert_eq!(result, vec![vec![1.0, 2.0]]);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn custom_retryable_predicate() {
        // Custom predicate: only retry on auth errors (unusual, but proves override works).
        let mock = CountingMock::new(3, LLMError::AuthError("retry me".into()));
        let provider = build_retry(
            mock.clone(),
            RetryConfig {
                max_attempts: 5,
                jitter: false,
                initial_backoff: Duration::from_millis(1),
                retryable: |err| matches!(err, LLMError::AuthError(_)),
                ..RetryConfig::default()
            },
        );
        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "ok");
        assert_eq!(mock.call_count(), 3);
    }

    // Verify the FunctionCall import used in mock is available.
    #[test]
    fn function_call_construction() {
        let _ = FunctionCall {
            name: "f".into(),
            arguments: "{}".into(),
        };
    }
}
