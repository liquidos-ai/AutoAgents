//! Fallback layer — routes to backup providers on failure.
//!
//! # Fallback semantics
//! - On a **fallbackable** error from the primary provider, each fallback is
//!   tried in the order they were added until one succeeds.
//! - On a **non-fallbackable** error (e.g. [`InvalidRequest`],
//!   [`AuthError`]) the error is propagated immediately without trying further
//!   providers — retrying a bad request on a different provider is wasteful.
//! - [`NoToolSupport`] *is* fallbackable so that a local/lite model can be
//!   backed by a full-featured remote provider for tool-calling tasks.
//! - Streaming methods fall back on the **initial async call** only.
//!
//! # Hot-path overhead
//! When `providers[0]` (primary) succeeds, the only overhead over a bare
//! provider is one call to the `fallbackable` function pointer and one slice
//! element access — no allocation, no iteration.
//!
//! # Composing with RetryLayer
//! `FallbackLayer` uses fallback providers exactly as passed to
//! [`FallbackLayer::new`]. Inner pipeline layers only wrap the primary `next`
//! provider.
//!
//! To retry each provider independently, pre-wrap each fallback with
//! [`RetryLayer`](super::RetryLayer) before passing it to fallback. If a single
//! outer retry is acceptable, add `RetryLayer` outside `FallbackLayer`.
//!
//! Example with a single outer retry:
//!
//! ```ignore
//! PipelineBuilder::new(openai)
//!     .add_layer(RetryLayer::with_defaults())
//!     .add_layer(FallbackLayer::new(vec![anthropic, ollama]))
//!     .build()
//! // Request flow: RetryLayer → FallbackLayer → primary/fallback providers
//! ```
//!
//! [`InvalidRequest`]: crate::error::LLMError::InvalidRequest
//! [`AuthError`]: crate::error::LLMError::AuthError
//! [`NoToolSupport`]: crate::error::LLMError::NoToolSupport

use std::{future::Future, pin::Pin, sync::Arc};

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

/// Configuration for [`FallbackLayer`].
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Returns `true` when a provider error should trigger a fallback attempt.
    ///
    /// Swap with a custom `fn` to adjust the policy without allocating a
    /// trait object.  The default is [`default_is_fallbackable`].
    pub fallbackable: fn(&LLMError) -> bool,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            fallbackable: default_is_fallbackable,
        }
    }
}

/// Default fallbackability predicate.
///
/// Falls back on network, provider, and structural-response errors as well as
/// [`NoToolSupport`](crate::error::LLMError::NoToolSupport) (enables routing
/// tool-calling tasks to a capable fallback).
///
/// Does **not** fall back on auth or invalid-request errors.
pub fn default_is_fallbackable(err: &LLMError) -> bool {
    matches!(
        err,
        LLMError::HttpError(_)
            | LLMError::ProviderError(_)
            | LLMError::Generic(_)
            | LLMError::ResponseFormatError { .. }
            | LLMError::NoToolSupport(_)
    )
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// An [`LLMLayer`] that routes to backup providers when the primary fails.
///
/// The fallback list is tried **in addition to** the primary provider injected
/// by [`PipelineBuilder`](crate::pipeline::PipelineBuilder) at build time, so
/// total providers = 1 (primary) + `fallbacks.len()`.
///
/// # Example
///
/// ```ignore
/// use autoagents_llm::{pipeline::PipelineBuilder, optim::FallbackLayer};
///
/// let llm = PipelineBuilder::new(openai)
///     .add_layer(FallbackLayer::new(vec![anthropic, ollama]))
///     .build();
/// ```
pub struct FallbackLayer {
    fallbacks: Vec<Arc<dyn LLMProvider>>,
    config: FallbackConfig,
}

impl FallbackLayer {
    /// Create a layer with the given fallback providers and default config.
    ///
    /// Fallback providers are used as-is. They are not automatically wrapped by
    /// other pipeline layers that may exist around the primary provider.
    pub fn new(fallbacks: Vec<Arc<dyn LLMProvider>>) -> Self {
        Self {
            fallbacks,
            config: FallbackConfig::default(),
        }
    }

    /// Create a layer with a single fallback provider.
    pub fn single(fallback: Arc<dyn LLMProvider>) -> Self {
        Self::new(vec![fallback])
    }

    /// Override the fallbackability predicate.
    pub fn with_config(mut self, config: FallbackConfig) -> Self {
        self.config = config;
        self
    }
}

impl LLMLayer for FallbackLayer {
    fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
        // Pre-build the provider list: primary first, then fallbacks.
        // Avoids any allocation on the hot call path.
        let mut providers = Vec::with_capacity(1 + self.fallbacks.len());
        providers.push(next);
        providers.extend(self.fallbacks);
        Arc::new(FallbackProvider {
            providers,
            config: self.config,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider wrapper
// ---------------------------------------------------------------------------

struct FallbackProvider {
    /// `providers[0]` is always the primary; the rest are fallbacks in order.
    providers: Vec<Arc<dyn LLMProvider>>,
    config: FallbackConfig,
}

// ---------------------------------------------------------------------------
// Core fallback loop
// ---------------------------------------------------------------------------

/// Try each provider with `f` in order.
///
/// Receives an owned `Arc<dyn LLMProvider>` (cloned from the slice) so that
/// callers can wrap the call in `async move { p.method(...).await }` without
/// the future ever borrowing from an iteration-scoped variable.
///
/// Returns the first `Ok`.  On a fallbackable `Err` logs a warning and
/// advances to the next provider.  On a non-fallbackable `Err` returns
/// immediately — retrying on a different provider would be pointless.
///
/// # Hot path (primary succeeds)
/// Single `f(providers[0].clone()).await` + one match arm.  No allocation
/// beyond the Arc clone.
async fn try_fallback<F, Fut, T>(
    providers: &[Arc<dyn LLMProvider>],
    config: &FallbackConfig,
    mut f: F,
) -> Result<T, LLMError>
where
    F: FnMut(Arc<dyn LLMProvider>) -> Fut,
    Fut: Future<Output = Result<T, LLMError>>,
{
    let mut last_err: Option<LLMError> = None;
    for (idx, provider) in providers.iter().enumerate() {
        match f(Arc::clone(provider)).await {
            Ok(v) => return Ok(v),
            Err(e) if (config.fallbackable)(&e) => {
                let label = if idx == 0 { "primary" } else { "fallback" };
                log::warn!(
                    "LLM {label}[{idx}] failed: {e}. Trying next provider ({}/{}).",
                    idx + 1,
                    providers.len(),
                );
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| LLMError::Generic("No providers available".into())))
}

// ---------------------------------------------------------------------------
// ChatProvider
// ---------------------------------------------------------------------------

// Each method uses `async move` so that the owned `Arc` (and any cloned data)
// are moved into the future rather than borrowing from the closure parameter.
// This is required because `async_trait` futures borrow `&self`, and if `p`
// were only borrowed from the closure's scope the future would not live long
// enough.

#[async_trait]
impl ChatProvider for FallbackProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_with_tools(messages, tools, js).await }
        })
        .await
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        try_fallback(&self.providers, &self.config, |p| {
            let input = input.clone();
            async move { p.chat_with_web_search(input).await }
        })
        .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_stream(messages, js).await }
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
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_stream_struct(messages, tools, js).await }
        })
        .await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.chat_stream_with_tools(messages, tools, js).await }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// CompletionProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionProvider for FallbackProvider {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        try_fallback(&self.providers, &self.config, |p| {
            let js = json_schema.clone();
            async move { p.complete(req, js).await }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// EmbeddingProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl EmbeddingProvider for FallbackProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        try_fallback(&self.providers, &self.config, |p| {
            let input = input.clone();
            async move { p.embed(input).await }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// ModelsProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl ModelsProvider for FallbackProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        // `Box<dyn ModelListResponse>` is !Send so cannot go through the generic
        // try_fallback helper.  Manual loop is equivalent for this low-frequency
        // administrative call.
        let mut last_err: Option<LLMError> = None;
        for (idx, provider) in self.providers.iter().enumerate() {
            match provider.list_models(request).await {
                Ok(r) => return Ok(r),
                Err(e) if (self.config.fallbackable)(&e) => {
                    let label = if idx == 0 { "primary" } else { "fallback" };
                    log::warn!("list_models {label}[{idx}] failed: {e}. Trying next provider.");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| LLMError::Generic("No providers available".into())))
    }
}

impl LLMProvider for FallbackProvider {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ToolCall,
        chat::{ChatResponse, StructuredOutputFormat, Tool},
        completion::CompletionRequest,
        error::LLMError,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    // -----------------------------------------------------------------------
    // Mock helpers
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

    /// Always fails with the given error.
    struct AlwaysFails {
        err_msg: String,
        calls: AtomicU32,
    }

    impl AlwaysFails {
        fn new(err_msg: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                err_msg: err_msg.into(),
                calls: AtomicU32::new(0),
            })
        }
        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl ChatProvider for AlwaysFails {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(LLMError::ProviderError(self.err_msg.clone()))
        }
    }
    #[async_trait]
    impl CompletionProvider for AlwaysFails {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(LLMError::ProviderError(self.err_msg.clone()))
        }
    }
    #[async_trait]
    impl EmbeddingProvider for AlwaysFails {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(LLMError::HttpError(self.err_msg.clone()))
        }
    }
    #[async_trait]
    impl ModelsProvider for AlwaysFails {}
    impl LLMProvider for AlwaysFails {}
    impl crate::HasConfig for AlwaysFails {
        type Config = crate::NoConfig;
    }

    /// Always succeeds with `response_text`.
    struct AlwaysSucceeds {
        text: String,
        calls: AtomicU32,
    }

    impl AlwaysSucceeds {
        fn new(text: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                text: text.into(),
                calls: AtomicU32::new(0),
            })
        }
        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl ChatProvider for AlwaysSucceeds {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(MockResponse(self.text.clone())))
        }
    }
    #[async_trait]
    impl CompletionProvider for AlwaysSucceeds {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(CompletionResponse {
                text: self.text.clone(),
            })
        }
    }
    #[async_trait]
    impl EmbeddingProvider for AlwaysSucceeds {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![vec![0.5]])
        }
    }
    #[async_trait]
    impl ModelsProvider for AlwaysSucceeds {}
    impl LLMProvider for AlwaysSucceeds {}
    impl crate::HasConfig for AlwaysSucceeds {
        type Config = crate::NoConfig;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    impl FallbackLayer {
        fn build_arc(self, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
            Box::new(self).build(next)
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn primary_success_no_fallback_called() {
        let primary = AlwaysSucceeds::new("primary");
        let fallback = AlwaysSucceeds::new("fallback");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .build_arc(primary.clone() as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "primary");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 0, "fallback must not be called");
    }

    #[tokio::test]
    async fn primary_fails_fallback_is_tried() {
        let primary = AlwaysFails::new("provider down");
        let fallback = AlwaysSucceeds::new("fallback_ok");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .build_arc(primary.clone() as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "fallback_ok");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn all_providers_fail_returns_last_error() {
        let p1 = AlwaysFails::new("p1 down");
        let p2 = AlwaysFails::new("p2 down");
        let p3 = AlwaysFails::new("p3 down");

        let provider = FallbackLayer::new(vec![
            p2.clone() as Arc<dyn LLMProvider>,
            p3.clone() as Arc<dyn LLMProvider>,
        ])
        .build_arc(p1.clone() as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let err = provider.chat(&[msg], None).await.unwrap_err();
        assert!(
            err.to_string().contains("p3 down"),
            "last error should be from p3: {err}"
        );
        assert_eq!(p1.call_count(), 1);
        assert_eq!(p2.call_count(), 1);
        assert_eq!(p3.call_count(), 1);
    }

    #[tokio::test]
    async fn non_fallbackable_error_stops_immediately() {
        let primary = Arc::new(AuthFailProvider);
        let fallback = AlwaysSucceeds::new("should_not_reach");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .build_arc(primary as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let err = provider.chat(&[msg], None).await.unwrap_err();
        assert!(matches!(err, LLMError::AuthError(_)));
        assert_eq!(
            fallback.call_count(),
            0,
            "fallback must not be called on auth error"
        );
    }

    #[tokio::test]
    async fn no_tool_support_triggers_fallback() {
        let primary = Arc::new(NoToolProvider);
        let fallback = AlwaysSucceeds::new("tool_capable");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .build_arc(primary as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "tool_capable");
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn fallback_second_in_chain_succeeds() {
        let p1 = AlwaysFails::new("p1 down");
        let p2 = AlwaysFails::new("p2 down");
        let p3 = AlwaysSucceeds::new("p3_ok");

        let provider = FallbackLayer::new(vec![
            p2.clone() as Arc<dyn LLMProvider>,
            p3.clone() as Arc<dyn LLMProvider>,
        ])
        .build_arc(p1.clone() as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "p3_ok");
        assert_eq!(p1.call_count(), 1);
        assert_eq!(p2.call_count(), 1);
        assert_eq!(p3.call_count(), 1);
    }

    #[tokio::test]
    async fn completion_fallback() {
        let primary = AlwaysFails::new("down");
        let fallback = AlwaysSucceeds::new("fallback_completion");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .build_arc(primary.clone() as Arc<dyn LLMProvider>);

        let req = CompletionRequest::new("prompt");
        let resp = provider.complete(&req, None).await.unwrap();
        assert_eq!(resp.text, "fallback_completion");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn embedding_fallback() {
        let primary = AlwaysFails::new("embed down");
        let fallback = AlwaysSucceeds::new("embed_ok");

        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .build_arc(primary.clone() as Arc<dyn LLMProvider>);

        let result = provider.embed(vec!["text".into()]).await.unwrap();
        assert_eq!(result, vec![vec![0.5_f32]]);
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn custom_fallbackable_predicate() {
        // Custom: only fallback on auth errors (unusual — proves override works).
        let primary = Arc::new(AuthFailProvider);
        let fallback = AlwaysSucceeds::new("custom_fallback");

        let config = FallbackConfig {
            fallbackable: |err| matches!(err, LLMError::AuthError(_)),
        };
        let provider = FallbackLayer::new(vec![fallback.clone() as Arc<dyn LLMProvider>])
            .with_config(config)
            .build_arc(primary as Arc<dyn LLMProvider>);

        let msg = ChatMessage::user().content("hi").build();
        let resp = provider.chat(&[msg], None).await.unwrap();
        assert_eq!(resp.text().unwrap(), "custom_fallback");
        assert_eq!(fallback.call_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Auxiliary mock providers
    // -----------------------------------------------------------------------

    struct AuthFailProvider;

    #[async_trait]
    impl ChatProvider for AuthFailProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Err(LLMError::AuthError("invalid key".into()))
        }
    }
    #[async_trait]
    impl CompletionProvider for AuthFailProvider {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            Err(LLMError::AuthError("invalid key".into()))
        }
    }
    #[async_trait]
    impl EmbeddingProvider for AuthFailProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Err(LLMError::AuthError("invalid key".into()))
        }
    }
    #[async_trait]
    impl ModelsProvider for AuthFailProvider {}
    impl LLMProvider for AuthFailProvider {}
    impl crate::HasConfig for AuthFailProvider {
        type Config = crate::NoConfig;
    }

    struct NoToolProvider;

    #[async_trait]
    impl ChatProvider for NoToolProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            Err(LLMError::NoToolSupport("no tools".into()))
        }
    }
    #[async_trait]
    impl CompletionProvider for NoToolProvider {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            Err(LLMError::NoToolSupport("no tools".into()))
        }
    }
    #[async_trait]
    impl EmbeddingProvider for NoToolProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Err(LLMError::NoToolSupport("no tools".into()))
        }
    }
    #[async_trait]
    impl ModelsProvider for NoToolProvider {}
    impl LLMProvider for NoToolProvider {}
    impl crate::HasConfig for NoToolProvider {
        type Config = crate::NoConfig;
    }

    // -----------------------------------------------------------------------
    // Default-predicate unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn fallbackable_errors() {
        assert!(default_is_fallbackable(&LLMError::HttpError(
            "timeout".into()
        )));
        assert!(default_is_fallbackable(&LLMError::ProviderError(
            "down".into()
        )));
        assert!(default_is_fallbackable(&LLMError::Generic(
            "network".into()
        )));
        assert!(default_is_fallbackable(&LLMError::NoToolSupport(
            "unsupported".into()
        )));
        assert!(default_is_fallbackable(&LLMError::ResponseFormatError {
            message: "bad".into(),
            raw_response: "{}".into()
        }));
    }

    #[test]
    fn non_fallbackable_errors() {
        assert!(!default_is_fallbackable(&LLMError::AuthError(
            "bad key".into()
        )));
        assert!(!default_is_fallbackable(&LLMError::InvalidRequest(
            "bad param".into()
        )));
        assert!(!default_is_fallbackable(&LLMError::JsonError(
            "parse".into()
        )));
        assert!(!default_is_fallbackable(&LLMError::ToolConfigError(
            "bad".into()
        )));
    }
}
