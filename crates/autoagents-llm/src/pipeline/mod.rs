//! LLM pipeline abstraction for composing optimization passes.
//!
//! # Overview
//! The `pipeline` module provides a composable layer system for LLM requests.
//! Any number of [`LLMLayer`] implementations can be stacked via [`PipelineBuilder`];
//! the resulting value is a plain `Arc<dyn LLMProvider>`, transparent to all existing
//! agent code.
//!
//! # Third-party extensibility
//! External crates only need to depend on `autoagents-llm` and implement [`LLMLayer`].
//! No sealed traits or private bounds — fully open.
//!
//! ```rust,ignore
//! use autoagents_llm::{LLMProvider, pipeline::LLMLayer};
//! use std::sync::Arc;
//!
//! pub struct MyLayer;
//!
//! impl LLMLayer for MyLayer {
//!     fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
//!         Arc::new(MyWrappedProvider::new(next))
//!     }
//! }
//! ```

use crate::LLMProvider;
use std::sync::Arc;

/// An optimization pass that wraps a downstream [`LLMProvider`].
///
/// Implement this trait to add any layer (cache, compression, routing, etc.)
/// from within this crate or from an external crate.
///
/// The only type in the signature is `Arc<dyn LLMProvider>` — fully public,
/// no internal coupling.
pub trait LLMLayer: Send + Sync + 'static {
    /// Consume this layer and produce a provider that wraps `next`.
    ///
    /// `Box<Self>` enables `Box<dyn LLMLayer>` storage while still consuming `self`.
    fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider>;
}

/// Builds a layered [`LLMProvider`] pipeline from a base provider and a sequence of layers.
///
/// Layers are applied so that the **first added** layer is the **outermost**
/// (i.e., the first to intercept incoming requests).
///
/// # Example
///
/// ```rust,ignore
/// use autoagents_llm::pipeline::PipelineBuilder;
/// use autoagents_llm::optim::{CacheLayer, CacheConfig};
/// use std::time::Duration;
///
/// let llm = PipelineBuilder::new(base_provider)
///     .add_layer(CacheLayer::new(CacheConfig {
///         ttl: Some(Duration::from_secs(3600)),
///         max_size: Some(500),
///         ..CacheConfig::default()
///     }))
///     .build();
/// // Result chain: CacheLayer → base_provider
/// ```
pub struct PipelineBuilder {
    base: Arc<dyn LLMProvider>,
    layers: Vec<Box<dyn LLMLayer>>,
}

impl PipelineBuilder {
    /// Create a new pipeline with the given base provider.
    pub fn new(provider: Arc<dyn LLMProvider>) -> Self {
        Self {
            base: provider,
            layers: Vec::new(),
        }
    }

    /// Add a layer to the pipeline.
    ///
    /// The first added layer will be the outermost (first to intercept requests).
    pub fn add_layer<L: LLMLayer>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Assemble the pipeline and return the resulting [`LLMProvider`].
    ///
    /// Layers are applied bottom-up so that the first-added layer wraps all others.
    pub fn build(self) -> Arc<dyn LLMProvider> {
        let mut provider = self.base;
        for layer in self.layers.into_iter().rev() {
            provider = layer.build(provider);
        }
        provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HasConfig, LLMProvider, NoConfig,
        chat::{ChatMessage, ChatProvider, ChatResponse, StructuredOutputFormat, Tool},
        completion::{CompletionProvider, CompletionRequest, CompletionResponse},
        embedding::EmbeddingProvider,
        error::LLMError,
        models::ModelsProvider,
    };
    use async_trait::async_trait;
    use std::sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    };

    // ------------------------------------------------------------------
    // Minimal mock provider
    // ------------------------------------------------------------------

    #[derive(Debug)]
    struct MockResponse(String);

    impl ChatResponse for MockResponse {
        fn text(&self) -> Option<String> {
            Some(self.0.clone())
        }
        fn tool_calls(&self) -> Option<Vec<crate::ToolCall>> {
            None
        }
    }

    impl std::fmt::Display for MockResponse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    struct MockProvider {
        label: String,
        call_count: Arc<AtomicU32>,
    }

    impl MockProvider {
        fn new(label: impl Into<String>) -> Self {
            Self {
                label: label.into(),
                call_count: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(MockResponse(self.label.clone())))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockProvider {
        async fn complete(
            &self,
            _req: &CompletionRequest,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            Ok(CompletionResponse {
                text: self.label.clone(),
            })
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl ModelsProvider for MockProvider {}

    impl LLMProvider for MockProvider {}

    impl HasConfig for MockProvider {
        type Config = NoConfig;
    }

    // ------------------------------------------------------------------
    // Tag layer: prepends "[TAG] " to chat responses to verify order
    // ------------------------------------------------------------------

    struct TagLayer {
        tag: String,
        intercepted: Arc<AtomicU32>,
    }

    impl TagLayer {
        fn new(tag: impl Into<String>, intercepted: Arc<AtomicU32>) -> Self {
            Self {
                tag: tag.into(),
                intercepted,
            }
        }
    }

    struct TagProvider {
        tag: String,
        intercepted: Arc<AtomicU32>,
        inner: Arc<dyn LLMProvider>,
    }

    #[async_trait]
    impl ChatProvider for TagProvider {
        async fn chat_with_tools(
            &self,
            messages: &[ChatMessage],
            tools: Option<&[Tool]>,
            json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            self.intercepted.fetch_add(1, Ordering::SeqCst);
            let inner_resp = self
                .inner
                .chat_with_tools(messages, tools, json_schema)
                .await?;
            let text = format!("[{}] {}", self.tag, inner_resp.text().unwrap_or_default());
            Ok(Box::new(MockResponse(text)))
        }
    }

    #[async_trait]
    impl CompletionProvider for TagProvider {
        async fn complete(
            &self,
            req: &CompletionRequest,
            json_schema: Option<StructuredOutputFormat>,
        ) -> Result<CompletionResponse, LLMError> {
            self.inner.complete(req, json_schema).await
        }
    }

    #[async_trait]
    impl EmbeddingProvider for TagProvider {
        async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            self.inner.embed(input).await
        }
    }

    #[async_trait]
    impl ModelsProvider for TagProvider {}

    impl LLMProvider for TagProvider {}

    impl LLMLayer for TagLayer {
        fn build(self: Box<Self>, next: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
            Arc::new(TagProvider {
                tag: self.tag,
                intercepted: self.intercepted,
                inner: next,
            })
        }
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_pipeline_no_layers_returns_base() {
        let mock = Arc::new(MockProvider::new("base"));
        let call_count = mock.call_count.clone();
        let pipeline = PipelineBuilder::new(mock).build();

        let messages = vec![ChatMessage::user().content("hello").build()];
        let response = pipeline
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();

        assert_eq!(response.text(), Some("base".to_string()));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_pipeline_layer_order_first_added_is_outermost() {
        let base = Arc::new(MockProvider::new("base"));
        let intercepted_a = Arc::new(AtomicU32::new(0));
        let intercepted_b = Arc::new(AtomicU32::new(0));

        let pipeline = PipelineBuilder::new(base)
            .add_layer(TagLayer::new("A", intercepted_a.clone()))
            .add_layer(TagLayer::new("B", intercepted_b.clone()))
            .build();

        let messages = vec![ChatMessage::user().content("test").build()];
        let response = pipeline
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();

        // A is outermost: A wraps B wraps base → "[A] [B] base"
        assert_eq!(response.text(), Some("[A] [B] base".to_string()));
        assert_eq!(intercepted_a.load(Ordering::SeqCst), 1);
        assert_eq!(intercepted_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_pipeline_single_layer() {
        let base = Arc::new(MockProvider::new("inner"));
        let intercepted = Arc::new(AtomicU32::new(0));

        let pipeline = PipelineBuilder::new(base)
            .add_layer(TagLayer::new("X", intercepted.clone()))
            .build();

        let messages = vec![ChatMessage::user().content("hi").build()];
        let response = pipeline
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();

        assert_eq!(response.text(), Some("[X] inner".to_string()));
        assert_eq!(intercepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_pipeline_three_layers_order() {
        let base = Arc::new(MockProvider::new("base"));

        let pipeline = PipelineBuilder::new(base)
            .add_layer(TagLayer::new("1", Arc::new(AtomicU32::new(0))))
            .add_layer(TagLayer::new("2", Arc::new(AtomicU32::new(0))))
            .add_layer(TagLayer::new("3", Arc::new(AtomicU32::new(0))))
            .build();

        let messages = vec![ChatMessage::user().content("x").build()];
        let response = pipeline
            .chat_with_tools(&messages, None, None)
            .await
            .unwrap();

        // 1 is outermost: "[1] [2] [3] base"
        assert_eq!(response.text(), Some("[1] [2] [3] base".to_string()));
    }
}
