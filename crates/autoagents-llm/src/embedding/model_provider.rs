use std::marker::PhantomData;

use crate::embedding::EmbeddingProvider;

/// Builder for creating embedding-only providers without going through the LLM builder.
pub struct EmbeddingBuilder<P: EmbeddingProvider> {
    backend: PhantomData<P>,
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) embedding_encoding_format: Option<String>,
    pub(crate) embedding_dimensions: Option<u32>,
    pub(crate) api_version: Option<String>,
    pub(crate) deployment_id: Option<String>,
    pub(crate) timeout_seconds: Option<u64>,
}

impl<P: EmbeddingProvider> Default for EmbeddingBuilder<P> {
    fn default() -> Self {
        Self {
            backend: PhantomData,
            api_key: None,
            base_url: None,
            model: None,
            embedding_encoding_format: None,
            embedding_dimensions: None,
            api_version: None,
            deployment_id: None,
            timeout_seconds: None,
        }
    }
}

impl<P: EmbeddingProvider> EmbeddingBuilder<P> {
    /// Create a new embedding provider builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the API key for the provider.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set a custom base URL for the provider.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model identifier for embeddings.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the encoding format for embeddings.
    pub fn embedding_encoding_format(mut self, format: impl Into<String>) -> Self {
        self.embedding_encoding_format = Some(format.into());
        self
    }

    /// Set the dimensions for embeddings.
    pub fn embedding_dimensions(mut self, dimensions: u32) -> Self {
        self.embedding_dimensions = Some(dimensions);
        self
    }

    /// Set the API version (used by Azure OpenAI).
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// Set the deployment ID (used by Azure OpenAI).
    pub fn deployment_id(mut self, deployment_id: impl Into<String>) -> Self {
        self.deployment_id = Some(deployment_id.into());
        self
    }

    /// Set a request timeout in seconds.
    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = Some(timeout);
        self
    }
}
