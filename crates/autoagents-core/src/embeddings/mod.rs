use std::sync::Arc;

use autoagents_llm::embedding::EmbeddingProvider;
use autoagents_llm::error::LLMError;
use serde::{Deserialize, Serialize};

use crate::one_or_many::OneOrMany;

pub mod distance;

pub type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider + Send + Sync>;
pub type VecArc = Arc<[f32]>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Embedding {
    pub document: String,
    pub vec: VecArc,
}

impl distance::VectorDistance for Embedding {
    fn cosine_similarity(&self, other: &Self, normalize: bool) -> f32 {
        self.vec
            .as_ref()
            .cosine_similarity(other.vec.as_ref(), normalize)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Embedding provider error: {0}")]
    Provider(#[from] LLMError),

    #[error("No content to embed")]
    Empty,

    #[error("Embedding failed: {0}")]
    EmbedFailure(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Debug, Default)]
pub struct TextEmbedder {
    parts: Vec<String>,
}

impl TextEmbedder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn embed(&mut self, text: impl Into<String>) {
        self.parts.push(text.into());
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub fn parts(&self) -> &[String] {
        &self.parts
    }

    pub fn into_parts(self) -> Vec<String> {
        self.parts
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("{0}")]
    Message(String),
}

pub trait Embed {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError>;
}

pub struct EmbeddingsBuilder<T> {
    provider: SharedEmbeddingProvider,
    documents: Vec<T>,
}

impl<T> EmbeddingsBuilder<T>
where
    T: Embed + Clone,
{
    pub fn new(provider: SharedEmbeddingProvider) -> Self {
        Self {
            provider,
            documents: Vec::default(),
        }
    }

    pub fn documents(mut self, docs: impl IntoIterator<Item = T>) -> Result<Self, EmbeddingError> {
        self.documents.extend(docs);
        if self.documents.is_empty() {
            return Err(EmbeddingError::Empty);
        }
        Ok(self)
    }

    pub async fn build(self) -> Result<Vec<(T, OneOrMany<Embedding>)>, EmbeddingError> {
        if self.documents.is_empty() {
            return Err(EmbeddingError::Empty);
        }

        let mut texts = Vec::default();
        let mut ranges = Vec::default();
        for doc in &self.documents {
            let mut embedder = TextEmbedder::default();
            doc.embed(&mut embedder)
                .map_err(|err| EmbeddingError::EmbedFailure(err.to_string()))?;

            if embedder.is_empty() {
                return Err(EmbeddingError::Empty);
            }

            let start = texts.len();
            let count = embedder.len();
            let parts = embedder.into_parts();
            texts.extend(parts);
            ranges.push((start, count));
        }

        let text_copy = texts.clone();
        let vectors = self
            .provider
            .embed(text_copy)
            .await
            .map_err(EmbeddingError::Provider)?;

        let mut cursor = 0usize;
        let mut results = Vec::with_capacity(self.documents.len());
        for (doc, (start, len)) in self.documents.into_iter().zip(ranges.into_iter()) {
            let slice = &vectors[start..start + len];
            let embeddings: Vec<Embedding> = slice
                .iter()
                .enumerate()
                .map(|(offset, vector)| Embedding {
                    document: texts[start + offset].clone(),
                    vec: vector.clone().into(),
                })
                .collect();
            cursor += len;
            results.push((doc, OneOrMany::from(embeddings)));
        }

        if cursor == 0 {
            return Err(EmbeddingError::Empty);
        }

        Ok(results)
    }
}
