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

#[cfg(test)]
impl Embed for String {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::distance::VectorDistance;
    use super::*;

    #[test]
    fn test_text_embedder_default() {
        let embedder = TextEmbedder::default();
        assert!(embedder.is_empty());
        assert_eq!(embedder.len(), 0);
    }

    #[test]
    fn test_text_embedder_embed_and_parts() {
        let mut embedder = TextEmbedder::default();
        embedder.embed("hello");
        embedder.embed("world");
        assert_eq!(embedder.len(), 2);
        assert!(!embedder.is_empty());
        assert_eq!(embedder.parts(), &["hello", "world"]);
    }

    #[test]
    fn test_text_embedder_into_parts() {
        let mut embedder = TextEmbedder::default();
        embedder.embed("a");
        embedder.embed("b");
        let parts = embedder.into_parts();
        assert_eq!(parts, vec!["a", "b"]);
    }

    #[test]
    fn test_embedding_creation() {
        let e = Embedding {
            document: "doc".to_string(),
            vec: vec![1.0, 0.0, 0.0].into(),
        };
        assert_eq!(e.document, "doc");
        assert_eq!(e.vec.len(), 3);
    }

    #[test]
    fn test_embedding_cosine_similarity_identical() {
        let a = Embedding {
            document: "a".to_string(),
            vec: vec![1.0, 0.0, 0.0].into(),
        };
        let b = Embedding {
            document: "b".to_string(),
            vec: vec![1.0, 0.0, 0.0].into(),
        };
        let sim = a.cosine_similarity(&b, true);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_cosine_similarity_orthogonal() {
        let a = Embedding {
            document: "a".to_string(),
            vec: vec![1.0, 0.0, 0.0].into(),
        };
        let b = Embedding {
            document: "b".to_string(),
            vec: vec![0.0, 1.0, 0.0].into(),
        };
        let sim = a.cosine_similarity(&b, true);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_embed_trait_for_string() {
        let s = "hello world".to_string();
        let mut embedder = TextEmbedder::default();
        s.embed(&mut embedder).unwrap();
        assert_eq!(embedder.len(), 1);
        assert_eq!(embedder.parts()[0], "hello world");
    }

    #[tokio::test]
    async fn test_embeddings_builder_empty_error() {
        use autoagents_test_utils::llm::MockLLMProvider;
        let provider: SharedEmbeddingProvider = Arc::new(MockLLMProvider {});
        let builder = EmbeddingsBuilder::<String>::new(provider);
        let result = builder.build().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embeddings_builder_success() {
        use autoagents_test_utils::llm::MockLLMProvider;
        let provider: SharedEmbeddingProvider = Arc::new(MockLLMProvider {});
        let result = EmbeddingsBuilder::new(provider)
            .documents(vec!["hello".to_string()])
            .unwrap()
            .build()
            .await;
        assert!(result.is_ok());
        let items = result.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, "hello");
    }

    #[test]
    fn test_embeddings_builder_documents_empty_error() {
        use autoagents_test_utils::llm::MockLLMProvider;
        let provider: SharedEmbeddingProvider = Arc::new(MockLLMProvider {});
        let result = EmbeddingsBuilder::<String>::new(provider).documents(Vec::<String>::new());
        assert!(result.is_err());
    }
}
