pub use payload::{
    NamedVectorPayloadDocument, PayloadDocument, PreparedNamedVectorPayloadDocument,
    PreparedPayloadDocument, embed_documents_with_payload_fields, embed_named_payload_documents,
    embed_payload_documents, mirrored_payload_fields, mirrored_payload_fields_for,
};
pub use request::VectorSearchRequest;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::document::Document;
use crate::embeddings::{Embed, Embedding, EmbeddingError, SharedEmbeddingProvider};
use crate::one_or_many::OneOrMany;
use crate::vector_store::request::{FilterError, SearchFilter};

pub mod in_memory_store;
pub mod payload;
pub mod request;

pub const DEFAULT_VECTOR_NAME: &str = "default";

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    #[error("Json error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Filter error: {0}")]
    FilterError(#[from] FilterError),

    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[error("Error while building VectorSearchRequest: {0}")]
    BuilderError(String),
}

#[async_trait]
pub trait VectorStoreIndex: Send + Sync {
    type Filter: SearchFilter + Send + Sync;

    async fn insert_documents<T>(&self, documents: Vec<T>) -> Result<(), VectorStoreError>
    where
        T: Embed + Serialize + Send + Sync + Clone;

    async fn insert_documents_with_ids<T>(
        &self,
        documents: Vec<(String, T)>,
    ) -> Result<(), VectorStoreError>
    where
        T: Embed + Serialize + Send + Sync + Clone;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'de> Deserialize<'de> + Send + Sync;

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError>;

    async fn insert_documents_with_named_vectors<T>(
        &self,
        documents: Vec<NamedVectorDocument<T>>,
    ) -> Result<(), VectorStoreError>
    where
        T: Serialize + Send + Sync + Clone;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreOutput {
    pub score: f64,
    pub id: String,
    pub document: Document,
}

#[derive(Debug, Clone)]
pub struct PreparedDocument {
    pub id: String,
    pub raw: serde_json::Value,
    pub embeddings: OneOrMany<Embedding>,
}

#[derive(Debug, Clone)]
pub struct NamedVectorDocument<T> {
    pub id: String,
    pub raw: T,
    pub vectors: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PreparedNamedVectorDocument {
    pub id: String,
    pub raw: serde_json::Value,
    pub vectors: HashMap<String, Vec<f32>>,
}

pub async fn embed_documents<T>(
    provider: &SharedEmbeddingProvider,
    documents: Vec<(String, T)>,
) -> Result<Vec<PreparedDocument>, VectorStoreError>
where
    T: Embed + Serialize + Send + Sync + Clone,
{
    let prepared =
        embed_documents_with_payload_fields(provider, documents, std::iter::empty::<&str>())
            .await?;
    Ok(prepared
        .into_iter()
        .map(|doc| PreparedDocument {
            id: doc.id,
            raw: doc.raw,
            embeddings: doc.embeddings,
        })
        .collect())
}

pub async fn embed_named_documents<T>(
    provider: &SharedEmbeddingProvider,
    documents: Vec<NamedVectorDocument<T>>,
) -> Result<Vec<PreparedNamedVectorDocument>, VectorStoreError>
where
    T: Serialize + Send + Sync + Clone,
{
    let documents = documents
        .into_iter()
        .map(|doc| NamedVectorPayloadDocument {
            id: doc.id,
            raw: doc.raw,
            vectors: doc.vectors,
            payload_fields: HashMap::new(),
        })
        .collect();

    let prepared = embed_named_payload_documents(provider, documents).await?;
    Ok(prepared
        .into_iter()
        .map(|doc| PreparedNamedVectorDocument {
            id: doc.id,
            raw: doc.raw,
            vectors: doc.vectors,
        })
        .collect())
}

pub fn normalize_id(id: Option<String>) -> String {
    id.unwrap_or_else(|| Uuid::new_v4().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;
    use crate::embeddings::{Embed, EmbedError, TextEmbedder};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::error::LLMError;
    use serde::Serialize;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct DummyEmbeddingProvider {
        vectors: Vec<Vec<f32>>,
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for DummyEmbeddingProvider {
        async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(self.vectors.clone())
        }
    }

    #[derive(Debug, Clone, Serialize)]
    struct MultiPartDoc {
        parts: Vec<String>,
    }

    impl Embed for MultiPartDoc {
        fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            for part in &self.parts {
                embedder.embed(part.clone());
            }
            Ok(())
        }
    }

    #[derive(Debug, Clone, Serialize)]
    struct EmptyDoc;

    impl Embed for EmptyDoc {
        fn embed(&self, _embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            Ok(())
        }
    }

    #[test]
    fn test_normalize_id_none_generates_uuid() {
        let id = normalize_id(None);
        assert!(!id.is_empty());
        assert!(uuid::Uuid::parse_str(&id).is_ok());
    }

    #[test]
    fn test_normalize_id_some_returns_value() {
        let id = normalize_id(Some("custom-id".to_string()));
        assert_eq!(id, "custom-id");
    }

    #[tokio::test]
    async fn test_embed_documents_with_mock() {
        use crate::tests::MockLLMProvider;
        let provider: SharedEmbeddingProvider = Arc::new(MockLLMProvider {});
        let docs = vec![("id1".to_string(), Document::new("hello"))];
        let result = embed_documents(&provider, docs).await;
        assert!(result.is_ok());
        let prepared = result.unwrap();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].id, "id1");
    }

    #[tokio::test]
    async fn test_embed_documents_empty_embedder() {
        let provider: SharedEmbeddingProvider =
            Arc::new(DummyEmbeddingProvider { vectors: vec![] });
        let docs = vec![("id1".to_string(), EmptyDoc)];
        let err = embed_documents(&provider, docs).await.unwrap_err();
        assert!(err.to_string().contains("No content to embed"));
    }

    #[tokio::test]
    async fn test_embed_documents_fewer_vectors_than_expected() {
        let provider: SharedEmbeddingProvider = Arc::new(DummyEmbeddingProvider {
            vectors: vec![vec![0.1_f32]],
        });
        let docs = vec![(
            "id1".to_string(),
            MultiPartDoc {
                parts: vec!["a".to_string(), "b".to_string()],
            },
        )];
        let err = embed_documents(&provider, docs).await.unwrap_err();
        assert!(err.to_string().contains("fewer vectors"));
    }

    #[tokio::test]
    async fn test_embed_named_documents_success() {
        let provider: SharedEmbeddingProvider = Arc::new(DummyEmbeddingProvider {
            vectors: vec![vec![0.1_f32], vec![0.2_f32]],
        });
        let docs = vec![NamedVectorDocument {
            id: "doc-1".to_string(),
            raw: "raw".to_string(),
            vectors: HashMap::from([
                ("title".to_string(), "hello".to_string()),
                ("body".to_string(), "world".to_string()),
            ]),
        }];
        let prepared = embed_named_documents(&provider, docs).await.unwrap();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].vectors.len(), 2);
    }

    #[tokio::test]
    async fn test_embed_named_documents_empty_vectors() {
        let provider: SharedEmbeddingProvider =
            Arc::new(DummyEmbeddingProvider { vectors: vec![] });
        let docs = vec![NamedVectorDocument {
            id: "doc-1".to_string(),
            raw: "raw".to_string(),
            vectors: HashMap::new(),
        }];
        let err = embed_named_documents(&provider, docs).await.unwrap_err();
        assert!(err.to_string().contains("No content to embed"));
    }

    #[tokio::test]
    async fn test_embed_named_documents_fewer_vectors() {
        let provider: SharedEmbeddingProvider = Arc::new(DummyEmbeddingProvider {
            vectors: vec![vec![0.1_f32]],
        });
        let docs = vec![NamedVectorDocument {
            id: "doc-1".to_string(),
            raw: "raw".to_string(),
            vectors: HashMap::from([
                ("title".to_string(), "hello".to_string()),
                ("body".to_string(), "world".to_string()),
            ]),
        }];
        let err = embed_named_documents(&provider, docs).await.unwrap_err();
        assert!(err.to_string().contains("fewer vectors"));
    }
}
