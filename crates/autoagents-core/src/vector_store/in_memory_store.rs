//! In-memory implementation of a vector store backed by shared embedding providers.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::embeddings::distance::VectorDistance;
use crate::embeddings::{Embedding, EmbeddingError, SharedEmbeddingProvider};
use crate::vector_store::request::Filter;
use crate::vector_store::{
    PreparedDocument, VectorSearchRequest, VectorStoreError, VectorStoreIndex, embed_documents,
    normalize_id,
};

#[derive(Clone)]
pub struct InMemoryVectorStore {
    provider: SharedEmbeddingProvider,
    embeddings: Arc<RwLock<HashMap<String, StoredEntry>>>,
}

#[derive(Clone)]
struct StoredEntry {
    raw: serde_json::Value,
    embeddings: crate::one_or_many::OneOrMany<Embedding>,
}

impl InMemoryVectorStore {
    pub fn new(provider: SharedEmbeddingProvider) -> Self {
        Self {
            provider,
            embeddings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn insert_prepared(&self, documents: Vec<PreparedDocument>) {
        let mut guard = self.embeddings.write().expect("lock poisoned");
        for doc in documents {
            guard.insert(
                doc.id,
                StoredEntry {
                    raw: doc.raw,
                    embeddings: doc.embeddings,
                },
            );
        }
    }

    fn best_similarity(entry: &StoredEntry, query: &Embedding) -> Option<f32> {
        entry
            .embeddings
            .iter()
            .map(|embedding| embedding.cosine_similarity(query, true))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[async_trait]
impl VectorStoreIndex for InMemoryVectorStore {
    type Filter = Filter<serde_json::Value>;

    async fn insert_documents<T>(&self, documents: Vec<T>) -> Result<(), VectorStoreError>
    where
        T: crate::embeddings::Embed + serde::Serialize + Send + Sync + Clone,
    {
        let docs_with_ids: Vec<(String, T)> = documents
            .into_iter()
            .map(|doc| (normalize_id(None), doc))
            .collect();

        let prepared = embed_documents(&self.provider, docs_with_ids).await?;
        self.insert_prepared(prepared);
        Ok(())
    }

    async fn insert_documents_with_ids<T>(
        &self,
        documents: Vec<(String, T)>,
    ) -> Result<(), VectorStoreError>
    where
        T: crate::embeddings::Embed + serde::Serialize + Send + Sync + Clone,
    {
        let normalized: Vec<(String, T)> = documents
            .into_iter()
            .map(|(id, doc)| (normalize_id(Some(id)), doc))
            .collect();
        let prepared = embed_documents(&self.provider, normalized).await?;
        self.insert_prepared(prepared);
        Ok(())
    }

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'de> serde::Deserialize<'de> + Send + Sync,
    {
        let vectors = self
            .provider
            .embed(vec![req.query().to_string()])
            .await
            .map_err(EmbeddingError::Provider)?;

        let Some(vector) = vectors.into_iter().next() else {
            return Ok(Vec::new());
        };

        let query_embedding = Embedding {
            document: req.query().to_string(),
            vec: vector,
        };

        let guard = self.embeddings.read().expect("lock poisoned");
        let mut matches = Vec::new();

        for (id, entry) in guard.iter() {
            if let Some(filter) = req.filter()
                && !filter.satisfies(&entry.raw)
            {
                continue;
            }

            if let Some(score) = Self::best_similarity(entry, &query_embedding) {
                if let Some(threshold) = req.threshold()
                    && (score as f64) < threshold
                {
                    continue;
                }

                let parsed: T = serde_json::from_value(entry.raw.clone())?;
                matches.push((score as f64, id.clone(), parsed));
            }
        }

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(req.samples() as usize);

        Ok(matches)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let vectors = self
            .provider
            .embed(vec![req.query().to_string()])
            .await
            .map_err(EmbeddingError::Provider)?;

        let Some(vector) = vectors.into_iter().next() else {
            return Ok(Vec::new());
        };

        let query_embedding = Embedding {
            document: req.query().to_string(),
            vec: vector,
        };

        let guard = self.embeddings.read().expect("lock poisoned");
        let mut matches = Vec::new();

        for (id, entry) in guard.iter() {
            if let Some(filter) = req.filter()
                && !filter.satisfies(&entry.raw)
            {
                continue;
            }

            if let Some(score) = Self::best_similarity(entry, &query_embedding) {
                if let Some(threshold) = req.threshold()
                    && (score as f64) < threshold
                {
                    continue;
                }

                matches.push((score as f64, id.clone()));
            }
        }

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(req.samples() as usize);

        Ok(matches)
    }
}
