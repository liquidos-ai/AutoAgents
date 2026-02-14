//! In-memory implementation of a vector store backed by shared embedding providers.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::embeddings::distance::VectorDistance;
use crate::embeddings::{Embedding, EmbeddingError, SharedEmbeddingProvider, VecArc};
use crate::vector_store::request::Filter;
use crate::vector_store::{
    DEFAULT_VECTOR_NAME, NamedVectorDocument, PreparedDocument, PreparedNamedVectorDocument,
    VectorSearchRequest, VectorStoreError, VectorStoreIndex, embed_documents,
    embed_named_documents, normalize_id,
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
    named_vectors: HashMap<String, VecArc>,
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
            let mut combined =
                vec![0.0f32; doc.embeddings.iter().next().map_or(0, |e| e.vec.len())];
            let mut count = 0usize;
            for embedding in doc.embeddings.iter() {
                for (i, value) in embedding.vec.iter().enumerate() {
                    combined[i] += value;
                }
                count += 1;
            }
            if count > 0 {
                for value in &mut combined {
                    *value /= count as f32;
                }
            }

            let mut named_vectors = HashMap::new();
            if !combined.is_empty() {
                named_vectors.insert(DEFAULT_VECTOR_NAME.to_string(), combined.into());
            }

            guard.insert(
                doc.id,
                StoredEntry {
                    raw: doc.raw,
                    embeddings: doc.embeddings,
                    named_vectors,
                },
            );
        }
    }

    fn insert_prepared_named(&self, documents: Vec<PreparedNamedVectorDocument>) {
        let mut guard = self.embeddings.write().expect("lock poisoned");
        for doc in documents {
            let PreparedNamedVectorDocument { id, raw, vectors } = doc;
            let named_vectors: HashMap<String, VecArc> = vectors
                .into_iter()
                .map(|(name, vec)| (name, vec.into()))
                .collect();

            guard.insert(
                id,
                StoredEntry {
                    raw,
                    // Named-vector entries can be scored directly from `named_vectors`.
                    // Keep `embeddings` empty to avoid duplicating vector storage.
                    embeddings: crate::one_or_many::OneOrMany::Many(Vec::new()),
                    named_vectors,
                },
            );
        }
    }

    fn best_similarity(entry: &StoredEntry, query: &Embedding) -> Option<f32> {
        if !entry.embeddings.is_empty() {
            return entry
                .embeddings
                .iter()
                .map(|embedding| embedding.cosine_similarity(query, true))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        entry
            .named_vectors
            .values()
            .map(|vector| vector.as_ref().cosine_similarity(query.vec.as_ref(), true))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn named_similarity(
        entry: &StoredEntry,
        query_vector_name: &str,
        query: &Embedding,
    ) -> Option<f32> {
        let vector = entry.named_vectors.get(query_vector_name)?;
        Some(vector.as_ref().cosine_similarity(query.vec.as_ref(), true))
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
            vec: vector.into(),
        };

        let guard = self.embeddings.read().expect("lock poisoned");
        let mut matches = Vec::new();

        for (id, entry) in guard.iter() {
            if let Some(filter) = req.filter()
                && !filter.satisfies(&entry.raw)
            {
                continue;
            }

            let score = if let Some(vector_name) = req.query_vector_name()
                && vector_name != DEFAULT_VECTOR_NAME
            {
                Self::named_similarity(entry, vector_name, &query_embedding)
            } else {
                Self::best_similarity(entry, &query_embedding)
            };

            if let Some(score) = score {
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
            vec: vector.into(),
        };

        let guard = self.embeddings.read().expect("lock poisoned");
        let mut matches = Vec::new();

        for (id, entry) in guard.iter() {
            if let Some(filter) = req.filter()
                && !filter.satisfies(&entry.raw)
            {
                continue;
            }

            let score = if let Some(vector_name) = req.query_vector_name()
                && vector_name != DEFAULT_VECTOR_NAME
            {
                Self::named_similarity(entry, vector_name, &query_embedding)
            } else {
                Self::best_similarity(entry, &query_embedding)
            };

            if let Some(score) = score {
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

    async fn insert_documents_with_named_vectors<T>(
        &self,
        documents: Vec<NamedVectorDocument<T>>,
    ) -> Result<(), VectorStoreError>
    where
        T: serde::Serialize + Send + Sync + Clone,
    {
        let normalized = documents
            .into_iter()
            .map(|doc| NamedVectorDocument {
                id: normalize_id(Some(doc.id)),
                raw: doc.raw,
                vectors: doc.vectors,
            })
            .collect::<Vec<_>>();
        let prepared = embed_named_documents(&self.provider, normalized).await?;
        self.insert_prepared_named(prepared);
        Ok(())
    }
}
