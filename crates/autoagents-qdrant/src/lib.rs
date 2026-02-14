use std::collections::HashMap;

use async_trait::async_trait;
use autoagents_core::embeddings::{Embed, Embedding, EmbeddingError, SharedEmbeddingProvider};
use autoagents_core::one_or_many::OneOrMany;
use autoagents_core::vector_store::request::{Filter, FilterError};
use autoagents_core::vector_store::{
    PreparedDocument, VectorSearchRequest, VectorStoreError, VectorStoreIndex, embed_documents,
    normalize_id,
};
use qdrant_client::Qdrant;
use qdrant_client::client::Payload;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter as QdrantFilter,
    PointStruct, Range, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder, condition,
    with_payload_selector,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone)]
pub struct QdrantVectorStore {
    client: Qdrant,
    collection_name: String,
    provider: SharedEmbeddingProvider,
}

impl QdrantVectorStore {
    fn stable_point_id(source_id: &str) -> String {
        // Qdrant point ids are UUID/u64. Convert arbitrary logical ids
        // (e.g. "path:start:end") into a deterministic UUIDv5.
        Uuid::new_v5(&Uuid::NAMESPACE_URL, source_id.as_bytes()).to_string()
    }

    pub fn new(
        provider: SharedEmbeddingProvider,
        url: impl Into<String>,
        collection_name: impl Into<String>,
    ) -> Result<Self, VectorStoreError> {
        Self::with_api_key(provider, url, collection_name, None)
    }

    pub fn with_api_key(
        provider: SharedEmbeddingProvider,
        url: impl Into<String>,
        collection_name: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self, VectorStoreError> {
        let url = url.into();
        let builder = Qdrant::from_url(&url);
        let client = if let Some(key) = api_key {
            builder
                .api_key(key)
                .build()
                .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?
        } else {
            builder
                .build()
                .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?
        };

        Ok(Self {
            client,
            collection_name: collection_name.into(),
            provider,
        })
    }

    async fn ensure_collection(&self, dimension: u64) -> Result<(), VectorStoreError> {
        let request = CreateCollectionBuilder::new(self.collection_name.clone())
            .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine))
            .build();

        let result = self.client.create_collection(request).await;
        if let Err(err) = result {
            // Ignore already existing collections to keep the operation idempotent.
            let message = err.to_string();
            if !message.contains("already exists") {
                return Err(VectorStoreError::DatastoreError(Box::new(err)));
            }
        }

        Ok(())
    }

    fn payload_for(doc: &PreparedDocument) -> Result<Payload, VectorStoreError> {
        let payload = serde_json::json!({
            "raw": doc.raw,
            "source_id": doc.id,
        });

        Payload::try_from(payload).map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))
    }

    fn decode_id(payload: &HashMap<String, qdrant_client::qdrant::Value>) -> Option<String> {
        payload
            .get("source_id")
            .and_then(|value| serde_json::to_value(value).ok())
            .and_then(|v| v.as_str().map(|id| id.to_string()))
    }

    fn decode_raw<T>(
        payload: &HashMap<String, qdrant_client::qdrant::Value>,
    ) -> Result<Option<T>, VectorStoreError>
    where
        T: for<'de> Deserialize<'de>,
    {
        if let Some(raw) = payload.get("raw") {
            let value = serde_json::to_value(raw).map_err(VectorStoreError::JsonError)?;
            let parsed = serde_json::from_value(value)?;
            Ok(Some(parsed))
        } else {
            Ok(None)
        }
    }

    /// Deletes documents using their logical/source IDs (the IDs used for upsert).
    pub async fn delete_documents_by_ids(
        &self,
        source_ids: &[String],
    ) -> Result<(), VectorStoreError> {
        if source_ids.is_empty() {
            return Ok(());
        }

        let point_ids = source_ids
            .iter()
            .map(|source_id| Self::stable_point_id(source_id))
            .collect::<Vec<_>>();

        self.client
            .delete_points(
                DeletePointsBuilder::new(self.collection_name.clone())
                    .points(point_ids)
                    .wait(true),
            )
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
    }
}

#[async_trait]
impl VectorStoreIndex for QdrantVectorStore {
    type Filter = Filter<serde_json::Value>;

    async fn insert_documents<T>(&self, documents: Vec<T>) -> Result<(), VectorStoreError>
    where
        T: Embed + Serialize + Send + Sync + Clone,
    {
        let docs: Vec<(String, T)> = documents
            .into_iter()
            .map(|doc| (normalize_id(None), doc))
            .collect();
        self.insert_documents_with_ids(docs).await
    }

    async fn insert_documents_with_ids<T>(
        &self,
        documents: Vec<(String, T)>,
    ) -> Result<(), VectorStoreError>
    where
        T: Embed + Serialize + Send + Sync + Clone,
    {
        let normalized: Vec<(String, T)> = documents
            .into_iter()
            .map(|(id, doc)| (normalize_id(Some(id)), doc))
            .collect();
        let prepared = embed_documents(&self.provider, normalized).await?;
        let Some(first) = prepared.first() else {
            return Ok(());
        };

        let dim = first
            .embeddings
            .iter()
            .next()
            .map(|e| e.vec.len())
            .unwrap_or(0);
        self.ensure_collection(dim as u64).await?;

        let mut points = Vec::new();
        for doc in prepared {
            let payload = Self::payload_for(&doc)?;
            let vector = combine_embeddings(&doc.embeddings)?;

            // Keep logical id in payload and map point id to a stable UUID.
            let point_id = Self::stable_point_id(&doc.id);

            points.push(PointStruct::new(point_id, vector, payload.clone()));
        }

        let request = UpsertPointsBuilder::new(self.collection_name.clone(), points).build();
        self.client
            .upsert_points(request)
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
    }

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'de> Deserialize<'de> + Send + Sync,
    {
        let vectors = self
            .provider
            .embed(vec![req.query().to_string()])
            .await
            .map_err(EmbeddingError::Provider)?;

        let Some(vector) = vectors.into_iter().next() else {
            return Ok(Vec::new());
        };

        let mut search =
            SearchPointsBuilder::new(self.collection_name.clone(), vector, req.samples())
                .with_payload(with_payload_selector::SelectorOptions::Enable(true));

        if let Some(filter) = req.filter() {
            search = search.filter(to_qdrant_filter(filter.clone())?);
        }

        if let Some(threshold) = req.threshold() {
            search = search.score_threshold(threshold as f32);
        }

        let response = self
            .client
            .search_points(search)
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        let mut results = Vec::new();
        for point in response.result {
            let id = Self::decode_id(&point.payload)
                .or_else(|| point.id.map(|id| format!("{id:?}")))
                .unwrap_or_default();

            if let Some(raw) = Self::decode_raw::<T>(&point.payload)? {
                results.push((point.score as f64, id, raw));
            }
        }

        Ok(results)
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

        let mut search =
            SearchPointsBuilder::new(self.collection_name.clone(), vector, req.samples())
                .with_payload(with_payload_selector::SelectorOptions::Enable(true));

        if let Some(filter) = req.filter() {
            search = search.filter(to_qdrant_filter(filter.clone())?);
        }

        if let Some(threshold) = req.threshold() {
            search = search.score_threshold(threshold as f32);
        }

        let response = self
            .client
            .search_points(search)
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        let mut results = Vec::new();
        for point in response.result {
            let id = Self::decode_id(&point.payload)
                .or_else(|| point.id.map(|id| format!("{id:?}")))
                .unwrap_or_default();
            results.push((point.score as f64, id));
        }

        Ok(results)
    }
}

fn to_qdrant_filter(filter: Filter<serde_json::Value>) -> Result<QdrantFilter, VectorStoreError> {
    use Filter::*;

    let empty = || QdrantFilter {
        must: Vec::new(),
        should: Vec::new(),
        must_not: Vec::new(),
        min_should: None,
    };

    match filter {
        Eq(key, value) => {
            let mut filter = empty();
            filter
                .must
                .push(Condition::matches(key, value_to_match_value(value)?));
            Ok(filter)
        }
        Gt(key, value) => {
            let mut filter = empty();
            filter.must.push(Condition::range(
                key,
                Range {
                    gt: Some(number_to_f64(&value)?),
                    gte: None,
                    lt: None,
                    lte: None,
                },
            ));
            Ok(filter)
        }
        Lt(key, value) => {
            let mut filter = empty();
            filter.must.push(Condition::range(
                key,
                Range {
                    lt: Some(number_to_f64(&value)?),
                    lte: None,
                    gt: None,
                    gte: None,
                },
            ));
            Ok(filter)
        }
        And(lhs, rhs) => {
            let mut left = to_qdrant_filter(*lhs)?;
            let right = to_qdrant_filter(*rhs)?;

            left.must.extend(right.must);
            left.must.extend(right.should);
            Ok(left)
        }
        Or(lhs, rhs) => {
            let left = to_qdrant_filter(*lhs)?;
            let right = to_qdrant_filter(*rhs)?;

            Ok(QdrantFilter {
                should: vec![
                    Condition {
                        condition_one_of: Some(condition::ConditionOneOf::Filter(left)),
                    },
                    Condition {
                        condition_one_of: Some(condition::ConditionOneOf::Filter(right)),
                    },
                ],
                must: Vec::new(),
                must_not: Vec::new(),
                min_should: None,
            })
        }
    }
}

fn value_to_match_value(
    value: serde_json::Value,
) -> Result<qdrant_client::qdrant::r#match::MatchValue, VectorStoreError> {
    use qdrant_client::qdrant::r#match::MatchValue;
    match value {
        serde_json::Value::String(s) => Ok(MatchValue::Keyword(s)),
        serde_json::Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                Ok(MatchValue::Integer(i))
            } else if let Some(f) = num.as_f64() {
                Ok(MatchValue::Keyword(f.to_string()))
            } else {
                Err(FilterError::TypeError("Unsupported number".into()).into())
            }
        }
        serde_json::Value::Bool(b) => Ok(MatchValue::Boolean(b)),
        other => Err(FilterError::TypeError(format!("Unsupported filter value {other:?}")).into()),
    }
}

fn number_to_f64(value: &serde_json::Value) -> Result<f64, VectorStoreError> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|v| v as f64))
        .ok_or_else(|| FilterError::TypeError(format!("Expected number, got {value:?}")).into())
}

fn combine_embeddings(embeddings: &OneOrMany<Embedding>) -> Result<Vec<f32>, VectorStoreError> {
    match embeddings {
        OneOrMany::One(embedding) => Ok(embedding.vec.clone()),
        OneOrMany::Many(list) => {
            let Some(first) = list.first() else {
                return Err(VectorStoreError::EmbeddingError(
                    EmbeddingError::EmbedFailure("no embeddings".into()),
                ));
            };

            let dim = first.vec.len();
            let mut sum = vec![0.0; dim];
            for embedding in list {
                if embedding.vec.len() != dim {
                    return Err(VectorStoreError::EmbeddingError(
                        EmbeddingError::EmbedFailure("inconsistent embedding dimensions".into()),
                    ));
                }
                for (i, value) in embedding.vec.iter().enumerate() {
                    sum[i] += value;
                }
            }

            let count = list.len() as f32;
            for value in &mut sum {
                *value /= count;
            }

            Ok(sum)
        }
    }
}
