use std::collections::HashMap;

use async_trait::async_trait;
use autoagents_core::embeddings::{Embed, Embedding, EmbeddingError, SharedEmbeddingProvider};
use autoagents_core::one_or_many::OneOrMany;
use autoagents_core::vector_store::request::{Filter, FilterError};
use autoagents_core::vector_store::{
    DEFAULT_VECTOR_NAME, NamedVectorDocument, PreparedDocument, VectorSearchRequest,
    VectorStoreError, VectorStoreIndex, embed_documents, embed_named_documents, normalize_id,
};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter as QdrantFilter,
    PointStruct, Range, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    VectorsConfigBuilder, condition, with_payload_selector,
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

    async fn ensure_named_collection(
        &self,
        dimensions: &HashMap<String, u64>,
    ) -> Result<(), VectorStoreError> {
        let request = Self::named_collection_request(&self.collection_name, dimensions);

        let result = self.client.create_collection(request).await;
        if let Err(err) = result {
            let message = err.to_string();
            if !message.contains("already exists") {
                return Err(VectorStoreError::DatastoreError(Box::new(err)));
            }
        }

        Ok(())
    }

    pub async fn recreate_named_collection(
        &self,
        dimensions: HashMap<String, u64>,
    ) -> Result<(), VectorStoreError> {
        let exists = self
            .client
            .collection_exists(self.collection_name.clone())
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        if exists {
            self.client
                .delete_collection(self.collection_name.clone())
                .await
                .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;
        }

        let request = Self::named_collection_request(&self.collection_name, &dimensions);
        self.client
            .create_collection(request)
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
    }

    fn named_collection_request(
        collection_name: &str,
        dimensions: &HashMap<String, u64>,
    ) -> qdrant_client::qdrant::CreateCollection {
        let mut config = VectorsConfigBuilder::default();
        for (name, dimension) in dimensions {
            config.add_named_vector_params(
                name.clone(),
                VectorParamsBuilder::new(*dimension, Distance::Cosine),
            );
        }

        CreateCollectionBuilder::new(collection_name.to_string())
            .vectors_config(config)
            .build()
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

    fn named_dimensions(vectors: &HashMap<String, Vec<f32>>) -> HashMap<String, u64> {
        vectors
            .iter()
            .map(|(name, vector)| (name.clone(), vector.len() as u64))
            .collect()
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

        if let Some(vector_name) = req.query_vector_name()
            && vector_name != DEFAULT_VECTOR_NAME
        {
            search = search.vector_name(vector_name.to_string());
        }

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

        if let Some(vector_name) = req.query_vector_name()
            && vector_name != DEFAULT_VECTOR_NAME
        {
            search = search.vector_name(vector_name.to_string());
        }

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

    async fn insert_documents_with_named_vectors<T>(
        &self,
        documents: Vec<NamedVectorDocument<T>>,
    ) -> Result<(), VectorStoreError>
    where
        T: Serialize + Send + Sync + Clone,
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
        let Some(first) = prepared.first() else {
            return Ok(());
        };

        let dimensions = Self::named_dimensions(&first.vectors);
        self.ensure_named_collection(&dimensions).await?;

        let mut points = Vec::new();
        for doc in prepared {
            let source_id = doc.id.clone();
            let payload = Payload::try_from(serde_json::json!({
                "raw": doc.raw,
                "source_id": source_id,
            }))
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;
            let point_id = Self::stable_point_id(&source_id);
            points.push(PointStruct::new(point_id, doc.vectors, payload));
        }

        let request = UpsertPointsBuilder::new(self.collection_name.clone(), points).build();
        self.client
            .upsert_points(request)
            .await
            .map_err(|err| VectorStoreError::DatastoreError(Box::new(err)))?;

        Ok(())
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
        OneOrMany::One(embedding) => Ok(embedding.vec.to_vec()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents_core::embeddings::Embedding;
    use autoagents_core::one_or_many::OneOrMany;
    use autoagents_core::vector_store::request::{Filter, SearchFilter};
    use autoagents_llm::embedding::EmbeddingProvider;
    use autoagents_llm::error::LLMError;
    use std::sync::Arc;

    #[derive(Debug)]
    struct DummyEmbeddingProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for DummyEmbeddingProvider {
        async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn test_stable_point_id_deterministic() {
        let id1 = QdrantVectorStore::stable_point_id("doc:1");
        let id2 = QdrantVectorStore::stable_point_id("doc:1");
        let id3 = QdrantVectorStore::stable_point_id("doc:2");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_payload_encode_decode() {
        #[derive(Debug, Clone, serde::Deserialize)]
        struct TestDoc {
            name: String,
        }

        let doc = PreparedDocument {
            id: "doc-1".to_string(),
            raw: serde_json::json!({"name":"alpha"}),
            embeddings: OneOrMany::One(Embedding {
                document: "alpha".to_string(),
                vec: Arc::from(vec![0.1_f32, 0.2_f32]),
            }),
        };

        let payload = QdrantVectorStore::payload_for(&doc).unwrap();
        let payload_map: HashMap<String, qdrant_client::qdrant::Value> = payload.clone().into();
        let decoded_id = QdrantVectorStore::decode_id(&payload_map).unwrap();
        assert_eq!(decoded_id, "doc-1");

        let decoded: Option<TestDoc> = QdrantVectorStore::decode_raw(&payload_map).unwrap();
        assert_eq!(decoded.unwrap().name, "alpha");
    }

    #[test]
    fn test_named_dimensions() {
        let vectors = HashMap::from([
            ("a".to_string(), vec![0.1_f32, 0.2_f32]),
            ("b".to_string(), vec![1.0_f32]),
        ]);
        let dims = QdrantVectorStore::named_dimensions(&vectors);
        assert_eq!(dims.get("a"), Some(&2));
        assert_eq!(dims.get("b"), Some(&1));
    }

    #[test]
    fn test_number_to_f64() {
        assert_eq!(number_to_f64(&serde_json::json!(1)).unwrap(), 1.0);
        assert_eq!(number_to_f64(&serde_json::json!(1.5)).unwrap(), 1.5);
        assert!(number_to_f64(&serde_json::json!("x")).is_err());
    }

    #[test]
    fn test_value_to_match_value() {
        let m = value_to_match_value(serde_json::json!("a")).unwrap();
        match m {
            qdrant_client::qdrant::r#match::MatchValue::Keyword(val) => assert_eq!(val, "a"),
            _ => panic!("expected keyword"),
        }

        let m = value_to_match_value(serde_json::json!(true)).unwrap();
        match m {
            qdrant_client::qdrant::r#match::MatchValue::Boolean(val) => assert!(val),
            _ => panic!("expected boolean"),
        }
    }

    #[test]
    fn test_value_to_match_value_numbers_and_errors() {
        let m = value_to_match_value(serde_json::json!(42)).unwrap();
        match m {
            qdrant_client::qdrant::r#match::MatchValue::Integer(val) => assert_eq!(val, 42),
            _ => panic!("expected integer"),
        }

        let m = value_to_match_value(serde_json::json!(1.5)).unwrap();
        match m {
            qdrant_client::qdrant::r#match::MatchValue::Keyword(val) => assert_eq!(val, "1.5"),
            _ => panic!("expected keyword"),
        }

        assert!(value_to_match_value(serde_json::json!([1, 2, 3])).is_err());
    }

    #[test]
    fn test_to_qdrant_filter_lt() {
        let filter = Filter::Lt("num".to_string(), serde_json::json!(10));
        let qdrant = to_qdrant_filter(filter).unwrap();
        assert_eq!(qdrant.must.len(), 1);
    }

    #[test]
    fn test_to_qdrant_filter_and_or() {
        let filter = Filter::Eq("field".to_string(), serde_json::json!("x"))
            .and(Filter::Gt("num".to_string(), serde_json::json!(2)));
        let qdrant = to_qdrant_filter(filter).unwrap();
        assert_eq!(qdrant.must.len(), 2);

        let filter = Filter::Eq("field".to_string(), serde_json::json!("x"))
            .or(Filter::Lt("num".to_string(), serde_json::json!(10)));
        let qdrant = to_qdrant_filter(filter).unwrap();
        assert_eq!(qdrant.should.len(), 2);
    }

    #[test]
    fn test_decode_helpers_missing_fields() {
        let payload: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
        assert!(QdrantVectorStore::decode_id(&payload).is_none());
        let raw: Option<serde_json::Value> = QdrantVectorStore::decode_raw(&payload).unwrap();
        assert!(raw.is_none());
    }

    #[test]
    fn test_to_qdrant_filter_eq_and_gt() {
        let filter = Filter::Eq("tag".to_string(), serde_json::json!("alpha"));
        let qdrant = to_qdrant_filter(filter).unwrap();
        assert_eq!(qdrant.must.len(), 1);

        let filter = Filter::Gt("score".to_string(), serde_json::json!(1.5));
        let qdrant = to_qdrant_filter(filter).unwrap();
        assert_eq!(qdrant.must.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_documents_by_ids_empty_is_noop() {
        let provider = Arc::new(DummyEmbeddingProvider);
        let store =
            QdrantVectorStore::new(provider, "http://localhost:6333", "collection").unwrap();
        let result = store.delete_documents_by_ids(&[]).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_combine_embeddings() {
        let one = OneOrMany::One(Embedding {
            document: "doc".to_string(),
            vec: Arc::from(vec![1.0_f32, 2.0_f32]),
        });
        let combined = combine_embeddings(&one).unwrap();
        assert_eq!(combined, vec![1.0, 2.0]);

        let many = OneOrMany::Many(vec![
            Embedding {
                document: "a".to_string(),
                vec: Arc::from(vec![1.0_f32, 3.0_f32]),
            },
            Embedding {
                document: "b".to_string(),
                vec: Arc::from(vec![3.0_f32, 5.0_f32]),
            },
        ]);
        let combined = combine_embeddings(&many).unwrap();
        assert_eq!(combined, vec![2.0, 4.0]);
    }

    #[test]
    fn test_combine_embeddings_dimension_mismatch() {
        let many = OneOrMany::Many(vec![
            Embedding {
                document: "a".to_string(),
                vec: Arc::from(vec![1.0_f32, 2.0_f32]),
            },
            Embedding {
                document: "b".to_string(),
                vec: Arc::from(vec![1.0_f32]),
            },
        ]);
        let err = combine_embeddings(&many).unwrap_err();
        assert!(
            err.to_string()
                .contains("inconsistent embedding dimensions")
        );
    }
}
