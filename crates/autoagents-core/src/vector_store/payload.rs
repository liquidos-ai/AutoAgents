use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::embeddings::{Embed, Embedding, EmbeddingError, SharedEmbeddingProvider, TextEmbedder};
use crate::one_or_many::OneOrMany;

use super::{NamedVectorDocument, VectorStoreError};

#[derive(Debug, Clone)]
pub struct PayloadDocument<T> {
    pub id: String,
    pub raw: T,
    pub payload_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct PreparedPayloadDocument {
    pub id: String,
    pub raw: serde_json::Value,
    pub payload_fields: HashMap<String, serde_json::Value>,
    pub embeddings: OneOrMany<Embedding>,
}

#[derive(Debug, Clone)]
pub struct NamedVectorPayloadDocument<T> {
    pub id: String,
    pub raw: T,
    pub vectors: HashMap<String, String>,
    pub payload_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct PreparedNamedVectorPayloadDocument {
    pub id: String,
    pub raw: serde_json::Value,
    pub payload_fields: HashMap<String, serde_json::Value>,
    pub vectors: HashMap<String, Vec<f32>>,
}

impl<T> PayloadDocument<T> {
    pub fn new(id: impl Into<String>, raw: T) -> Self {
        Self {
            id: id.into(),
            raw,
            payload_fields: HashMap::new(),
        }
    }

    pub fn with_payload_fields(
        mut self,
        payload_fields: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.payload_fields = payload_fields;
        self
    }
}

impl<T> PayloadDocument<T>
where
    T: Serialize,
{
    pub fn with_mirrored_payload_fields(
        mut self,
        fields: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, serde_json::Error> {
        self.payload_fields = mirrored_payload_fields_for(&self.raw, fields)?;
        Ok(self)
    }
}

impl<T> NamedVectorDocument<T> {
    pub fn with_payload_fields(
        self,
        payload_fields: HashMap<String, serde_json::Value>,
    ) -> NamedVectorPayloadDocument<T> {
        NamedVectorPayloadDocument {
            id: self.id,
            raw: self.raw,
            vectors: self.vectors,
            payload_fields,
        }
    }
}

impl<T> NamedVectorPayloadDocument<T> {
    pub fn new(id: impl Into<String>, raw: T, vectors: HashMap<String, String>) -> Self {
        Self {
            id: id.into(),
            raw,
            vectors,
            payload_fields: HashMap::new(),
        }
    }

    pub fn with_payload_fields(
        mut self,
        payload_fields: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.payload_fields = payload_fields;
        self
    }
}

impl<T> NamedVectorPayloadDocument<T>
where
    T: Serialize,
{
    pub fn with_mirrored_payload_fields(
        mut self,
        fields: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, serde_json::Error> {
        self.payload_fields = mirrored_payload_fields_for(&self.raw, fields)?;
        Ok(self)
    }
}

pub fn mirrored_payload_fields(
    raw: &serde_json::Value,
    fields: impl IntoIterator<Item = impl AsRef<str>>,
) -> HashMap<String, serde_json::Value> {
    let Some(raw_object) = raw.as_object() else {
        return HashMap::new();
    };

    let mut mirrored = HashMap::new();
    let mut seen = HashSet::new();
    for field in fields {
        let field = field.as_ref();
        if !seen.insert(field.to_string()) {
            continue;
        }

        if field == "raw" || field == "source_id" {
            continue;
        }

        if let Some(value) = raw_object.get(field) {
            mirrored.insert(field.to_string(), value.clone());
        }
    }

    mirrored
}

pub fn mirrored_payload_fields_for<T>(
    raw: &T,
    fields: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<HashMap<String, serde_json::Value>, serde_json::Error>
where
    T: Serialize,
{
    let raw = serde_json::to_value(raw)?;
    Ok(mirrored_payload_fields(&raw, fields))
}

pub async fn embed_documents_with_payload_fields<T, I, S>(
    provider: &SharedEmbeddingProvider,
    documents: Vec<(String, T)>,
    payload_fields: I,
) -> Result<Vec<PreparedPayloadDocument>, VectorStoreError>
where
    T: Embed + Serialize + Send + Sync + Clone,
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut all_texts = Vec::new();
    let mut ranges = Vec::new();
    let mut raws = Vec::new();
    let mut ids = Vec::new();
    let payload_field_names = payload_fields
        .into_iter()
        .map(|field| field.as_ref().to_string())
        .collect::<Vec<_>>();
    let mut mirrored_payloads = Vec::new();

    for (id, doc) in documents.iter() {
        let mut embedder = TextEmbedder::default();
        doc.embed(&mut embedder).map_err(|err| {
            VectorStoreError::EmbeddingError(EmbeddingError::EmbedFailure(err.to_string()))
        })?;

        if embedder.is_empty() {
            return Err(VectorStoreError::EmbeddingError(EmbeddingError::Empty));
        }

        let start = all_texts.len();
        let count = embedder.len();
        all_texts.extend(embedder.into_parts());
        ranges.push((start, count));
        let raw = serde_json::to_value(doc)?;
        mirrored_payloads.push(mirrored_payload_fields(&raw, &payload_field_names));
        raws.push(raw);
        ids.push(id.clone());
    }

    let vectors = provider
        .embed(all_texts.clone())
        .await
        .map_err(EmbeddingError::Provider)?;

    let mut prepared = Vec::with_capacity(ids.len());
    let mut vectors_iter = vectors.into_iter();
    let mut expected_start = 0usize;
    for (((id, raw), payload_fields), (start, count)) in ids
        .into_iter()
        .zip(raws)
        .zip(mirrored_payloads)
        .zip(ranges.into_iter())
    {
        if start != expected_start {
            return Err(VectorStoreError::EmbeddingError(
                EmbeddingError::EmbedFailure("embedding ranges are inconsistent".into()),
            ));
        }

        let mut embeddings = Vec::with_capacity(count);
        for offset in 0..count {
            let Some(vector) = vectors_iter.next() else {
                return Err(VectorStoreError::EmbeddingError(
                    EmbeddingError::EmbedFailure(
                        "embedding provider returned fewer vectors than expected".into(),
                    ),
                ));
            };

            embeddings.push(Embedding {
                document: all_texts[start + offset].clone(),
                vec: vector.into(),
            });
        }
        expected_start += count;

        prepared.push(PreparedPayloadDocument {
            id,
            raw,
            payload_fields,
            embeddings: OneOrMany::from(embeddings),
        });
    }

    Ok(prepared)
}

pub async fn embed_payload_documents<T>(
    provider: &SharedEmbeddingProvider,
    documents: Vec<PayloadDocument<T>>,
) -> Result<Vec<PreparedPayloadDocument>, VectorStoreError>
where
    T: Embed + Serialize + Send + Sync + Clone,
{
    let mut all_texts = Vec::new();
    let mut ranges = Vec::new();
    let mut raws = Vec::new();
    let mut ids = Vec::new();
    let mut mirrored_payloads = Vec::new();

    for doc in documents.iter() {
        let mut embedder = TextEmbedder::default();
        doc.raw.embed(&mut embedder).map_err(|err| {
            VectorStoreError::EmbeddingError(EmbeddingError::EmbedFailure(err.to_string()))
        })?;

        if embedder.is_empty() {
            return Err(VectorStoreError::EmbeddingError(EmbeddingError::Empty));
        }

        let start = all_texts.len();
        let count = embedder.len();
        all_texts.extend(embedder.into_parts());
        ranges.push((start, count));
        raws.push(serde_json::to_value(&doc.raw)?);
        mirrored_payloads.push(doc.payload_fields.clone());
        ids.push(doc.id.clone());
    }

    let vectors = provider
        .embed(all_texts.clone())
        .await
        .map_err(EmbeddingError::Provider)?;

    let mut prepared = Vec::with_capacity(ids.len());
    let mut vectors_iter = vectors.into_iter();
    let mut expected_start = 0usize;
    for (((id, raw), payload_fields), (start, count)) in ids
        .into_iter()
        .zip(raws)
        .zip(mirrored_payloads)
        .zip(ranges.into_iter())
    {
        if start != expected_start {
            return Err(VectorStoreError::EmbeddingError(
                EmbeddingError::EmbedFailure("embedding ranges are inconsistent".into()),
            ));
        }

        let mut embeddings = Vec::with_capacity(count);
        for offset in 0..count {
            let Some(vector) = vectors_iter.next() else {
                return Err(VectorStoreError::EmbeddingError(
                    EmbeddingError::EmbedFailure(
                        "embedding provider returned fewer vectors than expected".into(),
                    ),
                ));
            };

            embeddings.push(Embedding {
                document: all_texts[start + offset].clone(),
                vec: vector.into(),
            });
        }
        expected_start += count;

        prepared.push(PreparedPayloadDocument {
            id,
            raw,
            payload_fields,
            embeddings: OneOrMany::from(embeddings),
        });
    }

    Ok(prepared)
}

pub async fn embed_named_payload_documents<T>(
    provider: &SharedEmbeddingProvider,
    documents: Vec<NamedVectorPayloadDocument<T>>,
) -> Result<Vec<PreparedNamedVectorPayloadDocument>, VectorStoreError>
where
    T: Serialize + Send + Sync + Clone,
{
    let mut all_texts = Vec::new();
    let mut ranges = Vec::new();
    let mut raws = Vec::new();
    let mut ids = Vec::new();
    let mut names_by_doc = Vec::new();
    let mut mirrored_payloads = Vec::new();

    for doc in documents {
        if doc.vectors.is_empty() {
            return Err(VectorStoreError::EmbeddingError(EmbeddingError::Empty));
        }

        let mut names = Vec::with_capacity(doc.vectors.len());
        let start = all_texts.len();

        for (name, text) in doc.vectors {
            names.push(name);
            all_texts.push(text);
        }

        ranges.push((start, names.len()));
        names_by_doc.push(names);
        let raw = serde_json::to_value(doc.raw)?;
        mirrored_payloads.push(doc.payload_fields);
        raws.push(raw);
        ids.push(doc.id);
    }

    let vectors = provider
        .embed(all_texts.clone())
        .await
        .map_err(EmbeddingError::Provider)?;

    let mut prepared = Vec::with_capacity(ids.len());
    let mut vectors_iter = vectors.into_iter();
    let mut expected_start = 0usize;
    for ((((id, raw), payload_fields), (start, count)), names) in ids
        .into_iter()
        .zip(raws)
        .zip(mirrored_payloads)
        .zip(ranges.into_iter())
        .zip(names_by_doc.into_iter())
    {
        if start != expected_start {
            return Err(VectorStoreError::EmbeddingError(
                EmbeddingError::EmbedFailure("embedding ranges are inconsistent".into()),
            ));
        }

        let mut mapped = HashMap::with_capacity(count);
        for name in names.into_iter() {
            let Some(vector) = vectors_iter.next() else {
                return Err(VectorStoreError::EmbeddingError(
                    EmbeddingError::EmbedFailure(
                        "embedding provider returned fewer vectors than expected".into(),
                    ),
                ));
            };
            mapped.insert(name, vector);
        }
        expected_start += count;

        prepared.push(PreparedNamedVectorPayloadDocument {
            id,
            raw,
            payload_fields,
            vectors: mapped,
        });
    }

    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirrored_payload_fields_extracts_selected_root_keys() {
        let raw = serde_json::json!({
            "workspace_id": "ws-1",
            "project_id": "proj-1",
            "file_path": "src/lib.rs",
            "body": "very large text"
        });

        let mirrored = mirrored_payload_fields(&raw, ["workspace_id", "file_path", "missing"]);
        assert_eq!(mirrored.len(), 2);
        assert_eq!(
            mirrored.get("workspace_id"),
            Some(&serde_json::json!("ws-1"))
        );
        assert_eq!(
            mirrored.get("file_path"),
            Some(&serde_json::json!("src/lib.rs"))
        );
    }

    #[test]
    fn test_mirrored_payload_fields_for_serializable_value() {
        #[derive(Serialize)]
        struct IndexedDoc {
            workspace_id: &'static str,
            project_id: &'static str,
            body: &'static str,
        }

        let mirrored = mirrored_payload_fields_for(
            &IndexedDoc {
                workspace_id: "ws-1",
                project_id: "proj-1",
                body: "large text",
            },
            ["workspace_id", "project_id"],
        )
        .unwrap();

        assert_eq!(
            mirrored.get("workspace_id"),
            Some(&serde_json::json!("ws-1"))
        );
        assert_eq!(
            mirrored.get("project_id"),
            Some(&serde_json::json!("proj-1"))
        );
        assert!(!mirrored.contains_key("body"));
    }
}
