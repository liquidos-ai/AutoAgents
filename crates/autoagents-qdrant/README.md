# AutoAgents Qdrant

Vector store index integration for [Qdrant](https://qdrant.tech/). This adapter works with AutoAgents embedding providers and supports dense retrieval as well as Qdrant's [hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/).

## Named vectors

`autoagents-qdrant` supports Qdrant named vectors via the vector-store API:

- Use `VectorStoreIndex::insert_documents_with_named_vectors(...)` to upsert points with multiple named vector spaces.
- Use `VectorSearchRequest::builder().query_vector_name("symbol")` to select the vector space at query time.
- Keep omitting `query_vector_name` (or use `"default"`) for backward-compatible single-vector behavior.
