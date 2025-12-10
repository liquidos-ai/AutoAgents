use anyhow::Context;
use autoagents_core::document::Document;
use autoagents_core::embeddings::SharedEmbeddingProvider;
use autoagents_core::readers::simple_directory_reader::SimpleDirectoryReader;
use autoagents_core::vector_store::in_memory_store::InMemoryVectorStore;
use autoagents_core::vector_store::request::VectorSearchRequest;
use autoagents_core::vector_store::VectorStoreIndex;
use autoagents_llm::backends::openai::OpenAI;
use autoagents_llm::embedding::EmbeddingBuilder;
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is required")?;
    let provider: SharedEmbeddingProvider = EmbeddingBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("text-embedding-3-small")
        .build()
        .context("failed to build OpenAI embedding client")?;

    let documents = SimpleDirectoryReader::new("examples/vector_store_in_memory/data")
        .with_extensions(["txt"])
        .load_data()
        .context("failed to read local documents")?;

    let store = InMemoryVectorStore::new(provider.clone());
    store
        .insert_documents(documents)
        .await
        .context("failed to insert documents into in-memory store")?;

    let request = VectorSearchRequest::builder()
        .query("How do I reset my account password?")
        .samples(2)
        .build()
        .context("failed to build search request")?;

    let results = store
        .top_n::<Document>(request)
        .await
        .context("search against in-memory store failed")?;

    println!("Found {} related documents", results.len());
    for (score, id, doc) in results {
        println!(
            "- score: {score:.4}, id: {id}, source: {}, content: {}",
            doc.metadata["source"].as_str().unwrap_or("unknown"),
            doc.page_content
        );
    }

    Ok(())
}
