use anyhow::{Context, Error};
use autoagents_core::embeddings::{Embed, EmbedError, TextEmbedder};
use autoagents_core::vector_store::request::VectorSearchRequest;
use autoagents_core::vector_store::VectorStoreIndex;
use autoagents_llm::backends::openai::OpenAI;
use autoagents_llm::builder::LLMBuilder;
use autoagents_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SupportArticle {
    title: String,
    body: String,
}

impl Embed for SupportArticle {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.title.clone());
        embedder.embed(self.body.clone());
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is required")?;
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".into());
    let qdrant_collection =
        env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "autoagents_demo".into());
    let qdrant_api_key = env::var("QDRANT_API_KEY").ok();

    let provider: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("text-embedding-3-small")
        .build()
        .map_err(Error::new)?;

    let store = if let Some(key) = qdrant_api_key {
        QdrantVectorStore::with_api_key(
            provider.clone(),
            qdrant_url.clone(),
            qdrant_collection.clone(),
            Some(key),
        )?
    } else {
        QdrantVectorStore::new(
            provider.clone(),
            qdrant_url.clone(),
            qdrant_collection.clone(),
        )?
    };

    let documents = vec![
        SupportArticle {
            title: "Resetting your password".into(),
            body: "Open account settings, choose security, and use the reset password button."
                .into(),
        },
        SupportArticle {
            title: "Enabling notifications".into(),
            body: "Go to preferences, toggle notifications on, and choose your channels.".into(),
        },
        SupportArticle {
            title: "Exporting a workspace".into(),
            body: "From the workspace menu choose Export to create a compressed archive.".into(),
        },
    ];

    store
        .insert_documents(documents)
        .await
        .context("failed to upsert documents into Qdrant")?;

    let request = VectorSearchRequest::builder()
        .query("How do I recover my password?")
        .samples(2)
        .build()
        .context("failed to build search request")?;

    let results = store
        .top_n::<SupportArticle>(request)
        .await
        .context("search request failed")?;

    println!("Found {} matching documents", results.len());
    for (score, id, doc) in results {
        println!("- score: {score:.4}, id: {id}, title: {}", doc.title);
    }

    Ok(())
}
