use anyhow::{Context, Error};
use autoagents_core::embeddings::{Embed, EmbedError, TextEmbedder};
use autoagents_core::vector_store::request::VectorSearchRequest;
use autoagents_core::vector_store::VectorStoreIndex;
use autoagents_llm::backends::openai::OpenAI;
use autoagents_llm::builder::LLMBuilder;
use autoagents_llm::chat::{ChatMessage, ChatProvider, ChatRole, MessageType};
use autoagents_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct KnowledgeDoc {
    title: String,
    body: String,
}

impl Embed for KnowledgeDoc {
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
        env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "autoagents_rag_qa".into());
    let qdrant_api_key = env::var("QDRANT_API_KEY").ok();
    let question = env::var("QUESTION").unwrap_or_else(|_| "How do I reset my password?".into());

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key.clone())
        .model("gpt-4o-mini")
        .build()
        .map_err(Error::new)?;

    let embedder: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("text-embedding-3-small")
        .build()
        .map_err(Error::new)?;

    let store = if let Some(key) = qdrant_api_key {
        QdrantVectorStore::with_api_key(
            embedder.clone(),
            qdrant_url.clone(),
            qdrant_collection.clone(),
            Some(key),
        )?
    } else {
        QdrantVectorStore::new(
            embedder.clone(),
            qdrant_url.clone(),
            qdrant_collection.clone(),
        )?
    };

    let documents = vec![
        KnowledgeDoc {
            title: "Resetting your password".into(),
            body: "Open account settings, choose security, and use the reset password button."
                .into(),
        },
        KnowledgeDoc {
            title: "Two-factor authentication".into(),
            body:
                "Enable 2FA from the security tab and scan the QR code with your authenticator app."
                    .into(),
        },
        KnowledgeDoc {
            title: "Exporting workspaces".into(),
            body: "From the workspace menu choose Export to download a ZIP archive of all files."
                .into(),
        },
        KnowledgeDoc {
            title: "Sharing files".into(),
            body: "Click Share on a file, choose recipients, and set view or edit permissions."
                .into(),
        },
    ];

    store
        .insert_documents(documents)
        .await
        .context("failed to upsert documents into Qdrant")?;

    let request = VectorSearchRequest::builder()
        .query(question.clone())
        .samples(3)
        .build()
        .context("failed to build vector search request")?;

    let hits = store
        .top_n::<KnowledgeDoc>(request)
        .await
        .context("vector search failed")?;

    if hits.is_empty() {
        println!("No matches found in Qdrant");
        return Ok(());
    }

    println!("Top matches:");
    let mut context_blocks = Vec::new();
    for (score, id, doc) in &hits {
        println!("- score: {score:.4}, id: {id}, title: {}", doc.title);
        context_blocks.push(format!("Title: {}\nBody: {}", doc.title, doc.body));
    }

    let context = context_blocks.join("\n\n");
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            message_type: MessageType::Text,
            content: "You are a helpful support assistant. Use ONLY the provided context to answer. If the answer is not in the context, say you don't know.".into(),
        },
        ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: format!("Context:\n{}\n\nQuestion: {}", context, question),
        },
    ];

    let reply = llm
        .chat(&messages, None, None)
        .await
        .context("chat completion failed")?;

    if let Some(text) = reply.text() {
        println!("\nAnswer:\n{}", text);
    } else {
        println!("\nAnswer unavailable from LLM");
    }

    Ok(())
}
