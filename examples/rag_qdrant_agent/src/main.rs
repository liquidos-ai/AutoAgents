use anyhow::{Context, Error};
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::embeddings::{Embed, EmbedError, TextEmbedder};
use autoagents::core::vector_store::VectorStoreIndex;
use autoagents::core::vector_store::request::VectorSearchRequest;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::embedding::EmbeddingBuilder;
use autoagents::prelude::AgentHooks;
use autoagents_derive::agent;
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

#[agent(
    name = "support_agent",
    description = "Answer user questions using only the supplied context. If unsure, say you don't know.",
    tools = []
)]
#[derive(Default, Clone, AgentHooks)]
struct SupportAgent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is required")?;
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".into());
    let qdrant_collection =
        env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "autoagents_rag_agent".into());
    let qdrant_api_key = env::var("QDRANT_API_KEY").ok();
    let question = "How do I reset my password?";

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key.clone())
        .model("gpt-4o-mini")
        .build()
        .map_err(Error::new)?;

    let embedder = EmbeddingBuilder::<OpenAI>::new()
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
        .query(question)
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

    let mut context_blocks = Vec::new();
    for (_, _, doc) in &hits {
        context_blocks.push(format!("Title: {}\nBody: {}", doc.title, doc.body));
    }

    let context = context_blocks.join("\n\n");
    let prompt = format!(
        "You must answer using only the provided context.\nContext:\n{}\n\nQuestion: {}",
        context, question
    );

    let agent = BasicAgent::new(SupportAgent {});
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .build()
        .await?;

    let output = handle.agent.run(Task::new(prompt)).await?;
    println!("\nAnswer:\n{}", output);

    Ok(())
}
