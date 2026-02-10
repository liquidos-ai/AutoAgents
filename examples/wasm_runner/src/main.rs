use autoagents::{
    core::error::Error,
    init_logging,
    llm::{backends::openai::OpenAI, builder::LLMBuilder},
};
use std::sync::Arc;
mod wasm;

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    init_logging();
    // Check if API key is set
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".into());

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key) // Set the API key
        .model("gpt-4o") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .build()
        .expect("Failed to build LLM");

    wasm::wasm_agent(llm).await?;
    Ok(())
}
