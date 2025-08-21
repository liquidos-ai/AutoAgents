use clap::{Parser, ValueEnum};
use std::sync::Arc;
mod chaining;
mod simple;
use autoagents::{
    core::error::Error,
    init_logging,
    llm::{backends::openai::OpenAI, builder::LLMBuilder},
};
mod actor;
mod liquid_edge;
mod streaming;

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Simple,
    Chaining,
    Edge,
    Streaming,
    Actor,
}

#[derive(Debug, Clone, ValueEnum)]
enum EdgeDevice {
    CPU,
    CUDA,
}

/// Simple program to demonstrate AutoAgents functionality
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "usecase")]
    usecase: UseCase,
    #[arg(short, long, help = "device", default_value = "cpu")]
    device: EdgeDevice,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    init_logging();
    let args = Args::parse();
    // Check if API key is set
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key) // Set the API key
        .model("gpt-4o") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Simple => simple::simple_agent(llm).await?,
        UseCase::Chaining => chaining::run(llm).await?,
        UseCase::Edge => liquid_edge::edge_agent(args.device).await?,
        UseCase::Streaming => streaming::run(llm).await?,
        UseCase::Actor => actor::run(llm).await?,
    }

    Ok(())
}
