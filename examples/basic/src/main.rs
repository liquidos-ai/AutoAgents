use clap::{Parser, ValueEnum};
use std::sync::Arc;
mod simple;
use autoagents::{
    core::error::Error,
    init_logging,
    llm::{backends::openai::OpenAI, builder::LLMBuilder},
};
mod actor;
mod basic;
mod hooks;
mod manual_tool_agent;
mod onnx;
mod react_token_tracking;
mod react_token_tracking_ollama;
mod streaming;
mod token_tracking;
mod token_tracking_ollama;
mod toolkit;
mod utils;

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Simple,
    Basic,
    Edge,
    Streaming,
    Actor,
    Hooks,
    ManualToolAgent,
    Toolkit,
    TokenTracking,
    ReactTokenTracking,
    TokenTrackingOllama,
    ReactTokenTrackingOllama,
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
    #[arg(short, long, help = "use case")]
    usecase: UseCase,
    #[arg(short, long, help = "device", default_value = "cpu")]
    device: EdgeDevice,
    #[arg(
        short,
        long,
        help = "The Mode for the manual tool agent example",
        default_value = "addition"
    )]
    mode: String,
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
        .build()
        .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Simple => simple::simple_agent(llm).await?,
        UseCase::Basic => basic::basic_agent(llm).await?,
        UseCase::Edge => onnx::edge_agent(args.device).await?,
        UseCase::Streaming => streaming::run(llm).await?,
        UseCase::Actor => actor::run(llm).await?,
        UseCase::Hooks => hooks::hooks_agent(llm).await?,
        UseCase::ManualToolAgent => manual_tool_agent::run_agent(llm, &args.mode).await?,
        UseCase::Toolkit => toolkit::run_agent(llm).await?,
        UseCase::TokenTracking => token_tracking::token_tracking_example(llm).await?,
        UseCase::ReactTokenTracking => react_token_tracking::react_token_tracking_example(llm).await?,
        UseCase::TokenTrackingOllama => token_tracking_ollama::token_tracking_ollama_example().await?,
        UseCase::ReactTokenTrackingOllama => react_token_tracking_ollama::react_token_tracking_ollama_example().await?,
    }

    Ok(())
}
