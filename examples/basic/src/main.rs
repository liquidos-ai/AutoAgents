use clap::{Parser, ValueEnum};
use std::sync::Arc;
mod simple;
use autoagents::prelude::*;
mod actor;
mod basic;
mod hooks;
mod llm;
mod manual_tool_agent;
#[cfg(any(feature = "onnx", feature = "onnx-cuda"))]
mod onnx;
mod streaming;
mod toolkit;
mod utils;

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Simple,
    Basic,
    #[cfg(any(feature = "onnx", feature = "onnx-cuda"))]
    Edge,
    Streaming,
    Actor,
    Hooks,
    ManualToolAgent,
    Toolkit,
    Llm,
}

#[cfg(any(feature = "onnx", feature = "onnx-cuda"))]
#[derive(Debug, Clone, ValueEnum)]
enum EdgeDevice {
    CPU,
    #[cfg(feature = "onnx-cuda")]
    CUDA,
}

/// Simple program to demonstrate AutoAgents functionality
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "use case")]
    usecase: UseCase,
    #[cfg(any(feature = "onnx", feature = "onnx-cuda"))]
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
    let llm: Arc<autoagents::llm::backends::openai::OpenAI> =
        LLMBuilder::<autoagents::llm::backends::openai::OpenAI>::new()
            .api_key(api_key) // Set the API key
            .model("gpt-4o") // Use GPT-4o-mini model
            .max_tokens(512) // Limit response length
            .temperature(0.2) // Control response randomness (0.0-1.0)
            .build()
            .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Simple => simple::simple_agent(llm).await?,
        UseCase::Basic => basic::basic_agent(llm).await?,
        #[cfg(any(feature = "onnx", feature = "onnx-cuda"))]
        UseCase::Edge => onnx::edge_agent(args.device).await?,
        UseCase::Streaming => streaming::run(llm).await?,
        UseCase::Actor => actor::run(llm).await?,
        UseCase::Hooks => hooks::hooks_agent(llm).await?,
        UseCase::ManualToolAgent => manual_tool_agent::run_agent(llm, &args.mode).await?,
        UseCase::Toolkit => toolkit::run_agent(llm).await?,
        UseCase::Llm => llm::run_llm(llm).await?,
    }

    Ok(())
}
