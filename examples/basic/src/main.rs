use clap::{Parser, ValueEnum};
mod chat;
mod simple;
use autoagents::{core::error::Error, llm::backends::openai::OpenAI, llm::builder::LLMBuilder};

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Simple,
}

/// Simple program to demonstrate AutoAgents functionality
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "usecase")]
    usecase: UseCase,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    // Check if API key is set
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(api_key) // Set the API key
        .model("gpt-4o") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Simple => simple::simple_agent(llm).await?,
    }

    Ok(())
}
