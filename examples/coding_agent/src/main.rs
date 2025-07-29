use clap::{Parser, ValueEnum};
use std::sync::Arc;
mod agent;
mod interactive;
mod tools;
use autoagents::{core::error::Error, llm::backends::openai::OpenAI, llm::builder::LLMBuilder};

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Interactive,
}

/// Coding agent that demonstrates file manipulation capabilities
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "usecase to run")]
    usecase: UseCase,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    // Check if API key is set
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .max_tokens(2048)
        .temperature(0.1) // Lower temperature for more consistent code generation
        .stream(false)
        .build()
        .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Interactive => interactive::run_interactive_session(llm).await?,
    }

    Ok(())
}
