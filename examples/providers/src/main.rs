//! This example demonstrates different Backends

mod anthropic;
mod groq;
mod minimax;
mod ollama;
mod openai;
mod openrouter;

use autoagents::init_logging;
use clap::{Parser, ValueEnum};

#[derive(Debug, ValueEnum, Clone)]
pub enum Backend {
    OpenAI,
    OpenRouter,
    Anthropic,
    Ollama,
    Groq,
    MiniMax,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "LLM Backend")]
    backend: Backend,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();
    // Parse command line arguments
    let args = Args::parse();

    println!("Running example with backend: {:?}", args.backend);

    match args.backend {
        Backend::OpenAI => {
            println!("Using OpenAI backend (requires OPENAI_API_KEY)");
            openai::run().await?;
        }
        Backend::OpenRouter => {
            println!("Using OpenRouter backend (requires OPENROUTER_API_KEY)");
            openrouter::run().await?;
        }
        Backend::Anthropic => {
            println!("Using Anthropic backend (requires ANTHROPIC_API_KEY)");
            anthropic::run().await?;
        }
        Backend::Ollama => {
            println!("Using Ollama backend (requires local Ollama server)");
            ollama::run().await?;
        }
        Backend::Groq => {
            println!("Using Groq backend (requires GROQ_API_KEY)");
            groq::run().await?;
        }
        Backend::MiniMax => {
            println!("Using MiniMax backend (requires MINIMAX_API_KEY)");
            minimax::run().await?;
        }
    }

    Ok(())
}
