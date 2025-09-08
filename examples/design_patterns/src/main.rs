//! Design Patterns Example
//!
//! This example demonstrates different Agentic AI Design Patterns

mod chaining;
mod parallel;
mod planning;
mod reflection;
mod routing;

use autoagents::init_logging;
use autoagents::llm::{backends::openai::OpenAI, builder::LLMBuilder};
use clap::{Parser, ValueEnum};

#[derive(Debug, ValueEnum, Clone)]
pub enum DesignPattern {
    Chaining,
    Routing,
    Parallel,
    Reflection,
    Planning,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "design pattern")]
    design_pattern: DesignPattern,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();
    // Parse command line arguments
    let args = Args::parse();

    // Create the OpenAI LLM client
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(&api_key)
        .model("gpt-4")
        .temperature(0.2)
        .build()?;

    match args.design_pattern {
        DesignPattern::Chaining => chaining::run(llm).await?,
        DesignPattern::Routing => routing::run(llm).await?,
        DesignPattern::Parallel => parallel::run(llm).await?,
        DesignPattern::Reflection => reflection::run(llm).await?,
        DesignPattern::Planning => planning::run(llm).await?,
    }

    Ok(())
}
