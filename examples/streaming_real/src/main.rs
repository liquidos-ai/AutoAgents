use clap::{Parser, ValueEnum};
mod simple;
mod streaming;
use autoagents::{
    core::error::Error,
    init_logging,
};

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Simple,
    Streaming,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "usecase")]
    usecase: UseCase,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    init_logging();
    let args = Args::parse();
    
    
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("".into());
    if api_key.is_empty() {
        eprintln!("❌ OPENAI_API_KEY environment variable is required");
        eprintln!("Please set it with: export OPENAI_API_KEY='your-api-key-here'");
        std::process::exit(1);
    }

    match args.usecase {
        UseCase::Simple => {
            println!(" Running Simple agent");
            simple::simple_streaming_demo().await?;
        }
        UseCase::Streaming => {
            println!(" Running agents with Tools");
            streaming::streaming_agent_with_real_llm().await?;
        }
    }

    Ok(())
} 