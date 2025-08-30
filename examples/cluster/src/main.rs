mod agents;

use autoagents::llm::{backends::openai::OpenAI, builder::LLMBuilder};
use clap::{Parser, Subcommand};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run ResearchAgent in a cluster node (gathers information)
    Research {
        /// Port for this node
        #[arg(short = 'p', long, default_value = "9001")]
        port: u16,
        /// Remote node address to connect to (e.g., localhost:9002)
        #[arg(short = 'r', long)]
        remote: Option<String>,
        /// Node name
        #[arg(short = 'n', long, default_value = "research")]
        name: String,

        #[arg(long, default_value = "localhost")]
        host: String,
    },
    /// Run AnalysisAgent in a cluster node (analyzes research data)
    Analysis {
        /// Port for this node
        #[arg(short = 'p', long, default_value = "9002")]
        port: u16,
        /// Remote node address to connect to (e.g., localhost:9001)
        #[arg(short = 'r', long)]
        remote: Option<String>,
        /// Node name
        #[arg(short = 'n', long, default_value = "analysis")]
        name: String,

        #[arg(long, default_value = "localhost")]
        host: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Create LLM provider
    let llm = create_llm_provider()?;

    match args.command {
        Commands::Research {
            port,
            remote,
            name,
            host,
        } => {
            println!(
                "🔍 Starting ResearchAgent on port {} with name {}",
                port, name
            );
            agents::run_research_agent(llm, name, port, remote, host).await?;
        }
        Commands::Analysis {
            port,
            remote,
            name,
            host,
        } => {
            println!(
                "🧠 Starting AnalysisAgent on port {} with name {}",
                port, name
            );
            agents::run_analysis_agent(llm, name, port, remote, host).await?;
        }
    }
    Ok(())
}

fn create_llm_provider() -> Result<Arc<OpenAI>, Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o-mini")
        .max_tokens(512)
        .temperature(0.2)
        .build()
        .expect("Failed to build LLM");

    Ok(llm)
}
