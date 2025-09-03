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
    /// Run the cluster host runtime (coordinates all client agents)
    Host {
        /// Port for the host node
        #[arg(short = 'p', long, default_value = "9000")]
        port: u16,
        /// Node name
        #[arg(short = 'n', long, default_value = "cluster-host")]
        name: String,
        /// Host address
        #[arg(long, default_value = "localhost")]
        host: String,
    },
    /// Run ResearchAgent as a cluster client (gathers information)
    Research {
        /// Port for this node
        #[arg(short = 'p', long, default_value = "9001")]
        port: u16,
        /// Host address to connect to (e.g., localhost:9000)
        #[arg(short = 'r', long, default_value = "localhost:9000")]
        host_addr: String,
        /// Node name
        #[arg(short = 'n', long, default_value = "research")]
        name: String,
        /// Local host address
        #[arg(long, default_value = "localhost")]
        host: String,
    },
    /// Run AnalysisAgent as a cluster client (analyzes research data)
    Analysis {
        /// Port for this node
        #[arg(short = 'p', long, default_value = "9002")]
        port: u16,
        /// Host address to connect to (e.g., localhost:9000)
        #[arg(short = 'r', long, default_value = "localhost:9000")]
        host_addr: String,
        /// Node name
        #[arg(short = 'n', long, default_value = "analysis")]
        name: String,
        /// Local host address
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
        Commands::Host { port, name, host } => {
            println!(
                "🏠 Starting ClusterHost on port {} with name {}",
                port, name
            );
            agents::run_cluster_host(name, port, host).await?;
        }
        Commands::Research {
            port,
            host_addr,
            name,
            host,
        } => {
            println!(
                "🔍 Starting ResearchAgent on port {} with name {}",
                port, name
            );
            agents::run_research_agent(llm, name, port, host_addr, host).await?;
        }
        Commands::Analysis {
            port,
            host_addr,
            name,
            host,
        } => {
            println!(
                "🧠 Starting AnalysisAgent on port {} with name {}",
                port, name
            );
            agents::run_analysis_agent(llm, name, port, host_addr, host).await?;
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
