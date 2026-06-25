use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;

mod agent;
mod interactive;
mod tools;

use autoagents::{core::error::Error, llm::backends::openai::OpenAI, llm::builder::LLMBuilder};

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Interactive,
}

/// Coding agent that demonstrates sandboxed file manipulation capabilities
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "usecase to run")]
    usecase: UseCase,
    #[arg(
        long,
        value_name = "PATH",
        help = "Workspace root for sandboxed file tools (defaults to the current directory)"
    )]
    workspace: Option<PathBuf>,
}

fn workspace_path(path: Option<PathBuf>) -> PathBuf {
    path.unwrap_or_else(|| {
        std::env::current_dir().expect("failed to read current working directory")
    })
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let workspace = workspace_path(args.workspace);

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .max_tokens(2048)
        .temperature(0.1)
        .build()
        .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Interactive => {
            interactive::run_interactive_session(llm, workspace).await?;
        }
    }

    Ok(())
}
