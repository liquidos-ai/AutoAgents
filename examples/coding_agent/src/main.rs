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

/// Coding agent that demonstrates file manipulation capabilities
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "usecase to run")]
    usecase: UseCase,
    #[arg(
        long,
        default_value = ".",
        help = "Workspace root for all file operations"
    )]
    workspace_root: PathBuf,
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    let workspace_root = args.workspace_root.canonicalize().map_err(|e| {
        Error::CustomError(format!(
            "Failed to resolve workspace root {}: {}",
            args.workspace_root.display(),
            e
        ))
    })?;

    if !workspace_root.is_dir() {
        return Err(Error::CustomError(format!(
            "Workspace root must be a directory: {}",
            workspace_root.display()
        )));
    }

    let workspace_root = workspace_root.to_string_lossy().to_string();

    // Check if API key is set
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Initialize and configure the LLM client
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model("gpt-4o")
        .max_tokens(2048)
        .temperature(0.1) // Lower temperature for more consistent code generation
        .build()
        .expect("Failed to build LLM");

    match args.usecase {
        UseCase::Interactive => interactive::run_interactive_session(llm, workspace_root).await?,
    }

    Ok(())
}
