#![recursion_limit = "256"]

use autoagents::core::error::Error;
use autoagents::init_logging;
use clap::{Parser, Subcommand};

mod commands;
use crate::commands::run_frombytes;
use commands::{run_fromfile, run_pretrained};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "autoagents-burn-example")]
#[command(about = "AutoAgents Burn example with TinyLlama model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run with local model files
    FromFile {
        /// Model type to use
        #[arg(long, default_value = "tiny")]
        model: Model,

        /// Path to the model file
        #[arg(
            short,
            long,
            default_value = "./examples/burn/model/TinyLlama-1.1B/model.mpk"
        )]
        model_path: String,

        /// Path to the tokenizer file
        #[arg(
            short,
            long,
            default_value = "./examples/burn/model/TinyLlama-1.1B/tokenizer.json"
        )]
        tokenizer_path: String,

        /// Prompt to send to the agent
        #[arg(short, long, default_value = "Tell me a poem?")]
        prompt: String,
    },
    /// Run with local model files with Bytes
    FromBytes {
        /// Model type to use
        #[arg(long, default_value = "llama3")]
        model: Model,

        /// Path to the model file
        #[arg(
            short,
            long,
            default_value = "./examples/burn/model/Llama-3.2-1B-Instruct/model.mpk"
        )]
        model_path: String,

        /// Path to the tokenizer file
        #[arg(
            short,
            long,
            default_value = "./examples/burn/model/Llama-3.2-1B-Instruct/tokenizer.model"
        )]
        tokenizer_path: String,

        /// Prompt to send to the agent
        #[arg(short, long, default_value = "Tell me a poem?")]
        prompt: String,
    },
    /// Run with pretrained model (downloads automatically)
    Pretrained {
        /// Model type to use
        #[arg(long, default_value = "tiny")]
        model: Model,

        /// Prompt to send to the agent
        #[arg(short, long, default_value = "Tell me a poem?")]
        prompt: String,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Model {
    /// TinyLlama-1.1B model
    Tiny,
    /// Llama3-2-3B model
    Llama3,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    init_logging();

    let cli = Cli::parse();

    match cli.command {
        Commands::FromFile {
            model,
            model_path,
            tokenizer_path,
            prompt,
        } => {
            run_fromfile(model, model_path, tokenizer_path, prompt).await?;
        }
        Commands::FromBytes {
            model,
            model_path,
            tokenizer_path,
            prompt,
        } => {
            run_frombytes(model, model_path, tokenizer_path, prompt).await?;
        }
        Commands::Pretrained { model, prompt } => {
            run_pretrained(model, prompt).await?;
        }
    }

    Ok(())
}
