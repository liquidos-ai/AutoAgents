//! Example: Using MistralRsProvider with AutoAgents - CLI Edition
//!
//! This example demonstrates all ways to use the local LLM backend with mistral.rs:
//! 1. Text models from HuggingFace
//! 2. Vision models from HuggingFace
//! 3. GGUF quantized models from local files
//! 4. Tool calling with capable models
//!
//! Run with:
//! ```bash
//! # Text model (default)
//! cargo run --package mistral_rs --release -- --model-type text
//!
//! # Vision model
//! cargo run --package mistral_rs --release -- --model-type vision
//!
//! # GGUF model (requires local files)
//! cargo run --package mistral_rs --release -- --model-type gguf --model-dir models/phi-3.5
//!
//! # Tool calling model
//! cargo run --package mistral_rs --release -- --model-type tools
//! ```

mod gguf;
mod text;
mod tool;
mod vision;

use autoagents::core::error::Error;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
enum ModelTypeArg {
    /// Text model from HuggingFace (Phi-3.5-mini-instruct)
    Text,
    /// Vision model from HuggingFace (SmolVLM)
    Vision,
    /// GGUF quantized model from local files
    Gguf,
    /// Tool calling model (Mistral-7B-Instruct)
    Tools,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Type of model to load
    #[arg(short = 't', long, value_enum, default_value = "text")]
    model_type: ModelTypeArg,

    /// Model directory for GGUF models
    #[arg(
        short = 'd',
        long,
        default_value = "examples/mistral_rs/models/phi-3.5"
    )]
    model_dir: String,

    /// HuggingFace repo ID (overrides defaults)
    #[arg(short = 'r', long)]
    repo_id: Option<String>,

    /// GGUF quantization level
    #[arg(short = 'q', long, default_value = "q4-k-m")]
    quant: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "1024")]
    max_tokens: u32,

    /// Sampling temperature (0.0 - 2.0)
    #[arg(long, default_value = "0.2")]
    temperature: f32,

    /// Enable paged attention (not compatible with GGUF + CUDA)
    #[arg(long, default_value = "false")]
    paged_attention: bool,

    /// Enable detailed mistral.rs logging
    #[arg(long, default_value = "false")]
    verbose: bool,
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    // Initialize logging
    autoagents::init_logging();

    // Parse command-line arguments
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        MistralRs Local LLM Backend - CLI Example             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Run example based on model type
    match args.model_type {
        ModelTypeArg::Text => {
            println!("Loading TEXT model from HuggingFace...");
            let text_args = text::TextArgs {
                repo_id: args.repo_id,
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                paged_attention: args.paged_attention,
                verbose: args.verbose,
            };
            let llm = text::load_model(&text_args).await?;
            println!("✅ Model loaded successfully!\n");
            text::run_example(llm).await?;
        }
        ModelTypeArg::Vision => {
            println!("Loading VISION model from HuggingFace...");
            let vision_args = vision::VisionArgs {
                repo_id: args.repo_id,
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                verbose: args.verbose,
            };
            let llm = vision::load_model(&vision_args).await?;
            println!("Model loaded successfully!\n");
            vision::run_example(llm).await?;
        }
        ModelTypeArg::Gguf => {
            println!("Loading GGUF quantized model from local files...");
            let gguf_args = gguf::GgufArgs {
                model_dir: args.model_dir,
                quant: args.quant,
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                paged_attention: args.paged_attention,
                verbose: args.verbose,
            };
            let llm = gguf::load_model(&gguf_args).await?;
            println!("Model loaded successfully!\n");
            gguf::run_example(llm).await?;
        }
        ModelTypeArg::Tools => {
            println!("Loading TOOL-CALLING model from HuggingFace...");
            let tool_args = tool::ToolArgs {
                repo_id: args.repo_id,
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                paged_attention: args.paged_attention,
                verbose: args.verbose,
            };
            let llm = tool::load_model(&tool_args).await?;
            println!("Model loaded successfully!\n");
            tool::run_example(llm).await?;
        }
    }

    println!("\nExample completed successfully!");
    Ok(())
}
