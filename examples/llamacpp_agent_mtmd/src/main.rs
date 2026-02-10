#![allow(unused_imports)]
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::ToolCallError;
use autoagents::prelude::{ReActAgent, ReActAgentOutput};
use autoagents::prelude::{ToolInputT, ToolRuntime, ToolT};
use autoagents::{async_trait, init_logging};
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use autoagents_llamacpp::{LlamaCppProvider, ModelSource};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio_stream::StreamExt;

#[derive(Parser, Debug)]
#[command(version, about = "Run an AutoAgents math agent with llama.cpp", long_about = None)]
struct Args {
    #[arg(long, default_value = "What is 20 + 10?")]
    prompt: String,

    #[arg(long, help = "Optional chat template name or inline template")]
    chat_template: Option<String>,

    #[arg(long, help = "Context size override")]
    n_ctx: Option<u32>,

    #[arg(long, help = "Batch size override")]
    n_batch: Option<u32>,

    #[arg(long, help = "Thread count override")]
    n_threads: Option<i32>,

    #[arg(long, default_value_t = 256, help = "Max tokens to generate")]
    max_tokens: u32,

    #[arg(long, default_value_t = 0.2, help = "Sampling temperature")]
    temperature: f32,

    #[arg(long, help = "Path to an image for multimodal prompts")]
    image: PathBuf,

    #[arg(long, help = "Path to the MTMD mmproj file")]
    mmproj: Option<PathBuf>,

    #[arg(long, help = "MTMD media marker override")]
    media_marker: Option<String>,

    #[arg(
        long,
        default_value_t = true,
        help = "Enable GPU offload for MTMD projection"
    )]
    mmproj_use_gpu: bool,
}

#[agent(
    name = "mtmd_agent",
    description = "You are an helpful agent, answer user question and also explain them why you got the answer"
)]
#[derive(Default, Clone, AgentHooks)]
struct MathAgent {}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), Error> {
    init_logging();
    let args = Args::parse();

    let mut builder = LlamaCppProvider::builder()
        .model_source(ModelSource::HuggingFace {
            repo_id: "Qwen/Qwen3-VL-2B-Instruct-GGUF".to_string(),
            filename: Some("Qwen3VL-2B-Instruct-Q8_0.gguf".to_string()),
            mmproj_filename: Some("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf".to_string()),
        })
        .max_tokens(args.max_tokens)
        .temperature(args.temperature);

    if let Some(template) = args.chat_template {
        builder = builder.chat_template(template);
    }
    if let Some(n_ctx) = args.n_ctx {
        builder = builder.n_ctx(n_ctx);
    }
    if let Some(n_batch) = args.n_batch {
        builder = builder.n_batch(n_batch);
    }
    if let Some(n_threads) = args.n_threads {
        builder = builder.n_threads(n_threads);
    }
    if let Some(mmproj) = &args.mmproj {
        builder = builder.mmproj_path(mmproj.to_string_lossy().to_string());
    }
    if let Some(marker) = &args.media_marker {
        builder = builder.media_marker(marker.clone());
    }
    builder = builder.mmproj_use_gpu(args.mmproj_use_gpu);

    let llm = builder
        .build()
        .await
        .expect("Failed to build llama.cpp provider");
    let llm: Arc<dyn autoagents::llm::LLMProvider> = Arc::new(llm);

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = ReActAgent::new(MathAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let image_bytes = fs::read(&args.image)
        .await
        .map_err(|err| Error::CustomError(err.to_string()))?;

    let result = agent_handle
        .agent
        .run(Task::new_with_image(
            args.prompt.clone(),
            autoagents::protocol::ImageMime::JPEG,
            image_bytes.clone(),
        ))
        .await?;

    println!("Result: {:?}", result);

    // Process the stream directly
    {
        let mut stream = agent_handle
            .agent
            .run_stream(Task::new_with_image(
                args.prompt.clone(),
                autoagents::protocol::ImageMime::JPEG,
                image_bytes,
            ))
            .await?;

        println!("🌊 Agent Streaming Example");
        println!("🔄 Processing stream tokens...\n");

        while let Some(result) = stream.next().await {
            if let Ok(output) = result {
                print!("{}", output);
            }
        }
    };

    Ok(())
}
