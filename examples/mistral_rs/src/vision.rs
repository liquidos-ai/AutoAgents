//! Vision model example

use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::BasicAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents_mistral_rs::models::ModelType;
use autoagents_mistral_rs::{MistralRsProvider, ModelSource};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;

pub use crate::text::DemoAgent;

pub struct VisionArgs {
    pub repo_id: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub verbose: bool,
}

/// Load a vision model from HuggingFace
pub async fn load_model(args: &VisionArgs) -> Result<Arc<MistralRsProvider>, Error> {
    let repo_id = args
        .repo_id
        .clone()
        .unwrap_or_else(|| "HuggingFaceTB/SmolVLM-Instruct".to_string());

    println!("   Repository: {}", repo_id);
    println!("   Model type: Vision (VisionModelBuilder)");
    println!("   Max tokens: {}", args.max_tokens);
    println!("   Temperature: {}", args.temperature);
    println!("   Verbose logging: {}\n", args.verbose);

    let mut builder = MistralRsProvider::builder()
        .model_source(ModelSource::HuggingFace {
            repo_id,
            revision: None,
            model_type: ModelType::Vision,
        })
        .max_tokens(args.max_tokens)
        .temperature(args.temperature);

    if args.verbose {
        builder = builder.with_logging();
    }

    let provider = builder
        .build()
        .await
        .map_err(|e| Error::CustomError(e.to_string()))?;

    Ok(Arc::new(provider))
}

/// Run queries suitable for vision models
pub async fn run_example(llm: Arc<MistralRsProvider>) -> Result<(), Error> {
    println!("Running Vision Queries ...");

    let image_path = {
        let local_path = PathBuf::from("./test_img.jpg");
        if local_path.exists() {
            local_path
        } else {
            // Fall back to path when running from project root
            PathBuf::from("./examples/mistral_rs/test_img.jpg")
        }
    };

    let image_bytes = match fs::read(&image_path).await {
        Ok(bytes) => {
            println!("Loaded image: {} bytes\n", bytes.len());
            bytes
        }
        Err(e) => {
            return Err(Error::CustomError(format!("Failed to load image: {}", e)));
        }
    };

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let agent = BasicAgent::new(DemoAgent {});
    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new_with_image(
            "Describe the image?",
            autoagents::protocol::ImageMime::JPEG,
            image_bytes,
        ))
        .await?;

    println!("Response: {:?}\n", result);

    Ok(())
}
