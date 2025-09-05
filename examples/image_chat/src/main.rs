//! Image Chat Example
//!
//! This example demonstrates how to use AutoAgents with image messages.
//! It shows how to send images to LLMs that support vision capabilities.
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, DirectAgent};
use autoagents::core::tool::ToolT;
use autoagents::llm::{
    backends::openai::OpenAI,
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider, ImageMime},
};
use autoagents_derive::agent;
use clap::Parser;
use serde_json::Value;
use std::path::PathBuf;
use tokio::fs;

#[agent(name = "image_agent", description = "You are an Image analysis agent")]
#[derive(Default, Clone)]
pub struct ImageAgent {}

/// Image Chat Example - Analyze images using LLM vision capabilities
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the image file to analyze
    #[arg(short, long, help = "Path to the image file to analyze")]
    image: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Get image path from arguments or use default
    let image_path = args.image.unwrap_or_else(|| {
        println!("No image path provided, using default test image.");
        // Try relative path first (when running from examples/image_chat)
        let local_path = PathBuf::from("./test_img.jpg");
        if local_path.exists() {
            local_path
        } else {
            // Fall back to path when running from project root
            PathBuf::from("./examples/image_chat/test_img.jpg")
        }
    });

    println!("Reading image from: {}", image_path.display());

    // Read the image file
    let image_bytes = match fs::read(&image_path).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to read image file: {}", e));
        }
    };
    println!("Image size: {} bytes", image_bytes.len());

    // Create the OpenAI LLM client
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(&api_key)
        .model("gpt-4o")
        .build()?;

    // Create the message with image
    let message = ChatMessage::user()
        .content("What do you see in this image?")
        .image(ImageMime::JPEG, image_bytes.clone())
        .build();

    println!("\nSending image to OpenAI...");
    println!("Processing...\n");

    // Send the message and get response
    let response = llm.chat(&[message], None, None).await?;

    println!("Response:\n{}", response.text().unwrap_or_default());
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(ImageAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()?;

    println!("Running agent with image");
    let agent_result = agent
        .run(Task::new_with_image(
            "What do you see in this image?",
            ImageMime::JPEG,
            image_bytes,
        ))
        .await?;

    println!("Agent Response: {}", agent_result);

    Ok(())
}
