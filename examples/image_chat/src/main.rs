//! Image Chat Example
//!
//! This example demonstrates how to use AutoAgents with image messages.
//! It shows how to send images to LLMs that support vision capabilities.
use anyhow::Result;
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT};
use autoagents::core::environment::Environment;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::ToolT;

use autoagents::llm::{
    backends::openai::OpenAI,
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider, ImageMime},
};
use autoagents_derive::agent;
use serde_json::Value;

use tokio::fs;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

#[agent(name = "image_agent", description = "You are an Image analysis agent")]
#[derive(Default, Clone)]
pub struct ImageAgent {}

impl ReActExecutor for ImageAgent {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Read the test image
    let image_path = "./examples/image_chat/test_img.jpg";
    println!("Reading image from: {}", image_path);

    let image_bytes = fs::read(image_path).await?;
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

    let agent = ImageAgent {};

    let runtime = SingleThreadedRuntime::new(None);

    let test_topic = Topic::<Task>::new("test");

    let agent_handle = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(test_topic.clone())
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    let _addr = agent_handle.addr();

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    let _ = handle_events(receiver)?;

    // Publish message with image data to all the subscribing actors
    runtime
        .publish(
            &Topic::<Task>::new("test"),
            Task::new_with_image(
                "What do you see in this image?",
                ImageMime::JPEG,
                image_bytes,
            ),
        )
        .await?;

    let _ = environment.run().await;
    Ok(())
}

fn handle_events(event_stream: ReceiverStream<Event>) -> Result<()> {
    let mut event_stream = event_stream;
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    actor_id,
                    task_description,
                    ..
                } => {
                    println!(
                        "🎯 Task Started - Actor: {:?}, Task: {}",
                        actor_id, task_description
                    );
                }
                Event::TaskComplete { result, .. } => match result {
                    TaskResult::Value(val) => {
                        let react_output: ReActAgentOutput = serde_json::from_value(val).unwrap();
                        println!("✅ Task Complete - Result: {}", react_output.response);
                    }
                    TaskResult::Failure(err) => {
                        println!("❌ Task Error: {}", err);
                    }
                    TaskResult::Aborted => {
                        println!("⚠️ Task Aborted");
                    }
                },
                _ => {
                    println!("📋 Event: {:?}", event);
                }
            }
        }
    });
    Ok(())
}
