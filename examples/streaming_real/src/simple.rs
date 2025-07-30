use autoagents::core::agent::prebuilt::react::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::core::tool::ToolT;
use autoagents::llm::{LLMProvider, builder::LLMBuilder, backends::openai::OpenAI};
use autoagents_derive::{agent, AgentOutput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

/// Simple agent output for streaming demonstration
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct SimpleAgentOutput {
    #[output(description = "The response to the question")]
    response: String,
}

#[agent(
    name = "simple_agent",
    description = "You are a helpful assistant that explains technical concepts clearly and thoroughly.",
    tools = [],
    output = SimpleAgentOutput
)]
pub struct SimpleAgent {}

impl ReActExecutor for SimpleAgent {}

pub async fn simple_streaming_demo() -> Result<(), Error> {
    
    
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| Error::LLMError(autoagents_llm::error::LLMError::InvalidRequest("OPENAI_API_KEY environment variable is required".to_string())))?;
    
       println!(" Starting...");
    
    // Create real OpenAI provider with streaming enabled
    let llm: Arc<dyn LLMProvider> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key.clone())
        .model("gpt-4o-mini")
        .stream(true)
        .temperature(0.7)
        .max_tokens(1000)
        .system("You are a helpful assistant that explains technical concepts clearly and thoroughly. Provide detailed, well-structured responses.")
        .build()?;

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
    let agent = SimpleAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("simple_demo")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_simple_streaming_events(receiver);

    // Send the question about Rust agents
    println!("\n📝 Sending question about Rust agents...");
    
    runtime
        .publish_message("What are the benefits of agents written in Rust? Please provide a comprehensive explanation.".into(), "simple_demo".into())
        .await?;

    let _ = environment.run().await;
    Ok(())
}

fn handle_simple_streaming_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    agent_id,
                    task_description,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "📋 Task Started - Agent: {:?}, Task: {}",
                            agent_id, task_description
                        )
                        .green()
                    );
                }
                
                // Streaming events - this is where you'll see the real streaming
                Event::StreamTextChunk { chunk, is_final, .. } => {
                    print!("{}", chunk);
                    if is_final {
                        println!();
                    }
                }
                
                Event::StreamThinkingChunk { chunk, is_final, .. } => {
                    print!("{}", format!("🧠 {}", chunk).blue());
                    if is_final {
                        println!();
                    }
                }
                
                Event::TaskComplete { result, .. } => match result {
                    TaskResult::Value(val) => {
                        let val_clone = val.clone();
                        match serde_json::from_value::<ReActAgentOutput>(val) {
                            Ok(agent_out) => {
                                match serde_json::from_str::<SimpleAgentOutput>(&agent_out.response) {
                                    Ok(simple_out) => {
                                        println!(
                                            "\n{}",
                                            format!("🎉 Task Complete - Response: {}", simple_out.response)
                                                .green()
                                        );
                                    }
                                    Err(_) => {
                                        println!(
                                            "\n{}",
                                            format!("🎉 Task Complete - Response: {}", agent_out.response)
                                                .green()
                                        );
                                    }
                                }
                            }
                            Err(_) => {
                                println!(
                                    "\n{}",
                                    format!("🎉 Task Complete - Raw Result: {:?}", val_clone).green()
                                );
                            }
                        }
                    }
                    TaskResult::Failure(error) => {
                        println!("{}", format!("❌ Task Failed: {}", error).red());
                    }
                    TaskResult::Aborted => {
                        println!("{}", format!("⏹️ Task Aborted").yellow());
                    }
                },
                
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("🔄 Turn {}/{} started", turn_number + 1, max_turns).cyan()
                    );
                }
                
                Event::TurnCompleted {
                    turn_number,
                    final_turn,
                } => {
                    println!(
                        "{}",
                        format!(
                            "✅ Turn {} completed{}",
                            turn_number + 1,
                            if final_turn { " (final)" } else { "" }
                        )
                        .cyan()
                    );
                }
                
                _ => {
                    // Ignore other events for cleaner output
                }
            }
        }
    });
} 