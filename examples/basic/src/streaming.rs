use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
/// This example demonstrates Agent Streaming using the new runtime architecture
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::ToolT;
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, AgentOutput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

/// Streaming agent output
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct StreamingAgentOutput {
    #[output(description = "The streamed response")]
    response: String,
    #[output(description = "Response timestamp")]
    timestamp: String,
}

#[agent(
    name = "streaming_agent",
    description = "You are a math expert and knowledgeable assistant that provides detailed explanations. Respond in a conversational manner.",
    tools = [],
    output = StreamingAgentOutput
)]
pub struct StreamingAgent {}

impl ReActExecutor for StreamingAgent {}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🌊 Agent Streaming Example");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = StreamingAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for streaming agent
    let streaming_topic = Topic::<Task>::new("streaming_agent");

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(streaming_topic.clone())
        .stream(true) // Enable streaming for this agent
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    let receiver = environment.take_event_receiver(None).await?;
    handle_streaming_events(receiver);

    // Start the environment
    let _handle = environment.run();

    // Send multiple messages to demonstrate streaming
    println!("\n📤 Sending streaming tasks...");

    let tasks = vec![
        "What is Vector Calculus and why is it important in mathematics?",
        "Explain the concept of derivatives in simple terms",
        "How do integrals relate to the area under a curve?",
        "What are some real-world applications of linear algebra?",
    ];

    for (i, task_content) in tasks.iter().enumerate() {
        println!("\n💬 Sending task {}: {}", i + 1, task_content);

        let task = Task::new(*task_content);

        // Publish to topic
        runtime.publish(&streaming_topic, task).await?;

        // Give some time between tasks to see streaming effect
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    // Give time for processing
    tokio::time::sleep(tokio::time::Duration::from_secs(20)).await;

    println!("\n✅ Streaming example completed!");
    Ok(())
}

fn handle_streaming_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        let mut task_counter = 0;

        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    actor_id,
                    task_description,
                    ..
                } => {
                    task_counter += 1;
                    println!(
                        "{}",
                        format!(
                            "🎯 Task {} Started - Agent: {:?}\n   📝 Task: {}",
                            task_counter, actor_id, task_description
                        )
                            .cyan()
                    );
                }
                Event::TaskComplete { result, .. } => {
                    match result {
                        TaskResult::Value(val) => {
                            match serde_json::from_value::<ReActAgentOutput>(val) {
                                Ok(agent_out) => {
                                    // Try to parse as streaming output
                                    if let Ok(streaming_output) = serde_json::from_str::<StreamingAgentOutput>(&agent_out.response) {
                                        println!(
                                            "{}",
                                            format!(
                                                "🌊 Streaming Response ({}): {}",
                                                streaming_output.timestamp,
                                                streaming_output.response
                                            )
                                                .green()
                                        );
                                    } else {
                                        // Fallback to regular output
                                        println!(
                                            "{}",
                                            format!("💭 Response: {}", agent_out.response).green()
                                        );
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "{}",
                                        format!("❌ Failed to parse response: {}", e).red()
                                    );
                                }
                            }
                        }
                        TaskResult::Failure(error) => {
                            println!("{}", format!("❌ Task failed: {}", error).red());
                        }
                        TaskResult::Aborted => todo!()
                    }
                }
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!("🔧 Tool Call: {} with args: {}", tool_name, arguments)
                            .yellow()
                    );
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!("✅ Tool Completed: {} - Result: {:?}", tool_name, result)
                            .yellow()
                    );
                }
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("🔄 Turn {}/{} started", turn_number + 1, max_turns).magenta()
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
                            .magenta()
                    );
                }
                _ => {
                    // Handle other streaming-specific events if they exist
                }
            }
        }
    });
}