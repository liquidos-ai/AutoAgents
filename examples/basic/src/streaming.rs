use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
/// This example demonstrates Agent Streaming using the new runtime architecture
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT, RunnableAgent};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::SingleThreadedRuntime;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt as TokioStreamExt};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
struct Addition {}

impl ToolRuntime for Addition {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        println!("Tool Call Executed: {}", result);
        Ok(result.into())
    }
}

/// Streaming agent output
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct StreamingAgentOutput {
    #[output(description = "Response to the user query")]
    response: String,
}

#[agent(
    name = "streaming_agent",
    description = "You are a math expert and knowledgeable assistant that provides detailed explanations. Respond in a conversational manner.",
    tools = [Addition],
    output = StreamingAgentOutput
)]
#[derive(Clone)]
pub struct StreamingAgent {}

impl ReActExecutor for StreamingAgent {}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🌊 Agent Streaming Example");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = StreamingAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for streaming agent
    let streaming_topic = Topic::<Task>::new("streaming_agent");

    let agent_handler = AgentBuilder::new(agent)
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

    let agent = agent_handler.agent;
    let task = Task::new("What is 2 + 2?");

    // Process the stream directly
    let mut stream = agent.run_stream(task).await?;
    println!("🔄 Processing stream tokens...\n");

    // Print each stream token as it arrives
    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!("TSE: {:?}", output);
            }
            Err(e) => {
                println!("🔴 {}", format!("Stream error: {}", e).red());
            }
        }
    }

    println!("\n✅ Streaming example completed!");
    Ok(())
}

fn handle_streaming_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        let mut task_counter = 0;

        while let Some(event) = TokioStreamExt::next(&mut event_stream).await {
            match event {
                Event::TaskComplete { result, .. } => {
                    match result {
                        TaskResult::Value(val) => {
                            match serde_json::from_value::<ReActAgentOutput>(val) {
                                Ok(agent_out) => {
                                    // Try to parse as streaming output
                                    if let Ok(streaming_output) =
                                        serde_json::from_str::<StreamingAgentOutput>(
                                            &agent_out.response,
                                        )
                                    {
                                        println!(
                                            "{}",
                                            format!(
                                                "🌊 Streaming Response ({})",
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
                        TaskResult::Aborted => todo!(),
                    }
                }
                Event::StreamChunk { sub_id, chunk } => {
                    println!(
                        "{}",
                        format!("📦 Stream chunk ({}): {}", sub_id, chunk).cyan()
                    );
                }
                _ => {
                    // Handle other streaming-specific events if they exist
                }
            }
        }
    });
}
