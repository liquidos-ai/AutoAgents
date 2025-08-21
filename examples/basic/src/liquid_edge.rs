use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::backends::liquid_edge::LiquidEdge;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{agent, tool, ToolInput};
use colored::*;
use liquid_edge::cpu;
use liquid_edge::device::cuda_default;
use liquid_edge::runtime::onnx::onnx_model;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::EdgeDevice;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[allow(dead_code)]
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
        Ok(result.into())
    }
}

#[agent(
    name = "chat_agent",
    description = "You are ChatBOT, a helpful, friendly, and knowledgeable AI assistant. Your name is ChatBOT.",
    tools = []
)]
pub struct ChatAgent {}

impl ReActExecutor for ChatAgent {}

pub async fn edge_agent(device: EdgeDevice) -> Result<(), Error> {
    println!("🚀 Liquid Edge Local AI Example");

    // Create ONNX model abstraction
    let model_path = Path::new("./demo_models/tinyllama");
    let model = match onnx_model(model_path) {
        Ok(model) => model,
        Err(e) => {
            return Err(Error::LLMError(
                autoagents::llm::error::LLMError::ProviderError(format!(
                    "Model loading failed: {}. Make sure the model is available at ./models/tinyllama",
                    e
                )),
            ));
        }
    };

    // Use CUDA device (will fallback to CPU if CUDA is not available)
    let device = match device {
        EdgeDevice::CPU => cpu(),
        EdgeDevice::CUDA => cuda_default(),
    };
    println!("🔧 Using device: {}", device);

    // Initialize and configure the LLM client with device
    let llm: Arc<LiquidEdge> = LLMBuilder::<LiquidEdge>::new()
        .with_model(model)
        .with_device(device)
        .max_tokens(100) // Limit response length for faster testing
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .await
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = ChatAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for chat agent
    let chat_topic = Topic::<Task>::new("chat");

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(chat_topic.clone())
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    tokio::spawn(async move { environment.run().await });

    // Send chat message using the new messaging system
    println!("\n💬 Sending chat message to local AI...");
    let chat_task = Task::new("Hello! What is your name and how can you help me?");

    runtime.publish(&chat_topic, chat_task).await?;

    println!("\n✅ Liquid Edge example completed!");
    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    actor_id,
                    task_description,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "🎯 Task Started - Agent: {:?}, Task: {}",
                            actor_id, task_description
                        )
                        .cyan()
                    );
                }
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "🔧 Tool Call Started: {} with args: {}",
                            tool_name, arguments
                        )
                        .yellow()
                    );
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "✅ Tool Call Completed: {} - Result: {:?}",
                            tool_name, result
                        )
                        .yellow()
                    );
                }
                Event::TaskComplete { result, .. } => match result {
                    TaskResult::Value(val) => {
                        match serde_json::from_value::<ReActAgentOutput>(val) {
                            Ok(agent_out) => {
                                println!(
                                    "{}",
                                    format!("🤖 Local AI Response: {}", agent_out.response).green()
                                );
                            }
                            Err(e) => {
                                println!(
                                    "{}",
                                    format!("❌ Failed to parse agent output: {}", e).red()
                                );
                            }
                        }
                    }
                    TaskResult::Failure(e) => {
                        println!("{}", format!("❌ Task failed: {}", e).red());
                    }
                    _ => {
                        println!("🚫 Task aborted");
                    }
                },
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("🔄 Turn {}/{} started", turn_number + 1, max_turns).blue()
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
                        .blue()
                    );
                }
                _ => {
                    // Handle other events if needed
                }
            }
        }
    });
}
