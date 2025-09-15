use crate::EdgeDevice;
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::Event;
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::core::utils::BoxEventStream;
use autoagents_derive::{agent, tool, AgentHooks, ToolInput};
use autoagents_onnx::chat::{LiquidEdgeBuilder, OnnxEdge};
use autoagents_onnx::cpu;
use autoagents_onnx::device::cuda_default;
use autoagents_onnx::runtime::onnx::onnx_model;
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use tokio_stream::StreamExt;

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
#[derive(Clone, AgentHooks)]
pub struct ChatAgent {}

pub async fn edge_agent(device: EdgeDevice) -> Result<(), Error> {
    println!("🚀 Onnx ORT Edge Local AI Example");

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

    println!("Using Device: {device}");

    // Initialize and configure the LLM client with device
    let llm: Arc<OnnxEdge> = LiquidEdgeBuilder::new()
        .model(Box::new(model))
        .device(device)
        .max_tokens(100) // Limit response length for faster testing
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .build()
        .await
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = ReActAgent::new(ChatAgent {});
    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for chat agent
    let chat_topic = Topic::<Task>::new("chat");

    let _ = AgentBuilder::new(agent)
        .llm(llm)
        .runtime(runtime.clone())
        .subscribe(chat_topic.clone())
        .memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Send chat message using the new messaging system
    println!("\n💬 Sending chat message to local AI...");
    let chat_task = Task::new("Hello! What is your name and how can you help me?");

    runtime.publish(&chat_topic, chat_task).await?;

    let _ = environment.run().await;

    Ok(())
}

fn handle_events(mut event_stream: BoxEventStream<Event>) {
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
                Event::TaskComplete { result, .. } => {
                    match serde_json::from_str::<ReActAgentOutput>(&result) {
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
