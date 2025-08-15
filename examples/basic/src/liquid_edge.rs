use autoagents::core::agent::prebuilt::react::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::backends::liquid_edge::LiquidEdge;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
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

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
}

#[agent(
    name = "math_agent",
    description = "You are ChatBOT, a helpful, friendly, and knowledgeable AI assistant. Your name is ChatBOT."
    tools = []
)]
pub struct MathAgent {}

impl ReActExecutor for MathAgent {}

pub async fn edge_agent(device: EdgeDevice) -> Result<(), Error> {
    // Create ONNX model abstraction
    let model_path = Path::new("./demo_models/tinyllama");
    let model = match onnx_model(model_path) {
        Ok(model) => model,
        Err(e) => {
            return Err(Error::LLMError(
                autoagents::llm::error::LLMError::ProviderError(format!(
                    "Model loading failed: {}",
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
    println!("Using device: {}", device);

    // Initialize and configure the LLM client with device
    let llm: Arc<LiquidEdge> = LLMBuilder::<LiquidEdge>::new()
        .with_model(model)
        .with_device(device)
        .max_tokens(50) // Limit response length for faster testing
        .temperature(0.2) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .await
        .expect("Failed to build LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let runtime = SingleThreadedRuntime::new(None);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("test")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    runtime
        .publish_message("Heyyy, What is your name?".into(), "test".into())
        .await
        .unwrap();

    let _ = environment.run().await;
    Ok(())
}

fn handle_events(mut event_stream: ReceiverStream<Event>) {
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
                Event::ToolCallRequested {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!("Tool Call Started: {} with args: {}", tool_name, arguments)
                            .green()
                    );
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!("Tool Call Completed: {} - Result: {:?}", tool_name, result)
                            .green()
                    );
                }
                Event::TaskComplete { result, .. } => match result {
                    TaskResult::Value(val) => {
                        match serde_json::from_value::<ReActAgentOutput>(val) {
                            Ok(agent_out) => {
                                println!("Agent response: {}", agent_out.response);
                            }
                            Err(e) => {
                                println!(
                                    "{}",
                                    format!("Failed to parse ReActAgentOutput: {}", e).red()
                                );
                            }
                        }
                    }
                    TaskResult::Failure(e) => {
                        println!("{}", format!("Error!!! {e}").red());
                    }
                    _ => {
                        println!("Aborted")
                    }
                },
                Event::TurnStarted {
                    turn_number,
                    max_turns,
                } => {
                    println!(
                        "{}",
                        format!("Turn {}/{} started", turn_number + 1, max_turns).green()
                    );
                }
                Event::TurnCompleted {
                    turn_number,
                    final_turn,
                } => {
                    println!(
                        "{}",
                        format!(
                            "Turn {} completed{}",
                            turn_number + 1,
                            if final_turn { " (final)" } else { "" }
                        )
                        .green()
                    );
                }
                _ => {
                    println!("📡 Event: {:?}", event);
                }
            }
        }
    });
}
