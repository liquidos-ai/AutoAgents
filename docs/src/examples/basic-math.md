# Basic Weather Agent Example

This example demonstrates how to create a simple weather agent using AutoAgents. The agent can add two numbers (LOL!).

## Overview

The Math agent example showcases:

- Basic agent creation with the `#[agent]` macro
- Tool implementation with the `#[tool]` macro
- ReAct executor for reasoning and acting
- Memory management with sliding window
- Environment setup and task execution
- Type Safe Pub/Sub

## Complete Example

```rust
use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

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
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    generic: Option<String>,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition],
    output = MathAgentOutput
)]
pub struct MathAgent {}

impl ReActExecutor for MathAgent {}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let runtime = SingleThreadedRuntime::new(None);

    let test_topic = Topic::<Task>::new("test");

    let agent_handle = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic(test_topic.clone())
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    let _ = environment.register_runtime(runtime.clone()).await;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Publish message to all the subscribing actors
    runtime.publish(&Topic::<Task>::new("test"), Task::new("what is 2 + 2?")).await?;
    // Send a direct message for memory test
    println!("\n📧 Sending direct message to test memory...");
    runtime.send_message(Task::new("What was the question I asked?"), agent_handle.addr()).await?;

    let _ = environment.run().await;
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
                            "📋 Task Started - Agent: {:?}, Task: {}",
                            actor_id, task_description
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
                        let agent_out: ReActAgentOutput = serde_json::from_value(val).unwrap();
                        let math_out: MathAgentOutput =
                            serde_json::from_str(&agent_out.response).unwrap();
                        println!(
                            "{}",
                            format!(
                                "Math Value: {}, Explanation: {}, Generic: {:?}",
                                math_out.value, math_out.explanation, math_out.generic
                            )
                                .green()
                        );
                    }
                    _ => {
                        println!("{}", "Error!!!".to_string().red());
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
```