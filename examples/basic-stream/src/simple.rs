use autoagents::core::agent::prebuilt::react::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::protocol::{Event, TaskResult};
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
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
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent",
    tools = [Addition]
    output = MathAgentOutput
)]
pub struct MathAgent {}

impl ReActExecutor for MathAgent {}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
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
        .publish_message("What is 2 + 2?".into(), "test".into())
        .await
        .unwrap();
    runtime
        .publish_message("What did I ask before?".into(), "test".into())
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
                        let agent_out: ReActAgentOutput = serde_json::from_value(val).unwrap();
                        let math_out: MathAgentOutput =
                            serde_json::from_str(&agent_out.response).unwrap();
                        println!(
                            "{}",
                            format!(
                                "Math Value: {}, Explanation: {}",
                                math_out.value, math_out.explanation
                            )
                            .green()
                        );
                    }
                    _ => {
                        println!("{}", format!("Error!!!").red());
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
