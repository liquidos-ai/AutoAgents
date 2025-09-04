use autoagents::core::actor::Topic;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgentOutput, ReActExecutor};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{ActorAgent, AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::protocol::Event;
use autoagents::core::runtime::{SingleThreadedRuntime, TypedRuntime};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT, WasmRuntime};
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
    name = "WasmAddition",
    description = "Use this WASM tool to add two numbers using WebAssembly runtime",
    input = AdditionArgs,
)]
struct WasmAddition {}

impl ToolRuntime for WasmAddition {
    fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("🔧 Executing WASM Addition tool...");

        let runtime = WasmRuntime::builder()
            .source_file("./examples/wasm_tool/wasm/wasm_tool.wasm")
            .alloc_fn("alloc")
            .execute_fn("execute")
            .free_fn(Some("free".to_string()))
            .build()
            .map_err(|e| {
                ToolCallError::RuntimeError(format!("Failed to build WASM runtime: {}", e).into())
            })?;

        // Execute and get result
        match runtime.run(args) {
            Ok(result) => {
                println!("✅ WASM execution successful: {}", result);
                Ok(result)
            }
            Err(e) => {
                println!("❌ WASM execution failed: {}", e);
                Err(ToolCallError::RuntimeError(e.into()))
            }
        }
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct WasmMathAgentOutput {
    #[output(description = "The WASM computation result")]
    value: i64,
    #[output(description = "Explanation of the WASM computation")]
    explanation: String,
}

#[agent(
    name = "wasm_math_agent",
    description = "You are a Math agent that uses WebAssembly (WASM) tools for computations. You demonstrate the power of running secure, sandboxed code through WASM.",
    tools = [WasmAddition],
    output = WasmMathAgentOutput
)]
#[derive(Clone)]
pub struct WasmMathAgent {}

impl ReActExecutor for WasmMathAgent {}

pub async fn wasm_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🚀 WASM Agent Example - Math operations using WebAssembly");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = WasmMathAgent {};
    let runtime = SingleThreadedRuntime::new(None);

    // Create topic for WASM agent
    let wasm_topic = Topic::<Task>::new("wasm_math");

    let _ = AgentBuilder::<_, ActorAgent>::new(agent)
        .llm(llm)
        .runtime(runtime.clone())
        .subscribe(wasm_topic.clone())
        .memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);
    environment.register_runtime(runtime.clone()).await?;

    let receiver = environment.take_event_receiver(None).await?;
    handle_events(receiver);

    // Start the environment
    let _handle = environment.run();

    // Send WASM computation tasks
    println!("\n📤 Sending WASM computation tasks...");

    let tasks = vec![
        "Calculate 2 + 2 using the WASM addition tool",
        "What is 15 + 27? Use the WASM tool for this calculation",
        "Compute 100 + 200 and explain how WASM tools work",
    ];

    for (i, task_content) in tasks.iter().enumerate() {
        println!("\n💻 Sending WASM task {}: {}", i + 1, task_content);

        let task = Task::new(*task_content);

        // Publish to topic
        runtime.publish(&wasm_topic, task).await?;

        // Give time between tasks
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    }

    // Give time for processing
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

    println!("\n✅ WASM Agent example completed!");
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
                            "🎯 WASM Task Started - Agent: {:?}\n   📝 Task: {}",
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
                        format!("🔧 WASM Tool Call: {} with args: {}", tool_name, arguments)
                            .yellow()
                    );
                }
                Event::ToolCallCompleted {
                    tool_name, result, ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "✅ WASM Tool Completed: {} - Result: {:?}",
                            tool_name, result
                        )
                        .yellow()
                    );
                }
                Event::TaskComplete { result, .. } => {
                    match serde_json::from_str::<ReActAgentOutput>(&result) {
                        Ok(agent_out) => {
                            // Try to parse as WASM math output
                            if let Ok(wasm_output) =
                                serde_json::from_str::<WasmMathAgentOutput>(&agent_out.response)
                            {
                                println!(
                                    "{}",
                                    format!(
                                        "🧮 WASM Math Result:\n   Value: {}\n   Explanation: {}\n",
                                        wasm_output.value, wasm_output.explanation
                                    )
                                    .green()
                                );
                            } else {
                                // Fallback to regular output
                                println!(
                                    "{}",
                                    format!("💬 Agent Response: {}", agent_out.response).green()
                                );
                            }
                        }
                        Err(e) => {
                            println!("{}", format!("❌ Failed to parse response: {}", e).red());
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
