use autoagents::core::agent::prebuilt::react::ReActExecutor;
use autoagents::core::agent::{AgentBuilder, AgentDeriveT, AgentOutputT};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::core::runtime::{Runtime, SingleThreadedRuntime};
use autoagents::llm::{LLMProvider, ToolCallError, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

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
fn add(args: AdditionArgs) -> Result<i64, ToolCallError> {
    println!("ToolCall: Add {:?}", args);
    Ok(args.left + args.right)
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
    description = "You are a Math agent.",
    tools = [],
    executor = ReActExecutor,
    output = MathAgentOutput
)]
pub struct MathAgent {}

pub async fn distributed_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Math Agent Example...\n");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let runtime = SingleThreadedRuntime::new(10);

    let _ = AgentBuilder::new(agent)
        .with_llm(llm)
        .runtime(runtime.clone())
        .subscribe_topic("test")
        .with_memory(sliding_window_memory)
        .build()
        .await?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None);

    let _ = environment.run().await.unwrap();
    runtime
        .publish_message("What is 2 + 2?".into(), "test".into())
        .await
        .unwrap();
    runtime
        .publish_message("What did I ask before?".into(), "test".into())
        .await
        .unwrap();

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
}
