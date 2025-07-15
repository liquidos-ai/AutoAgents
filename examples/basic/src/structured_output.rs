use autoagents::core::agent::base::AgentBuilder;
use autoagents::core::agent::output::AgentOutputT;
use autoagents::core::agent::{AgentDeriveT, ReActAgentOutput, ReActExecutor};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
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
fn get_weather(args: AdditionArgs) -> Result<i64, ToolCallError> {
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
    tools = [Addition],
    executor = ReActExecutor,
    output = MathAgentOutput
)]
pub struct MathAgent {}

pub async fn math_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Math Agent Example...\n");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let agent = AgentBuilder::new(agent)
        .with_llm(llm)
        .with_memory(sliding_window_memory)
        .build()?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None).await;

    // Register the agent
    let agent_id = environment.register_agent(agent.clone(), None).await?;

    // Add tasks
    let _ = environment.add_task(agent_id, "What is 5 + 3?").await?;

    // Run all tasks
    let results = environment.run_all(agent_id, None).await?;
    let last_result = results.last().unwrap();
    let agent_output: MathAgentOutput =
        last_result.extract_agent_output(ReActAgentOutput::extract_agent_output)?;
    println!(
        "Results: {}, {}",
        agent_output.value, agent_output.explanation
    );

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
}
