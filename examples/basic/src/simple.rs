use crate::utils::handle_events;
use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
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
struct Addition {}

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("execute tool: {:?}", args);
        let typed_args: AdditionArgs = serde_json::from_value(args)?;
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct MultiplicationArgs {
    #[input(description = "Left operand for multiplication")]
    left: i32,
    #[input(description = "Right operand for multiplication")]
    right: i32,
}

#[tool(name = "multiplication", description = "Multiply two numbers together", input = MultiplicationArgs
)]
struct Multiplication {}

#[async_trait]
impl ToolRuntime for Multiplication {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        println!("Executing Multiplication");
        let typed_args: MultiplicationArgs = serde_json::from_value(args)?;
        let result = typed_args.left * typed_args.right;
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
    description = "You solve math problems by calling tools step-by-step.

RULES:
1. ALWAYS call tools for calculations - NEVER calculate yourself
2. For multi-step problems, call tools multiple times in sequence
3. After ALL calculations are done then only return the final answer.

Example for \"What is (20 + 30) * 10?\":
- Step 1: Call addition(20, 30) → get 50
- Step 2: Call multiplication(50, 10) → get 500
- Step 3: Return \"First added 20+30=10, then multiplied 50*100=500\"

CRITICAL: Your final response MUST be valid JSON with 'value' and 'explanation' fields.",
    tools = [Addition, Multiplication],
    output = MathAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

impl From<ReActAgentOutput> for MathAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        output.parse_or_map(|resp| MathAgentOutput {
            value: 0,
            explanation: resp.to_string(),
            generic: None,
        })
    }
}

pub async fn simple_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MathAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    println!("Running simple_agent with direct run method");

    //Handle Events sent from the executor, This is good for updating UI based on the events
    handle_events(agent_handle.rx);

    let result = agent_handle.agent.run(Task::new("What is 1 + 1?")).await?;
    println!("Result: {:?}", result);

    println!("Multi-step calculation");
    let result3 = agent_handle
        .agent
        .run(Task::new("What is (10 + 5) * 3?"))
        .await?;
    println!("Result: {:?}\n", result3);
    Ok(())
}
