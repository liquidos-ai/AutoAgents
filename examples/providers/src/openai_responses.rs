use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::ReActAgent;
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::ToolCallError;
use autoagents::llm::backends::openai::{OpenAI, OpenAIApiMode};
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::chat::ReasoningEffort;
use autoagents::prelude::{ToolRuntime, ToolT};
use autoagents_derive::{AgentHooks, ToolInput, agent, tool};
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

#[agent(
    name = "math_agent",
    description = "You are a Math agent. Use the provided tools when calculation is needed. After a tool returns the result, answer the user directly and do not call the same tool again for the same completed step.",
    tools = [Addition],
)]
#[derive(Default, Clone, AgentHooks)]
pub struct MathAgent {}

pub async fn run() -> Result<(), Error> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".into());

    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .api_mode(OpenAIApiMode::Responses)
        .model("gpt-5.2")
        .reasoning_effort(ReasoningEffort::Medium)
        .build()
        .expect("Failed to build OpenAI Responses LLM");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MathAgent {}))
        .llm(llm)
        .memory(sliding_window_memory)
        .build()
        .await?;

    let result = agent_handle
        .agent
        .run(Task::new("What is 20 + 10? Answer in one short sentence."))
        .await?;
    println!("Result: {:?}", result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn openai_responses_addition_tool_executes_expected_result() {
        let result = Addition {}
            .execute(json!({"left": 20, "right": 10}))
            .await
            .expect("addition should succeed");
        assert_eq!(result, json!(30));
    }
}
