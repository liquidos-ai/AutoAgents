/// This example demonstrates Agent Streaming
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::error::Error;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents_derive::{agent, tool, AgentHooks, AgentOutput, ToolInput};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_stream::StreamExt;

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
        println!("Tool Call Executed: {}", result);
        Ok(result.into())
    }
}

#[derive(Deserialize, Serialize, Debug, AgentOutput)]
pub struct AgentOutput {
    #[output(description = "The response of the query")]
    response: String,
}

impl From<ReActAgentOutput> for AgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        // For streaming: only try to parse JSON if the output is marked as done
        // and the response is not empty
        if output.done && !resp.trim().is_empty() {
            // Try to parse as structured JSON first
            if let Ok(value) = serde_json::from_str::<AgentOutput>(&resp) {
                return value;
            }
        }

        // For streaming chunks or unparseable content, create a default response
        AgentOutput {
            response: resp.to_string(),
        }
    }
}

#[agent(
    name = "streaming_agent",
    description = "You are a Math agent and can solve problems in addition",
    tools = [Addition]
    output = AgentOutput
)]
#[derive(Clone, AgentHooks)]
pub struct StreamingAgent {}

pub async fn run(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🌊 Agent Streaming Example");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = ReActAgent::new(StreamingAgent {});

    let agent_handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .stream(true) // Enable streaming for this agent
        .memory(sliding_window_memory)
        .build()
        .await?;

    let task = Task::new("What is 2 + 2?");

    // Process the stream directly
    let mut stream = agent_handle.agent.run_stream(task).await?;
    println!("🔄 Processing stream tokens...\n");

    while let Some(result) = stream.next().await {
        match result {
            Ok(output) => {
                println!(
                    "{}",
                    format!("🌊 Streaming Response: {}", output.response).green()
                );
            }
            _ => {
                //
            }
        }
    }

    println!("\n✅ Streaming example completed!");
    Ok(())
}
