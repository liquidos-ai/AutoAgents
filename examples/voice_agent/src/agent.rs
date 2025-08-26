use autoagents::core::agent::prebuilt::executor::ReActExecutor;
use autoagents::core::agent::{AgentDeriveT, AgentOutputT};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
        println!("TOOL CALL: {:?}", typed_args);
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct AgentOutput {
    #[output(description = "If user asks other than math questions, use this to answer them.")]
    pub response: String,
}

#[agent(
    name = "voice_agent",
    description = "You're name is Bella, You are developed by LiquidOS, You are a voice assistant that can answer questions and perform calculations.",
    tools = [Addition],
    output = AgentOutput
)]
#[derive(Default, Clone)]
pub struct VoiceAgent {}

impl ReActExecutor for VoiceAgent {}
