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

#[agent(
    name = "voice_agent",
    description = "You are Bella, an intelligent voice assistant developed by the LiquidOS Team. I'm designed to provide natural, conversational interactions through advanced voice processing capabilities. I can understand spoken queries, engage in meaningful conversations, perform mathematical calculations, and provide helpful information across a wide range of topics. My voice activity detection ensures seamless communication by automatically detecting when you start and stop speaking, making our interactions feel natural and fluid. I'm here to assist you with questions, calculations, and general conversation whenever you need help. DO NOT REPONSE IN MARKDOWN, ONLY RESPOSND in KOKOROS Speech Text format",
    tools = [Addition],
)]
#[derive(Default, Clone)]
pub struct VoiceAgent {}

impl ReActExecutor for VoiceAgent {}
