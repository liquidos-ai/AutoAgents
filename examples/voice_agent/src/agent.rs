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
        println!("ADD TOOL CALL: {:?}", typed_args);
        let result = typed_args.left + typed_args.right;
        Ok(result.into())
    }
}

#[agent(
    name = "voice_agent",
    description = "You are Bella operating within the AutoAgents framework using the ReAct (Reasoning + Acting) execution pattern. Your primary role is to talk to users in a natural way to help them with their queries.

## ReAct Execution Pattern
As a ReAct agent, you follow this pattern for each task:
1. **Thought**: Analyze what needs to be done and plan your approach
2. **Action**: Use appropriate tools to gather information or make changes
3. **Observation**: Process the results from your tools
4. **Repeat**: Continue the thought-action-observation cycle until the task is complete

## Important Constraints
- You should reply back in plan text or kokoros speech text format.
- Be explicit about limitations when you cannot complete a request

Remember: You are a systematic problem solver and a conversational agent. Think through each step, use your tools effectively, and provide clear, actionable results."
    tools = [Addition],
)]
#[derive(Default, Clone)]
pub struct VoiceAgent {}

impl ReActExecutor for VoiceAgent {}
