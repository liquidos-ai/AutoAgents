use async_trait::async_trait;
use autoagents_core::agent::AgentOutputT;
use autoagents_core::tool::{ToolCallError, ToolRuntime, ToolT};
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct MultiplyArgs {
    #[input(description = "Left operand")]
    left: i64,
    #[input(description = "Right operand")]
    right: i64,
}

#[tool(name = "multiply", description = "Multiply two numbers", input = MultiplyArgs)]
struct Multiplication;

#[async_trait]
impl ToolRuntime for Multiplication {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let parsed: MultiplyArgs = serde_json::from_value(args)?;
        Ok((parsed.left * parsed.right).into())
    }
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct ProductOut {
    #[output(description = "The computed product")]
    value: i64,
}

#[agent(
    name = "multiplier",
    description = "Multiplies numbers",
    tools = [Multiplication],
    output = ProductOut
)]
#[derive(Clone, AgentHooks, Default)]
struct MultiplierAgent;

fn main() {
    let tool = Multiplication;
    assert_eq!(tool.name(), "multiply");
    let schema = tool.args_schema();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["left"]["type"], "integer");

    let output = ProductOut::structured_output_format();
    assert_eq!(output["name"], "ProductOut");
    assert!(output["schema"]["properties"]["value"].is_object());

    let _agent = MultiplierAgent;
}
