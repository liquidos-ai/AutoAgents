#![allow(dead_code)]

use autoagents::async_trait;
use autoagents::core::agent::AgentOutputT;
use autoagents::core::tool::{ToolCallError, ToolRuntime, ToolT};
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct AddArgs {
    #[input(description = "Left operand")]
    left: i64,
    #[input(description = "Right operand")]
    right: i64,
}

#[tool(name = "add", description = "Add two numbers", input = AddArgs)]
struct Addition;

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let parsed: AddArgs = serde_json::from_value(args)?;
        Ok((parsed.left + parsed.right).into())
    }
}

#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct SumOut {
    #[output(description = "The computed sum")]
    value: i64,
}

#[agent(
    name = "adder",
    description = "Adds numbers",
    tools = [Addition],
    output = SumOut
)]
#[derive(Clone, AgentHooks, Default)]
struct AdderAgent;

#[cfg(test)]
mod smoke_tests {
    use super::*;

    #[test]
    fn generated_schemas_match_expectations() {
        let tool = Addition;
        assert_eq!(tool.name(), "add");
        let schema = tool.args_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["left"]["type"], "integer");

        let output = SumOut::structured_output_format();
        assert_eq!(output["name"], "SumOut");
        assert!(output["schema"]["properties"]["value"].is_object());

        let _agent = AdderAgent;
    }
}
