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
    #[input(description = "Mode", choice = ["sum", "avg"])]
    mode: String,
    #[input(description = "Weight", choice = [1, 2, 3])]
    weight: u32,
    #[input(description = "Large id", choice = [10000000000000000000])]
    large_id: u64,
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

/// Computes a sum
/// with optional metadata
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[strict(true)]
struct SumOut {
    #[output(description = "The computed sum")]
    value: i64,
    #[output(description = "Optional note")]
    note: Option<String>,
    #[output(description = "Tags")]
    tags: Vec<String>,
    #[output(description = "Small value")]
    small: i8,
    #[output(description = "Ratio")]
    ratio: f64,
    #[output(description = "Mode", choice = ["fast", "slow"])]
    mode: String,
    #[output(description = "Level", choice = [1, 2])]
    level: u32,
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
    use autoagents::core::agent::AgentDeriveT;

    #[test]
    fn generated_schemas_match_expectations() {
        let tool = Addition;
        assert_eq!(tool.name(), "add");
        let schema = tool.args_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["left"]["type"], "integer");
        assert_eq!(schema["properties"]["weight"]["enum"][0], 1);
        assert_eq!(
            schema["properties"]["large_id"]["enum"][0].as_u64(),
            Some(10_000_000_000_000_000_000)
        );

        let output = SumOut::structured_output_format();
        assert_eq!(output["name"], "SumOut");
        assert!(output["strict"].as_bool().unwrap());
        assert!(output["description"]
            .as_str()
            .unwrap()
            .contains("Computes a sum"));
        assert_eq!(output["schema"]["properties"]["tags"]["type"], "array");
        assert_eq!(output["schema"]["properties"]["small"]["type"], "integer");
        assert_eq!(output["schema"]["properties"]["ratio"]["type"], "number");
        assert_eq!(
            output["schema"]["properties"]["mode"]["enum"][0],
            "fast"
        );
        assert!(output["schema"]["properties"]["value"].is_object());
        assert!(
            output["schema"]["required"]
                .as_array()
                .expect("required array")
                .iter()
                .any(|field| field == "value")
        );
        assert!(
            !output["schema"]["required"]
                .as_array()
                .expect("required array")
                .iter()
                .any(|field| field == "note")
        );

        let agent = AdderAgent;
        assert_eq!(agent.name(), "adder");
        assert!(agent.output_schema().is_some());
    }
}
