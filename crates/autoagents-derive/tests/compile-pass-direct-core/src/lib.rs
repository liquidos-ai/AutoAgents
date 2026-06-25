#![allow(dead_code)]

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
    #[input(description = "Scale", choice = [1, 2])]
    scale: u8,
    #[input(description = "Large factor", choice = [10000000000000000000])]
    large_factor: u64,
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

/// Product result
/// with metadata
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[strict(true)]
struct ProductOut {
    #[output(description = "The computed product")]
    value: i64,
    #[output(description = "Optional note")]
    note: Option<String>,
    #[output(description = "Factors")]
    factors: Vec<i64>,
    #[output(description = "Small flag")]
    flag: i8,
    #[output(description = "Ratio")]
    ratio: f64,
    #[output(description = "Mode", choice = ["exact", "approx"])]
    mode: String,
}

#[agent(
    name = "multiplier",
    description = "Multiplies numbers",
    tools = [Multiplication],
    output = ProductOut
)]
#[derive(Clone, AgentHooks, Default)]
struct MultiplierAgent;

#[cfg(test)]
mod smoke_tests {
    use super::*;
    use autoagents_core::agent::AgentDeriveT;

    #[test]
    fn generated_schemas_match_expectations() {
        let tool = Multiplication;
        assert_eq!(tool.name(), "multiply");
        let schema = tool.args_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["left"]["type"], "integer");
        assert_eq!(schema["properties"]["scale"]["enum"][0], 1);
        assert_eq!(
            schema["properties"]["large_factor"]["enum"][0].as_u64(),
            Some(10_000_000_000_000_000_000)
        );

        let output = ProductOut::structured_output_format();
        assert_eq!(output["name"], "ProductOut");
        assert!(output["strict"].as_bool().unwrap());
        assert!(output["description"]
            .as_str()
            .unwrap()
            .contains("Product result"));
        assert_eq!(output["schema"]["properties"]["factors"]["type"], "array");
        assert_eq!(output["schema"]["properties"]["flag"]["type"], "integer");
        assert_eq!(output["schema"]["properties"]["ratio"]["type"], "number");
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

        let agent = MultiplierAgent;
        assert_eq!(agent.name(), "multiplier");
        assert!(agent.output_schema().is_some());
    }
}
