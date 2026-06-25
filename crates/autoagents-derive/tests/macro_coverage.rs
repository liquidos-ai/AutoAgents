//! In-process integration tests that exercise proc-macro expansion paths for coverage.
//! Uses the `autoagents` facade dependency layout (see `resolve_layout`).

#![allow(dead_code)]

use autoagents::async_trait;
use autoagents::core::agent::{AgentDeriveT, AgentOutputT};
use autoagents::core::tool::{
    ToolCallError, ToolInputSchema, ToolInputT, ToolOutputT, ToolRuntime, ToolT,
};
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct RichToolArgs {
    #[input(description = "Name")]
    name: String,
    #[input(description = "Small signed integer")]
    small: i8,
    #[input(description = "Wide unsigned integer")]
    wide: u64,
    #[input(description = "Height")]
    height: u16,
    #[input(description = "Ratio")]
    ratio: f64,
    #[input(description = "Enabled flag")]
    enabled: bool,
    #[input(description = "Optional note")]
    note: Option<String>,
    #[input(description = "Mode", choice = ["fast", "slow"])]
    mode: String,
    #[input(description = "Level", choice = [1, 2, 3])]
    level: i32,
    #[input(description = "Large id", choice = [10000000000000000000])]
    large_id: u64,
}

/// First line of docs
/// Second line of docs
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
#[strict(false)]
struct RichAgentOutput {
    #[output(description = "Name")]
    name: String,
    #[output(description = "Tags")]
    tags: Vec<String>,
    #[output(description = "Small signed integer")]
    small: i8,
    #[output(description = "Wide unsigned integer")]
    wide: u64,
    #[output(description = "Ratio")]
    ratio: f64,
    #[output(description = "Enabled flag")]
    enabled: bool,
    #[output(description = "Optional note")]
    note: Option<String>,
    #[output(description = "Mode", choice = ["a", "b"])]
    mode: String,
    #[output(description = "Level", choice = [1, 2])]
    level: u32,
    #[output(description = "Large id", choice = [10000000000000000000])]
    large: u64,
}

#[derive(Debug, Serialize, Deserialize, AgentOutput, JsonSchema)]
struct ToolResultOut {
    #[output(description = "Whether the tool succeeded")]
    ok: bool,
}

#[tool(
    name = "rich",
    description = "Rich tool with many field types",
    input = RichToolArgs,
    output = ToolResultOut
)]
struct RichTool;

#[async_trait]
impl ToolRuntime for RichTool {
    async fn execute(&self, _args: Value) -> Result<Value, ToolCallError> {
        Ok(serde_json::json!({"ok": true}))
    }
}

#[agent(name = "plain", description = "Agent without structured output", tools = [RichTool])]
#[derive(Clone, AgentHooks, Default)]
struct PlainAgent;

#[agent(
    name = "rich",
    description = "Agent with structured output",
    tools = [RichTool],
    output = RichAgentOutput
)]
#[derive(Clone, AgentHooks, Default)]
struct RichAgent;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_input_schema_covers_numeric_and_string_choices() {
        let tool = RichTool;
        assert_eq!(tool.name(), "rich");
        assert_eq!(tool.description(), "Rich tool with many field types");

        let schema = tool.args_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["small"]["type"], "integer");
        assert_eq!(schema["properties"]["wide"]["type"], "integer");
        assert_eq!(schema["properties"]["height"]["type"], "integer");
        assert_eq!(schema["properties"]["ratio"]["type"], "number");
        assert_eq!(schema["properties"]["enabled"]["type"], "boolean");

        let mode_enum = schema["properties"]["mode"]["enum"].as_array().unwrap();
        assert_eq!(mode_enum[0], "fast");
        let level_enum = schema["properties"]["level"]["enum"].as_array().unwrap();
        assert_eq!(level_enum[0], 1);
        let large_enum = schema["properties"]["large_id"]["enum"].as_array().unwrap();
        assert_eq!(large_enum[0].as_u64(), Some(10_000_000_000_000_000_000));

        let io = RichToolArgs::io_schema();
        assert!(io.contains("object"));
        let static_value = RichToolArgs::io_schema_value();
        assert_eq!(static_value["type"], "object");
    }

    #[test]
    fn tool_output_schema_is_exposed() {
        let tool = RichTool;
        let output_schema = tool.output_schema().expect("tool output schema");
        assert!(output_schema.is_object());
        let static_schema = ToolResultOut::io_schema();
        assert!(static_schema.is_object());
    }

    #[test]
    fn agent_output_schema_covers_vectors_choices_and_docs() {
        let output = RichAgentOutput::structured_output_format();
        assert_eq!(output["name"], "RichAgentOutput");
        assert_eq!(output["strict"].as_bool(), Some(false));
        let description = output["description"].as_str().unwrap();
        assert!(description.contains("First line"));
        assert!(description.contains("Second line"));
        assert_eq!(output["schema"]["properties"]["tags"]["type"], "array");
        assert_eq!(output["schema"]["properties"]["small"]["type"], "integer");
        assert_eq!(output["schema"]["properties"]["wide"]["type"], "integer");
        assert_eq!(output["schema"]["properties"]["ratio"]["type"], "number");
        assert_eq!(output["schema"]["properties"]["enabled"]["type"], "boolean");

        let large_enum = output["schema"]["properties"]["large"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(large_enum[0].as_u64(), Some(10_000_000_000_000_000_000));

        let raw = RichAgentOutput::output_schema();
        assert!(raw.contains("RichAgentOutput"));
    }

    #[test]
    fn agents_hooks_and_output_schema_branches() {
        let plain = PlainAgent;
        assert_eq!(plain.name(), "plain");
        assert_eq!(plain.description(), "Agent without structured output");
        assert!(plain.output_schema().is_none());
        assert_eq!(plain.tools().len(), 1);

        let rich = RichAgent;
        assert_eq!(rich.name(), "rich");
        assert!(rich.output_schema().is_some());
        assert_eq!(rich.tools().len(), 1);
    }
}
