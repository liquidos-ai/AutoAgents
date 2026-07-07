use async_trait::async_trait;
use autoagents_core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::tool;
use serde_json::Value;

// `#[tool]` requires an input type that implements both generated schema traits.
// Implementing only `ToolInputT` is not enough for runtime schema exposure.
struct ManualArgs;

impl ToolInputT for ManualArgs {
    fn io_schema() -> &'static str {
        r#"{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}"#
    }
}

#[tool(name = "manual", description = "Manual input schema", input = ManualArgs)]
struct ManualTool;

#[async_trait]
impl ToolRuntime for ManualTool {
    async fn execute(&self, _args: Value) -> Result<Value, ToolCallError> {
        Ok(Value::Null)
    }
}

fn main() {}
