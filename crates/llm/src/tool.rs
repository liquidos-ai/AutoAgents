use serde_json::Value;
use std::{fmt::Debug, sync::Arc};

use crate::chat::{FunctionTool, Tool};

#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    #[error("Runtime Error {0}")]
    RuntimeError(#[from] Box<dyn std::error::Error + Sync + Send>),

    #[error("Serde Error {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub trait ToolT: Send + Sync + Debug {
    /// The name of the tool.
    fn name(&self) -> &'static str;
    /// A description explaining the tool’s purpose.
    fn description(&self) -> &'static str;
    /// Return a description of the expected arguments.
    fn args_schema(&self) -> Value;
    /// Run the tool with the given arguments (in JSON) and return the result (in JSON).
    fn run(&self, args: Value) -> Result<Value, ToolCallError>;
}

pub trait ToolInputT {
    fn io_schema() -> &'static str;
}

impl From<&Arc<Box<dyn ToolT>>> for Tool {
    fn from(tool: &Arc<Box<dyn ToolT>>) -> Self {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.args_schema(),
            },
        }
    }
}
