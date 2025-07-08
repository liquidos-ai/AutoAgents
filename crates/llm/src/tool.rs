use serde_json::Value;
use std::fmt::Debug;

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
    fn name(&self) -> String;
    /// A description explaining the tool’s purpose.
    fn description(&self) -> String;
    /// Return a description of the expected arguments.
    fn args_schema(&self) -> Value;
    /// Run the tool with the given arguments (in JSON) and return the result (in JSON).
    fn run(&self, args: Value) -> Result<Value, ToolCallError>;

    /// Clones the tool into a boxed trait object.
    fn clone_box(&self) -> Box<dyn ToolT>;
}

impl Clone for Box<dyn ToolT> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub trait ToolInputT {
    fn io_schema() -> &'static str;
}

impl From<&Box<dyn ToolT>> for Tool {
    fn from(tool: &Box<dyn ToolT>) -> Self {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: tool.name(),
                description: tool.description(),
                parameters: tool.args_schema(),
            },
        }
    }
}
