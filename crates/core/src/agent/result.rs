use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunResult {
    /// Indicates if the run was successful
    pub success: bool,

    /// The actual output of the agent, if any
    pub output: Option<Value>,

    /// Optional error message if failed
    pub error_message: Option<String>,

    /// Additional metadata, e.g. timestamps, logs, etc.
    pub metadata: Option<Value>,
}

impl AgentRunResult {
    /// Helper to create a success result
    pub fn success(output: Value) -> Self {
        Self {
            success: true,
            output: Some(output),
            error_message: None,
            metadata: None,
        }
    }

    /// Helper to create a failure result
    pub fn failure(error_message: String) -> Self {
        Self {
            success: false,
            output: None,
            error_message: Some(error_message),
            metadata: None,
        }
    }
}
