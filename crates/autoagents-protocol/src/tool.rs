use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Result emitted after executing a single tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub success: bool,
    pub arguments: Value,
    pub result: Value,
}
