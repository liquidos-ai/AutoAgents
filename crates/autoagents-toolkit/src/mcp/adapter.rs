use autoagents::{
    async_trait,
    core::tool::{ToolCallError, ToolRuntime, ToolT},
};
use rmcp::{
    model::{CallToolRequestParams, CallToolResult, ClientInfo, Tool as McpTool},
    service::{RoleClient, RunningService},
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

/// MCP Tool Adapter that bridges MCP tools with AutoAgents tool system
#[derive(Debug)]
pub struct McpToolAdapter {
    name: String,
    description: String,
    args_schema: Value,
    service: Arc<RunningService<RoleClient, ClientInfo>>,
}

impl McpToolAdapter {
    /// Create a new MCP tool adapter from an MCP tool and service connection
    pub fn new(tool: McpTool, service: Arc<RunningService<RoleClient, ClientInfo>>) -> Self {
        let name = tool.name.to_string();
        let description = tool.description.unwrap_or_default().to_string();
        let args_schema = serde_json::to_value(&tool.input_schema).unwrap_or_else(|_| {
            json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        });

        Self {
            name,
            description,
            args_schema,
            service,
        }
    }

    /// Get the tool name as a string reference
    pub fn tool_name(&self) -> &str {
        &self.name
    }
}

impl ToolT for McpToolAdapter {
    fn name(&self) -> &str {
        // We need to use Box::leak to convert String to &'static str
        // This is safe because tools are typically long-lived
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn description(&self) -> &str {
        // Same approach for description
        Box::leak(self.description.clone().into_boxed_str())
    }

    fn args_schema(&self) -> Value {
        self.args_schema.clone()
    }
}

#[async_trait]
impl ToolRuntime for McpToolAdapter {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        // Convert args to the expected format for MCP
        let arguments = match args.as_object() {
            Some(obj) => Some(obj.clone()),
            None => {
                return Err(ToolCallError::RuntimeError(
                    "Arguments must be a JSON object".to_string().into(),
                ));
            }
        };

        // Prepare the call tool request
        let request = CallToolRequestParams {
            meta: None,
            name: self.name.clone().into(),
            arguments,
            task: None,
        };

        // Execute the tool via MCP
        match self.service.call_tool(request).await {
            Ok(result) => {
                // Convert MCP result to our format
                convert_mcp_result(result)
            }
            Err(e) => {
                log::error!("MCP tool execution failed for {}: {:?}", self.name, e);
                Err(ToolCallError::RuntimeError(
                    format!("MCP tool execution failed: {}", e).into(),
                ))
            }
        }
    }
}

/// Convert MCP CallToolResult to AutoAgents Value format
fn convert_mcp_result(result: CallToolResult) -> Result<Value, ToolCallError> {
    if result.is_error.unwrap_or(false) {
        // Handle error case
        let error_msg = result
            .content
            .first()
            .and_then(|c| {
                // Try to extract text from content
                // Since the content structure may vary, we'll be flexible
                match serde_json::to_value(&c.raw) {
                    Ok(val) => {
                        // Try to extract text field from the serialized content
                        val.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    }
                    Err(_) => None,
                }
            })
            .unwrap_or_else(|| "Unknown MCP error".to_string());

        return Err(ToolCallError::RuntimeError(
            format!("MCP tool error: {}", error_msg).into(),
        ));
    }

    // Convert successful result
    let mut result_data = HashMap::new();

    // Extract content from the result
    if !result.content.is_empty() {
        let mut contents = Vec::new();
        for content in &result.content {
            // Serialize the raw content to JSON for flexible handling
            match serde_json::to_value(&content.raw) {
                Ok(content_json) => {
                    contents.push(content_json);
                }
                Err(e) => {
                    log::warn!("Failed to serialize MCP content: {}", e);
                    // Fallback to a simple representation
                    contents.push(json!({
                        "type": "unknown",
                        "error": format!("Serialization failed: {}", e)
                    }));
                }
            }
        }
        result_data.insert("content", json!(contents));
    }

    // Add success indicator
    result_data.insert("success", json!(true));

    Ok(json!(result_data))
}

/// Wrapper to allow Arc<dyn ToolT> to be used as Box<dyn ToolT>
/// This is needed for the AgentDeriveT trait which expects Box<dyn ToolT>
#[derive(Debug)]
pub struct McpToolWrapper {
    tool: Arc<dyn ToolT>,
}

impl McpToolWrapper {
    /// Create a new wrapper around an Arc<dyn ToolT>
    pub fn new(tool: Arc<dyn ToolT>) -> Self {
        Self { tool }
    }
}

impl ToolT for McpToolWrapper {
    fn name(&self) -> &str {
        self.tool.name()
    }

    fn description(&self) -> &str {
        self.tool.description()
    }

    fn args_schema(&self) -> Value {
        self.tool.args_schema()
    }
}

#[async_trait]
impl ToolRuntime for McpToolWrapper {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        self.tool.execute(args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoagents::core::tool::{ToolCallError, ToolRuntime, ToolT};
    use rmcp::model::{CallToolResult, Content};
    use serde_json::json;

    #[derive(Debug)]
    struct DummyTool;

    impl ToolT for DummyTool {
        fn name(&self) -> &str {
            "dummy"
        }

        fn description(&self) -> &str {
            "dummy tool"
        }

        fn args_schema(&self) -> Value {
            json!({"type": "object"})
        }
    }

    #[autoagents::async_trait]
    impl ToolRuntime for DummyTool {
        async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
            Ok(args)
        }
    }

    #[tokio::test]
    async fn test_mcp_tool_wrapper_delegates() {
        let wrapper = McpToolWrapper::new(Arc::new(DummyTool));
        assert_eq!(wrapper.name(), "dummy");
        assert_eq!(wrapper.description(), "dummy tool");
        assert_eq!(wrapper.args_schema(), json!({"type": "object"}));

        let result = wrapper.execute(json!({"ok": true})).await.unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn test_convert_mcp_result_success() {
        let result = CallToolResult::success(vec![Content::text("ok".to_string())]);
        let value = convert_mcp_result(result).unwrap();
        assert!(matches!(value["success"].as_bool(), Some(true)));
        assert!(value["content"].is_array());
    }

    #[test]
    fn test_convert_mcp_result_error() {
        let result = CallToolResult::error(vec![Content::text("failed".to_string())]);
        let err = convert_mcp_result(result).unwrap_err();
        assert!(err.to_string().contains("MCP tool error"));
    }
}
