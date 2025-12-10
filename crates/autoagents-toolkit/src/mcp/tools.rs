use crate::mcp::{McpConfig, McpError, McpToolsManager};
use autoagents::core::tool::ToolT;
use std::path::Path;
use std::sync::Arc;

/// A collection of MCP tools that can be used in AutoAgents
#[derive(Debug, Clone)]
pub struct McpTools {
    manager: Arc<McpToolsManager>,
}

impl McpTools {
    /// Create MCP tools from a configuration file
    pub async fn from_config<P: AsRef<Path>>(config_path: P) -> Result<Self, McpError> {
        let manager = Arc::new(McpToolsManager::from_config_file(config_path).await?);
        Ok(Self { manager })
    }

    /// Create MCP tools from a configuration object
    pub async fn from_config_object(config: &McpConfig) -> Result<Self, McpError> {
        let manager = Arc::new(McpToolsManager::new());
        manager.connect_servers(config).await?;
        Ok(Self { manager })
    }

    /// Create an empty MCP tools instance
    pub fn new() -> Self {
        Self {
            manager: Arc::new(McpToolsManager::new()),
        }
    }

    /// Get all available tools
    pub async fn get_tools(&self) -> Vec<Arc<dyn ToolT>> {
        self.manager.get_tools().await
    }

    /// Get a specific tool by name
    pub async fn get_tool(&self, name: &str) -> Option<Arc<dyn ToolT>> {
        self.manager.get_tool(name).await
    }

    /// Get tool names
    pub async fn tool_names(&self) -> Vec<String> {
        self.manager.tool_names().await
    }

    /// Get the number of available tools
    pub async fn tool_count(&self) -> usize {
        self.manager.tool_count().await
    }

    /// Refresh tools from all connected servers
    pub async fn refresh(&self) -> Result<(), McpError> {
        self.manager.refresh_tools().await
    }

    /// Get connected server names
    pub async fn connected_servers(&self) -> Vec<String> {
        self.manager.connected_servers().await
    }

    /// Get the underlying manager for advanced operations
    pub fn manager(&self) -> Arc<McpToolsManager> {
        Arc::clone(&self.manager)
    }
}

impl Default for McpTools {
    fn default() -> Self {
        Self::new()
    }
}

/// A convenience macro to create McpTools from a config file path
/// This is used to support the syntax: McpTools::from_config("path/to/config.toml")
#[macro_export]
macro_rules! mcp_tools_from_config {
    ($config_path:expr) => {{
        // This needs to be handled at runtime since we can't do async in macro expansion
        // The agent system will need to handle this appropriately
        $crate::mcp::tools::McpTools::from_config($config_path)
    }};
}

// For easier integration with the agent system
impl McpTools {
    /// Convert to a vector of boxed tools for use with the agent system
    pub async fn to_boxed_tools(&self) -> Vec<Box<dyn ToolT>> {
        self.get_tools()
            .await
            .into_iter()
            .map(|tool| {
                // Create a wrapper that can be boxed
                Box::new(McpToolWrapper { tool }) as Box<dyn ToolT>
            })
            .collect()
    }
}

/// A wrapper to allow Arc<dyn ToolT> to be used as Box<dyn ToolT>
#[derive(Debug)]
struct McpToolWrapper {
    tool: Arc<dyn ToolT>,
}

impl ToolT for McpToolWrapper {
    fn name(&self) -> &str {
        self.tool.name()
    }

    fn description(&self) -> &str {
        self.tool.description()
    }

    fn args_schema(&self) -> serde_json::Value {
        self.tool.args_schema()
    }
}

#[autoagents::async_trait]
impl autoagents::core::tool::ToolRuntime for McpToolWrapper {
    async fn execute(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, autoagents::core::tool::ToolCallError> {
        self.tool.execute(args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_mcp_tools() {
        let tools = McpTools::new();
        assert_eq!(tools.tool_count().await, 0);
        assert!(tools.tool_names().await.is_empty());
        assert!(tools.connected_servers().await.is_empty());
    }

    #[tokio::test]
    async fn test_mcp_tools_boxed_conversion() {
        let tools = McpTools::new();
        let boxed_tools = tools.to_boxed_tools().await;
        assert!(boxed_tools.is_empty());
    }
}
