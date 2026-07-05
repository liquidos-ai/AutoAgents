use crate::mcp::{
    McpConfig, McpError, McpProcessPolicy, McpServerInstructions, McpServerStatus, McpToolsManager,
};
use autoagents::core::tool::ToolT;
use rmcp::model::{Prompt, Resource, ResourceTemplate};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A collection of MCP tools that can be used in AutoAgents
#[derive(Debug, Clone, Default)]
pub struct McpTools {
    manager: Arc<McpToolsManager>,
}

impl McpTools {
    /// Create an empty MCP tools instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Create MCP tools from a configuration file
    pub async fn from_config<P: AsRef<Path>>(config_path: P) -> Result<Self, McpError> {
        let manager = Arc::new(McpToolsManager::from_config_file(config_path).await?);
        Ok(Self { manager })
    }

    /// Create MCP tools from a configuration file with an explicit local process policy.
    pub async fn from_config_with_process_policy<P: AsRef<Path>>(
        config_path: P,
        process_policy: McpProcessPolicy,
    ) -> Result<Self, McpError> {
        let manager = Arc::new(
            McpToolsManager::from_config_file_with_process_policy(config_path, process_policy)
                .await?,
        );
        Ok(Self { manager })
    }

    /// Create MCP tools from a configuration object
    pub async fn from_config_object(config: &McpConfig) -> Result<Self, McpError> {
        let mut manager = McpToolsManager::new();
        if let Some(base_dir) = config.base_dir() {
            manager = manager.with_base_dir(base_dir.to_path_buf());
        }
        let manager = Arc::new(manager);
        manager.connect_servers(config).await?;
        Ok(Self { manager })
    }

    /// Create MCP tools from a configuration object with an explicit local process policy.
    pub async fn from_config_object_with_process_policy(
        config: &McpConfig,
        process_policy: McpProcessPolicy,
    ) -> Result<Self, McpError> {
        let mut manager = McpToolsManager::with_process_policy(process_policy);
        if let Some(base_dir) = config.base_dir() {
            manager = manager.with_base_dir(base_dir.to_path_buf());
        }
        let manager = Arc::new(manager);
        manager.connect_servers(config).await?;
        Ok(Self { manager })
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

    /// Get configured server statuses.
    pub async fn status(&self) -> HashMap<String, McpServerStatus> {
        self.manager.status().await
    }

    /// Connect a configured server.
    pub async fn connect(&self, name: &str) -> Result<(), McpError> {
        self.manager.connect(name).await
    }

    /// Disconnect a configured server.
    pub async fn disconnect(&self, name: &str) -> Result<(), McpError> {
        self.manager.disconnect(name).await
    }

    /// Get server-provided instructions for connected servers.
    pub async fn server_instructions(&self) -> Vec<McpServerInstructions> {
        self.manager.server_instructions().await
    }

    /// List prompts from connected servers.
    pub async fn prompts(&self) -> Result<HashMap<String, Prompt>, McpError> {
        self.manager.prompts().await
    }

    /// List resources from connected servers.
    pub async fn resources(
        &self,
        server_name: Option<&str>,
    ) -> Result<HashMap<String, Resource>, McpError> {
        self.manager.resources(server_name).await
    }

    /// List resource templates from connected servers.
    pub async fn resource_templates(
        &self,
        server_name: Option<&str>,
    ) -> Result<HashMap<String, ResourceTemplate>, McpError> {
        self.manager.resource_templates(server_name).await
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
    use crate::mcp::{McpProcessPolicy, McpServerConfig};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_empty_mcp_tools() {
        let tools = McpTools::default();
        assert_eq!(tools.tool_count().await, 0);
        assert!(tools.tool_names().await.is_empty());
        assert!(tools.connected_servers().await.is_empty());
    }

    #[tokio::test]
    async fn test_mcp_tools_boxed_conversion() {
        let tools = McpTools::default();
        let boxed_tools = tools.to_boxed_tools().await;
        assert!(boxed_tools.is_empty());
    }

    #[tokio::test]
    async fn from_config_object_honors_config_base_dir_for_relative_commands() {
        let dir = tempdir().unwrap();
        let bin_dir = dir.path().join("node_modules").join(".bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let server_bin = bin_dir.join("server");
        std::fs::write(&server_bin, "").unwrap();

        let mut config = McpConfig::new().with_base_dir(dir.path().to_path_buf());
        config.add_server(McpServerConfig::local(
            "local",
            vec!["./node_modules/.bin/server".to_string()],
        ));

        let err =
            McpTools::from_config_object_with_process_policy(&config, McpProcessPolicy::deny_all())
                .await
                .unwrap_err();

        assert!(err.to_string().contains(&server_bin.display().to_string()));
    }
}
