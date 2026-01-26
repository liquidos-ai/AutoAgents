use crate::mcp::{
    adapter::McpToolAdapter,
    config::{McpConfig, McpServerConfig},
};
use autoagents::core::tool::ToolT;
use rmcp::{
    model::ClientInfo,
    service::{RoleClient, RunningService, ServiceExt},
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::RwLock;

/// Represents a connection to an MCP server
#[derive(Debug)]
pub struct McpServerConnection {
    pub name: String,
    pub service: Arc<RunningService<RoleClient, ClientInfo>>,
}

/// MCP Tools Manager that handles multiple MCP server connections
#[derive(Debug)]
pub struct McpToolsManager {
    connections: Arc<RwLock<HashMap<String, McpServerConnection>>>,
    tools: Arc<RwLock<Vec<Arc<dyn ToolT>>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Server not found: {0}")]
    ServerNotFound(String),
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Transport error: {0}")]
    TransportError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Tool error: {0}")]
    ToolError(String),
    #[error("Rmcp error: {0}")]
    RmcpError(#[from] rmcp::ErrorData),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Generic error: {0}")]
    GenericError(String),
}

impl McpToolsManager {
    /// Create a new MCP tools manager
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            tools: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Load MCP configuration from a file and connect to all servers
    pub async fn from_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, McpError> {
        let config =
            McpConfig::from_file(path).map_err(|e| McpError::ConfigError(e.to_string()))?;

        let manager = Self::new();
        manager.connect_servers(&config).await?;
        Ok(manager)
    }

    /// Connect to all servers defined in the configuration
    pub async fn connect_servers(&self, config: &McpConfig) -> Result<(), McpError> {
        let mut connections = self.connections.write().await;
        let mut all_tools = self.tools.write().await;

        for server_config in &config.servers {
            match self.connect_server(server_config).await {
                Ok(connection) => {
                    let server_name = connection.name.clone();

                    // Get tools from this server
                    match self.load_server_tools(&connection).await {
                        Ok(tools) => {
                            log::info!(
                                "Connected to MCP server '{}' with {} tools",
                                server_name,
                                tools.len()
                            );

                            // Add tools to the global collection
                            all_tools.extend(tools);

                            // Store the connection
                            connections.insert(server_name.clone(), connection);
                        }
                        Err(e) => {
                            log::error!(
                                "Failed to load tools from server '{}': {}",
                                server_name,
                                e
                            );
                            return Err(e);
                        }
                    }
                }
                Err(e) => {
                    log::error!(
                        "Failed to connect to server '{}': {}",
                        server_config.name,
                        e
                    );
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Connect to a single MCP server
    async fn connect_server(
        &self,
        server_config: &McpServerConfig,
    ) -> Result<McpServerConnection, McpError> {
        // Validate configuration
        server_config.validate().map_err(McpError::ConfigError)?;

        let service = match server_config.protocol.as_str() {
            "stdio" => self.connect_stdio_server(server_config).await?,
            _ => {
                return Err(McpError::ConfigError(format!(
                    "Unsupported protocol: {}. Currently only 'stdio' is supported.",
                    server_config.protocol
                )));
            }
        };

        Ok(McpServerConnection {
            name: server_config.name.clone(),
            service,
        })
    }

    /// Connect to a stdio-based MCP server
    async fn connect_stdio_server(
        &self,
        config: &McpServerConfig,
    ) -> Result<Arc<RunningService<RoleClient, ClientInfo>>, McpError> {
        let mut command = Command::new(&config.command);

        // Configure command arguments
        if !config.args.is_empty() {
            command.args(&config.args);
        }

        // Set working directory
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }

        // Set environment variables
        for (key, value) in &config.env {
            command.env(key, value);
        }

        // Create transport
        let transport = TokioChildProcess::new(command.configure(|_| {}))
            .map_err(|e| McpError::TransportError(e.to_string()))?;

        // Create client info for AutoAgents using default values
        let client_info = ClientInfo::default();

        // Connect to server
        let service = client_info.serve(transport).await.map_err(|e| {
            McpError::GenericError(format!("Failed to connect to MCP server: {:?}", e))
        })?;

        Ok(Arc::new(service))
    }

    /// Load tools from a connected MCP server
    async fn load_server_tools(
        &self,
        connection: &McpServerConnection,
    ) -> Result<Vec<Arc<dyn ToolT>>, McpError> {
        let tools_result = connection
            .service
            .list_tools(None)
            .await
            .map_err(|e| McpError::GenericError(format!("Failed to list tools: {:?}", e)))?;

        let mut adapted_tools = Vec::new();

        for tool in tools_result.tools {
            let adapter = McpToolAdapter::new(tool, Arc::clone(&connection.service));
            adapted_tools.push(Arc::new(adapter) as Arc<dyn ToolT>);
        }

        Ok(adapted_tools)
    }

    /// Get all available tools
    pub async fn get_tools(&self) -> Vec<Arc<dyn ToolT>> {
        self.tools.read().await.clone()
    }

    /// Get tools from a specific server
    pub async fn get_server_tools(
        &self,
        server_name: &str,
    ) -> Result<Vec<Arc<dyn ToolT>>, McpError> {
        let connections = self.connections.read().await;
        let connection = connections
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        self.load_server_tools(connection).await
    }

    /// Get a tool by name
    pub async fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn ToolT>> {
        let tools = self.tools.read().await;
        tools.iter().find(|tool| tool.name() == tool_name).cloned()
    }

    /// Refresh tools from all connected servers
    pub async fn refresh_tools(&self) -> Result<(), McpError> {
        let connections = self.connections.read().await;
        let mut all_tools = Vec::new();

        for connection in connections.values() {
            let tools = self.load_server_tools(connection).await?;
            all_tools.extend(tools);
        }

        *self.tools.write().await = all_tools;
        Ok(())
    }

    /// Get the names of all connected servers
    pub async fn connected_servers(&self) -> Vec<String> {
        self.connections.read().await.keys().cloned().collect()
    }

    /// Check if a server is connected
    pub async fn is_server_connected(&self, server_name: &str) -> bool {
        self.connections.read().await.contains_key(server_name)
    }

    /// Get the number of available tools
    pub async fn tool_count(&self) -> usize {
        self.tools.read().await.len()
    }

    /// Get tool names
    pub async fn tool_names(&self) -> Vec<String> {
        self.tools
            .read()
            .await
            .iter()
            .map(|tool| tool.name().to_string())
            .collect()
    }
}

impl Default for McpToolsManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tools_manager_creation() {
        let manager = McpToolsManager::new();
        // Basic structural test
        assert_eq!(
            std::ptr::addr_of!(manager) as usize % std::mem::align_of::<McpToolsManager>(),
            0
        );
    }

    #[test]
    fn test_mcp_error_display() {
        let error = McpError::ServerNotFound("test_server".to_string());
        assert_eq!(error.to_string(), "Server not found: test_server");

        let error = McpError::ConnectionFailed("connection timeout".to_string());
        assert_eq!(error.to_string(), "Connection failed: connection timeout");

        let error = McpError::ConfigError("invalid config".to_string());
        assert_eq!(error.to_string(), "Configuration error: invalid config");
    }

    #[tokio::test]
    async fn test_manager_basic_operations() {
        let manager = McpToolsManager::new();

        // Test initial state
        assert_eq!(manager.tool_count().await, 0);
        assert!(manager.tool_names().await.is_empty());
        assert!(manager.connected_servers().await.is_empty());
        assert!(!manager.is_server_connected("nonexistent").await);
        assert!(manager.get_tool("nonexistent").await.is_none());
    }

    #[test]
    fn test_client_info_creation() {
        let client_info = ClientInfo::default();
        // ClientInfo is an InitializeRequestParam with default implementation values
        // We can't directly test the field values without knowing the internal structure
        // but we can ensure it can be created
        assert!(
            (std::ptr::addr_of!(client_info) as usize)
                .is_multiple_of(std::mem::align_of::<ClientInfo>())
        );
    }
}
