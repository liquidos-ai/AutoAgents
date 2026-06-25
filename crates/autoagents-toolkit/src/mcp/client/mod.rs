mod http;
mod stdio;
mod timeout;

use crate::mcp::{
    adapter::McpToolAdapter,
    config::{McpConfig, McpServerConfig},
    policy::McpSecurityPolicy,
    security::McpSecurityError,
};
use autoagents::core::tool::ToolT;
use rmcp::{
    model::ClientInfo,
    service::{RoleClient, RunningService},
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub use timeout::with_timeout;

/// Represents a connection to an MCP server
#[derive(Debug)]
pub struct McpServerConnection {
    pub name: String,
    pub service: Arc<RunningService<RoleClient, ClientInfo>>,
    pub timeout: Duration,
}

/// MCP Tools Manager that handles multiple MCP server connections
pub struct McpToolsManager {
    connections: Arc<RwLock<HashMap<String, McpServerConnection>>>,
    tools: Arc<RwLock<Vec<Arc<dyn ToolT>>>>,
    security_policy: McpSecurityPolicy,
    config_base: Option<PathBuf>,
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
    #[error("MCP operation '{operation}' timed out after {millis}ms")]
    Timeout {
        operation: &'static str,
        millis: u64,
    },
    #[error("Security error: {0}")]
    Security(#[from] McpSecurityError),
    #[error("Rmcp error: {0}")]
    RmcpError(#[from] rmcp::ErrorData),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Generic error: {0}")]
    GenericError(String),
}

impl std::fmt::Debug for McpToolsManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolsManager")
            .field("security_policy", &self.security_policy)
            .field("config_base", &self.config_base)
            .finish_non_exhaustive()
    }
}

impl Default for McpToolsManager {
    fn default() -> Self {
        Self::with_security_policy_and_base(McpSecurityPolicy::default(), None)
    }
}

impl McpToolsManager {
    /// Create a new MCP tools manager with secure-default stdio policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a manager with a custom security policy.
    pub fn with_security_policy(security_policy: McpSecurityPolicy) -> Self {
        Self::with_security_policy_and_base(security_policy, None)
    }

    /// Create a manager with a custom security policy and config base directory.
    pub fn with_security_policy_and_base(
        security_policy: McpSecurityPolicy,
        config_base: Option<PathBuf>,
    ) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            tools: Arc::new(RwLock::new(Vec::new())),
            security_policy,
            config_base,
        }
    }

    /// Return the active security policy.
    pub fn security_policy(&self) -> &McpSecurityPolicy {
        &self.security_policy
    }

    /// Return the config base directory used for path and cwd validation.
    pub fn config_base(&self) -> Option<&Path> {
        self.config_base.as_deref()
    }

    /// Load MCP configuration from a file and connect to all servers.
    pub async fn from_config_file<P: AsRef<Path>>(path: P) -> Result<Self, McpError> {
        Self::from_config_file_with_policy(path, McpSecurityPolicy::secure_default()?).await
    }

    /// Load MCP configuration from a file using a custom security policy.
    pub async fn from_config_file_with_policy<P: AsRef<Path>>(
        path: P,
        security_policy: McpSecurityPolicy,
    ) -> Result<Self, McpError> {
        let (mcp, base_dir) =
            McpConfig::load_from_file(path).map_err(|e| McpError::ConfigError(e.to_string()))?;

        mcp.validate_all(&security_policy, Some(&base_dir))
            .map_err(McpError::ConfigError)?;

        let manager = Self::with_security_policy_and_base(security_policy, Some(base_dir));
        manager.connect_servers(&mcp).await?;
        Ok(manager)
    }

    /// Connect to all servers defined in the configuration.
    pub async fn connect_servers(&self, config: &McpConfig) -> Result<(), McpError> {
        config
            .validate_all(&self.security_policy, self.config_base())
            .map_err(McpError::ConfigError)?;

        let mut pending_connections = Vec::new();
        let mut pending_tools = Vec::new();

        for server_config in &config.servers {
            let connection = match self.connect_server(server_config).await {
                Ok(connection) => connection,
                Err(e) => {
                    log::error!(
                        "Failed to connect to server '{}': {}",
                        server_config.name,
                        e
                    );
                    return Err(e);
                }
            };

            let server_name = connection.name.clone();
            match self.load_server_tools(&connection).await {
                Ok(tools) => {
                    log::info!(
                        "Connected to MCP server '{}' with {} tools",
                        server_name,
                        tools.len()
                    );
                    pending_connections.push(connection);
                    pending_tools.extend(tools);
                }
                Err(e) => {
                    log::error!("Failed to load tools from server '{}': {}", server_name, e);
                    return Err(e);
                }
            }
        }

        let mut connections = self.connections.write().await;
        let mut all_tools = self.tools.write().await;
        for connection in pending_connections {
            connections.insert(connection.name.clone(), connection);
        }
        all_tools.extend(pending_tools);

        Ok(())
    }

    async fn connect_server(
        &self,
        server_config: &McpServerConfig,
    ) -> Result<McpServerConnection, McpError> {
        server_config
            .validate(&self.security_policy, self.config_base())
            .map_err(McpError::ConfigError)?;

        let timeout = Duration::from_secs(server_config.timeout);

        let service = match server_config.protocol.as_str() {
            "stdio" => {
                stdio::connect_stdio_server(
                    server_config,
                    &self.security_policy,
                    self.config_base(),
                )
                .await?
            }
            "http" => {
                http::connect_http_server(
                    server_config,
                    false,
                    self.security_policy.allow_private_http_endpoints(),
                )
                .await?
            }
            "sse" => {
                http::connect_http_server(
                    server_config,
                    true,
                    self.security_policy.allow_private_http_endpoints(),
                )
                .await?
            }
            other => {
                return Err(McpError::ConfigError(format!(
                    "Unsupported protocol: {other}"
                )));
            }
        };

        Ok(McpServerConnection {
            name: server_config.name.clone(),
            service,
            timeout,
        })
    }

    async fn load_server_tools(
        &self,
        connection: &McpServerConnection,
    ) -> Result<Vec<Arc<dyn ToolT>>, McpError> {
        let service = Arc::clone(&connection.service);
        let timeout = connection.timeout;

        let tools_result = with_timeout(timeout, "list_tools", async {
            service
                .list_tools(None)
                .await
                .map_err(|e| McpError::GenericError(format!("Failed to list tools: {e:?}")))
        })
        .await?;

        let mut adapted_tools = Vec::new();

        for tool in tools_result.tools {
            let adapter = McpToolAdapter::new(tool, Arc::clone(&connection.service), timeout);
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
        let (service, timeout) = {
            let connections = self.connections.read().await;
            let connection = connections
                .get(server_name)
                .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;
            (Arc::clone(&connection.service), connection.timeout)
        };

        let tools_result = with_timeout(timeout, "list_tools", async {
            service
                .list_tools(None)
                .await
                .map_err(|e| McpError::GenericError(format!("Failed to list tools: {e:?}")))
        })
        .await?;

        let mut adapted_tools = Vec::new();
        for tool in tools_result.tools {
            let adapter = McpToolAdapter::new(tool, service.clone(), timeout);
            adapted_tools.push(Arc::new(adapter) as Arc<dyn ToolT>);
        }

        Ok(adapted_tools)
    }

    /// Get a tool by name
    pub async fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn ToolT>> {
        let tools = self.tools.read().await;
        tools.iter().find(|tool| tool.name() == tool_name).cloned()
    }

    /// Refresh tools from all connected servers
    pub async fn refresh_tools(&self) -> Result<(), McpError> {
        let connections_snapshot: Vec<McpServerConnection> = self
            .connections
            .read()
            .await
            .values()
            .map(|connection| McpServerConnection {
                name: connection.name.clone(),
                service: Arc::clone(&connection.service),
                timeout: connection.timeout,
            })
            .collect();

        let mut all_tools = Vec::new();
        for connection in &connections_snapshot {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::policy::{McpProcessApprover, McpProcessLaunchSpec};
    use autoagents::async_trait;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_manager_basic_operations() {
        let manager = McpToolsManager::default();

        assert_eq!(manager.tool_count().await, 0);
        assert!(manager.tool_names().await.is_empty());
        assert!(manager.connected_servers().await.is_empty());
        assert!(!manager.is_server_connected("nonexistent").await);
        assert!(manager.get_tool("nonexistent").await.is_none());
    }

    fn invalid_protocol_config() -> McpServerConfig {
        McpServerConfig::new(
            "bad_server".to_string(),
            "websocket".to_string(),
            "noop".to_string(),
        )
    }

    fn http_without_url_config() -> McpServerConfig {
        McpServerConfig::new_http("remote".to_string(), String::default())
    }

    #[tokio::test]
    async fn test_get_server_tools_missing() {
        let manager = McpToolsManager::default();
        let err = manager.get_server_tools("missing").await.unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound(_)));
    }

    #[tokio::test]
    async fn test_connect_server_rejects_invalid_protocol() {
        let manager = McpToolsManager::default();
        let config = invalid_protocol_config();
        let err = manager.connect_server(&config).await.unwrap_err();
        assert!(matches!(err, McpError::ConfigError(_)));
        assert!(err.to_string().contains("Unsupported protocol"));
    }

    #[tokio::test]
    async fn test_connect_server_rejects_http_without_url() {
        let manager = McpToolsManager::default();
        let config = http_without_url_config();
        let err = manager.connect_server(&config).await.unwrap_err();
        assert!(matches!(err, McpError::ConfigError(_)));
        assert!(err.to_string().contains("url is required"));
    }

    #[tokio::test]
    async fn test_connect_servers_returns_error_on_invalid_protocol() {
        let manager = McpToolsManager::default();
        let mut config = McpConfig::new();
        config.add_server(invalid_protocol_config());
        let err = manager.connect_servers(&config).await.unwrap_err();
        assert!(err.to_string().contains("Unsupported protocol"));
    }

    #[tokio::test]
    async fn test_connect_servers_leaves_manager_empty_on_failure() {
        let manager = McpToolsManager::default();
        let mut config = McpConfig::new();
        config.add_server(invalid_protocol_config());
        let _ = manager.connect_servers(&config).await.unwrap_err();
        assert_eq!(manager.tool_count().await, 0);
        assert!(manager.connected_servers().await.is_empty());
    }

    #[tokio::test]
    async fn test_refresh_tools_on_empty_manager() {
        let manager = McpToolsManager::default();
        manager.refresh_tools().await.unwrap();
        let tools = manager.get_tools().await;
        assert!(tools.is_empty());
    }

    struct AlwaysApprove;

    #[async_trait]
    impl McpProcessApprover for AlwaysApprove {
        async fn approve(&self, _launch: &McpProcessLaunchSpec) -> Result<(), McpSecurityError> {
            Ok(())
        }
    }

    #[test]
    fn validate_allows_non_allowlisted_command_when_approver_configured() {
        let policy = McpSecurityPolicy::secure_default()
            .unwrap()
            .with_approver(Arc::new(AlwaysApprove));
        let config = McpServerConfig::new(
            "custom".to_string(),
            "stdio".to_string(),
            "bash".to_string(),
        );
        assert!(config.validate(&policy, None).is_ok());
    }
}
