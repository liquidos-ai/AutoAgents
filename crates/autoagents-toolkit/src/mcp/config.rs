use tempfile::tempdir;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// MCP Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name
    pub name: String,
    /// Protocol type (stdio, sse, etc.)
    pub protocol: String,
    /// Command to execute
    pub command: String,
    /// Arguments for the command
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Working directory
    #[serde(default)]
    pub cwd: Option<String>,
    /// Connection timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

/// MCP configuration containing all servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(rename = "server")]
    pub servers: Vec<McpServerConfig>,
}

/// Top-level configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub mcp: McpConfig,
}

fn default_timeout() -> u64 {
    30
}

impl McpConfig {
    /// Load MCP configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config.mcp)
    }

    /// Create a new empty MCP configuration
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
        }
    }

    /// Add a server to the configuration
    pub fn add_server(&mut self, server: McpServerConfig) {
        self.servers.push(server);
    }

    /// Get a server by name
    pub fn get_server(&self, name: &str) -> Option<&McpServerConfig> {
        self.servers.iter().find(|s| s.name == name)
    }

    /// List all server names
    pub fn server_names(&self) -> Vec<&str> {
        self.servers.iter().map(|s| s.name.as_str()).collect()
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServerConfig {
    /// Create a new server configuration
    pub fn new(name: String, protocol: String, command: String) -> Self {
        Self {
            name,
            protocol,
            command,
            args: Vec::new(),
            env: HashMap::new(),
            cwd: None,
            timeout: default_timeout(),
        }
    }

    /// Set command arguments
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set environment variables
    pub fn with_env(mut self, env: HashMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Set working directory
    pub fn with_cwd<P: AsRef<Path>>(mut self, cwd: P) -> Self {
        self.cwd = Some(cwd.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = timeout;
        self
    }

    /// Validate the server configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Server name cannot be empty".to_string());
        }

        if self.command.is_empty() {
            return Err("Server command cannot be empty".to_string());
        }

        match self.protocol.as_str() {
            "stdio" | "sse" | "http" => Ok(()),
            _ => Err(format!("Unsupported protocol: {}", self.protocol)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mcp_server_config_creation() {
        let config = McpServerConfig::new(
            "test_server".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        );

        assert_eq!(config.name, "test_server");
        assert_eq!(config.protocol, "stdio");
        assert_eq!(config.command, "python");
        assert!(config.args.is_empty());
        assert!(config.env.is_empty());
        assert_eq!(config.timeout, 30);
    }

    #[test]
    fn test_mcp_server_config_builder() -> std::io::Result<()> {
        let dir = tempdir()?;
        let cwd = dir.path().to_str().unwrap().to_string();
        let mut env = HashMap::new();
        env.insert("PYTHONPATH".to_string(), "/path/to/modules".to_string());

        let config = McpServerConfig::new(
            "builder_test".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        )
        .with_args(vec!["-m".to_string(), "my_server".to_string()])
        .with_env(env.clone())
        .with_cwd(cwd.clone())
        .with_timeout(60);

        assert_eq!(config.name, "builder_test");
        assert_eq!(config.args, vec!["-m", "my_server"]);
        assert_eq!(config.env, env);
        assert_eq!(config.cwd, Some(cwd));
        assert_eq!(config.timeout, 60);
        Ok(())
    }

    #[test]
    fn test_mcp_server_config_validation() {
        let valid_config = McpServerConfig::new(
            "valid".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        );
        assert!(valid_config.validate().is_ok());

        let empty_name =
            McpServerConfig::new("".to_string(), "stdio".to_string(), "python".to_string());
        assert!(empty_name.validate().is_err());

        let empty_command =
            McpServerConfig::new("test".to_string(), "stdio".to_string(), "".to_string());
        assert!(empty_command.validate().is_err());

        let invalid_protocol = McpServerConfig::new(
            "test".to_string(),
            "invalid".to_string(),
            "python".to_string(),
        );
        assert!(invalid_protocol.validate().is_err());
    }

    #[test]
    fn test_mcp_config_operations() {
        let mut config = McpConfig::new();
        assert!(config.servers.is_empty());
        assert!(config.server_names().is_empty());

        let server = McpServerConfig::new(
            "test_server".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        );
        config.add_server(server);

        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.server_names(), vec!["test_server"]);
        assert!(config.get_server("test_server").is_some());
        assert!(config.get_server("nonexistent").is_none());
    }

    #[test]
    fn test_config_from_toml() {
        let toml_content = r#"
[mcp]
[[mcp.server]]
name = "test_server"
protocol = "stdio"
command = "python"
args = ["-m", "test_module"]
timeout = 45

[[mcp.server]]
name = "another_server"
protocol = "sse"
command = "node"
args = ["server.js"]
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let config = McpConfig::from_file(temp_file.path()).unwrap();
        assert_eq!(config.servers.len(), 2);

        let first_server = config.get_server("test_server").unwrap();
        assert_eq!(first_server.protocol, "stdio");
        assert_eq!(first_server.command, "python");
        assert_eq!(first_server.args, vec!["-m", "test_module"]);
        assert_eq!(first_server.timeout, 45);

        let second_server = config.get_server("another_server").unwrap();
        assert_eq!(second_server.protocol, "sse");
        assert_eq!(second_server.command, "node");
        assert_eq!(second_server.args, vec!["server.js"]);
    }

    #[test]
    fn test_config_from_invalid_toml() {
        let invalid_toml = r#"
[mcp]
[[mcp.server]]
name = "incomplete"
# Missing required fields
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(invalid_toml.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let result = McpConfig::from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_env_vars() {
        let dir = tempdir().unwrap();
        let cwd_str = dir.path().to_str().unwrap();

        let toml_content = format!(r#"
[mcp]
[[mcp.server]]
name = "env_server"
protocol = "stdio"
command = "python"
cwd = "{cwd}"

[mcp.server.env]
PYTHONPATH = "/path/to/modules"
DEBUG = "1"
        "#, cwd = cwd_str);

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let config = McpConfig::from_file(temp_file.path()).unwrap();
        let server = config.get_server("env_server").unwrap();

        assert_eq!(server.cwd, Some(cwd_str.to_string()));
        assert_eq!(
            server.env.get("PYTHONPATH"),
            Some(&"/path/to/modules".to_string())
        );
        assert_eq!(server.env.get("DEBUG"), Some(&"1".to_string()));
    }

    #[test]
    fn test_default_values() {
        let server = McpServerConfig::new(
            "default_test".to_string(),
            "stdio".to_string(),
            "command".to_string(),
        );

        assert_eq!(server.timeout, 30);
        assert!(server.args.is_empty());
        assert!(server.env.is_empty());
        assert!(server.cwd.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let mut env = HashMap::new();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());

        let server = McpServerConfig::new(
            "serialize_test".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        )
        .with_args(vec!["-m".to_string(), "server".to_string()])
        .with_env(env);

        let mut config = McpConfig::new();
        config.add_server(server);

        let serialized = toml::to_string(&Config {
            mcp: config.clone(),
        })
        .unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();

        assert_eq!(config.servers.len(), deserialized.mcp.servers.len());
        let original = &config.servers[0];
        let parsed = &deserialized.mcp.servers[0];

        assert_eq!(original.name, parsed.name);
        assert_eq!(original.protocol, parsed.protocol);
        assert_eq!(original.command, parsed.command);
        assert_eq!(original.args, parsed.args);
        assert_eq!(original.env, parsed.env);
    }
}
