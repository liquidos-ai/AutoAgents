use crate::mcp::policy::McpSecurityPolicy;
use crate::mcp::security::{
    looks_like_path, resolve_path, validate_command_allowlist, validate_command_is_bare_name,
    validate_http_url, validate_resolved_path_within_base,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// MCP Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name
    pub name: String,
    /// Protocol type (`stdio`, `http`, or legacy `sse`)
    pub protocol: String,
    /// Launcher command for stdio transport (bare name only, e.g. `python3`)
    #[serde(default)]
    pub command: String,
    /// Arguments for the stdio launcher
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables for stdio transport
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Working directory for stdio transport
    #[serde(default)]
    pub cwd: Option<String>,
    /// Remote MCP endpoint for `http` / `sse` transports
    #[serde(default)]
    pub url: Option<String>,
    /// HTTP headers for `http` / `sse` transports
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Per-operation timeout in seconds (connect, list_tools, call_tool)
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

/// MCP configuration containing all servers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    /// Load MCP configuration from a TOML file with secure-default validation.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_file_with_policy(path, &McpSecurityPolicy::secure_default())
    }

    /// Load MCP configuration from a TOML file using the provided security policy.
    pub fn from_file_with_policy<P: AsRef<Path>>(
        path: P,
        policy: &McpSecurityPolicy,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (mcp, base_dir) = Self::load_from_file(path)?;
        mcp.validate_all(policy, Some(&base_dir))?;
        Ok(mcp)
    }

    /// Parse and resolve a configuration file without validation.
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Self, PathBuf), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        let mut mcp = config.mcp;
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        mcp.resolve_paths(&base_dir);
        Ok((mcp, base_dir))
    }

    /// Create a new empty MCP configuration
    pub fn new() -> Self {
        Self::default()
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

    /// Resolve relative filesystem paths against the config file directory.
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        for server in &mut self.servers {
            server.resolve_paths(base_dir);
        }
    }

    /// Validate every server entry.
    pub fn validate_all(
        &self,
        policy: &McpSecurityPolicy,
        config_base: Option<&Path>,
    ) -> Result<(), String> {
        for server in &self.servers {
            server.validate(policy, config_base)?;
        }
        Ok(())
    }
}

impl McpServerConfig {
    /// Create a new stdio server configuration
    pub fn new(name: String, protocol: String, command: String) -> Self {
        Self {
            name,
            protocol,
            command,
            args: Vec::new(),
            env: HashMap::new(),
            cwd: None,
            url: None,
            headers: HashMap::new(),
            timeout: default_timeout(),
        }
    }

    /// Create a new remote HTTP server configuration
    pub fn new_http(name: String, url: String) -> Self {
        Self {
            name,
            protocol: "http".to_string(),
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            cwd: None,
            url: Some(url),
            headers: HashMap::new(),
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

    /// Set remote URL for HTTP/SSE transports
    pub fn with_url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    /// Set HTTP headers for remote transports
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    /// Resolve relative paths for stdio launch parameters.
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        if let Some(cwd) = &self.cwd {
            self.cwd = Some(resolve_path(base_dir, cwd));
        }

        self.args = self
            .args
            .iter()
            .map(|arg| {
                if looks_like_path(arg) {
                    resolve_path(base_dir, arg)
                } else {
                    arg.clone()
                }
            })
            .collect();
    }

    /// Validate the server configuration under the given security policy.
    pub fn validate(
        &self,
        policy: &McpSecurityPolicy,
        config_base: Option<&Path>,
    ) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Server name cannot be empty".to_string());
        }

        if self.timeout == 0 {
            return Err("Timeout must be greater than zero".to_string());
        }

        match self.protocol.as_str() {
            "stdio" => self.validate_stdio(policy, config_base),
            "http" | "sse" => self.validate_http(policy),
            other => Err(format!("Unsupported protocol: {other}")),
        }
    }

    fn validate_stdio(
        &self,
        policy: &McpSecurityPolicy,
        config_base: Option<&Path>,
    ) -> Result<(), String> {
        if self.command.is_empty() {
            return Err("command is required for stdio transport".to_string());
        }

        if self.url.is_some() {
            return Err("url must not be set for stdio transport".to_string());
        }

        validate_command_is_bare_name(&self.command).map_err(|e| e.to_string())?;

        if !policy.defers_allowlist_to_approver() {
            validate_command_allowlist(&self.command, policy.allowed_commands())
                .map_err(|e| e.to_string())?;
        }

        policy
            .validate_stdio_fields(&self.stdio_launch_spec(), config_base)
            .map_err(|e| e.to_string())?;

        if let Some(base) = config_base {
            for arg in &self.args {
                if looks_like_path(arg) {
                    validate_resolved_path_within_base(arg, base).map_err(|e| e.to_string())?;
                }
            }
        }

        Ok(())
    }

    fn validate_http(&self, policy: &McpSecurityPolicy) -> Result<(), String> {
        if !self.command.is_empty() {
            return Err("command must not be set for http/sse transport".to_string());
        }

        if !self.args.is_empty() {
            return Err("args must not be set for http/sse transport".to_string());
        }

        if self.cwd.is_some() {
            return Err("cwd must not be set for http/sse transport".to_string());
        }

        if !self.env.is_empty() {
            return Err("env must not be set for http/sse transport".to_string());
        }

        let url = self
            .url
            .as_ref()
            .filter(|url| !url.is_empty())
            .ok_or_else(|| "url is required for http/sse transport".to_string())?;

        validate_http_url(url, policy.allow_private_http_endpoints()).map_err(|e| e.to_string())
    }

    /// Build a launch spec for stdio authorization at connect time.
    pub fn stdio_launch_spec(&self) -> crate::mcp::policy::McpProcessLaunchSpec {
        crate::mcp::policy::McpProcessLaunchSpec {
            server_name: self.name.clone(),
            command: self.command.clone(),
            args: self.args.clone(),
            cwd: self.cwd.clone(),
            env: self.env.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tempfile::tempdir;

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
        assert!(
            valid_config
                .validate(&McpSecurityPolicy::secure_default(), None)
                .is_ok()
        );

        let empty_name =
            McpServerConfig::new("".to_string(), "stdio".to_string(), "python".to_string());
        assert!(
            empty_name
                .validate(&McpSecurityPolicy::secure_default(), None)
                .is_err()
        );

        let empty_command =
            McpServerConfig::new("test".to_string(), "stdio".to_string(), "".to_string());
        assert!(
            empty_command
                .validate(&McpSecurityPolicy::secure_default(), None)
                .is_err()
        );

        let invalid_protocol = McpServerConfig::new(
            "test".to_string(),
            "invalid".to_string(),
            "python".to_string(),
        );
        assert!(
            invalid_protocol
                .validate(&McpSecurityPolicy::secure_default(), None)
                .is_err()
        );
    }

    #[test]
    fn test_http_config_requires_url() {
        let config = McpServerConfig::new_http("remote".to_string(), String::new());
        let err = config
            .validate(&McpSecurityPolicy::secure_default(), None)
            .unwrap_err();
        assert!(err.contains("url is required"));
    }

    #[test]
    fn test_http_config_rejects_command() {
        let mut config =
            McpServerConfig::new_http("remote".to_string(), "https://example.com/mcp".to_string());
        config.command = "curl".to_string();
        let err = config
            .validate(&McpSecurityPolicy::secure_default(), None)
            .unwrap_err();
        assert!(err.contains("command must not be set"));
    }

    #[test]
    fn test_stdio_rejects_url_field() {
        let mut config = McpServerConfig::new(
            "stdio".to_string(),
            "stdio".to_string(),
            "python3".to_string(),
        );
        config.url = Some("https://example.com".to_string());
        let err = config
            .validate(&McpSecurityPolicy::secure_default(), None)
            .unwrap_err();
        assert!(err.contains("url must not be set"));
    }

    #[test]
    fn test_mcp_config_operations() {
        let mut config = McpConfig::default();
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
name = "remote"
protocol = "http"
url = "https://example.com/mcp"
timeout = 60
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

        let second_server = config.get_server("remote").unwrap();
        assert_eq!(second_server.protocol, "http");
        assert_eq!(
            second_server.url.as_deref(),
            Some("https://example.com/mcp")
        );
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

        let toml_content = format!(
            r#"
[mcp]
[[mcp.server]]
name = "env_server"
protocol = "stdio"
command = "python"
cwd = "{cwd}"

[mcp.server.env]
DEBUG = "1"
        "#,
            cwd = cwd_str
        );

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let config = McpConfig::from_file(temp_file.path()).unwrap();
        let server = config.get_server("env_server").unwrap();

        assert_eq!(server.cwd, Some(cwd_str.to_string()));
        assert_eq!(server.env.get("DEBUG"), Some(&"1".to_string()));
    }

    #[test]
    fn test_resolve_relative_script_path() {
        let dir = tempdir().unwrap();
        let base = dir.path().to_path_buf();
        let mut config = McpServerConfig::new(
            "echo".to_string(),
            "stdio".to_string(),
            "python3".to_string(),
        )
        .with_args(vec!["servers/echo_server.py".to_string()]);
        config.resolve_paths(&base);
        assert_eq!(
            config.args[0],
            base.join("servers/echo_server.py").to_string_lossy()
        );
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
        assert!(server.url.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let mut env = HashMap::default();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());

        let server = McpServerConfig::new(
            "serialize_test".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        )
        .with_args(vec!["-m".to_string(), "server".to_string()])
        .with_env(env);

        let mut config = McpConfig::default();
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
