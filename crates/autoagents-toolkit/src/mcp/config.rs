use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// MCP configuration containing all servers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(rename = "server")]
    pub servers: Vec<McpServerConfig>,
    #[serde(skip)]
    base_dir: Option<PathBuf>,
}

/// Top-level configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub mcp: McpConfig,
}

/// MCP server configuration.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub enabled: bool,
    pub timeout_ms: u64,
    pub transport: McpServerTransport,
}

/// Supported MCP transports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerTransport {
    Local(McpLocalServerConfig),
    Remote(McpRemoteServerConfig),
    Unsupported { protocol: String },
}

/// Local stdio MCP server configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpLocalServerConfig {
    pub command: Vec<String>,
    pub cwd: Option<String>,
    pub environment: HashMap<String, String>,
}

/// Remote Streamable HTTP MCP server configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpRemoteServerConfig {
    pub url: String,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawMcpServerConfig {
    name: String,
    #[serde(rename = "type")]
    transport_type: Option<String>,
    protocol: Option<String>,
    command: Option<CommandValue>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default, alias = "environment")]
    env: HashMap<String, String>,
    cwd: Option<String>,
    url: Option<String>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default = "default_enabled")]
    enabled: bool,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    timeout: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CommandValue {
    String(String),
    Vec(Vec<String>),
}

fn default_enabled() -> bool {
    true
}

pub fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

impl McpConfig {
    /// Load MCP configuration from a TOML file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;
        config.mcp.base_dir = path.parent().map(Path::to_path_buf);
        Ok(config.mcp)
    }

    /// Create a new empty MCP configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a server to the configuration.
    pub fn add_server(&mut self, server: McpServerConfig) {
        self.servers.push(server);
    }

    /// Get a server by name.
    pub fn get_server(&self, name: &str) -> Option<&McpServerConfig> {
        self.servers.iter().find(|server| server.name == name)
    }

    /// List all server names.
    pub fn server_names(&self) -> Vec<&str> {
        self.servers
            .iter()
            .map(|server| server.name.as_str())
            .collect()
    }

    /// Base directory used for resolving relative MCP process paths.
    pub fn base_dir(&self) -> Option<&Path> {
        self.base_dir.as_deref()
    }

    /// Set the base directory used for resolving relative process paths.
    pub fn with_base_dir<P: Into<PathBuf>>(mut self, base_dir: P) -> Self {
        self.base_dir = Some(base_dir.into());
        self
    }
}

impl McpServerConfig {
    /// Create a backward-compatible local stdio server configuration.
    pub fn new(name: String, protocol: String, command: String) -> Self {
        let transport = if protocol == "stdio" {
            McpServerTransport::Local(McpLocalServerConfig {
                command: vec![command],
                cwd: None,
                environment: HashMap::new(),
            })
        } else {
            McpServerTransport::Unsupported { protocol }
        };

        Self {
            name,
            enabled: true,
            timeout_ms: default_timeout_ms(),
            transport,
        }
    }

    /// Create a local stdio MCP server configuration.
    pub fn local(name: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            timeout_ms: default_timeout_ms(),
            transport: McpServerTransport::Local(McpLocalServerConfig {
                command,
                cwd: None,
                environment: HashMap::new(),
            }),
        }
    }

    /// Create a remote Streamable HTTP MCP server configuration.
    pub fn remote(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            timeout_ms: default_timeout_ms(),
            transport: McpServerTransport::Remote(McpRemoteServerConfig {
                url: url.into(),
                headers: HashMap::new(),
            }),
        }
    }

    /// Set local command arguments for backward-compatible callers.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        if let McpServerTransport::Local(local) = &mut self.transport {
            local.command.extend(args);
        }
        self
    }

    /// Set local environment variables.
    pub fn with_env(mut self, env: HashMap<String, String>) -> Self {
        if let McpServerTransport::Local(local) = &mut self.transport {
            local.environment = env;
        }
        self
    }

    /// Set local working directory.
    pub fn with_cwd<P: AsRef<Path>>(mut self, cwd: P) -> Self {
        if let McpServerTransport::Local(local) = &mut self.transport {
            local.cwd = Some(cwd.as_ref().to_string_lossy().to_string());
        }
        self
    }

    /// Set timeout in seconds.
    ///
    /// This preserves the pre-typed-config builder API semantics. Use
    /// [`Self::with_timeout_ms`] when configuring millisecond values.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_ms = timeout_secs.saturating_mul(1_000);
        self
    }

    /// Set timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set whether this server is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set remote HTTP headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        if let McpServerTransport::Remote(remote) = &mut self.transport {
            remote.headers = headers;
        }
        self
    }

    /// Validate the server configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("MCP server name cannot be empty".to_string());
        }

        if self.timeout_ms == 0 {
            return Err(format!(
                "MCP server '{}' timeout_ms must be greater than zero",
                self.name
            ));
        }

        match &self.transport {
            McpServerTransport::Local(local) => {
                if local.command.is_empty() || local.command[0].trim().is_empty() {
                    return Err(format!(
                        "MCP local server '{}' command cannot be empty",
                        self.name
                    ));
                }
                Ok(())
            }
            McpServerTransport::Remote(remote) => {
                if remote.url.trim().is_empty() {
                    return Err(format!(
                        "MCP remote server '{}' url cannot be empty",
                        self.name
                    ));
                }
                remote
                    .url
                    .parse::<http::Uri>()
                    .map_err(|error| {
                        format!("MCP remote server '{}' url is invalid: {error}", self.name)
                    })
                    .and_then(|uri| match uri.scheme_str() {
                        Some("http" | "https") => Ok(()),
                        Some(scheme) => Err(format!(
                            "MCP remote server '{}' url scheme '{}' is unsupported",
                            self.name, scheme
                        )),
                        None => Err(format!(
                            "MCP remote server '{}' url must include http or https scheme",
                            self.name
                        )),
                    })
            }
            McpServerTransport::Unsupported { protocol } => Err(format!(
                "Unsupported MCP protocol '{}'. Use type = 'local' for stdio or type = 'remote' for Streamable HTTP.",
                protocol
            )),
        }
    }

    pub fn local_config(&self) -> Option<&McpLocalServerConfig> {
        match &self.transport {
            McpServerTransport::Local(local) => Some(local),
            _ => None,
        }
    }

    pub fn remote_config(&self) -> Option<&McpRemoteServerConfig> {
        match &self.transport {
            McpServerTransport::Remote(remote) => Some(remote),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawMcpServerConfig::deserialize(deserializer)?;
        let timeout_ms = raw
            .timeout_ms
            .or_else(|| {
                raw.timeout
                    .map(|timeout_secs| timeout_secs.saturating_mul(1_000))
            })
            .unwrap_or_else(default_timeout_ms);

        let transport = match raw.transport_type.as_deref() {
            Some("local") => {
                let command = command_vec(raw.command, raw.args)
                    .map_err(<D::Error as serde::de::Error>::custom)?;
                McpServerTransport::Local(McpLocalServerConfig {
                    command,
                    cwd: raw.cwd,
                    environment: raw.env,
                })
            }
            Some("remote") => McpServerTransport::Remote(McpRemoteServerConfig {
                url: raw.url.ok_or_else(|| {
                    <D::Error as serde::de::Error>::custom("remote MCP server requires url")
                })?,
                headers: raw.headers,
            }),
            Some(other) => McpServerTransport::Unsupported {
                protocol: other.to_string(),
            },
            None => match raw.protocol.as_deref() {
                Some("stdio") => {
                    let command = command_vec(raw.command, raw.args)
                        .map_err(<D::Error as serde::de::Error>::custom)?;
                    McpServerTransport::Local(McpLocalServerConfig {
                        command,
                        cwd: raw.cwd,
                        environment: raw.env,
                    })
                }
                Some(protocol @ ("http" | "https" | "streamable_http")) => {
                    let url = raw.url.ok_or_else(|| {
                        <D::Error as serde::de::Error>::custom(format!(
                            "remote MCP protocol '{protocol}' requires url"
                        ))
                    })?;
                    McpServerTransport::Remote(McpRemoteServerConfig {
                        url,
                        headers: raw.headers,
                    })
                }
                Some(protocol) => McpServerTransport::Unsupported {
                    protocol: protocol.to_string(),
                },
                None => {
                    return Err(<D::Error as serde::de::Error>::custom(
                        "MCP server requires type or protocol",
                    ));
                }
            },
        };

        Ok(Self {
            name: raw.name,
            enabled: raw.enabled,
            timeout_ms,
            transport,
        })
    }
}

impl Serialize for McpServerConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut fields = match &self.transport {
            McpServerTransport::Local(_) => serializer.serialize_struct("McpServerConfig", 7)?,
            McpServerTransport::Remote(_) => serializer.serialize_struct("McpServerConfig", 6)?,
            McpServerTransport::Unsupported { .. } => {
                serializer.serialize_struct("McpServerConfig", 4)?
            }
        };
        fields.serialize_field("name", &self.name)?;
        fields.serialize_field("enabled", &self.enabled)?;
        fields.serialize_field("timeout_ms", &self.timeout_ms)?;
        match &self.transport {
            McpServerTransport::Local(local) => {
                fields.serialize_field("type", "local")?;
                fields.serialize_field("command", &local.command)?;
                fields.serialize_field("cwd", &local.cwd)?;
                fields.serialize_field("environment", &local.environment)?;
            }
            McpServerTransport::Remote(remote) => {
                fields.serialize_field("type", "remote")?;
                fields.serialize_field("url", &remote.url)?;
                fields.serialize_field("headers", &remote.headers)?;
            }
            McpServerTransport::Unsupported { protocol } => {
                fields.serialize_field("protocol", protocol)?;
            }
        }
        fields.end()
    }
}

fn command_vec(command: Option<CommandValue>, args: Vec<String>) -> Result<Vec<String>, String> {
    let mut command = match command {
        Some(CommandValue::String(command)) => vec![command],
        Some(CommandValue::Vec(command)) => command,
        None => return Err("local MCP server requires command".to_string()),
    };
    command.extend(args);
    Ok(command)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, tempdir};

    #[test]
    fn creates_local_config_with_legacy_constructor() {
        let config = McpServerConfig::new(
            "test_server".to_string(),
            "stdio".to_string(),
            "python".to_string(),
        );

        assert_eq!(config.name, "test_server");
        assert_eq!(config.timeout_ms, 30_000);
        let local = config.local_config().expect("local config");
        assert_eq!(local.command, vec!["python"]);
        assert!(local.environment.is_empty());
    }

    #[test]
    fn local_builder_sets_options() -> std::io::Result<()> {
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

        let local = config.local_config().expect("local config");
        assert_eq!(local.command, vec!["python", "-m", "my_server"]);
        assert_eq!(local.environment, env);
        assert_eq!(local.cwd, Some(cwd));
        assert_eq!(config.timeout_ms, 60_000);
        Ok(())
    }

    #[test]
    fn validates_config() {
        assert!(
            McpServerConfig::new(
                "valid".to_string(),
                "stdio".to_string(),
                "python".to_string()
            )
            .validate()
            .is_ok()
        );

        assert!(
            McpServerConfig::new("".to_string(), "stdio".to_string(), "python".to_string())
                .validate()
                .is_err()
        );

        assert!(
            McpServerConfig::new("test".to_string(), "stdio".to_string(), "".to_string())
                .validate()
                .is_err()
        );

        assert!(
            McpServerConfig::new("test".to_string(), "sse".to_string(), "node".to_string())
                .validate()
                .is_err()
        );

        assert!(
            McpServerConfig::remote("remote", "https://example.com/mcp")
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn config_operations_work() {
        let mut config = McpConfig::default();
        assert!(config.servers.is_empty());
        assert!(config.server_names().is_empty());

        config.add_server(McpServerConfig::local(
            "test_server",
            vec!["python".to_string()],
        ));

        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.server_names(), vec!["test_server"]);
        assert!(config.get_server("test_server").is_some());
        assert!(config.get_server("nonexistent").is_none());
    }

    #[test]
    fn parses_new_and_legacy_toml() {
        let toml_content = r#"
[mcp]
[[mcp.server]]
name = "test_server"
protocol = "stdio"
command = "python"
args = ["-m", "test_module"]
timeout = 45

[[mcp.server]]
name = "remote_server"
type = "remote"
url = "https://example.com/mcp"
timeout_ms = 5000

[mcp.server.headers]
Authorization = "Bearer token"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let config = McpConfig::from_file(temp_file.path()).unwrap();
        assert_eq!(config.servers.len(), 2);
        assert_eq!(config.base_dir(), temp_file.path().parent());

        let first_server = config.get_server("test_server").unwrap();
        let first_local = first_server.local_config().unwrap();
        assert_eq!(first_local.command, vec!["python", "-m", "test_module"]);
        assert_eq!(first_server.timeout_ms, 45_000);

        let second_server = config.get_server("remote_server").unwrap();
        let remote = second_server.remote_config().unwrap();
        assert_eq!(remote.url, "https://example.com/mcp");
        assert_eq!(
            remote.headers.get("Authorization"),
            Some(&"Bearer token".to_string())
        );
    }

    #[test]
    fn rejects_invalid_toml() {
        let invalid_toml = r#"
[mcp]
[[mcp.server]]
name = "incomplete"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(invalid_toml.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let result = McpConfig::from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn parses_environment_aliases() {
        let dir = tempdir().unwrap();
        let cwd_str = dir.path().to_str().unwrap();

        let toml_content = format!(
            r#"
[mcp]
[[mcp.server]]
name = "env_server"
type = "local"
command = ["python", "-m", "server"]
cwd = "{cwd}"

[mcp.server.environment]
PYTHONPATH = "/path/to/modules"
DEBUG = "1"
        "#,
            cwd = cwd_str
        );

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let config = McpConfig::from_file(temp_file.path()).unwrap();
        let local = config
            .get_server("env_server")
            .unwrap()
            .local_config()
            .unwrap();

        assert_eq!(local.cwd, Some(cwd_str.to_string()));
        assert_eq!(
            local.environment.get("PYTHONPATH"),
            Some(&"/path/to/modules".to_string())
        );
        assert_eq!(local.environment.get("DEBUG"), Some(&"1".to_string()));
    }

    #[test]
    fn serializes_config() {
        let mut env = HashMap::default();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());

        let server = McpServerConfig::local(
            "serialize_test",
            vec!["python".to_string(), "-m".to_string(), "server".to_string()],
        )
        .with_env(env);

        let mut config = McpConfig::default();
        config.add_server(server);

        let serialized = toml::to_string(&Config {
            mcp: config.clone(),
        })
        .unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();

        assert_eq!(config.servers.len(), deserialized.mcp.servers.len());
        let original = config.servers[0].local_config().unwrap();
        let parsed = deserialized.mcp.servers[0].local_config().unwrap();

        assert_eq!(original.command, parsed.command);
        assert_eq!(original.environment, parsed.environment);
    }
}
