use crate::mcp::{
    adapter::McpToolAdapter,
    config::{
        McpConfig, McpLocalServerConfig, McpRemoteServerConfig, McpServerConfig, McpServerTransport,
    },
};
use autoagents::core::tool::ToolT;
use http::{HeaderName, HeaderValue};
use rmcp::{
    model::{
        ClientInfo, PaginatedRequestParams, Prompt, Resource, ResourceTemplate, ServerCapabilities,
        Tool,
    },
    service::{RoleClient, RunningService, ServiceExt},
    transport::{
        ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use std::{
    collections::{HashMap, HashSet},
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::process::Command;
use tokio::sync::RwLock;

const MAX_LIST_PAGES: usize = 1_000;

type SharedTool = Arc<dyn ToolT>;
type ToolCache = HashMap<String, Vec<SharedTool>>;

/// Status for a configured MCP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerStatus {
    Connected,
    Disabled,
    Failed { error: String },
}

/// Server-provided MCP instructions with the exposed AutoAgents tool names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerInstructions {
    pub name: String,
    pub instructions: String,
    pub tools: Vec<String>,
}

/// Policy for approving local stdio MCP process execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct McpProcessPolicy {
    allow_all: bool,
    paths: HashSet<PathBuf>,
}

impl McpProcessPolicy {
    /// Deny every local MCP process.
    pub fn deny_all() -> Self {
        Self::default()
    }

    /// Allow every local MCP process. Use only for trusted local config.
    pub fn allow_all() -> Self {
        Self {
            allow_all: true,
            paths: HashSet::new(),
        }
    }

    /// Resolve bare commands through the current `PATH` and allow those executable paths.
    pub fn allow_commands<I, S>(commands: I) -> Result<Self, McpError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let env = HashMap::new();
        let mut policy = Self::deny_all();
        for command in commands {
            policy
                .paths
                .insert(resolve_command_on_path(command.as_ref(), &cwd, &env)?);
        }
        Ok(policy)
    }

    /// Allow executable paths. Existing paths are canonicalized.
    pub fn allow_paths<I, P>(paths: I) -> Result<Self, McpError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut policy = Self::deny_all();
        for path in paths {
            policy.paths.insert(canonicalize_existing(path.as_ref())?);
        }
        Ok(policy)
    }

    /// Resolve a bare command through the current `PATH` and allow that executable path.
    pub fn with_command(mut self, command: impl AsRef<str>) -> Result<Self, McpError> {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let env = HashMap::new();
        self.paths
            .insert(resolve_command_on_path(command.as_ref(), &cwd, &env)?);
        Ok(self)
    }

    /// Add an executable path. Existing paths are canonicalized.
    pub fn with_path(mut self, path: impl AsRef<Path>) -> Result<Self, McpError> {
        self.paths.insert(canonicalize_existing(path.as_ref())?);
        Ok(self)
    }

    fn approve(&self, server: &str, resolved: &ResolvedLocalCommand) -> Result<(), McpError> {
        if self.allow_all {
            return Ok(());
        }

        match &resolved.policy_subject {
            ProcessPolicySubject::Path(path) if self.paths.contains(path) => Ok(()),
            ProcessPolicySubject::Path(path) => Err(McpError::ProcessNotAllowed {
                server: server.to_string(),
                command: path.display().to_string(),
            }),
        }
    }
}

/// Represents a connection to an MCP server.
#[derive(Debug)]
pub struct McpServerConnection {
    pub name: String,
    pub service: Arc<RunningService<RoleClient, ClientInfo>>,
    pub timeout: Duration,
    pub instructions: Option<String>,
    pub capabilities: ServerCapabilities,
}

/// MCP Tools Manager that handles multiple MCP server connections.
#[derive(Debug)]
pub struct McpToolsManager {
    connections: Arc<RwLock<HashMap<String, McpServerConnection>>>,
    configs: Arc<RwLock<HashMap<String, McpServerConfig>>>,
    status: Arc<RwLock<HashMap<String, McpServerStatus>>>,
    tools: Arc<RwLock<ToolCache>>,
    process_policy: McpProcessPolicy,
    base_dir: PathBuf,
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
    #[error("Local MCP process for server '{server}' is not allowed by policy: {command}")]
    ProcessNotAllowed { server: String, command: String },
    #[error("MCP operation '{operation}' timed out for server '{server}' after {timeout:?}")]
    Timeout {
        server: String,
        operation: &'static str,
        timeout: Duration,
    },
    #[error("Tool error: {0}")]
    ToolError(String),
    #[error("Rmcp error: {0}")]
    RmcpError(#[from] rmcp::ErrorData),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Generic error: {0}")]
    GenericError(String),
}

impl Default for McpToolsManager {
    fn default() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            status: Arc::new(RwLock::new(HashMap::new())),
            tools: Arc::new(RwLock::new(HashMap::new())),
            process_policy: McpProcessPolicy::default(),
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

impl McpToolsManager {
    /// Create a new MCP tools manager. Local process execution is denied by default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an MCP tools manager with an explicit process policy.
    pub fn with_process_policy(process_policy: McpProcessPolicy) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            status: Arc::new(RwLock::new(HashMap::new())),
            tools: Arc::new(RwLock::new(HashMap::new())),
            process_policy,
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    /// Set the base directory used for resolving relative local MCP process paths.
    pub fn with_base_dir<P: Into<PathBuf>>(mut self, base_dir: P) -> Self {
        self.base_dir = base_dir.into();
        self
    }

    /// Load MCP configuration from a file and connect to enabled servers.
    pub async fn from_config_file<P: AsRef<Path>>(path: P) -> Result<Self, McpError> {
        Self::from_config_file_with_process_policy(path, McpProcessPolicy::deny_all()).await
    }

    /// Load MCP configuration from a file using an explicit local process policy.
    pub async fn from_config_file_with_process_policy<P: AsRef<Path>>(
        path: P,
        process_policy: McpProcessPolicy,
    ) -> Result<Self, McpError> {
        let config =
            McpConfig::from_file(path).map_err(|error| McpError::ConfigError(error.to_string()))?;

        let mut manager = Self::with_process_policy(process_policy);
        if let Some(base_dir) = config.base_dir() {
            manager.base_dir = base_dir.to_path_buf();
        }
        manager.connect_servers(&config).await?;
        Ok(manager)
    }

    /// Connect to all servers defined in the configuration.
    pub async fn connect_servers(&self, config: &McpConfig) -> Result<(), McpError> {
        self.connect_servers_inner(config, false).await
    }

    /// Connect to all servers while preserving partial startup on failures.
    ///
    /// Failed enabled servers are recorded in [`Self::status`], but this method
    /// continues connecting later servers and returns `Ok(())`.
    pub async fn connect_servers_allow_partial(&self, config: &McpConfig) -> Result<(), McpError> {
        self.connect_servers_inner(config, true).await
    }

    async fn connect_servers_inner(
        &self,
        config: &McpConfig,
        allow_partial: bool,
    ) -> Result<(), McpError> {
        {
            let mut configs = self.configs.write().await;
            for server_config in &config.servers {
                configs.insert(server_config.name.clone(), server_config.clone());
            }
        }

        for server_config in &config.servers {
            if !server_config.enabled {
                self.remove_server_connection_and_tools(&server_config.name)
                    .await;
                self.set_status(&server_config.name, McpServerStatus::Disabled)
                    .await;
                continue;
            }

            if let Err(error) = self.connect_and_store(server_config).await {
                log::error!(
                    "Failed to connect to MCP server '{}': {}",
                    server_config.name,
                    error
                );
                self.remove_server_connection_and_tools(&server_config.name)
                    .await;
                self.set_status(
                    &server_config.name,
                    McpServerStatus::Failed {
                        error: error.to_string(),
                    },
                )
                .await;
                if !allow_partial {
                    return Err(error);
                }
            }
        }

        Ok(())
    }

    /// Connect or reconnect one configured server.
    pub async fn connect(&self, server_name: &str) -> Result<(), McpError> {
        let config = self
            .configs
            .read()
            .await
            .get(server_name)
            .cloned()
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        if let Err(error) = self.connect_and_store(&config).await {
            self.remove_server_connection_and_tools(&config.name).await;
            self.set_status(
                &config.name,
                McpServerStatus::Failed {
                    error: error.to_string(),
                },
            )
            .await;
            return Err(error);
        }

        Ok(())
    }

    /// Disconnect one server and remove its tools.
    pub async fn disconnect(&self, server_name: &str) -> Result<(), McpError> {
        if !self.configs.read().await.contains_key(server_name) {
            return Err(McpError::ServerNotFound(server_name.to_string()));
        }

        self.remove_server_connection_and_tools(server_name).await;
        self.set_status(server_name, McpServerStatus::Disabled)
            .await;
        Ok(())
    }

    async fn connect_and_store(&self, server_config: &McpServerConfig) -> Result<(), McpError> {
        let connection = self.connect_server(server_config).await?;
        let server_name = connection.name.clone();
        let tools = self.load_server_tools(&connection).await?;

        self.connections
            .write()
            .await
            .insert(server_name.clone(), connection);
        self.replace_server_tools(&server_name, tools).await;
        self.set_status(&server_name, McpServerStatus::Connected)
            .await;

        Ok(())
    }

    /// Connect to a single MCP server.
    async fn connect_server(
        &self,
        server_config: &McpServerConfig,
    ) -> Result<McpServerConnection, McpError> {
        server_config.validate().map_err(McpError::ConfigError)?;
        let timeout = Duration::from_millis(server_config.timeout_ms);

        let service = match &server_config.transport {
            McpServerTransport::Local(local) => {
                self.connect_stdio_server(server_config, local, timeout)
                    .await?
            }
            McpServerTransport::Remote(remote) => {
                self.connect_remote_server(server_config, remote, timeout)
                    .await?
            }
            McpServerTransport::Unsupported { protocol } => {
                return Err(McpError::ConfigError(format!(
                    "Unsupported MCP protocol: {protocol}"
                )));
            }
        };

        let peer_info = service.peer().peer_info();
        let instructions = peer_info
            .as_ref()
            .and_then(|info| info.instructions.as_deref())
            .map(str::trim)
            .filter(|instructions| !instructions.is_empty())
            .map(str::to_string);
        let capabilities = peer_info
            .as_ref()
            .map(|info| info.capabilities.clone())
            .unwrap_or_default();

        Ok(McpServerConnection {
            name: server_config.name.clone(),
            service,
            timeout,
            instructions,
            capabilities,
        })
    }

    async fn connect_stdio_server(
        &self,
        server_config: &McpServerConfig,
        local: &McpLocalServerConfig,
        timeout: Duration,
    ) -> Result<Arc<RunningService<RoleClient, ClientInfo>>, McpError> {
        let resolved = self.resolve_local_command(&server_config.name, local)?;
        self.process_policy
            .approve(&server_config.name, &resolved)?;

        let mut command = Command::new(&resolved.executable);
        command.args(&resolved.args);
        command.current_dir(&resolved.cwd);
        for (key, value) in &local.environment {
            command.env(key, value);
        }

        let transport = TokioChildProcess::new(command.configure(|_| {}))
            .map_err(|error| McpError::TransportError(error.to_string()))?;
        let client_info = ClientInfo::default();

        let service = timeout_result(&server_config.name, "connect", timeout, async move {
            client_info.serve(transport).await.map_err(|error| {
                McpError::ConnectionFailed(format!(
                    "Failed to connect to local MCP server: {error:?}"
                ))
            })
        })
        .await?;

        Ok(Arc::new(service))
    }

    async fn connect_remote_server(
        &self,
        server_config: &McpServerConfig,
        remote: &McpRemoteServerConfig,
        timeout: Duration,
    ) -> Result<Arc<RunningService<RoleClient, ClientInfo>>, McpError> {
        let headers = parse_headers(&server_config.name, &remote.headers)?;
        let transport_config = StreamableHttpClientTransportConfig::with_uri(remote.url.clone())
            .custom_headers(headers);
        let transport = StreamableHttpClientTransport::from_config(transport_config);
        let client_info = ClientInfo::default();

        let service = timeout_result(&server_config.name, "connect", timeout, async move {
            client_info.serve(transport).await.map_err(|error| {
                McpError::ConnectionFailed(format!(
                    "Failed to connect to remote MCP server: {error:?}"
                ))
            })
        })
        .await?;

        Ok(Arc::new(service))
    }

    fn resolve_local_command(
        &self,
        server_name: &str,
        local: &McpLocalServerConfig,
    ) -> Result<ResolvedLocalCommand, McpError> {
        let executable = local.command.first().ok_or_else(|| {
            McpError::ConfigError(format!("MCP server '{server_name}' command is empty"))
        })?;

        let cwd = match &local.cwd {
            Some(cwd) => resolve_path(&self.base_dir, cwd),
            None => self.base_dir.clone(),
        };

        let canonical_executable = if is_path_command(executable) {
            let path = PathBuf::from(executable);
            let resolved = if path.is_absolute() {
                path
            } else {
                cwd.join(path)
            };
            canonicalize_existing(&resolved)?
        } else {
            resolve_command_on_path(executable, &cwd, &local.environment)?
        };

        Ok(ResolvedLocalCommand {
            executable: canonical_executable.to_string_lossy().to_string(),
            args: local.command.iter().skip(1).cloned().collect(),
            cwd,
            policy_subject: ProcessPolicySubject::Path(canonical_executable),
        })
    }

    /// Load tools from a connected MCP server.
    async fn load_server_tools(
        &self,
        connection: &McpServerConnection,
    ) -> Result<Vec<Arc<dyn ToolT>>, McpError> {
        if connection.capabilities.tools.is_none() {
            return Ok(Vec::new());
        }

        let tools = timeout_result(
            &connection.name,
            "list_tools",
            connection.timeout,
            list_all_tools(Arc::clone(&connection.service)),
        )
        .await?;

        Ok(tools
            .into_iter()
            .map(|tool| {
                Arc::new(McpToolAdapter::new(
                    &connection.name,
                    tool,
                    Arc::clone(&connection.service),
                    connection.timeout,
                )) as Arc<dyn ToolT>
            })
            .collect())
    }

    /// Get all available tools.
    pub async fn get_tools(&self) -> Vec<Arc<dyn ToolT>> {
        let tools = self.tools.read().await;
        flatten_tools_by_server_name(&tools)
    }

    /// Get tools from a specific server.
    pub async fn get_server_tools(
        &self,
        server_name: &str,
    ) -> Result<Vec<Arc<dyn ToolT>>, McpError> {
        if !self.connections.read().await.contains_key(server_name) {
            return Err(McpError::ServerNotFound(server_name.to_string()));
        }

        Ok(self
            .tools
            .read()
            .await
            .get(server_name)
            .cloned()
            .unwrap_or_default())
    }

    /// Get a tool by its exposed name.
    pub async fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn ToolT>> {
        let tools = self.tools.read().await;
        tools
            .values()
            .flat_map(|tools| tools.iter())
            .find(|tool| tool.name() == tool_name)
            .cloned()
    }

    /// Refresh tools from all connected servers.
    pub async fn refresh_tools(&self) -> Result<(), McpError> {
        let connections = self.connections.read().await;
        let mut all_tools = HashMap::new();

        for connection in connections.values() {
            match self.load_server_tools(connection).await {
                Ok(tools) => {
                    all_tools.insert(connection.name.clone(), tools);
                }
                Err(error) => {
                    self.remove_server_tools(&connection.name).await;
                    self.set_status(
                        &connection.name,
                        McpServerStatus::Failed {
                            error: error.to_string(),
                        },
                    )
                    .await;
                    return Err(error);
                }
            }
        }

        *self.tools.write().await = all_tools;
        Ok(())
    }

    /// Get the names of all connected servers.
    pub async fn connected_servers(&self) -> Vec<String> {
        let mut servers = self
            .connections
            .read()
            .await
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        servers.sort();
        servers
    }

    /// Check if a server is connected.
    pub async fn is_server_connected(&self, server_name: &str) -> bool {
        self.connections.read().await.contains_key(server_name)
    }

    /// Get all configured server statuses.
    pub async fn status(&self) -> HashMap<String, McpServerStatus> {
        self.status.read().await.clone()
    }

    /// Get the number of available tools.
    pub async fn tool_count(&self) -> usize {
        self.tools
            .read()
            .await
            .values()
            .map(Vec::len)
            .sum::<usize>()
    }

    /// Get exposed tool names.
    pub async fn tool_names(&self) -> Vec<String> {
        let tools = self.tools.read().await;
        flatten_tools_by_server_name(&tools)
            .into_iter()
            .map(|tool| tool.name().to_string())
            .collect()
    }

    /// Get connected server instructions.
    pub async fn server_instructions(&self) -> Vec<McpServerInstructions> {
        let connections = self.connections.read().await;
        let tools = self.tools.read().await;

        let mut instructions = connections
            .values()
            .filter_map(|connection| {
                let instructions = connection.instructions.as_ref()?;
                Some(McpServerInstructions {
                    name: connection.name.clone(),
                    instructions: instructions.clone(),
                    tools: tools
                        .get(&connection.name)
                        .into_iter()
                        .flat_map(|tools| tools.iter())
                        .map(|tool| tool.name().to_string())
                        .collect(),
                })
            })
            .collect::<Vec<_>>();
        instructions.sort_by(|left, right| left.name.cmp(&right.name));
        instructions
    }

    /// List prompts from connected servers, keyed as `server:prompt`.
    pub async fn prompts(&self) -> Result<HashMap<String, Prompt>, McpError> {
        let connections = self.connections.read().await;
        let mut result = HashMap::new();
        for connection in connections.values() {
            if connection.capabilities.prompts.is_none() {
                continue;
            }
            let prompts = timeout_result(
                &connection.name,
                "list_prompts",
                connection.timeout,
                list_all_prompts(Arc::clone(&connection.service)),
            )
            .await?;
            for prompt in prompts {
                result.insert(format!("{}:{}", connection.name, prompt.name), prompt);
            }
        }
        Ok(result)
    }

    /// List resources from connected servers, keyed as `server:uri`.
    pub async fn resources(
        &self,
        server_name: Option<&str>,
    ) -> Result<HashMap<String, Resource>, McpError> {
        let connections = self.connections.read().await;
        let mut result = HashMap::new();
        for connection in connections.values() {
            if server_name.is_some_and(|name| name != connection.name) {
                continue;
            }
            if connection.capabilities.resources.is_none() {
                continue;
            }
            let resources = timeout_result(
                &connection.name,
                "list_resources",
                connection.timeout,
                list_all_resources(Arc::clone(&connection.service)),
            )
            .await?;
            for resource in resources {
                result.insert(
                    format!("{}:{}", connection.name, resource.raw.uri),
                    resource,
                );
            }
        }
        Ok(result)
    }

    /// List resource templates from connected servers, keyed as `server:uri_template`.
    pub async fn resource_templates(
        &self,
        server_name: Option<&str>,
    ) -> Result<HashMap<String, ResourceTemplate>, McpError> {
        let connections = self.connections.read().await;
        let mut result = HashMap::new();
        for connection in connections.values() {
            if server_name.is_some_and(|name| name != connection.name) {
                continue;
            }
            if connection.capabilities.resources.is_none() {
                continue;
            }
            let templates = timeout_result(
                &connection.name,
                "list_resource_templates",
                connection.timeout,
                list_all_resource_templates(Arc::clone(&connection.service)),
            )
            .await?;
            for template in templates {
                result.insert(
                    format!("{}:{}", connection.name, template.raw.uri_template),
                    template,
                );
            }
        }
        Ok(result)
    }

    async fn set_status(&self, server_name: &str, status: McpServerStatus) {
        self.status
            .write()
            .await
            .insert(server_name.to_string(), status);
    }

    async fn remove_server_tools(&self, server_name: &str) {
        self.tools.write().await.remove(server_name);
    }

    async fn remove_server_connection_and_tools(&self, server_name: &str) {
        self.connections.write().await.remove(server_name);
        self.remove_server_tools(server_name).await;
    }

    async fn replace_server_tools(&self, server_name: &str, tools: Vec<Arc<dyn ToolT>>) {
        self.tools
            .write()
            .await
            .insert(server_name.to_string(), tools);
    }
}

#[derive(Debug)]
struct ResolvedLocalCommand {
    executable: String,
    args: Vec<String>,
    cwd: PathBuf,
    policy_subject: ProcessPolicySubject,
}

#[derive(Debug, PartialEq, Eq)]
enum ProcessPolicySubject {
    Path(PathBuf),
}

async fn timeout_result<T, F>(
    server: &str,
    operation: &'static str,
    timeout: Duration,
    future: F,
) -> Result<T, McpError>
where
    F: Future<Output = Result<T, McpError>>,
{
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| McpError::Timeout {
            server: server.to_string(),
            operation,
            timeout,
        })?
}

async fn list_all_tools(
    service: Arc<RunningService<RoleClient, ClientInfo>>,
) -> Result<Vec<Tool>, McpError> {
    let mut tools = Vec::new();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();

    for _ in 0..MAX_LIST_PAGES {
        let result = service
            .peer()
            .list_tools(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .map_err(|error| McpError::GenericError(format!("Failed to list tools: {error:?}")))?;
        tools.extend(result.tools);
        match result.next_cursor {
            Some(next_cursor) => {
                if !seen_cursors.insert(next_cursor.clone()) {
                    return Err(McpError::GenericError(format!(
                        "MCP tools/list returned duplicate cursor: {next_cursor}"
                    )));
                }
                cursor = Some(next_cursor);
            }
            None => return Ok(tools),
        }
    }

    Err(McpError::GenericError(format!(
        "MCP tools/list exceeded {MAX_LIST_PAGES} pages"
    )))
}

async fn list_all_prompts(
    service: Arc<RunningService<RoleClient, ClientInfo>>,
) -> Result<Vec<Prompt>, McpError> {
    let mut prompts = Vec::new();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();

    for _ in 0..MAX_LIST_PAGES {
        let result = service
            .peer()
            .list_prompts(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .map_err(|error| {
                McpError::GenericError(format!("Failed to list prompts: {error:?}"))
            })?;
        prompts.extend(result.prompts);
        match result.next_cursor {
            Some(next_cursor) => {
                if !seen_cursors.insert(next_cursor.clone()) {
                    return Err(McpError::GenericError(format!(
                        "MCP prompts/list returned duplicate cursor: {next_cursor}"
                    )));
                }
                cursor = Some(next_cursor);
            }
            None => return Ok(prompts),
        }
    }

    Err(McpError::GenericError(format!(
        "MCP prompts/list exceeded {MAX_LIST_PAGES} pages"
    )))
}

async fn list_all_resources(
    service: Arc<RunningService<RoleClient, ClientInfo>>,
) -> Result<Vec<Resource>, McpError> {
    let mut resources = Vec::new();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();

    for _ in 0..MAX_LIST_PAGES {
        let result = service
            .peer()
            .list_resources(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .map_err(|error| {
                McpError::GenericError(format!("Failed to list resources: {error:?}"))
            })?;
        resources.extend(result.resources);
        match result.next_cursor {
            Some(next_cursor) => {
                if !seen_cursors.insert(next_cursor.clone()) {
                    return Err(McpError::GenericError(format!(
                        "MCP resources/list returned duplicate cursor: {next_cursor}"
                    )));
                }
                cursor = Some(next_cursor);
            }
            None => return Ok(resources),
        }
    }

    Err(McpError::GenericError(format!(
        "MCP resources/list exceeded {MAX_LIST_PAGES} pages"
    )))
}

async fn list_all_resource_templates(
    service: Arc<RunningService<RoleClient, ClientInfo>>,
) -> Result<Vec<ResourceTemplate>, McpError> {
    let mut templates = Vec::new();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();

    for _ in 0..MAX_LIST_PAGES {
        let result = service
            .peer()
            .list_resource_templates(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .map_err(|error| {
                McpError::GenericError(format!("Failed to list resource templates: {error:?}"))
            })?;
        templates.extend(result.resource_templates);
        match result.next_cursor {
            Some(next_cursor) => {
                if !seen_cursors.insert(next_cursor.clone()) {
                    return Err(McpError::GenericError(format!(
                        "MCP resources/templates/list returned duplicate cursor: {next_cursor}"
                    )));
                }
                cursor = Some(next_cursor);
            }
            None => return Ok(templates),
        }
    }

    Err(McpError::GenericError(format!(
        "MCP resources/templates/list exceeded {MAX_LIST_PAGES} pages"
    )))
}

fn parse_headers(
    server_name: &str,
    headers: &HashMap<String, String>,
) -> Result<HashMap<HeaderName, HeaderValue>, McpError> {
    headers
        .iter()
        .map(|(key, value)| {
            let name = HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                McpError::ConfigError(format!(
                    "Invalid header name for MCP server '{server_name}': {error}"
                ))
            })?;
            let value = HeaderValue::from_str(value).map_err(|error| {
                McpError::ConfigError(format!(
                    "Invalid header value for MCP server '{server_name}' header '{key}': {error}"
                ))
            })?;
            Ok((name, value))
        })
        .collect()
}

fn resolve_path(base_dir: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn canonicalize_existing(path: &Path) -> Result<PathBuf, McpError> {
    path.canonicalize().map_err(|error| {
        McpError::ConfigError(format!(
            "Failed to canonicalize MCP process path '{}': {error}",
            path.display()
        ))
    })
}

fn resolve_command_on_path(
    command: &str,
    cwd: &Path,
    environment: &HashMap<String, String>,
) -> Result<PathBuf, McpError> {
    if command.is_empty() {
        return Err(McpError::ConfigError(
            "MCP local server command cannot be empty".to_string(),
        ));
    }
    if is_path_command(command) {
        return canonicalize_existing(Path::new(command));
    }

    let path_value = environment
        .get("PATH")
        .map(|path| path.as_str().into())
        .or_else(|| std::env::var_os("PATH"));

    let Some(path_value) = path_value else {
        return Err(McpError::ConfigError(format!(
            "MCP local server command '{command}' was not found because PATH is not set"
        )));
    };

    for dir in std::env::split_paths(&path_value) {
        let dir = if dir.as_os_str().is_empty() {
            cwd.to_path_buf()
        } else {
            dir
        };
        for candidate in command_candidates(&dir, command) {
            if candidate.is_file() {
                return canonicalize_existing(&candidate);
            }
        }
    }

    Err(McpError::ConfigError(format!(
        "MCP local server command '{command}' was not found on PATH"
    )))
}

fn command_candidates(dir: &Path, command: &str) -> Vec<PathBuf> {
    let command_path = Path::new(command);
    #[cfg(windows)]
    {
        if command_path.extension().is_some() {
            return vec![dir.join(command_path)];
        }
        let path_ext = std::env::var_os("PATHEXT")
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into())
            .to_string_lossy()
            .to_string();
        path_ext
            .split(';')
            .filter(|ext| !ext.is_empty())
            .map(|ext| dir.join(format!("{command}{ext}")))
            .chain(std::iter::once(dir.join(command_path)))
            .collect()
    }
    #[cfg(not(windows))]
    {
        vec![dir.join(command_path)]
    }
}

fn flatten_tools_by_server_name(tools: &ToolCache) -> Vec<Arc<dyn ToolT>> {
    let mut server_names = tools.keys().collect::<Vec<_>>();
    server_names.sort();
    server_names
        .into_iter()
        .flat_map(|server_name| tools[server_name].iter().cloned())
        .collect()
}

fn is_path_command(command: &str) -> bool {
    command.contains('/') || command.contains('\\') || Path::new(command).is_absolute()
}

fn sanitize_tool_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

pub(crate) fn tool_prefix(server_name: &str) -> String {
    format!(
        "{}_{}_",
        sanitize_tool_segment(server_name),
        hex_encode(server_name.as_bytes())
    )
}

pub(crate) fn exposed_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "{}{}",
        tool_prefix(server_name),
        sanitize_tool_segment(tool_name)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::McpServerTransport;
    use autoagents::core::tool::ToolCallError;
    use tempfile::tempdir;

    #[derive(Debug)]
    struct DummyTool {
        name: String,
    }

    #[autoagents::async_trait]
    impl autoagents::core::tool::ToolRuntime for DummyTool {
        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> Result<serde_json::Value, ToolCallError> {
            Ok(serde_json::Value::Null)
        }
    }

    impl ToolT for DummyTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "dummy tool"
        }

        fn args_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    #[tokio::test]
    async fn manager_basic_operations() {
        let manager = McpToolsManager::default();

        assert_eq!(manager.tool_count().await, 0);
        assert!(manager.tool_names().await.is_empty());
        assert!(manager.connected_servers().await.is_empty());
        assert!(!manager.is_server_connected("nonexistent").await);
        assert!(manager.get_tool("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn get_server_tools_missing() {
        let manager = McpToolsManager::default();
        let err = manager.get_server_tools("missing").await.unwrap_err();
        assert!(matches!(err, McpError::ServerNotFound(_)));
    }

    #[tokio::test]
    async fn connect_servers_records_invalid_protocol_as_failed_status() {
        let manager = McpToolsManager::default();
        let mut config = McpConfig::new();
        config.add_server(McpServerConfig::new(
            "bad_server".to_string(),
            "sse".to_string(),
            "noop".to_string(),
        ));

        let err = manager.connect_servers(&config).await.unwrap_err();
        assert!(err.to_string().contains("Unsupported"));
        let status = manager.status().await;
        assert!(matches!(
            status.get("bad_server"),
            Some(McpServerStatus::Failed { error }) if error.contains("Unsupported")
        ));
    }

    #[tokio::test]
    async fn connect_servers_allow_partial_records_failures_without_error() {
        let manager = McpToolsManager::default();
        let mut config = McpConfig::new();
        config.add_server(McpServerConfig::new(
            "bad_server".to_string(),
            "sse".to_string(),
            "noop".to_string(),
        ));

        manager
            .connect_servers_allow_partial(&config)
            .await
            .unwrap();
        let status = manager.status().await;
        assert!(matches!(
            status.get("bad_server"),
            Some(McpServerStatus::Failed { error }) if error.contains("Unsupported")
        ));
    }

    #[tokio::test]
    async fn connect_servers_marks_disabled_without_connecting() {
        let manager = McpToolsManager::default();
        let mut config = McpConfig::new();
        config.add_server(
            McpServerConfig::local("disabled", vec!["echo".to_string()]).with_enabled(false),
        );

        manager.connect_servers(&config).await.unwrap();
        let status = manager.status().await;
        assert_eq!(status.get("disabled"), Some(&McpServerStatus::Disabled));
    }

    #[tokio::test]
    async fn connect_servers_removes_existing_tools_when_server_is_disabled() {
        let manager = McpToolsManager::default();
        manager.tools.write().await.insert(
            "disabled".to_string(),
            vec![Arc::new(DummyTool {
                name: exposed_tool_name("disabled", "stale"),
            })],
        );
        assert_eq!(manager.tool_count().await, 1);

        let mut config = McpConfig::new();
        config.add_server(
            McpServerConfig::local("disabled", vec!["echo".to_string()]).with_enabled(false),
        );

        manager.connect_servers(&config).await.unwrap();

        assert_eq!(manager.tool_count().await, 0);
        assert!(
            manager
                .get_tool(&exposed_tool_name("disabled", "stale"))
                .await
                .is_none()
        );
        assert!(!manager.is_server_connected("disabled").await);
        let status = manager.status().await;
        assert_eq!(status.get("disabled"), Some(&McpServerStatus::Disabled));
    }

    #[tokio::test]
    async fn remove_server_tools_does_not_remove_sibling_prefix_tools() {
        let manager = McpToolsManager::default();
        manager.tools.write().await.extend([
            (
                "my".to_string(),
                vec![Arc::new(DummyTool {
                    name: exposed_tool_name("my", "search"),
                }) as Arc<dyn ToolT>],
            ),
            (
                "my_team".to_string(),
                vec![Arc::new(DummyTool {
                    name: exposed_tool_name("my_team", "search"),
                }) as Arc<dyn ToolT>],
            ),
        ]);

        manager.remove_server_tools("my").await;

        assert!(
            manager
                .get_tool(&exposed_tool_name("my", "search"))
                .await
                .is_none()
        );
        assert!(
            manager
                .get_tool(&exposed_tool_name("my_team", "search"))
                .await
                .is_some()
        );
        assert_eq!(manager.tool_count().await, 1);
    }

    #[test]
    fn exposed_tool_names_do_not_collide_for_sanitized_server_name_matches() {
        assert_ne!(
            exposed_tool_name("github.tools", "search"),
            exposed_tool_name("github_tools", "search")
        );
    }

    #[tokio::test]
    async fn public_tool_lists_are_sorted_by_server_name() {
        let manager = McpToolsManager::default();
        manager.tools.write().await.extend([
            (
                "z_server".to_string(),
                vec![Arc::new(DummyTool {
                    name: exposed_tool_name("z_server", "search"),
                }) as Arc<dyn ToolT>],
            ),
            (
                "a_server".to_string(),
                vec![Arc::new(DummyTool {
                    name: exposed_tool_name("a_server", "lookup"),
                }) as Arc<dyn ToolT>],
            ),
        ]);

        assert_eq!(
            manager.tool_names().await,
            vec![
                exposed_tool_name("a_server", "lookup"),
                exposed_tool_name("z_server", "search")
            ]
        );
        assert_eq!(
            manager
                .get_tools()
                .await
                .iter()
                .map(|tool| tool.name().to_string())
                .collect::<Vec<_>>(),
            vec![
                exposed_tool_name("a_server", "lookup"),
                exposed_tool_name("z_server", "search")
            ]
        );
    }

    #[tokio::test]
    async fn failed_reconnect_removes_existing_tools() {
        let manager = McpToolsManager::default();
        manager.tools.write().await.insert(
            "local".to_string(),
            vec![Arc::new(DummyTool {
                name: exposed_tool_name("local", "stale"),
            })],
        );

        let mut config = McpConfig::new();
        config.add_server(McpServerConfig::local(
            "local",
            vec!["definitely-missing-autoagents-mcp-server".to_string()],
        ));

        let err = manager.connect_servers(&config).await.unwrap_err();
        assert!(err.to_string().contains("not found") || err.to_string().contains("not allowed"));
        assert!(
            manager
                .get_tool(&exposed_tool_name("local", "stale"))
                .await
                .is_none()
        );
        assert_eq!(manager.tool_count().await, 0);
        let status = manager.status().await;
        assert!(matches!(
            status.get("local"),
            Some(McpServerStatus::Failed { .. })
        ));
    }

    #[tokio::test]
    async fn default_policy_denies_local_processes() {
        let manager = McpToolsManager::default();
        let mut config = McpConfig::new();
        config.add_server(McpServerConfig::local("local", vec!["echo".to_string()]));

        let err = manager.connect_servers(&config).await.unwrap_err();
        assert!(err.to_string().contains("not allowed"));
        let status = manager.status().await;
        assert!(matches!(
            status.get("local"),
            Some(McpServerStatus::Failed { error }) if error.contains("not allowed")
        ));
    }

    #[test]
    fn policy_allows_canonical_paths() {
        let dir = tempdir().unwrap();
        let executable = dir.path().join("server");
        std::fs::write(&executable, "").unwrap();
        let executable = executable.canonicalize().unwrap();

        let policy = McpProcessPolicy::allow_paths([&executable]).unwrap();
        let resolved = ResolvedLocalCommand {
            executable: executable.to_string_lossy().to_string(),
            args: Vec::new(),
            cwd: PathBuf::from("."),
            policy_subject: ProcessPolicySubject::Path(executable),
        };

        assert!(policy.approve("server", &resolved).is_ok());
    }

    #[test]
    fn bare_command_resolves_through_child_path_before_policy_approval() {
        let dir = tempdir().unwrap();
        let trusted_bin = dir.path().join("trusted");
        let shadow_bin = dir.path().join("shadow");
        std::fs::create_dir_all(&trusted_bin).unwrap();
        std::fs::create_dir_all(&shadow_bin).unwrap();
        let trusted = trusted_bin.join("server");
        let shadow = shadow_bin.join("server");
        std::fs::write(&trusted, "").unwrap();
        std::fs::write(&shadow, "").unwrap();

        let manager = McpToolsManager {
            base_dir: dir.path().to_path_buf(),
            process_policy: McpProcessPolicy::allow_paths([&trusted]).unwrap(),
            ..McpToolsManager::default()
        };
        let local = McpLocalServerConfig {
            command: vec!["server".to_string()],
            cwd: None,
            environment: HashMap::from([(
                "PATH".to_string(),
                shadow_bin.to_string_lossy().to_string(),
            )]),
        };

        let resolved = manager.resolve_local_command("server", &local).unwrap();
        assert!(matches!(
            resolved.policy_subject,
            ProcessPolicySubject::Path(ref path) if path == &shadow.canonicalize().unwrap()
        ));
        assert!(manager.process_policy.approve("server", &resolved).is_err());
    }

    #[test]
    fn relative_cwd_and_path_commands_resolve_from_base_dir() {
        let dir = tempdir().unwrap();
        let bin_dir = dir.path().join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let executable = bin_dir.join("server");
        std::fs::write(&executable, "").unwrap();

        let manager = McpToolsManager {
            base_dir: dir.path().to_path_buf(),
            ..McpToolsManager::default()
        };
        let local = McpLocalServerConfig {
            command: vec!["./server".to_string()],
            cwd: Some("bin".to_string()),
            environment: HashMap::new(),
        };

        let resolved = manager.resolve_local_command("server", &local).unwrap();
        assert_eq!(resolved.cwd, bin_dir);
        assert!(matches!(
            resolved.policy_subject,
            ProcessPolicySubject::Path(path) if path == executable.canonicalize().unwrap()
        ));
    }

    #[test]
    fn tool_names_are_prefixed_and_sanitized() {
        assert_eq!(
            exposed_tool_name("my.special-server", "tool.name"),
            "my_special-server_6d792e7370656369616c2d736572766572_tool_name"
        );
    }

    #[test]
    fn remote_header_parsing_rejects_invalid_names() {
        let mut headers = HashMap::new();
        headers.insert("bad header".to_string(), "value".to_string());
        assert!(parse_headers("server", &headers).is_err());
    }

    #[test]
    fn validate_remote_config_from_transport() {
        let config = McpServerConfig {
            name: "remote".to_string(),
            enabled: true,
            timeout_ms: crate::mcp::config::default_timeout_ms(),
            transport: McpServerTransport::Remote(McpRemoteServerConfig {
                url: "https://example.com/mcp".to_string(),
                headers: HashMap::new(),
            }),
        };
        assert!(config.validate().is_ok());
    }
}
