use std::sync::Arc;
use std::time::Duration;

use rmcp::{
    model::ClientInfo,
    service::{RoleClient, RunningService, ServiceExt},
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use tokio::process::Command;

use crate::mcp::config::McpServerConfig;
use crate::mcp::policy::McpSecurityPolicy;

use super::{McpError, with_timeout};

pub async fn connect_stdio_server(
    config: &McpServerConfig,
    policy: &McpSecurityPolicy,
    config_base: Option<&std::path::Path>,
) -> Result<Arc<RunningService<RoleClient, ClientInfo>>, McpError> {
    let launch = config.stdio_launch_spec();
    policy
        .authorize_stdio_launch(&launch, config_base)
        .await
        .map_err(McpError::from)?;

    let mut command = Command::new(&config.command);

    if !config.args.is_empty() {
        command.args(&config.args);
    }

    if let Some(cwd) = &config.cwd {
        command.current_dir(cwd);
    }

    for (key, value) in &config.env {
        command.env(key, value);
    }

    let transport = TokioChildProcess::new(command.configure(|_| {}))
        .map_err(|e| McpError::TransportError(e.to_string()))?;

    let client_info = ClientInfo::default();
    let timeout = Duration::from_secs(config.timeout);

    let service = with_timeout(timeout, "connect", async {
        client_info.serve(transport).await.map_err(|e| {
            McpError::ConnectionFailed(format!("Failed to connect to MCP server: {e:?}"))
        })
    })
    .await?;

    Ok(Arc::new(service))
}
