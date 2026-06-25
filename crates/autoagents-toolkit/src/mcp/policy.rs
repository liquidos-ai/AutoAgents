use crate::mcp::security::{
    DEFAULT_STDIO_ALLOWED_COMMANDS, McpSecurityError, effective_allowed_commands,
    validate_command_allowlist, validate_stdio_args, validate_stdio_cwd, validate_stdio_env,
};
use autoagents::async_trait;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Specification of a stdio MCP server process about to be launched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpProcessLaunchSpec {
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub env: HashMap<String, String>,
}

impl fmt::Display for McpProcessLaunchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.command, self.args.join(" "))
    }
}

/// Hook invoked before spawning a non-allowlisted stdio process.
#[async_trait]
pub trait McpProcessApprover: Send + Sync {
    async fn approve(&self, launch: &McpProcessLaunchSpec) -> Result<(), McpSecurityError>;
}

/// Security policy governing MCP stdio process execution and HTTP endpoint restrictions.
#[derive(Clone)]
pub struct McpSecurityPolicy {
    allowed_commands: HashSet<String>,
    approver: Option<Arc<dyn McpProcessApprover>>,
    allow_private_http_endpoints: bool,
}

impl fmt::Debug for McpSecurityPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpSecurityPolicy")
            .field("allowed_commands", &self.allowed_commands)
            .field("has_approver", &self.approver.is_some())
            .field(
                "allow_private_http_endpoints",
                &self.allow_private_http_endpoints,
            )
            .finish()
    }
}

impl Default for McpSecurityPolicy {
    fn default() -> Self {
        Self {
            allowed_commands: DEFAULT_STDIO_ALLOWED_COMMANDS
                .iter()
                .map(|c| (*c).to_string())
                .collect(),
            approver: None,
            allow_private_http_endpoints: false,
        }
    }
}

impl McpSecurityPolicy {
    /// Secure-by-default policy: enforce the built-in launcher allowlist and block private HTTP hosts.
    ///
    /// Also merges entries from `AUTOAGENTS_MCP_STDIO_EXTRA_COMMANDS` when set.
    pub fn secure_default() -> Result<Self, McpSecurityError> {
        Ok(Self {
            allowed_commands: effective_allowed_commands()?,
            approver: None,
            allow_private_http_endpoints: false,
        })
    }

    /// Permissive policy for trusted local development and tests only.
    ///
    /// Disables the command allowlist and permits private-network HTTP endpoints.
    /// Do not use in production deployments.
    pub fn permissive() -> Self {
        Self {
            allowed_commands: HashSet::new(),
            approver: None,
            allow_private_http_endpoints: true,
        }
    }

    /// Replace the effective allowlist.
    pub fn with_allowed_commands(mut self, commands: HashSet<String>) -> Self {
        self.allowed_commands = commands;
        self
    }

    /// Allow non-allowlisted stdio commands after explicit approval at spawn time.
    pub fn with_approver(mut self, approver: Arc<dyn McpProcessApprover>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Permit HTTP connections to localhost and private-network addresses.
    pub fn with_allow_private_http_endpoints(mut self, allow: bool) -> Self {
        self.allow_private_http_endpoints = allow;
        self
    }

    pub fn allowed_commands(&self) -> &HashSet<String> {
        &self.allowed_commands
    }

    pub fn allow_private_http_endpoints(&self) -> bool {
        self.allow_private_http_endpoints
    }

    /// Whether non-allowlisted stdio commands may proceed via an approval hook.
    pub fn defers_allowlist_to_approver(&self) -> bool {
        self.approver.is_some()
    }

    /// Validate stdio config fields that are enforced regardless of approval.
    pub fn validate_stdio_fields(
        &self,
        spec: &McpProcessLaunchSpec,
        config_base: Option<&Path>,
    ) -> Result<(), McpSecurityError> {
        validate_stdio_args(&spec.command, &spec.args)?;
        validate_stdio_env(&spec.env)?;
        if let Some(cwd) = &spec.cwd {
            validate_stdio_cwd(cwd, config_base)?;
        }
        Ok(())
    }

    /// Validate and authorize a stdio launch under this policy.
    pub async fn authorize_stdio_launch(
        &self,
        spec: &McpProcessLaunchSpec,
        config_base: Option<&Path>,
    ) -> Result<(), McpSecurityError> {
        self.validate_stdio_fields(spec, config_base)?;

        match validate_command_allowlist(&spec.command, &self.allowed_commands) {
            Ok(()) => Ok(()),
            Err(err) => {
                if let Some(approver) = &self.approver {
                    approver.approve(spec).await
                } else {
                    Err(err)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysApprove;

    #[async_trait]
    impl McpProcessApprover for AlwaysApprove {
        async fn approve(&self, _launch: &McpProcessLaunchSpec) -> Result<(), McpSecurityError> {
            Ok(())
        }
    }

    struct AlwaysDeny;

    #[async_trait]
    impl McpProcessApprover for AlwaysDeny {
        async fn approve(&self, _launch: &McpProcessLaunchSpec) -> Result<(), McpSecurityError> {
            Err(McpSecurityError::NotApproved("denied".to_string()))
        }
    }

    fn launch_spec(command: &str) -> McpProcessLaunchSpec {
        McpProcessLaunchSpec {
            server_name: "test".to_string(),
            command: command.to_string(),
            args: vec![],
            cwd: None,
            env: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn secure_default_rejects_bash() {
        let policy = McpSecurityPolicy::secure_default().unwrap();
        let err = policy
            .authorize_stdio_launch(&launch_spec("bash"), None)
            .await
            .unwrap_err();
        assert!(matches!(err, McpSecurityError::CommandNotAllowed { .. }));
    }

    #[tokio::test]
    async fn permissive_allows_bash() {
        let policy = McpSecurityPolicy::permissive();
        policy
            .authorize_stdio_launch(&launch_spec("bash"), None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn approver_allows_non_allowlisted_command() {
        let policy = McpSecurityPolicy::secure_default()
            .unwrap()
            .with_approver(Arc::new(AlwaysApprove));
        policy
            .authorize_stdio_launch(&launch_spec("bash"), None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn approver_denied_returns_error() {
        let policy = McpSecurityPolicy::secure_default()
            .unwrap()
            .with_approver(Arc::new(AlwaysDeny));
        let err = policy
            .authorize_stdio_launch(&launch_spec("bash"), None)
            .await
            .unwrap_err();
        assert!(matches!(err, McpSecurityError::NotApproved(_)));
    }
}
