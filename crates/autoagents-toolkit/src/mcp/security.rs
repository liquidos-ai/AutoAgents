use std::collections::HashSet;
use std::env;
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};

/// Default launcher names permitted for MCP stdio transport.
pub const DEFAULT_STDIO_ALLOWED_COMMANDS: &[&str] =
    &["npx", "uvx", "python", "python3", "node", "docker", "deno"];

const ENV_EXTRA_COMMANDS: &str = "AUTOAGENTS_MCP_STDIO_EXTRA_COMMANDS";

const SHELL_METACHARACTERS: &[char] = &[';', '|', '&', '`', '$', '\n', '\r'];

const DANGEROUS_ENV_VARS: &[&str] = &[
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "NODE_OPTIONS",
    "NODE_EXTRA_CA_CERTS",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "BASH_ENV",
    "ENV",
    "GODEBUG",
];

/// Build the effective stdio command allowlist from defaults and environment.
pub fn effective_allowed_commands() -> Result<HashSet<String>, McpSecurityError> {
    let mut allowed: HashSet<String> = DEFAULT_STDIO_ALLOWED_COMMANDS
        .iter()
        .map(|c| (*c).to_string())
        .collect();

    if let Ok(extra) = env::var(ENV_EXTRA_COMMANDS) {
        for command in extra.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            validate_command_is_bare_name(command)?;
            allowed.insert(command.to_string());
        }
    }

    Ok(allowed)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum McpSecurityError {
    #[error("stdio command must be a bare launcher name, not a path: '{0}'")]
    CommandMustBeBareName(String),
    #[error(
        "command '{command}' is not in the stdio allowlist. Allowed: {allowed:?}. \
         Extend via AUTOAGENTS_MCP_STDIO_EXTRA_COMMANDS, configure an McpProcessApprover, \
         or use McpSecurityPolicy::permissive() for local development."
    )]
    CommandNotAllowed {
        command: String,
        allowed: Vec<String>,
    },
    #[error("dangerous argument '{arg}' is not permitted for launcher '{command}'")]
    DangerousArgument { command: String, arg: String },
    #[error("argument contains shell metacharacters: '{0}'")]
    ShellMetacharacters(String),
    #[error("process launch was not approved: {0}")]
    NotApproved(String),
    #[error("environment variable '{0}' is not permitted for stdio transport")]
    DangerousEnvVar(String),
    #[error("working directory '{0}' is not permitted")]
    InvalidWorkingDirectory(String),
    #[error("http url is not permitted: {0}")]
    DisallowedHttpUrl(String),
}

/// Validate that a stdio launcher name is a bare command (no path components).
pub fn validate_command_is_bare_name(command: &str) -> Result<(), McpSecurityError> {
    if command.is_empty() {
        return Err(McpSecurityError::CommandMustBeBareName(command.to_string()));
    }

    if Path::new(command).is_absolute() || command.contains('/') || command.contains('\\') {
        return Err(McpSecurityError::CommandMustBeBareName(command.to_string()));
    }

    Ok(())
}

/// Validate stdio command against the provided allowlist.
///
/// An empty allowlist disables allowlist enforcement (explicit opt-out).
pub fn validate_command_allowlist(
    command: &str,
    allowed: &HashSet<String>,
) -> Result<(), McpSecurityError> {
    if allowed.is_empty() {
        return Ok(());
    }

    validate_command_is_bare_name(command)?;

    if allowed.contains(command) {
        return Ok(());
    }

    Err(McpSecurityError::CommandNotAllowed {
        command: command.to_string(),
        allowed: {
            let mut list: Vec<String> = allowed.iter().cloned().collect();
            list.sort();
            list
        },
    })
}

/// Validate stdio arguments for injection-prone patterns.
pub fn validate_stdio_args(command: &str, args: &[String]) -> Result<(), McpSecurityError> {
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];

        if arg.chars().any(|c| SHELL_METACHARACTERS.contains(&c)) {
            return Err(McpSecurityError::ShellMetacharacters(arg.clone()));
        }

        if is_dangerous_arg(command, arg) {
            return Err(McpSecurityError::DangerousArgument {
                command: command.to_string(),
                arg: arg.clone(),
            });
        }

        if requires_restricted_value(command, arg) {
            let value = args
                .get(index + 1)
                .ok_or_else(|| McpSecurityError::DangerousArgument {
                    command: command.to_string(),
                    arg: arg.clone(),
                })?;
            validate_restricted_arg_value(command, arg, value)?;
            index += 2;
            continue;
        }

        index += 1;
    }

    Ok(())
}

/// Validate stdio environment variables.
pub fn validate_stdio_env(
    env: &std::collections::HashMap<String, String>,
) -> Result<(), McpSecurityError> {
    for key in env.keys() {
        if DANGEROUS_ENV_VARS
            .iter()
            .any(|blocked| key.eq_ignore_ascii_case(blocked))
        {
            return Err(McpSecurityError::DangerousEnvVar(key.clone()));
        }
    }
    Ok(())
}

/// Validate stdio working directory exists and stays within the config base when provided.
pub fn validate_stdio_cwd(cwd: &str, config_base: Option<&Path>) -> Result<(), McpSecurityError> {
    let cwd_path = PathBuf::from(cwd);
    if !cwd_path.is_dir() {
        return Err(McpSecurityError::InvalidWorkingDirectory(format!(
            "cwd does not exist or is not a directory: {cwd}"
        )));
    }

    let canonical_cwd = cwd_path.canonicalize().map_err(|_| {
        McpSecurityError::InvalidWorkingDirectory(format!("cwd is not accessible: {cwd}"))
    })?;

    if let Some(base) = config_base {
        let canonical_base = base.canonicalize().map_err(|_| {
            McpSecurityError::InvalidWorkingDirectory(format!(
                "config base directory is not accessible: {}",
                base.display()
            ))
        })?;

        if !canonical_cwd.starts_with(&canonical_base) {
            return Err(McpSecurityError::InvalidWorkingDirectory(format!(
                "cwd must be inside the config directory ({})",
                canonical_base.display()
            )));
        }
    }

    Ok(())
}

/// Validate remote MCP HTTP(S) URLs and block SSRF to private networks by default.
pub fn validate_http_url(url: &str, allow_private_networks: bool) -> Result<(), McpSecurityError> {
    let parsed =
        url::Url::parse(url).map_err(|e| McpSecurityError::DisallowedHttpUrl(e.to_string()))?;

    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(McpSecurityError::DisallowedHttpUrl(format!(
                "url scheme must be http or https, got '{scheme}'"
            )));
        }
    }

    let host = parsed.host().ok_or_else(|| {
        McpSecurityError::DisallowedHttpUrl("url must include a host".to_string())
    })?;

    if allow_private_networks {
        return Ok(());
    }

    let host_label = host.to_string();
    if is_blocked_http_host(host) {
        return Err(McpSecurityError::DisallowedHttpUrl(format!(
            "connections to private or link-local hosts are not permitted: {host_label}"
        )));
    }

    Ok(())
}

fn is_blocked_http_host(host: url::Host<&str>) -> bool {
    match host {
        url::Host::Domain(name) => is_blocked_http_domain(name),
        url::Host::Ipv4(v4) => is_private_or_link_local_ip(IpAddr::V4(v4)),
        url::Host::Ipv6(v6) => is_private_or_link_local_ip(IpAddr::V6(v6)),
    }
}

fn is_blocked_http_domain(host: &str) -> bool {
    let host_lower = host.to_ascii_lowercase();
    if matches!(
        host_lower.as_str(),
        "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" | "[::1]"
    ) {
        return true;
    }

    if host_lower == "metadata.google.internal" {
        return true;
    }

    if let Ok(ip) = host.parse::<IpAddr>() {
        return is_private_or_link_local_ip(ip);
    }

    false
}

fn is_private_or_link_local_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.octets() == [169, 254, 169, 254]
        }
        IpAddr::V6(v6) => v6.is_loopback() || v6.is_unique_local() || v6.is_unicast_link_local(),
    }
}

fn is_dangerous_arg(command: &str, arg: &str) -> bool {
    let arg_lower = arg.to_ascii_lowercase();

    match command {
        "npx" | "npm" => matches!(arg_lower.as_str(), "-c" | "--call"),
        "node" | "deno" => matches!(
            arg_lower.as_str(),
            "-e" | "--eval" | "-p" | "--print" | "-r" | "--require" | "--import"
        ),
        "python" | "python3" => matches!(arg_lower.as_str(), "-c"),
        "docker" => matches!(
            arg_lower.as_str(),
            "--privileged" | "--cap-add" | "--network=host" | "--pid=host" | "--ipc=host"
        ),
        _ => false,
    }
}

fn requires_restricted_value(command: &str, arg: &str) -> bool {
    if command != "docker" {
        return false;
    }

    matches!(
        arg.to_ascii_lowercase().as_str(),
        "-v" | "--volume" | "--mount" | "--network" | "--pid" | "--ipc"
    )
}

fn validate_restricted_arg_value(
    command: &str,
    flag: &str,
    value: &str,
) -> Result<(), McpSecurityError> {
    if command != "docker" {
        return Ok(());
    }

    let flag_lower = flag.to_ascii_lowercase();
    let value_lower = value.to_ascii_lowercase();

    if matches!(flag_lower.as_str(), "--network" | "--pid" | "--ipc") && value_lower == "host" {
        return Err(McpSecurityError::DangerousArgument {
            command: command.to_string(),
            arg: format!("{flag} {value}"),
        });
    }

    if matches!(flag_lower.as_str(), "-v" | "--volume" | "--mount")
        && (value_lower.starts_with('/') || value_lower.contains(":/"))
    {
        return Err(McpSecurityError::DangerousArgument {
            command: command.to_string(),
            arg: format!("{flag} {value}"),
        });
    }

    Ok(())
}

/// Returns true when the value should be resolved relative to the config file directory.
pub fn looks_like_path(value: &str) -> bool {
    if value.is_empty() || value.starts_with('-') {
        return false;
    }

    let path = Path::new(value);
    if path.is_absolute() {
        return true;
    }

    if value.starts_with("./") || value.starts_with("../") {
        return true;
    }

    if value.ends_with(".py") || value.ends_with(".js") || value.ends_with(".ts") {
        return true;
    }

    if value.contains('/') || value.contains('\\') {
        return path.extension().is_some();
    }

    false
}

/// Resolve a relative path against the config file directory.
pub fn resolve_path(base_dir: &Path, value: &str) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        base_dir.join(path).to_string_lossy().to_string()
    }
}

/// Ensure a resolved path stays within the config base directory.
pub fn validate_resolved_path_within_base(
    path: &str,
    config_base: &Path,
) -> Result<(), McpSecurityError> {
    let candidate = PathBuf::from(path);
    let canonical_base = config_base.canonicalize().map_err(|_| {
        McpSecurityError::InvalidWorkingDirectory(format!(
            "config base directory is not accessible: {}",
            config_base.display()
        ))
    })?;

    let resolved = if candidate.exists() {
        candidate
            .canonicalize()
            .map_err(|_| McpSecurityError::DangerousArgument {
                command: "stdio".to_string(),
                arg: format!("path does not exist or is not accessible: {path}"),
            })?
    } else {
        normalize_lexical_path(&candidate)
    };

    if !resolved.starts_with(&canonical_base) {
        return Err(McpSecurityError::DangerousArgument {
            command: "stdio".to_string(),
            arg: format!("path escapes config directory: {path}"),
        });
    }

    Ok(())
}

fn normalize_lexical_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn effective_allowed_commands_includes_defaults() {
        let allowed = effective_allowed_commands().unwrap();
        assert!(allowed.contains("python3"));
        assert!(allowed.contains("npx"));
    }

    #[test]
    fn rejects_absolute_command_paths() {
        let allowed = effective_allowed_commands().unwrap();
        assert!(validate_command_allowlist("/usr/bin/python3", &allowed).is_err());
        assert!(validate_command_allowlist("../../bin/sh", &allowed).is_err());
        assert!(validate_command_allowlist("bin/python", &allowed).is_err());
    }

    #[test]
    fn allows_bare_launcher_names() {
        let allowed = effective_allowed_commands().unwrap();
        assert!(validate_command_allowlist("python3", &allowed).is_ok());
    }

    #[test]
    fn rejects_disallowed_commands() {
        let allowed = effective_allowed_commands().unwrap();
        assert!(validate_command_allowlist("bash", &allowed).is_err());
        assert!(validate_command_allowlist("sh", &allowed).is_err());
    }

    #[test]
    fn empty_allowlist_permits_any_bare_command() {
        let allowed = HashSet::new();
        assert!(validate_command_allowlist("bash", &allowed).is_ok());
    }

    #[test]
    fn rejects_dangerous_npx_args() {
        assert!(validate_stdio_args("npx", &["-c".to_string(), "id".to_string()]).is_err());
        assert!(validate_stdio_args("npx", &["-y".to_string(), "pkg".to_string()]).is_ok());
    }

    #[test]
    fn rejects_python_c_flag() {
        assert!(
            validate_stdio_args("python3", &["-c".to_string(), "print(1)".to_string()]).is_err()
        );
        assert!(
            validate_stdio_args("python3", &["-m".to_string(), "my_server".to_string()]).is_ok()
        );
    }

    #[test]
    fn rejects_shell_metacharacters_in_args() {
        assert!(validate_stdio_args("python3", &["foo;rm -rf /".to_string()]).is_err());
    }

    #[test]
    fn rejects_dangerous_env_vars() {
        let mut env = HashMap::new();
        env.insert("LD_PRELOAD".to_string(), "/evil.so".to_string());
        assert!(validate_stdio_env(&env).is_err());
    }

    #[test]
    fn rejects_private_http_hosts() {
        assert!(validate_http_url("http://127.0.0.1/mcp", false).is_err());
        assert!(validate_http_url("http://169.254.169.254/mcp", false).is_err());
        assert!(validate_http_url("https://api.example.com/mcp", false).is_ok());
    }

    #[test]
    fn allows_private_http_hosts_when_enabled() {
        assert!(validate_http_url("http://127.0.0.1/mcp", true).is_ok());
    }

    #[test]
    fn rejects_docker_space_separated_network_host() {
        assert!(
            validate_stdio_args("docker", &["--network".to_string(), "host".to_string()]).is_err()
        );
    }

    #[test]
    fn rejects_docker_privileged_args() {
        assert!(validate_stdio_args("docker", &["--privileged".to_string()]).is_err());
        assert!(validate_stdio_args("docker", &["-v".to_string(), "/:/mnt".to_string()]).is_err());
    }

    #[test]
    fn rejects_ipv6_link_local_http_hosts() {
        assert!(validate_http_url("http://[fe80::1]/mcp", false).is_err());
    }

    #[test]
    fn looks_like_path_distinguishes_flags_from_paths() {
        assert!(!looks_like_path("--format=json/xml"));
        assert!(looks_like_path("servers/echo_server.py"));
        assert!(looks_like_path("./local/script.py"));
    }

    #[test]
    fn validates_cwd_within_config_base() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("servers");
        std::fs::create_dir_all(&nested).unwrap();
        let cwd = nested.to_str().unwrap();
        validate_stdio_cwd(cwd, Some(dir.path())).unwrap();
    }
}
